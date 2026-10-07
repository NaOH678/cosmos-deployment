"""One joint sample per fixed case, shared by PointFlow/FK metrics and media."""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.callbacks.fk_eval import FKEvalCallback
from cosmos_framework.callbacks.pointflow_eval_cases import evaluation_rng
from cosmos_framework.callbacks.pointflow_visualize import render_case, stitch_windows
from cosmos_framework.utils import distributed, log, misc


def numpy(value):
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def sample_joint_case(model, batch, *, seed, steps):
    if not batch.get("pointflow") or not batch.get("fk"):
        raise ValueError("Unified eval requires both PointFlow and FK")
    pf, fk = batch["pointflow"][0], batch["fk"][0]
    if not np.array_equal(numpy(pf["metadata"]["raw_frame_ids"]), numpy(fk["metadata"]["raw_frame_ids"])):
        raise ValueError("Unified eval requires identical PointFlow/FK source frames")
    samples = model.generate_samples_from_batch(
        batch, guidance=1.0, seed=[seed], n_sample=1, has_negative_prompt=False, num_steps=steps
    )
    for key in ("vision", "action", "pointflow", "fk"):
        if len(samples.get(key, [])) != 1:
            raise ValueError(f"Unified eval expected one {key} result")
        if not torch.isfinite(samples[key][0]).all():
            raise FloatingPointError(f"Unified eval produced non-finite {key}")
    return samples


def attach_fk_overlay(record, fk, prediction, pointflow, *, dream_hw=None):
    # Reuse the established FK projector; PointFlow keeps its own DA3 intrinsics.
    from tools.verify_fk_camera_projection import IMG_H, IMG_W, project

    anchor = numpy(fk["inputs"]["anchor_xyz"])
    target, pred = numpy(fk["targets"]["displacement"]), numpy(prediction)
    if pred.shape != target.shape or len(record["frames_bgr"]) != len(pred) + 1:
        raise ValueError("FK overlay timing/shape differs from PointFlow video")
    valid = numpy(fk["targets"]["valid"]).astype(bool)
    valid = np.concatenate([valid[:1], valid])
    for name, displacement in (("gt", target), ("pred", pred)):
        xyz = anchor[None] + np.concatenate([np.zeros_like(anchor)[None], displacement])
        if "intrinsics_px" in fk.get("metadata", {}):
            k = numpy(fk["metadata"]["intrinsics_px"])
            projected = xyz.reshape(-1, 3) @ k.T
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = projected[:, :2] / projected[:, 2:]
            front = projected[:, 2] > 0
            source_wh = numpy(fk["metadata"]["image_size_wh"])
        else:
            uv, front = project(xyz.reshape(-1, 3))
            source_wh = np.array([IMG_W, IMG_H])
        uv, front = uv.reshape(*xyz.shape[:-1], 2), front.reshape(xyz.shape[:-1])
        height, width = record["frames_bgr"][0].shape[:2]
        if dream_hw is None:
            uv = (uv + 0.5) * np.array([width, height]) / source_wh - 0.5
        else:
            tracker_wh = numpy(pointflow["inputs"]["image_size_wh"])
            affine = numpy(pointflow["metadata"]["uv_to_video"])
            tracker_centers = (uv + 0.5) * tracker_wh / source_wh
            uv = (tracker_centers @ affine[:, :2].T + affine[:, 2]) * [width / dream_hw[1], height / dream_hw[0]] - 0.5
        record[f"fk_{name}_uv"] = uv
        record[f"fk_{name}_valid"] = valid & front & np.isfinite(uv).all(-1)


def make_fk_writer(config):
    writer = FKEvalCallback(joint_only=True, make_figures=False, make_video=False)
    # Callback constructors do not install runtime metadata; CallbackDict normally does.
    writer.config = config
    return writer


def run_joint_eval(callback, model, iteration):
    from cosmos_framework.callbacks.pointflow_eval import _decode_dream_frames, _write_video, make_preview

    root = Path(callback.config.job.path_local)
    step_root = root / "pointflow_eval" / f"step_{iteration:07d}"
    writer = make_fk_writer(callback.config)
    if distributed.is_rank0():
        fk_root = root / "fk_eval"
        fk_root.mkdir(parents=True, exist_ok=True)
        (fk_root / "fixed_cases.json").write_text(json.dumps(callback._fixed[0], indent=2) + "\n")
    media = []
    for identity, cpu_batch in zip(*callback._fixed, strict=True):
        seed, case_id = identity["seed"], identity["case_id"]
        log.info(f"Unified joint eval {case_id}: {callback.sampling_steps} steps, seed={seed}")
        with evaluation_rng(seed):
            batch = misc.to(copy.deepcopy(cpu_batch), device="cuda")
            samples = sample_joint_case(model, batch, seed=seed, steps=callback.sampling_steps)
        if distributed.is_rank0():
            pf, fk = cpu_batch["pointflow"][0], cpu_batch["fk"][0]
            pred, fk_pred = samples["pointflow"][0], samples["fk"][0]
            joint_id = dict(identity, case_id=case_id + "_joint")
            conditions = "shared joint video/action/PointFlow/FK sample; first frame and state conditioned"
            record = make_preview(pf, pred, 1.0, prediction_kind="joint_unipc_shared")
            record.update(seed=seed, sampler="unipc", sampling_steps=callback.sampling_steps, conditions=conditions)
            attach_fk_overlay(record, fk, fk_pred, pf)
            paths = callback._save_case(record, joint_id, None, iteration)
            writer._save_case(
                identity, cpu_batch, fk_pred, None, iteration, vision=samples["vision"], subtitle=conditions
            )
            directory = Path(paths["video"]).parent
            provenance = dict(
                identity,
                sampling_steps=callback.sampling_steps,
                sampling_mode="shared_joint",
                modalities=["video", "action", "pointflow", "fk"],
                sampling_calls=1,
            )
            (directory / "joint_sample.json").write_text(json.dumps(provenance, indent=2) + "\n")
            fk_directory = root / "fk_eval" / f"step_{iteration:07d}" / case_id
            (fk_directory / "joint_sample.json").write_text(json.dumps(provenance, indent=2) + "\n")
            np.savez_compressed(directory / "joint_prediction.npz", **{k: numpy(v[0]) for k, v in samples.items()})
            frames = _decode_dream_frames(model, samples)
            dream_dir = directory / "dream_canvas"
            dream_dir.mkdir(parents=True, exist_ok=True)
            _write_video(frames, dream_dir / "dream.mp4", float(record["fps"]))
            dream = make_preview(pf, pred, 1.0, prediction_kind="joint_unipc_shared_dream", frames_bgr_override=frames)
            dream.update(seed=seed, conditions=conditions)
            attach_fk_overlay(dream, fk, fk_pred, pf, dream_hw=frames[0].shape[:2])
            dream_paths, _ = render_case(dream, None, dream_dir, max_points=callback.max_points)
            if identity["split"] == "val":
                media.append((identity, paths["video"], dream_paths["video"]))
        del samples, batch
    if distributed.is_rank0():
        for episode in dict.fromkeys(row[0]["episode"] for row in media):
            prefix = f"{episode}_" if callback.val_episodes > 1 else ""
            for stage in ("early", "middle", "late"):
                clips = [row for row in media if row[0]["episode"] == episode and row[0]["stage"] == stage]
                for index, suffix in ((1, "joint"), (2, "joint_dream")):
                    stitch_windows([row[index] for row in clips], step_root / f"{prefix}{stage}_stitched_{suffix}.mp4")
