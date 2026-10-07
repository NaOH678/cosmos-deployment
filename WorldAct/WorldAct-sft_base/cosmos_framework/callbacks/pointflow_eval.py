"""PointFlow validation metrics and LingBot-style RGB comparison media."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from PIL import Image

from cosmos_framework.callbacks.pointflow_eval_cases import evaluation_rng, fixed_cases
from cosmos_framework.callbacks.pointflow_visualize import (
    point_video_grid_diagnostic,
    render_case,
    stage_viewer,
    stitch_windows,
    trajectory_metrics,
)
from cosmos_framework.data.pointflow_window import _episode_metadata, read_frame
from cosmos_framework.utils import callback, distributed, log, misc

# The labeled sandwich delivery (report.json) drops intrinsics.npy entirely: the
# tracker grid is fixed at 640x448 and the export pipeline's normalized
# intrinsics are constant across frames (measured on
# pf_out/sandwich/efep_labeled/episode_0013_20260731_133649/intrinsics.npy).
# Used ONLY for projecting preview overlays into pixels; metrics are computed in
# metric xyz and never touch this.
_LABELED_INTRINSICS = np.array(
    [[0.81309587, 0.0, 0.5], [0.0, 1.1615655, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32
)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def make_preview(
    sample,
    prediction,
    sigma,
    width=320,
    prediction_kind="one_step_denoising_estimate",
    canvas=None,
    frames_bgr_override=None,
):
    """Adapt one ragged Cosmos sample to the LingBot rendering contract.

    ``frames_bgr_override`` replaces the GT video decode (used to render joint
    rollouts on the model's OWN generated video: points dreamed together with
    pixels only overlay consistently on the dreamed canvas)."""
    metadata, inputs, targets = sample["metadata"], sample["inputs"], sample["targets"]
    source = Path(metadata["source_path"])
    video, tracker_w, tracker_h = _episode_metadata(source)
    frame_ids = _numpy(metadata["raw_frame_ids"]).astype(int)
    source_ids = np.load(source / "frame_indices.npy", allow_pickle=False)
    rows = np.searchsorted(source_ids, frame_ids)
    if np.any(rows >= len(source_ids)) or not np.array_equal(source_ids[rows], frame_ids):
        raise ValueError("PointFlow preview frames do not match tracker source")
    ids = _numpy(inputs["point_ids"]).astype(int)
    xyz0 = _numpy(inputs["anchor_xyz"]).astype(np.float32)
    future_gt = _numpy(targets["displacement"]).astype(np.float32)
    prediction = _numpy(prediction).astype(np.float32)
    if prediction.shape != future_gt.shape or len(frame_ids) != len(prediction) + 1:
        raise ValueError("PointFlow preview must have H predictions and H+1 raw frames")
    zero = np.zeros_like(xyz0)[None]
    flow, gt = np.concatenate([zero, prediction]), np.concatenate([zero, future_gt])
    valid = np.concatenate([np.ones((1, len(ids)), bool), _numpy(targets["valid"]).astype(bool)])
    moving = np.max(np.where(valid, np.linalg.norm(gt, axis=-1), 0), axis=0) >= 0.01
    intrinsics_path = source / "intrinsics.npy"
    frames, gt_uv, pred_uv = [], [], []
    if frames_bgr_override is not None and len(frames_bgr_override) != len(frame_ids):
        raise ValueError(
            f"Dream canvas frames {len(frames_bgr_override)} != window frames {len(frame_ids)}"
        )
    cap = None if frames_bgr_override is not None else cv2.VideoCapture(video)
    try:
        if cap is not None and not cap.isOpened():
            raise FileNotFoundError(video)
        for t, (raw_id, row) in enumerate(zip(frame_ids, rows, strict=True)):
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(raw_id))
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode preview frame {raw_id}")
            else:
                bgr = frames_bgr_override[t]
            height = max(2, round(bgr.shape[0] * width / bgr.shape[1] / 2) * 2)
            frames.append(cv2.resize(bgr, (width, height)))
            uv = read_frame(source / "uv_px.npy", int(row)).reshape(-1, 2)[ids]
            k = read_frame(intrinsics_path, int(row)) if intrinsics_path.is_file() else _LABELED_INTRINSICS
            xyz = xyz0 + flow[t]
            projected = xyz @ k.T
            with np.errstate(divide="ignore", invalid="ignore"):
                pixels = projected[:, :2] / projected[:, 2:3]
            tracker_wh = np.array([tracker_w, tracker_h])
            panel_wh = np.array([width, height])
            if frames_bgr_override is not None:
                # The dream canvas is the COMPOSED view (wrist strip on top + head
                # below), not the head-only tracker frame: route uv through the
                # recorded tracker->video affine, then scale canvas -> panel.
                affine = _numpy(metadata["uv_to_video"]).astype(np.float64)
                canvas_wh = _numpy(metadata["video_size_wh"]).astype(np.float64)
                panel_scale = panel_wh / canvas_wh
                gt_uv.append(((uv + 0.5) @ affine[:, :2].T + affine[:, 2]) * panel_scale - 0.5)
                pred_canvas = (pixels * tracker_wh + 0.5) @ affine[:, :2].T + affine[:, 2]
                pred_uv.append(pred_canvas * panel_scale - 0.5)
            else:
                gt_uv.append((uv + 0.5) * panel_wh / tracker_wh - 0.5)
                pred_uv.append((pixels * tracker_wh + 0.5) * panel_wh / tracker_wh - 0.5)
    finally:
        if cap is not None:
            cap.release()
    record = dict(
        flow=flow,
        flow_gt=gt,
        xyz0=xyz0,
        valid=valid,
        moving=moving,
        query_uv=_numpy(inputs["anchor_uv"]),
        point_ids=ids,
        raw_frame_ids=frame_ids,
        sigma=float(sigma),
        fps=metadata["timing"].fps,
        prediction_kind=prediction_kind,
        frames_bgr=frames,
        gt_uv=np.asarray(gt_uv),
        pred_uv=np.asarray(pred_uv),
    )
    if canvas is not None:
        # The overlay panels above read the tracker's own uv_px and re-project the
        # 3D labels, so they cannot show a canvas mismatch; this one replays the
        # model's own coordinate arithmetic on the model's own video.
        panel, diagnostics = point_video_grid_diagnostic(canvas, metadata["uv_to_video"], inputs["anchor_uv"])
        record["position_panel"] = panel
        record["position_diagnostics"] = diagnostics
    return record


def zero_actions_(data_batch):
    """Zero every action tensor in the batch, in place.

    The design premise is that point tokens are the bridge between video and action,
    and this eval hands the model the *ground-truth* future action.  So a point
    prediction that does not move when the action is zeroed is not reading it: the
    correct answer was sitting in the context and changed nothing.

    ``data_batch["action"]`` is the dataloader's nested ``[[tensor], [None], ...]``
    form (see ``OmniMoTModel._normalize_action_databatch``), hence the recursion.
    """

    def walk(value):
        if isinstance(value, torch.Tensor):
            return value.zero_()
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, tuple):
            return tuple(walk(item) for item in value)
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    if data_batch.get("action") is None:
        raise ValueError("Cannot ablate the action: the batch carries no 'action'")
    data_batch["action"] = walk(data_batch["action"])


def _walk_video(data_batch, fn):
    """Apply ``fn`` to every video tensor in the batch, in place.

    Handles both a single tensor and the nested list forms; the time axis is -3
    in both [C,T,H,W] and [B,C,T,H,W] layouts.
    """

    def walk(value):
        if isinstance(value, torch.Tensor):
            if value.ndim < 4:
                raise ValueError(f"Expected video tensor [...,C,T,H,W], got {tuple(value.shape)}")
            fn(value)
            return value
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, tuple):
            return tuple(walk(item) for item in value)
        return value

    if data_batch.get("video") is None:
        raise ValueError("Cannot ablate the video: the batch carries no 'video'")
    data_batch["video"] = walk(data_batch["video"])


def zero_video_(data_batch):
    """Black out every frame: no observation content reaches the VAE at all."""
    _walk_video(data_batch, lambda v: v.zero_())
    _drop_latent_cache(data_batch)


def freeze_video_first_frame_(data_batch):
    """Repeat frame 0 for the whole window: the first frame stays a clean condition,
    future frames carry no information (a static scene)."""
    _walk_video(data_batch, lambda v: v.copy_(v.narrow(-3, 0, 1).expand_as(v)))
    _drop_latent_cache(data_batch)


def _drop_latent_cache(data_batch):
    """Pixel-space video ablations must also drop the precomputed window latents:
    `get_data_and_condition` prefers `vae_latent_cache` over encoding the raw video,
    so without this the model still sees the un-ablated latents and the ablation is
    a silent no-op (measured: video/first_frame dependence exactly 0.00)."""
    data_batch.pop("vae_latent_cache", None)


ABLATION_TRANSFORMS = {"action": zero_actions_, "video": zero_video_, "first_frame": freeze_video_first_frame_}


def _decode_dream_frames(model, samples):
    """Decode the jointly generated video to BGR uint8 frames (the dream canvas)."""
    vision = samples.get("vision")
    if not vision:
        return None
    decoded = model.decode(vision[0])
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if decoded.ndim == 5:
        decoded = decoded[0]
    if decoded.ndim != 4:
        raise ValueError(f"Unexpected decoded dream shape: {tuple(decoded.shape)}")
    frames = (decoded.float().cpu().clamp(-1, 1).permute(1, 2, 3, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]


def _write_video(frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class PointFlowEvalCallback(callback.Callback):
    """Fixed fitting/generalization cases; all ranks participate in sampling."""

    def __init__(
        self,
        every_n=1,
        max_points=128,
        max_cases=2,
        sampling_steps=16,
        comparison_steps=32,
        sampler="unipc",
        comparison_sampler="euler",
        seed=42,
        val_stage_fractions=(0.2, 0.5, 0.8),
        ablate_action=False,
        ablate_modes="",
        joint_rollout=False,
        sigma_scan=False,
        config=None,
        trainer=None,
    ):
        super().__init__(config, trainer)
        self.every_n, self.max_points, self.max_cases = int(every_n), int(max_points), int(max_cases)
        self.sampling_steps, self.comparison_steps, self.seed = int(sampling_steps), int(comparison_steps), int(seed)
        # The main trajectory must use the solver the deployed system runs (UniPC, the
        # same loop as video/action); the comparison keeps the legacy Euler reference so
        # one eval answers "was the solver the problem" directly.
        self.sampler, self.comparison_sampler = str(sampler), str(comparison_sampler)
        self.val_stage_fractions = tuple(float(value) for value in val_stage_fractions)
        # `${oc.env:...}` yields a string, so "false" would be truthy without this.
        if str(ablate_action).lower() == "true" if isinstance(ablate_action, str) else bool(ablate_action):
            ablate_modes = f"{ablate_modes},action"
        # Comma-separated ablation passes, each an extra sampling of every case with
        # one conditioning channel neutralized (see ABLATION_TRANSFORMS):
        #   action      - zero the GT action (does the branch read it at all?)
        #   video       - black out every frame (does it read ANY video content?)
        #   first_frame - freeze the window to frame 0 (does it read FUTURE frames?
        #                 in wam training those are noisy targets, never clean context)
        modes = [m.strip() for m in str(ablate_modes).split(",") if m.strip()]
        unknown = [m for m in modes if m not in ABLATION_TRANSFORMS]
        if unknown:
            raise ValueError(f"Unknown ablation modes {unknown}; known: {sorted(ABLATION_TRANSFORMS)}")
        self.ablate_modes = modes
        # Route B: also evaluate the deployment-shaped condition — point tokens
        # denoised inside the MAIN video/action sampling loop (first frame + state
        # action clean, future video/action/point jointly sampled), one extra
        # generate_samples_from_batch call per case.
        self.joint_rollout = (
            str(joint_rollout).lower() == "true" if isinstance(joint_rollout, str) else bool(joint_rollout)
        )
        # Diagnostic: sweep the one-step clean estimate over sigma.  The sampled
        # trajectory ends at sigma = 0, so where the direct estimate breaks matters more
        # than its value at the sigma the batch happened to draw.
        self.sigma_scan = str(sigma_scan).lower() == "true" if isinstance(sigma_scan, str) else bool(sigma_scan)
        if min(self.every_n, self.max_points, self.max_cases, self.sampling_steps, self.comparison_steps) <= 0:
            raise ValueError("PointFlow eval limits must be positive")
        self._fixed = None
        self._done = False
        self._compared = False

    def on_validation_start(self, model, dataloader, iteration=0):
        self._done = False

    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration % self.every_n or self._done:
            return
        value = output_batch.get("flow_matching_loss_pointflow")
        if value is not None and distributed.is_rank0() and wandb.run is not None:
            wandb.log({"pointflow/val_loss": float(value.detach().cpu())}, step=iteration)
        # Called inside the trainer's eval/EMA context, on EVERY rank. Do not
        # launch FSDP forwards only on rank zero.
        with evaluation_rng(self.seed):
            if self._fixed is None:
                self._fixed = fixed_cases(
                    self.config, count=self.max_cases, seed=self.seed, val_stage_fractions=self.val_stage_fractions
                )
                if distributed.is_rank0():
                    root = Path(self.config.job.path_local) / "pointflow_eval"
                    root.mkdir(parents=True, exist_ok=True)
                    temporary = root / "fixed_cases_stages_4windows.tmp.json"
                    temporary.write_text(json.dumps(self._fixed[0], indent=2) + "\n")
                    temporary.replace(root / "fixed_cases_stages_4windows.json")
            stage_media = []
            joint_stage_media = []
            dream_stage_media = []
            for identity, cpu_batch in zip(*self._fixed, strict=True):
                case_id, seed = identity["case_id"], identity["seed"]
                log.info(f"PointFlow eval {case_id}: {self.sampler} {self.sampling_steps} steps, seed={seed}")
                with evaluation_rng(seed):
                    batch = misc.to(copy.deepcopy(cpu_batch), device="cuda")
                    prediction = model.sample_pointflow(
                        batch, steps=self.sampling_steps, seed=seed, sampler=self.sampler
                    )
                ablated_flows = {}
                for ablate_mode in self.ablate_modes:
                    # Same seed, so the start noise is identical and the only difference
                    # from `prediction` is that one conditioning channel carries nothing.
                    ablated_batch = misc.to(copy.deepcopy(cpu_batch), device="cuda")
                    ABLATION_TRANSFORMS[ablate_mode](ablated_batch)
                    with evaluation_rng(seed):
                        ablated_flows[ablate_mode] = model.sample_pointflow(
                            ablated_batch, steps=self.sampling_steps, seed=seed, sampler=self.sampler
                        )
                scan = None
                if self.sigma_scan:
                    with evaluation_rng(seed):
                        scan = model.pointflow_sigma_scan(
                            misc.to(copy.deepcopy(cpu_batch), device="cuda"), seed=seed
                        )
                joint_samples = None
                if self.joint_rollout:
                    # Route B: point tokens denoise inside the main video/action loop.
                    # The wam plan keeps the first frame and state action clean; future
                    # video/action and point displacements are jointly sampled.
                    joint_batch = misc.to(copy.deepcopy(cpu_batch), device="cuda")
                    with evaluation_rng(seed):
                        joint_samples = model.generate_samples_from_batch(
                            joint_batch,
                            guidance=1.0,
                            seed=[seed],
                            n_sample=1,
                            has_negative_prompt=False,
                            num_steps=self.sampling_steps,
                        )
                reference = None
                if not self._compared:
                    with evaluation_rng(seed):
                        reference = model.sample_pointflow(
                            misc.to(copy.deepcopy(cpu_batch), device="cuda"), steps=self.comparison_steps, seed=seed,
                            sampler=self.comparison_sampler,
                        )
                if distributed.is_rank0():
                    sample = cpu_batch["pointflow"][0]
                    times = _numpy(sample["metadata"]["timestamps_sec"])
                    videos = cpu_batch.get("video") or cpu_batch.get("images")
                    record = make_preview(
                        sample,
                        prediction,
                        1.0,
                        prediction_kind=f"{self.sampler}_{self.sampling_steps}_pure_noise",
                        canvas=videos[0] if videos else None,
                    )
                    record.update(
                        seed=seed,
                        sampler=self.sampler,
                        comparison_sampler=self.comparison_sampler,
                        sampling_steps=self.sampling_steps,
                        conditions="clean GT video/action + text + anchor geometry; conditional fit, not rollout",
                    )
                    if scan is not None:
                        record["sigma_scan"] = scan
                    for ablate_mode, ablated in ablated_flows.items():
                        # Shaped like `flow`: the anchor step is zero by construction.
                        record[f"flow_ablated_{ablate_mode}"] = np.concatenate(
                            [np.zeros_like(record["xyz0"])[None], _numpy(ablated)]
                        )
                    paths = self._save_case(record, identity, reference, iteration)
                    if joint_samples is not None:
                        joint_prediction = joint_samples["pointflow"][0]
                        joint_record = make_preview(
                            sample,
                            joint_prediction,
                            1.0,
                            prediction_kind=f"joint_{self.sampler}_{self.sampling_steps}",
                            canvas=videos[0] if videos else None,
                        )
                        joint_record.update(
                            seed=seed,
                            sampler=self.sampler,
                            comparison_sampler=self.comparison_sampler,
                            sampling_steps=self.sampling_steps,
                            conditions="first frame + state action clean; future video/action/point joint rollout",
                        )
                        joint_paths = self._save_case(
                            joint_record, dict(identity, case_id=f"{case_id}_joint"), None, iteration
                        )
                        if identity["split"] == "val":
                            times = _numpy(sample["metadata"]["timestamps_sec"])
                            joint_stage_media.append(
                                dict(
                                    stage=identity["stage"],
                                    video=joint_paths["video"],
                                    start_time=float(times[0]),
                                    end_time=float(times[-1]),
                                    label=f"{identity['stage']} | joint rollout | frame {identity['start_frame']}",
                                )
                            )
                        # Dream-canvas view: the joint rollout's points were generated
                        # against the model's OWN dreamed video, not the GT video, so
                        # overlaying them on GT frames measures dream-vs-reality, not
                        # joint consistency.  Render them again on the dreamed frames.
                        dream_frames = _decode_dream_frames(model, joint_samples)
                        if dream_frames is not None:
                            dream_dir = Path(joint_paths["video"]).parent / "dream_canvas"
                            dream_dir.mkdir(parents=True, exist_ok=True)
                            _write_video(dream_frames, dream_dir / "dream.mp4", float(joint_record["fps"]))
                            log.info(
                                f"PointFlow joint dream canvas: frame {dream_frames[0].shape[1]}x{dream_frames[0].shape[0]}, "
                                f"affine canvas {sample['metadata']['video_size_wh']}"
                            )
                            dream_record = make_preview(
                                sample,
                                joint_prediction,
                                1.0,
                                prediction_kind=f"joint_{self.sampler}_{self.sampling_steps}_dream_canvas",
                                frames_bgr_override=dream_frames,
                            )
                            dream_record.update(
                                seed=seed,
                                fps=joint_record["fps"],
                                conditions="joint rollout overlaid on the DREAMED video (internal consistency)",
                            )
                            dream_paths, _ = render_case(dream_record, None, dream_dir, max_points=self.max_points)
                            if identity["split"] == "val":
                                dream_stage_media.append(
                                    dict(
                                        stage=identity["stage"],
                                        video=dream_paths["video"],
                                        start_time=float(times[0]),
                                        end_time=float(times[-1]),
                                        label=f"{identity['stage']} | joint dream canvas | frame {identity['start_frame']}",
                                    )
                                )
                    if identity["split"] == "val":
                        stage_media.append(
                            dict(
                                stage=identity["stage"],
                                video=paths["video"],
                                start_time=float(times[0]),
                                end_time=float(times[-1]),
                                label=f"{identity['stage']} | {identity['episode']} | "
                                f"frame {identity['start_frame']} | {times[0]:.2f}-{times[-1]:.2f}s",
                            )
                        )
                del prediction, reference, batch
            if distributed.is_rank0():
                output = Path(self.config.job.path_local) / "pointflow_eval" / f"step_{iteration:07d}"
                stitched = []
                for stage in ("early", "middle", "late"):
                    clips = [item for item in stage_media if item["stage"] == stage]
                    video = stitch_windows([item["video"] for item in clips], output / f"{stage}_stitched.mp4")
                    stitched.append(
                        dict(
                            stage=stage,
                            kind="conditional",
                            video=video,
                            label=f"{clips[0]['label']} | full interval "
                            f"{clips[0]['start_time']:.2f}-{clips[-1]['end_time']:.2f}s | 4 windows",
                        )
                    )
                    for suffix, media in (("joint", joint_stage_media), ("joint_dream", dream_stage_media)):
                        clips = [item for item in media if item["stage"] == stage]
                        if not clips:
                            continue
                        video = stitch_windows(
                            [item["video"] for item in clips], output / f"{stage}_stitched_{suffix}.mp4"
                        )
                        stitched.append(
                            dict(
                                stage=stage,
                                kind=suffix,
                                video=video,
                                label=f"{clips[0]['label']} | {clips[0]['start_time']:.2f}"
                                f"-{clips[-1]['end_time']:.2f}s | 4 windows",
                            )
                        )
                viewer = stage_viewer(stitched, output / "validation_stages.html")
                if wandb.run is not None:
                    wandb.log(
                        {"pointflow/val_stages": wandb.Html(Path(viewer).read_text(), inject=False)}, step=iteration
                    )
        self._done = True
        self._compared = True

    def _save_case(self, record, identity, reference, iteration):
        from cosmos_framework.model.generator.pointflow_attention import default_attention_mode

        case_id = identity["case_id"]
        directory = Path(self.config.job.path_local) / "pointflow_eval" / f"step_{iteration:07d}" / case_id
        directory.mkdir(parents=True, exist_ok=True)
        metrics = trajectory_metrics(record["flow"], record["flow_gt"], record["valid"], record["moving"])
        # Alias the main trajectory under its sampler name too: bare ``all_ade_mm`` does
        # not say which solver produced it, while every comparison key already carries
        # its own (``euler16_all_ade_mm``).  The bare names stay for compatibility.
        metrics.update({f"{self.sampler}{self.sampling_steps}_" + k: v for k, v in metrics.items()})
        baseline = trajectory_metrics(
            np.zeros_like(record["flow"]), record["flow_gt"], record["valid"], record["moving"]
        )
        metrics.update({"zero_" + k: v for k, v in baseline.items()})
        # Keys are already ``pointflow_sigma_scan_*``; they are scalars, so the wandb
        # block below picks them up without needing "loss" in the name.
        metrics.update(record.get("sigma_scan", {}))
        static = record["valid"][1:] & ~record["moving"][None]
        if static.any():
            metrics["static_drift_mm"] = float(np.linalg.norm(record["flow"][1:], axis=-1)[static].mean() * 1000)
        if reference is not None:
            ref = np.concatenate([np.zeros_like(record["xyz0"])[None], _numpy(reference)])
            record["flow_comparison"] = ref
            record["comparison_steps"] = self.comparison_steps
            metrics.update(
                {
                    f"{self.comparison_sampler}{self.comparison_steps}_" + k: v
                    for k, v in trajectory_metrics(ref, record["flow_gt"], record["valid"], record["moving"]).items()
                }
            )
            mask = record["valid"][1:]
            if mask.any():
                metrics["sampling_difference_mm"] = float(
                    np.linalg.norm(ref[1:] - record["flow"][1:], axis=-1)[mask].mean() * 1000
                )
        for ablate_mode in self.ablate_modes:
            key = f"flow_ablated_{ablate_mode}"
            if key not in record:
                continue
            ablated_flow = record[key]
            metrics.update(
                {
                    f"ablation_{ablate_mode}_" + k: v
                    for k, v in trajectory_metrics(
                        ablated_flow, record["flow_gt"], record["valid"], record["moving"]
                    ).items()
                }
            )
            mask = record["valid"][1:]
            if mask.any():
                moved = np.linalg.norm(ablated_flow[1:] - record["flow"][1:], axis=-1)[mask]
                scale = np.linalg.norm(record["flow"][1:], axis=-1)[mask].mean()
                metrics[f"{ablate_mode}_ablation_delta_mm"] = float(moved.mean() * 1000)
                # ~0 means neutralizing the channel did not move the prediction at
                # all, i.e. the branch is not reading it.
                metrics[f"{ablate_mode}_ablation_dependence"] = float(moved.mean() / max(scale, 1e-9))
        np.savez_compressed(
            directory / "prediction.npz",
            **{k: v for k, v in record.items() if k not in ("frames_bgr", "position_panel")},
        )
        paths, diagnostics = render_case(record, None, directory, max_points=self.max_points)
        paths["error_curve"] = save_error_curve(record, directory)
        metrics.update(diagnostics)
        if "position_panel" in record:
            position_path = directory / "position_grid.png"
            Image.fromarray(cv2.cvtColor(record["position_panel"], cv2.COLOR_BGR2RGB)).save(position_path)
            paths["position_grid"] = str(position_path)
            metrics.update(record["position_diagnostics"])
        metrics.update(
            prediction_kind=record["prediction_kind"],
            episode=identity["episode"],
            conditions=record["conditions"],
            seed=identity["seed"],
            # Name the position rule that actually ran, not the one the design
            # intends: the pairwise reference attention is opt-in and a run that
            # leaves it off is the legacy all-modalities mRoPE ablation.
            attention_mode=default_attention_mode(),
        )
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if wandb.run is not None:
            caption = f"{case_id}: {identity['episode']}; {record['prediction_kind']}; clean GT video/action"
            table = wandb.Table(
                columns=["case", "comparison", "error_map", "position_grid", "video", "ade_mm", "zero_ade_mm"]
            )
            comparison = wandb.Image(paths["comparison"], caption=caption)
            error = wandb.Image(paths["error_map"], caption=caption)
            video = wandb.Video(paths["video"], format="mp4")
            position = (
                wandb.Image(paths["position_grid"], caption="point tokens on the model's video canvas")
                if "position_grid" in paths
                else None
            )
            table.add_data(
                case_id,
                comparison,
                error,
                position,
                video,
                metrics.get("all_ade_mm"),
                metrics.get("zero_all_ade_mm"),
            )
            wandb.log(
                {
                    f"pointflow/{case_id}/comparison": comparison,
                    f"pointflow/{case_id}/error_map": error,
                    f"pointflow/{case_id}/video": video,
                    f"pointflow/{case_id}/error_curve": wandb.Image(paths["error_curve"]),
                    f"pointflow/{case_id}/position_grid": position,
                    f"pointflow/{case_id}/table": table,
                    **{f"pointflow/{case_id}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                },
                step=iteration,
            )
        log.info(f"PointFlow eval visualization saved: {directory}")
        return paths


def save_error_curve(record, directory):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 4))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    valid = record["valid"][1:]
    times = np.arange(1, len(record["flow"])) / record["fps"]
    # Label every curve with the solver that produced it; the comparison used to be
    # hard-coded to "Euler", which would lie as soon as the two sides were swapped.
    main = f"{record.get('sampler', 'pred')} {record.get('sampling_steps', '')}".strip()
    predictions = [(main, record["flow"]), ("Zero", np.zeros_like(record["flow"]))]
    if "flow_comparison" in record:
        other = f"{record.get('comparison_sampler', 'euler')} {record['comparison_steps']}".strip()
        predictions.append((other, record["flow_comparison"]))
    curves = {"seconds": times.tolist()}
    for name, prediction in predictions:
        errors = np.linalg.norm(prediction[1:] - record["flow_gt"][1:], axis=-1) * 1000
        for group, subset in (("moving", record["moving"]), ("static", ~record["moving"])):
            mask = valid & subset[None]
            counts = mask.sum(1)
            means = np.divide(
                np.where(mask, errors, 0).sum(1), counts, out=np.full(len(times), np.nan), where=counts > 0
            )
            ax.plot(times, means, label=f"{name} {group}", linestyle="--" if name == "Zero" else "-")
            curves[f"{name}_{group}_mm"] = [None if not np.isfinite(x) else float(x) for x in means]
    ax.set(
        xlabel="Prediction horizon (s)",
        ylabel="Position error (mm)",
        title="Fixed conditions; pure-noise PointFlow generation",
    )
    ax.legend(fontsize=8)
    figure.tight_layout()
    path = Path(directory) / "error_curve.png"
    figure.savefig(path)
    (Path(directory) / "error_curve.json").write_text(json.dumps(curves, indent=2) + "\n")
    return str(path)
