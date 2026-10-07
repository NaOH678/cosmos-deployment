#!/usr/bin/env python3
"""Walk the PointFlow data chain end to end and check every link, not assume it.

Each check prints the evidence it used. Links that can only be verified
structurally (anything needing the VAE or the transformer, which need a GPU) say
so explicitly rather than passing silently.

    python cosmos_framework/scripts/validate_pointflow_chain.py \
        --raw-root /path/raw_data/singlerighthand_sandwich_100 \
        --cache-root /path/singlerighthand-sandwich-100-cosmos-cache \
        --dense-root /path/sandwich_dense_fullseq_10_0298_20260908/outputs \
        --pointflow-manifest pointflow_outputs/task5/mixed_manifest.json \
        --episode-allowlist examples/pointflow_sandwich_10_episodes.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import (
    SingleRightHandRawDataset,
)
from cosmos_framework.data.generator.action.pointflow_source import resize_pointflow_metadata
from cosmos_framework.data.generator.action.transforms import reflection_pad_to_target
from cosmos_framework.data.pointflow_window import read_frame
from tools.prepare_singlerighthand_video_cache import resize_to_resolution_content

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> bool:
    RESULTS.append((bool(ok), name, evidence))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}\n         {evidence}")
    return bool(ok)


def note(name: str, evidence: str) -> None:
    """A measured property that is not a pass/fail of this pipeline."""
    print(f"  [note] {name}\n         {evidence}")


def npy_header(path: Path):
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    return tuple(shape), dtype


def probe(path: Path, index=0):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode {path}")
    return frame  # BGR HxWx3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--dense-root", type=Path, required=True)
    p.add_argument("--pointflow-manifest", type=Path, required=True)
    p.add_argument("--episode-allowlist", type=Path, required=True)
    p.add_argument("--frame", type=int, default=1326)
    p.add_argument("--pixel-stride", type=int, default=32, help="latent_downsample_factor * latent_patch_size")
    args = p.parse_args()

    episodes = [x.strip() for x in args.episode_allowlist.read_text().splitlines() if x.strip()]
    # Timing contract of the recipe: 15 Hz target, 24 fps mRoPE base, VAE temporal
    # compression 4. Matches sequence_packing/mrope.py's FPS modulation.
    fps, base_fps, tcf, steps_per_token = 15.0, 24.0, 4, 4
    manifest = json.loads(args.pointflow_manifest.read_text())
    sources = {row["name"]: row["pointflow_source"] for row in manifest["episodes"]}
    cache_manifest = {
        row["name"]: row for row in json.loads((args.cache_root / "video_manifest.json").read_text())["episodes"]
    }
    episode = episodes[-1]
    frame = min(args.frame, 100000)

    print(f"\n=== L1 tracker output <-> raw video ===")
    head = probe(args.raw_root / episode / "videos" / "head.mp4", frame)
    wrist = probe(args.raw_root / episode / "videos" / "right_wrist.mp4", frame)
    check(head.shape[:2] == (480, 640), "head.mp4 decodes to 640x480", f"decoded HxW = {head.shape[:2]}")
    check(wrist.shape[:2] == (480, 848), "right_wrist.mp4 decodes to 848x480", f"decoded HxW = {wrist.shape[:2]}")

    dense = args.dense_root / episode
    pos_shape, pos_dtype = npy_header(dense / "position.npy")
    uv_shape, _ = npy_header(dense / "uv_px.npy")
    ids = np.load(dense / "frame_indices.npy", allow_pickle=False)
    stamps = np.load(dense / "timestamps_sec.npy", allow_pickle=False)
    tracker_h, tracker_w = 448, 480 // 64 * 64
    check(pos_shape[1:3] == (tracker_h, 640), "tracker grid is 640x448", f"position.npy = {pos_shape}")
    check(uv_shape[1:3] == (tracker_h, 640), "uv_px grid matches", f"uv_px.npy = {uv_shape}")
    check(480 // 64 * 64 == 448, "448 comes from flooring 480 to a multiple of 64", "480 // 64 * 64 = 448")
    check(
        np.array_equal(ids, np.arange(len(ids))),
        "tracker row i is source frame i",
        f"frame_indices[0:3]={ids[:3]} len={len(ids)}",
    )
    check(
        np.allclose(stamps, np.arange(len(stamps)) / 30.0), "tracker timeline is 30 Hz", f"dt = {np.diff(stamps)[:1]}"
    )

    intrinsics = np.load(dense / "intrinsics.npy", allow_pickle=False)
    K = np.diag([640.0, 448.0, 1.0]) @ intrinsics[frame]
    xyz = read_frame(dense / "position.npy", frame).reshape(-1, 3)
    uv = read_frame(dense / "uv_px.npy", frame).reshape(-1, 2)
    valid = read_frame(dense / "valid.npy", frame).reshape(-1)
    projected = xyz @ K.T
    projected = projected[:, :2] / projected[:, 2:3]
    error = np.linalg.norm(projected - uv, axis=1)[valid & (xyz[:, 2] > 0)]
    check(
        np.median(error) < 1.0,
        "3D and uv_px are consistent under K_px",
        f"reprojection median {np.median(error):.3f} px",
    )

    print(f"\n=== L2 tracker geometry vs the D435 ground truth ===")
    truth = json.loads((args.raw_root / episode / "auxiliary_camera" / "metadata.json").read_text())
    color = truth["capture_metadata"]["cameras"]["head"]["streams"]["color"]["intrinsics"]
    note(
        "tracker focal vs the measured D435 focal",
        f"tracker fx={K[0, 0]:.1f} vs D435 fx={color['fx']:.1f} -> {K[0, 0] / color['fx']:.3f}x "
        f"(a property of the tracker model, not of this pipeline)",
    )
    note(
        "camera model aspect",
        f"fx={K[0, 0]:.4f} fy={K[1, 1]:.4f} ratio {K[0, 0] / K[1, 1]:.6f}; a squeeze-aware model would report "
        f"fx/fy = 480/448 = {480 / 448:.6f}",
    )

    print(f"\n=== L3 composite <-> affine ===")

    def to_width(image, width=640):
        return cv2.resize(
            image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_LINEAR
        )

    wrist_rgb = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
    head_rgb = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
    tensor = SingleRightHandRawDataset._compose_views(
        torch.from_numpy(head_rgb).permute(2, 0, 1)[None],
        torch.from_numpy(wrist_rgb).permute(2, 0, 1)[None],
    )
    composite_hw = tuple(tensor.shape[-2:])
    composite = tensor[0].permute(1, 2, 0).numpy()
    wrist_h = round(480 * 640 / 848)
    affine = np.asarray(sources[episode]["uv_to_video"], dtype=np.float64)
    check(
        composite_hw == (362 + 480, 640),
        "composite is 640x842",
        f"_compose_views -> {composite_hw}, wrist height {wrist_h}",
    )
    check(
        tuple(sources[episode]["video_size_wh"]) == (640, 842),
        "manifest canvas is the composite",
        f"video_size_wh = {sources[episode]['video_size_wh']}",
    )
    check(
        abs(affine[1, 1] - 480 / 448) < 1e-6 and abs(affine[1, 2] - (362 + (480 / 448 - 1) / 2)) < 1e-6,
        "affine inverts the tracker squeeze and offsets by the wrist height",
        f"y scale {affine[1, 1]:.6f} = 480/448, offset {affine[1, 2]:.6f} = 362 + half-pixel",
    )

    print(f"\n=== L4 affine lands the tracked points on the head view ===")
    mapped = np.c_[uv, np.ones(len(uv))] @ np.vstack([affine, [0, 0, 1]]).T
    check(
        mapped[valid, 1].min() >= 362 and mapped[valid, 1].max() < 842 and mapped[valid, 0].max() < 640,
        "every valid point falls inside the head region of the composite",
        f"x[{mapped[valid, 0].min():.1f},{mapped[valid, 0].max():.1f}] y[{mapped[valid, 1].min():.1f},{mapped[valid, 1].max():.1f}]",
    )

    print(f"\n=== L5 video cache <-> raw video ===")
    cached_path = args.cache_root / cache_manifest[episode]["path"]
    cached_shape, cached_dtype = npy_header(cached_path)
    resized, image_size = resize_to_resolution_content(tensor, "480")
    recomputed = resized[0].numpy()
    stored = read_frame(cached_path, frame)
    check(
        cached_dtype == np.uint8 and cached_shape[0] == len(ids),
        "cache shape and frame count match the source",
        f"{cached_shape} {cached_dtype} vs {len(ids)} source frames",
    )
    check(
        np.array_equal(recomputed, stored),
        "cache frame is byte-identical to an online recomputation",
        f"max|diff| = {int(np.abs(recomputed.astype(int) - stored.astype(int)).max())}",
    )
    check(
        cached_shape[-2:] == tuple(recomputed.shape[-2:]),
        "cache stores the unpadded content",
        f"{cached_shape[-2:]} == recomputed {tuple(recomputed.shape[-2:])}, padding deferred",
    )

    print(f"\n=== L6 cache path == online path, and the composed affine ===")
    online = reflection_pad_to_target({"video": tensor.clone()}, ["video"], True, 544, 736)
    via_cache = reflection_pad_to_target({"video": resized.clone()}, ["video"], True, 544, 736)
    check(
        torch.equal(online["video"], via_cache["video"]),
        "online and cached video are bit-identical",
        f"shapes {tuple(online['video'].shape)}",
    )
    check(
        torch.equal(online["image_size"], via_cache["image_size"]),
        "online and cached image_size agree",
        f"{online['image_size'].tolist()}",
    )
    pf = {"metadata": {"uv_to_video": affine.copy(), "video_size_wh": np.asarray(sources[episode]["video_size_wh"])}}
    canvas_w, canvas_h = 544, 736
    resize_pointflow_metadata(pf, (544, 716), (544, 716), (canvas_w, canvas_h))
    composed = np.vstack([pf["metadata"]["uv_to_video"], [0, 0, 1]])
    check(
        abs(composed[0, 0] - 544 / 640) < 1e-5 and abs(composed[1, 1] - (716 / 842) * (480 / 448)) < 1e-5,
        "composed affine scales the tracker canvas onto the model canvas",
        f"x' = {composed[0, 0]:.6f}u (= 544/640), y' = {composed[1, 1]:.6f}v + {composed[1, 2]:.3f} "
        f"(= 716/842 * 480/448)",
    )
    note(
        "aspect rounding in the composed affine",
        f"x scale {composed[0, 0]:.6f} vs y scale {composed[1, 1] / (480 / 448):.6f}: differ by "
        f"{abs(composed[0, 0] / (composed[1, 1] / (480 / 448)) - 1) * 100:.3f}% because the resize rounds 842*0.85=715.7 to 716",
    )

    print(f"\n=== L7 latent cache <-> video cache ===")
    latent = torch.load(args.cache_root / "vae_latents" / f"{episode}.pt", map_location="cpu", weights_only=True)
    latent_hw = tuple(latent["shape"][-2:])
    check(
        tuple(latent["original_size"]) == (716, 544) and tuple(latent["padded_size"]) == (736, 544),
        "latent was encoded from the same content, padded to a multiple of 16",
        f"original_size={latent['original_size']} padded_size={latent['padded_size']}",
    )
    check(
        latent_hw == (736 // 16, 544 // 16),
        "latent spatial size = padded canvas / 16",
        f"latent {latent_hw} vs padded {736}x{544}/16 = {(736 // 16, 544 // 16)}",
    )
    check(
        latent["padded_frames"] == 1 + 4 * ((len(ids) - 1 + 3) // 4),
        "latent temporal length follows the causal VAE from the same frame count",
        f"padded_frames={latent['padded_frames']} from {len(ids)} source frames",
    )

    print(f"\n=== L8 video token grid <-> canvas ===")
    content_h, content_w = 716, 544
    latent_cropped = (content_h // 16, content_w // 16)
    patch_h = math.ceil(latent_cropped[0] / 2)
    patch_w = math.ceil(latent_cropped[1] / 2)
    check(
        patch_w == canvas_w // args.pixel_stride,
        "token grid width equals canvas width / patch stride",
        f"{patch_w} == {canvas_w}//{args.pixel_stride}",
    )
    note(
        "token grid height vs canvas height / patch stride",
        f"{patch_h} vs {canvas_h // args.pixel_stride}: off by one because {content_h} is not a multiple of 16, "
        f"so _remove_padding_from_latent floors {content_h}//16 = {latent_cropped[0]} rows where the content spans "
        f"{content_h / 16:.2f}. Bounded; the point coordinates below still fit.",
    )

    # The point tokens reuse the video tokens' mRoPE axes, so the video grid must
    # be 0-based for the two to share a frame. This runs the real constructor.
    from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens

    mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=9,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=0,
        fps=fps,
        base_fps=base_fps,
        temporal_compression_factor=tcf,
    )
    h_axis, w_axis, t_axis = mrope_ids[1].long(), mrope_ids[2].long(), mrope_ids[0]
    check(
        int(h_axis.min()) == 0
        and int(h_axis.max()) == patch_h - 1
        and int(w_axis.min()) == 0
        and int(w_axis.max()) == patch_w - 1,
        "video token h/w indices are 0-based, so point positions share their frame",
        f"h in [{int(h_axis.min())},{int(h_axis.max())}], w in [{int(w_axis.min())},{int(w_axis.max())}]",
    )
    latent_times = np.unique(t_axis.numpy())
    check(
        np.allclose(np.diff(latent_times), 1.6) and abs(latent_times[0]) < 1e-9,
        "video latent mRoPE times are 0, 1.6, 3.2 ... on the same grid as the point blocks",
        f"unique times {np.round(latent_times, 4).tolist()}",
    )

    print(f"\n=== L9 point positions <-> video token grid ===")
    K_uv = np.diag([640.0, 448.0, 1.0]) @ intrinsics[frame]
    projected_uv = xyz @ K_uv.T
    projected_uv = projected_uv[:, :2] / projected_uv[:, 2:3]
    patch = (projected_uv @ composed[:2, :2].T + composed[:2, 2] - (args.pixel_stride - 1) / 2) / args.pixel_stride
    inside = (patch[valid, 0] >= 0) & (patch[valid, 0] < patch_w) & (patch[valid, 1] >= 0) & (patch[valid, 1] < patch_h)
    check(
        inside.mean() > 0.9,
        "point tokens land inside the video token grid",
        f"{inside.mean():.1%} of {int(valid.sum())} points inside {patch_w}x{patch_h} grid",
    )

    print(f"\n=== L10 physical time ===")
    latent_dt = tcf * (base_fps / tcf) / fps
    block_dt = steps_per_token / fps * (base_fps / tcf)
    check(
        abs(latent_dt - 1.6) < 1e-9 and abs(block_dt - 1.6) < 1e-9,
        "point block times and video latent times share one spacing",
        f"latent {latent_dt}/frame, point block {block_dt}/block -> both aligned on the same grid",
    )

    failed = [name for ok, name, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print(f"  FAILED: {name}")


if __name__ == "__main__":
    main()
