#!/usr/bin/env python3
"""Animate the points the training selection actually keeps, on CPU.

``examples/launch_pointflow_motion_scan.sh`` renders PTv3 *clusters* and needs a
GPU.  The GT-ranked point selection (``select_top_n`` / ``min_voxel_members``)
happens inside ``prepare_window``, before the encoder -- so what training actually
supervises can be rendered without Sonata at all.

This script does NOT re-implement the selection: it instantiates
``PointFlowSource`` with the real training manifest and calls ``load()``, so the
per-window seed derivation (sha256 of recipe seed, episode and start frame), the
window/video alignment assertions and the selection knobs are the very code the
trainer runs.  What you see is what the loss sees.

    python tools/visualize_pointflow_selection.py \
        --episode episode_0013_20260731_133649 --start-frames 0 400 800 \
        --output pointflow_outputs/selection_vis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cosmos_framework.data.generator.action.pointflow_source import PointFlowSource
from cosmos_framework.data.pointflow_window import PointFlowTiming, _episode_metadata, read_frame


def selection_stats(sample):
    displacement = np.asarray(sample["targets"]["displacement"], dtype=np.float64)
    valid = np.asarray(sample["targets"]["valid"])
    magnitude = np.linalg.norm(displacement, axis=-1)
    counts = valid.sum(0)
    per_point = np.where(counts > 0, (magnitude * valid).sum(0) / np.maximum(counts, 1), 0.0)
    return per_point * 1000.0, float(valid.mean())  # mm, fraction


def load_training_sample(source: PointFlowSource, episode: str, start_frame: int, timing):
    """Call the training data path with the window's source frame IDs.

    ``load`` asserts its own window rows equal these IDs, so a wrong stride fails
    loudly instead of rendering a window the trainer would never see.
    """
    entry = source.entries[episode]
    if entry is None:
        print(f"  {episode}: no PointFlow labels in the manifest, skipped")
        return None, None
    episode_dir = entry["path"]
    stamps = np.load(episode_dir / "timestamps_sec.npy", allow_pickle=False)
    source_fps = 1.0 / float(np.median(np.diff(stamps)))
    stride = int(round(source_fps / timing.fps))
    if stride < 1:
        raise ValueError(f"{episode}: source FPS {source_fps:.2f} below Cosmos FPS {timing.fps}")
    frame_ids = start_frame + np.arange(timing.steps + 1) * stride
    sample = source.load(episode, frame_ids, source_fps, entry["video_size_wh"])
    return sample, episode_dir


def render_window(sample, episode_dir: Path, start_frame: int, args, timing):
    ids = np.asarray(sample["inputs"]["point_ids"])
    if len(ids) == 0:
        print(f"  w{start_frame}: selection kept no points, skipped")
        return None
    per_point_mm, valid_fraction = selection_stats(sample)

    video, width, height = _episode_metadata(episode_dir)
    source_ids = np.load(episode_dir / "frame_indices.npy", allow_pickle=False)
    frame_ids = np.asarray(sample["metadata"]["raw_frame_ids"])
    rows = np.searchsorted(source_ids, frame_ids)
    if np.any(rows >= len(source_ids)) or not np.array_equal(source_ids[rows], frame_ids):
        raise ValueError("Window frame IDs do not match source")

    all_uv, all_valid = [], []
    for row in rows:
        uv = read_frame(episode_dir / "uv_px.npy", int(row)).reshape(-1, 2)[ids]
        valid = read_frame(episode_dir / "valid.npy", int(row)).reshape(-1)[ids]
        xyz = read_frame(episode_dir / "position.npy", int(row)).reshape(-1, 3)[ids]
        valid &= np.isfinite(uv).all(1) & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        all_uv.append(np.rint(np.where(valid[:, None], uv, 0)).astype(int))
        all_valid.append(valid)
    all_uv, all_valid = np.array(all_uv), np.array(all_valid)

    # Color by mean GT displacement: still points are dark, the fastest are red.
    scale = max(np.percentile(per_point_mm, 95), 1.0)
    colors = cv2.applyColorMap(
        np.clip(per_point_mm / scale * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO
    )[:, 0, :]

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open head video: {video}")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{sample['metadata']['episode']}_w{start_frame}_top{args.select_top_n or 'all'}"
    if args.phantom_guard:
        stem += "_noghost"
    video_path = args.output / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), timing.fps, (width * 2, height + 40))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open MP4 writer: {video_path}")

    frames = []
    next_frame = 0
    timestamps = np.asarray(sample["metadata"]["timestamps_sec"])
    try:
        for t, raw_id in enumerate(frame_ids):
            while next_frame < raw_id:
                if not capture.grab():
                    raise RuntimeError("Video ended before requested frame")
                next_frame += 1
            ok, frame = capture.read()
            next_frame += 1
            if not ok:
                raise RuntimeError(f"Cannot decode frame {raw_id}")
            overlay = cv2.resize(frame, (width, height)).copy()
            for i in range(len(ids)):
                if not all_valid[t, i]:
                    continue
                color = tuple(int(c) for c in colors[i])
                # Never bridge an invalid interval, including after reappearance.
                for j in range(t, max(0, t - args.trail_steps), -1):
                    if not (all_valid[j, i] and all_valid[j - 1, i]):
                        break
                    cv2.line(overlay, tuple(all_uv[j - 1, i]), tuple(all_uv[j, i]), color, 1)
                cv2.circle(overlay, tuple(all_uv[t, i]), 2, color, -1)
            canvas = np.zeros((height + 40, width * 2, 3), dtype=np.uint8)
            canvas[40:] = np.concatenate((cv2.resize(frame, (width, height)), overlay), axis=1)
            labels = [
                f"RGB | frame {raw_id} | +{timestamps[t] - timestamps[0]:.2f}s",
                f"selected {len(ids)} pts | visible {int(all_valid[t].sum())} | color = mean motion (max {scale:.0f}mm)",
            ]
            for panel, label in enumerate(labels):
                cv2.putText(
                    canvas,
                    label,
                    (panel * width + 8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
            frames.append(
                Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                .resize((1280, round((height + 40) * 1280 / (2 * width))))
                .quantize(colors=128)
            )
    finally:
        writer.release()
        capture.release()

    gif_path = args.output / f"{stem}.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / timing.fps), loop=0)
    frames[len(frames) // 2].convert("RGB").save(args.output / f"{stem}_mid.jpg", quality=90)
    print(
        f"  w{start_frame}: kept {len(ids)} points, mean motion p50 "
        f"{np.median(per_point_mm):.1f}mm / max {per_point_mm.max():.0f}mm, valid fraction {valid_fraction:.2f}"
        f" -> {video_path.name}, {gif_path.name}"
    )
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", required=True, help="episode name as listed in the manifest")
    parser.add_argument("--start-frames", type=int, nargs="+", default=[0], help="source frame indices")
    parser.add_argument("--manifest", type=Path, default=Path("pointflow_outputs/task5/mixed_manifest.json"))
    parser.add_argument("--select-top-n", type=int, default=300, help="0 = keep the full cloud")
    parser.add_argument("--min-voxel-members", type=int, default=3)
    parser.add_argument("--min-valid-steps", type=int, default=16, help="the recipe's select_min_valid_steps")
    parser.add_argument("--regions", default="", help="comma-separated region filter, empty = off (as in training)")
    parser.add_argument(
        "--phantom-guard",
        action="store_true",
        help="demote phantom-drift points (uv glued <2px while 3D drifts >30mm) in the motion ranking",
    )
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42, help="the recipe's pointflow_seed")
    parser.add_argument("--trail-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.select_top_n < 0 or args.trail_steps < 0:
        parser.error("Invalid selection size or trail length")
    timing = PointFlowTiming()
    source = PointFlowSource(
        args.manifest,
        timing=timing,
        max_points=args.max_points,
        voxel_size=args.voxel_size,
        seed=args.seed,
        select_top_n=args.select_top_n,
        min_voxel_members=args.min_voxel_members,
        select_min_valid_steps=args.min_valid_steps,
        select_regions=tuple(r for r in args.regions.split(",") if r),
        select_phantom_guard=args.phantom_guard,
    )
    for start in args.start_frames:
        try:
            sample, episode_dir = load_training_sample(source, args.episode, start, timing)
            if sample is not None:
                render_window(sample, episode_dir, start, args, timing)
        except ValueError as exc:
            print(f"  w{start}: SKIPPED ({exc})")


if __name__ == "__main__":
    main()
