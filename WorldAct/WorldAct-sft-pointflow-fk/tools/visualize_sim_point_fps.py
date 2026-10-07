"""Preview causal anchor FPS on a complete simulation episode (no cache mutation)."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from cosmos_framework.data.pointflow_anchor_selection import anchor_fps, hand_guided_fps
from cosmos_framework.data.pointflow_window import read_frame
from tools.visualize_pointflow_selection import selection_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--episode", default="episode_000000")
    p.add_argument("--points", type=int, default=1024)
    p.add_argument("--mode", choices=["global", "hand_guided"], default="global")
    p.add_argument("--hand-radius", type=float, default=0.05)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.points < 1 or not np.isfinite(args.hand_radius) or args.hand_radius <= 0:
        p.error("--points and --hand-radius must be positive")
    sys.path.append(str(args.bundle / "code/bench2dex"))
    import torch
    from utils.sim_pointfk_dataset import SimPointFKDataset

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.bundle / "datasets/bench2dex-task21-cosmos-cache"
    index = json.loads((cache / "window_index.json").read_text())["windows"]
    split = next(r["split"] for r in index if r["episode"] == args.episode)
    ds = SimPointFKDataset(args.bundle, split)
    windows, selections, selection_details = [], {}, {}
    for i, row in enumerate(ds.rows):
        if row["episode"] != args.episode:
            continue
        sample = ds[i]
        s = sample["pointflow"]
        if args.mode == "hand_guided":
            assert sample["fk"]["metadata"]["hand_order"] == ["left", "right"]
            keep, details = hand_guided_fps(
                s["inputs"]["anchor_xyz"], sample["fk"]["inputs"]["anchor_xyz"], args.points, args.hand_radius
            )
            selection_details[str(row["start_frame"])] = details
        else:
            keep = anchor_fps(s["inputs"]["anchor_xyz"], args.points)
        ids = s["inputs"]["point_ids"][keep]
        assert len(ids) == len(np.unique(ids)) == min(args.points, len(s["inputs"]["point_ids"]))
        raw = args.bundle / row["raw_path"]
        uv = np.stack([read_frame(raw / "uv_px.npy", t).reshape(-1, 2)[ids] for t in range(33)])
        valid = np.stack([read_frame(raw / "valid.npy", t).reshape(-1)[ids] for t in range(33)])
        valid &= (
            np.isfinite(uv).all(2) & (uv[..., 0] >= 0) & (uv[..., 0] < 640) & (uv[..., 1] >= 0) & (uv[..., 1] < 480)
        )
        # Future motion controls visualization color only; it never enters FPS.
        motion, _ = selection_stats(s)
        # Match the global-FPS preview's color scale for a fair visual comparison.
        reference = anchor_fps(s["inputs"]["anchor_xyz"], args.points) if args.mode == "hand_guided" else keep
        scale = max(float(np.percentile(motion[reference], 95)), 1.0)
        motion = motion[keep]
        colors = cv2.applyColorMap(np.clip(motion / scale * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[
            :, 0
        ]
        start = int(row["start_frame"])
        windows.append(
            dict(
                start=start,
                end=int(row["frame_ids"][-1]),
                uv=np.rint(np.nan_to_num(uv)).astype(int),
                valid=valid,
                colors=colors,
                scale=scale,
                candidates=len(s["inputs"]["point_ids"]),
            )
        )
        selections[str(start)] = ids.tolist()
        print(f"{args.episode}@{start}: {len(s['inputs']['point_ids'])} -> {len(ids)}", flush=True)
    windows.sort(key=lambda w: w["start"])
    starts = np.array([w["start"] for w in windows])
    frames_path = cache / "video_frames" / f"{args.episode}.npy"
    manifest = json.loads((cache / "video_manifest.json").read_text())
    frames = next(r["shape"][0] for r in manifest["episodes"] if r["name"] == args.episode)
    stem = f"{args.episode}_fps{args.points}_full"
    if args.mode == "hand_guided":
        stem = f"{args.episode}_handguided{args.points}_full"
    path = args.output / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 520))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    annotated = 0
    try:
        for frame in range(frames):
            rgb = read_frame(frames_path, frame)[:, :480, :].transpose(1, 2, 0)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = bgr.copy()
            offset = int(np.searchsorted(starts, frame, side="right") - 1)
            text = "No supervised window (initial / homing frame)"
            if offset >= 0 and frame <= windows[offset]["end"]:
                w = windows[offset]
                t = frame - w["start"]
                for k in np.flatnonzero(w["valid"][t]):
                    color = tuple(int(c) for c in w["colors"][k])
                    for j in range(t, max(0, t - 5), -1):
                        if not (w["valid"][j, k] and w["valid"][j - 1, k]):
                            break
                        cv2.line(overlay, tuple(w["uv"][j - 1, k]), tuple(w["uv"][j, k]), color, 1)
                    cv2.circle(overlay, tuple(w["uv"][t, k]), 2, color, -1)
                label = "FK-FPS" if args.mode == "hand_guided" else "FPS"
                text = f"{label} {len(w['colors'])}/{w['candidates']} | visible {w['valid'][t].sum()} | window {w['start']}"
                annotated += 1
            canvas = np.zeros((520, 1280, 3), dtype=np.uint8)
            canvas[40:] = np.concatenate([bgr, overlay], axis=1)
            for x, label in [(8, f"RGB | frame {frame} | {frame / 20:.2f}s"), (648, text)]:
                cv2.putText(canvas, label, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
            if frame == frames // 2:
                cv2.imwrite(str(args.output / f"{stem}_mid.jpg"), canvas)
    finally:
        writer.release()
    result = dict(
        episode=args.episode,
        frames=frames,
        fps=20,
        annotated_frames=annotated,
        windows=len(windows),
        points=args.points,
        selection=(
            "512 global + up to 256 per hand within current FK radius; remainder global FPS"
            if args.mode == "hand_guided" and args.points == 1024
            else args.mode + " current anchor XYZ FPS"
        ),
        selection_details=selection_details,
        color="mean GT motion TURBO per window; visualization only",
        trail_steps=5,
        transitions="latest window at each frame; IDs/trails restart at window boundaries",
        canvas="head only, original left, selected points right",
        point_ids_by_start_frame=selections,
    )
    (args.output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {path}: {frames} frames, {annotated} annotated", flush=True)


if __name__ == "__main__":
    main()
