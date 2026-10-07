"""Animate offline Track4World labels with fixed anchor cluster identities."""

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cosmos_framework.data.pointflow_window import read_frame


def motion_ranked_clusters(window, mapping, sizes, top_k, min_members):
    """Cluster IDs ranked by how far their members actually travel in this window.

    Cluster IDs are only meaningful inside one window -- PTv3 derives them from that
    window's point cloud -- so the choice has to be a rule, not a fixed list.  Rank
    by ground-truth displacement magnitude, averaged over each cluster's members and
    only over the steps where a point is valid.

    A floor on membership matters: the top of a naive ranking is single-point
    "clusters" whose mean is one point's tracking noise, not a coherent part.
    """
    displacement = np.asarray(window["target_displacement"], dtype=np.float64)
    valid = np.asarray(window["target_valid"])
    magnitude = np.linalg.norm(displacement, axis=-1)
    counts = valid.sum(0)
    per_point = np.where(counts > 0, (magnitude * valid).sum(0) / np.maximum(counts, 1), 0.0)
    per_cluster = np.bincount(mapping, weights=per_point, minlength=len(sizes)) / np.maximum(sizes, 1)
    candidates = np.flatnonzero(sizes >= min_members)
    if len(candidates) == 0:
        # Mirror the --focus-uv default: a window whose clusters all fell under the
        # floor still has clusters, so rank those rather than failing the render.
        candidates = np.arange(len(sizes))
    order = candidates[np.argsort(-per_cluster[candidates], kind="stable")]
    chosen = order[:top_k]
    return chosen.tolist(), per_cluster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--encoding-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(5), default=3)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cluster-id", type=int)
    selection.add_argument("--cluster-ids", nargs="+", type=int)
    selection.add_argument(
        "--top-motion-clusters",
        type=int,
        help="rank this window's own clusters by ground-truth motion and keep the top K",
    )
    parser.add_argument("--min-members", type=int, default=8, help="membership floor for --top-motion-clusters")
    parser.add_argument("--full-sequence", action="store_true")
    parser.add_argument("--focus-uv", nargs=2, type=float, default=[400, 320])
    parser.add_argument("--max-display-points", type=int, default=2048)
    parser.add_argument("--trail-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_display_points < 1 or args.trail_steps < 0:
        parser.error("Invalid display budget or trail length")
    window = np.load(args.encoding_dir / "window.npz", allow_pickle=False)
    encoding = np.load(args.encoding_dir / f"encoding_enc{args.stage}.npz", allow_pickle=False)
    ids = window["point_ids"]
    if not np.array_equal(ids, encoding["point_ids"]):
        raise ValueError("Encoding and window point IDs differ")
    mapping = encoding["original_to_cluster"]
    count = len(encoding["cluster_counts"])
    if args.top_motion_clusters is not None:
        if args.top_motion_clusters < 1:
            parser.error("--top-motion-clusters must be positive")
        clusters, per_cluster = motion_ranked_clusters(
            window, mapping, encoding["cluster_counts"], args.top_motion_clusters, args.min_members
        )
        print(
            "motion-ranked clusters: "
            + ", ".join(
                f"{c}({int(encoding['cluster_counts'][c])} members, {per_cluster[c] * 1000:.0f}mm)" for c in clusters
            )
        )
    elif args.cluster_id is None and args.cluster_ids is None:
        candidates = np.flatnonzero(encoding["cluster_counts"] >= 8)
        if len(candidates) == 0:
            candidates = np.arange(count)
        cluster = int(
            candidates[np.argmin(np.linalg.norm(encoding["cluster_uv"][candidates] - np.array(args.focus_uv), axis=1))]
        )
        clusters = [cluster]
    else:
        clusters = list(dict.fromkeys(args.cluster_ids if args.cluster_ids is not None else [args.cluster_id]))
    if any(not 0 <= c < count for c in clusters):
        parser.error("cluster ID is out of range")
    focused = np.flatnonzero(np.isin(mapping, clusters))
    palette = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255), (0, 255, 0)]
    if len(clusters) > len(palette):
        palette = [
            tuple(round(v * 255) for v in colorsys.hsv_to_rgb(j / len(clusters), 0.9, 1)[::-1])
            for j in range(len(clusters))
        ]
    highlight = {c: palette[j] for j, c in enumerate(clusters)}
    rng = np.random.default_rng(args.seed)
    shown = np.sort(rng.choice(len(ids), min(args.max_display_points, len(ids)), replace=False))
    colors = rng.integers(40, 256, (count, 3))
    meta = json.loads((args.episode / "COMPLETE.json").read_text())
    width, height = meta["inference_width"], meta["inference_height"]
    source_ids = np.load(args.episode / "frame_indices.npy", allow_pickle=False)
    frame_ids = source_ids if args.full_sequence else window["raw_frame_ids"]
    rows = np.searchsorted(source_ids, frame_ids)
    if np.any(rows >= len(source_ids)) or not np.array_equal(source_ids[rows], frame_ids):
        raise ValueError("Window frame IDs do not match source")
    all_uv, all_valid = [], []
    for row in rows:
        uv = read_frame(args.episode / "uv_px.npy", int(row)).reshape(-1, 2)[ids]
        valid = read_frame(args.episode / "valid.npy", int(row)).reshape(-1)[ids]
        xyz = read_frame(args.episode / "position.npy", int(row)).reshape(-1, 3)[ids]
        valid &= np.isfinite(uv).all(1) & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        all_uv.append(np.rint(np.where(valid[:, None], uv, 0)).astype(int))
        all_valid.append(valid)
    all_uv, all_valid = np.array(all_uv), np.array(all_valid)
    timestamps = np.load(args.episode / "timestamps_sec.npy", allow_pickle=False)[rows]
    intervals = np.diff(timestamps)
    if not np.allclose(intervals, intervals[0], atol=1e-4) or intervals[0] <= 0:
        raise ValueError("Video export requires a uniform positive frame interval")
    fps = 1 / intervals[0]
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(str(c) for c in clusters)
    stem = f"motion_enc{args.stage}_cluster{suffix}" + ("_full" if args.full_sequence else "")
    video_path = args.output / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height + 40))
    capture = cv2.VideoCapture(meta["video"])
    if not writer.isOpened() or not capture.isOpened():
        writer.release()
        capture.release()
        raise RuntimeError("Cannot open source video or MP4 writer")
    frames = []
    next_frame = 0
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
            frame = cv2.resize(frame, (width, height))
            panels = [frame.copy(), frame.copy(), frame.copy()]
            for i in shown:
                if all_valid[t, i]:
                    xy = tuple(all_uv[t, i])
                    cv2.circle(panels[1], xy, 2, tuple(int(c) for c in colors[mapping[i]]), -1)
                    cv2.circle(panels[2], xy, 1, (140, 140, 140), -1)
            for i in focused:
                if not all_valid[t, i]:
                    continue
                color = highlight[int(mapping[i])]
                # Never bridge an invalid interval, including after reappearance.
                for j in range(t, max(0, t - args.trail_steps), -1):
                    if not (all_valid[j, i] and all_valid[j - 1, i]):
                        break
                    cv2.line(panels[2], tuple(all_uv[j - 1, i]), tuple(all_uv[j, i]), color, 1)
                cv2.circle(panels[2], tuple(all_uv[t, i]), 3, color, -1)
            canvas = np.zeros((height + 40, width * 3, 3), dtype=np.uint8)
            canvas[40:] = np.concatenate(panels, axis=1)
            labels = [
                f"GT RGB | frame {raw_id} | +{timestamps[t] - timestamps[0]:.2f}s",
                f"Track4World GT | enc{args.stage} fixed colors",
                f"Clusters {suffix} | visible {all_valid[t, focused].sum()}/{len(focused)}",
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
            for j, c in enumerate(clusters):
                cv2.putText(
                    canvas,
                    f"ID {c}",
                    (width * 2 + 8 + (j % 5) * 110, 60 + (j // 5) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    highlight[c],
                    1,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
            frames.append(
                Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                .resize((960, round((height + 40) * 960 / (3 * width))))
                .quantize(colors=128)
            )
    finally:
        capture.release()
        writer.release()
    gif_path = args.output / f"{stem}.gif"
    # GIF has 10 ms timing granularity; distribute rounding rather than speeding up playback.
    ticks = np.rint(np.arange(len(frames) + 1) * 100 / fps).astype(int)
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], loop=0, duration=(np.diff(ticks) * 10).tolist(), disposal=2
    )
    strip = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for row, index in enumerate([0, len(frames) // 2, len(frames) - 1]):
        strip.paste(frames[index], (0, row * frames[0].height))
    strip.save(args.output / f"{stem}_preview.jpg")
    record = {
        "source": "offline Track4World labels, not model predictions",
        "stage": args.stage,
        "cluster_ids": clusters,
        "anchor_frame": int(window["raw_frame_ids"][0]),
        "full_sequence": args.full_sequence,
        "highlight_colors_bgr": highlight,
        "cluster_members": len(focused),
        "frames": len(frames),
        "fps": fps,
        "raw_frame_ids": frame_ids.tolist(),
        "display_points": len(shown),
        "selection": f"top {args.top_motion_clusters} by ground-truth motion, >= {args.min_members} members"
        if args.top_motion_clusters is not None
        else (
            "explicit cluster ID"
            if args.cluster_id is not None or args.cluster_ids is not None
            else "nearest anchor UV centroid with >=8 members when available"
        ),
        "focus_uv": args.focus_uv,
        "seed": args.seed,
        "mp4": str(video_path),
        "gif": str(gif_path),
    }
    (args.output / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
