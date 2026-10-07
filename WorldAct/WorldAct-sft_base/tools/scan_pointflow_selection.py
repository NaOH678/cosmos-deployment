#!/usr/bin/env python3
"""Report what the GT-ranked point selection keeps, across the training data.

This is the training-side counterpart to ``examples/launch_pointflow_motion_scan.sh``.
That one renders clusters and needs a GPU because PointTransformerV3 runs on spconv.
This one only exercises the *data* path, because the selection happens before PTv3 --
so it runs on CPU over the whole dataset, which is what actually matters when you are
choosing ``POINTFLOW_SELECT_MOTION_FRACTION``:

  * how many points survive, i.e. whether the ragged pack stays a sane size
  * whether any sampled window degenerates (too few kept points for PTv3 to pool)
  * how much ground-truth motion the selection actually retains

    python tools/scan_pointflow_selection.py \
        --manifest pointflow_outputs/task5/mixed_manifest.json \
        --episode-allowlist examples/pointflow_sandwich_10_episodes.txt \
        --fractions 0,0.02,0.05,0.10 --windows-per-episode 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cosmos_framework.data.generator.action.pointflow_source import PointFlowSource
from cosmos_framework.data.pointflow_window import PointFlowTiming

# Below this many kept voxels PointTransformerV3 has no neighbourhood to pool, so a
# selection that leaves fewer is reported as degenerate rather than merely small.
DEGENERATE_VOXELS = 8


def sampled_windows(num_frames, source_fps, fps, chunk_length, per_episode):
    """Evenly spaced window starts, leaving room for a full window at the end."""
    source_stride = int(round(source_fps / fps))
    last = num_frames - 1 - chunk_length * source_stride
    if last < 0:
        return []
    if per_episode >= last + 1:
        return list(range(last + 1))
    return np.linspace(0, last, per_episode, dtype=int).tolist()


def summarize(sample, timing, source_fps):
    """Kept-point statistics for one window."""
    displacement = np.asarray(sample["targets"]["displacement"], dtype=np.float64)
    valid = np.asarray(sample["targets"]["valid"])
    xyz = np.asarray(sample["inputs"]["anchor_xyz"], dtype=np.float64)
    magnitude = np.linalg.norm(displacement, axis=-1)
    counts = valid.sum(0)
    per_point = np.where(counts > 0, (magnitude * valid).sum(0) / np.maximum(counts, 1), 0.0) * 1000.0
    spacing = float("nan")
    if len(xyz) > 1:
        distances, _ = cKDTree(xyz).query(xyz, k=2)
        spacing = float(np.median(distances[:, 1]) * 1000.0)
    # Per-element scale of the supervised target.  This is the quantity that must
    # match the unit-variance sampling noise: ``pointflow_displacement_scale`` in
    # metres per model unit.  The (n, mean, std) triple pools exactly across
    # windows; the raw values are handed back alongside it so the caller can take
    # exact pooled quantiles, which cannot be recovered from summary statistics.
    values = displacement[valid]
    row = {
        "points": len(sample["inputs"]["point_ids"]),
        "voxels": len(sample["inputs"]["coord"]),
        "bbox_m": np.ptp(xyz, axis=0).round(3).tolist() if len(xyz) else [0, 0, 0],
        "motion_p50_mm": float(np.median(per_point)) if len(per_point) else 0.0,
        "motion_max_mm": float(per_point.max()) if len(per_point) else 0.0,
        "spacing_mm": spacing,
        "valid_fraction": float(valid.mean()),
        "displacement_n": int(values.size),
        "displacement_mean_m": float(values.mean()) if values.size else 0.0,
        "displacement_std_m": float(values.std()) if values.size else 0.0,
    }
    return row, values.reshape(-1).astype(np.float32)


def pooled_std(rows):
    """Exact pooled std over windows from their (n, mean, std) triples."""
    n = np.array([r["displacement_n"] for r in rows], dtype=np.float64)
    mean = np.array([r["displacement_mean_m"] for r in rows], dtype=np.float64)
    std = np.array([r["displacement_std_m"] for r in rows], dtype=np.float64)
    total = n.sum()
    if total == 0:
        return 0.0, 0.0
    grand = float((n * mean).sum() / total)
    variance = float((n * (std**2 + (mean - grand) ** 2)).sum() / total)
    return grand, float(np.sqrt(variance))


# q01..q99 of a standard normal spans 4.6524 sigma; dividing the signed 1-99
# range by it reproduces the std for a Gaussian while ignoring extreme tails.
GAUSSIAN_Q01_Q99_SPAN = 4.6524


def scale_candidates(values):
    """Candidate ``pointflow_displacement_scale`` values from one pooled sample.

    ``std`` matches the unit-variance noise exactly but is inflated by tracking
    outliers.  ``robust`` divides the signed q01..q99 span by its Gaussian value,
    so it equals ``std`` for clean data and shrugs off the tail.  ``quantile``
    is the convention the action normalizers use -- q01/q99 mapped to +/-1 --
    which leaves a Gaussian at std ~0.43, i.e. 2.3x below the noise.
    """
    if values.size == 0:
        return {}
    q01, q50, q99 = (float(v) for v in np.quantile(values, [0.01, 0.5, 0.99]))
    absolute = np.abs(values)
    a50, a99, a999 = (float(v) for v in np.quantile(absolute, [0.5, 0.99, 0.999]))
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q01": q01,
        "q50": q50,
        "q99": q99,
        "abs_p50": a50,
        "abs_p99": a99,
        "abs_p999": a999,
        "abs_max": float(absolute.max()),
        "robust": (q99 - q01) / GAUSSIAN_Q01_Q99_SPAN,
        "quantile": (q99 - q01) / 2.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--episode-allowlist", type=Path, required=True)
    parser.add_argument(
        "--fractions",
        default="0,0.02,0.05,0.10,0.25",
        help="select_motion_fraction values; 0 = off (full cloud)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="select_top_n value; when > 0 it replaces --fractions with this single setting",
    )
    parser.add_argument("--min-voxel-members", type=int, default=0)
    parser.add_argument(
        "--regions",
        default="",
        help="comma-separated region labels for anchor candidates (1=fingertip, 2=hand, 3+=objects); empty = all",
    )
    parser.add_argument("--min-valid-steps", type=int, default=0)
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON dump of every row")
    args = parser.parse_args()

    state = json.loads((args.cache_root / "manifest.json").read_text())
    by_name = {str(row["name"]): row for row in state["episodes"]}
    names = [
        line.strip()
        for line in args.episode_allowlist.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"allowlist names not in the state manifest: {missing[:3]}")

    timing = PointFlowTiming(args.fps, args.chunk_length, 4)
    if args.top_n > 0:
        # (label, select_motion_fraction, select_top_n)
        settings = [(f"top{args.top_n}", 0.0, args.top_n)]
    else:
        settings = [(("off (full)" if v == 0 else f"{v:.0%}"), v, 0) for v in map(float, args.fractions.split(","))]
    rows = []
    # Raw valid displacement values per setting, kept only to take exact pooled
    # quantiles.  float32, ~0.3 MB per window at 410 points -- a scan of a few
    # hundred windows stays well under 100 MB.
    pooled = {}
    print(f"  {len(names)} episodes x {args.windows_per_episode} windows x {len(settings)} settings\n")

    for name in names:
        episode = by_name[name]
        source_fps = float(episode["source_fps"])
        windows = sampled_windows(
            int(episode["num_frames"]), source_fps, args.fps, args.chunk_length, args.windows_per_episode
        )
        stride = int(round(source_fps / args.fps))
        for window in windows:
            frame_ids = window + stride * np.arange(args.chunk_length + 1, dtype=np.int64)
            for label, keep, top_n in settings:
                source = PointFlowSource(
                    str(args.manifest),
                    timing=timing,
                    max_points=args.max_points,
                    voxel_size=args.voxel_size,
                    seed=args.seed,
                    select_motion_fraction=keep,
                    select_top_n=top_n,
                    min_voxel_members=args.min_voxel_members,
                    select_regions=args.regions,
                    select_min_valid_steps=args.min_valid_steps,
                )
                sample = source.load(name, frame_ids, source_fps, (544, 736))
                row, values = summarize(sample, timing, source_fps)
                row.update(episode=name, window=int(window), setting=label)
                rows.append(row)
                pooled.setdefault(label, []).append(values)
        print(f"  {name}: {len(windows)} windows done", flush=True)

    print(
        f"\n  {'select':>9}{'pts min':>9}{'pts p50':>9}{'pts max':>9}{'vox p50':>9}"
        f"{'motion p50':>12}{'motion max':>12}{'间距 p50':>10}{'退化窗口':>10}{'位移std':>10}"
    )
    print("  " + "-" * 98)
    for label, keep, top_n in settings:
        subset = [r for r in rows if r["setting"] == label]
        if not subset:
            continue
        points = np.array([r["points"] for r in subset])
        voxels = np.array([r["voxels"] for r in subset])
        motion = np.array([r["motion_p50_mm"] for r in subset])
        peak = np.array([r["motion_max_mm"] for r in subset])
        spacing = np.array([r["spacing_mm"] for r in subset])
        degenerate = int((voxels < DEGENERATE_VOXELS).sum())
        print(
            f"  {label:>9}{points.min():>9}{int(np.median(points)):>9}{points.max():>9}"
            f"{int(np.median(voxels)):>9}{np.median(motion):>12.1f}{peak.max():>12.1f}"
            f"{np.nanmedian(spacing):>10.1f}{degenerate:>10}{pooled_std(subset)[1]:>10.4f}"
        )

    print("\n  pointflow_displacement_scale 候选值(= 单位:米 / 模型单位):")
    for label, keep, top_n in settings:
        subset = [r for r in rows if r["setting"] == label]
        if not subset:
            continue
        stats = scale_candidates(np.concatenate(pooled[label]) if pooled.get(label) else np.empty(0, np.float32))
        if not stats:
            continue
        print(f"\n    select={label}   n={stats['n']}")
        print(
            f"      分布   均值 {stats['mean']:+.5f}   std {stats['std']:.5f}"
            f"    q01 {stats['q01']:+.5f}   q50 {stats['q50']:+.5f}   q99 {stats['q99']:+.5f}"
        )
        print(
            f"      尾部   |v| p50 {stats['abs_p50']:.5f}   p99 {stats['abs_p99']:.5f}"
            f"   p99.9 {stats['abs_p999']:.5f}   max {stats['abs_max']:.4f}"
        )
        print(f"      候选   std      {stats['std']:.5f}   → clean/噪声 {stats['std']:.3f}")
        print(
            f"             robust   {stats['robust']:.5f}   → clean/噪声 {stats['robust']:.3f}"
            f"   (=(q99-q01)/4.6524,高斯下等于 std)"
        )
        print(
            f"             quantile {stats['quantile']:.5f}   → clean/噪声 {stats['quantile']:.3f}"
            f"   (=(q99-q01)/2,action 归一化器的约定)"
        )
        print(f"             std 被尾部抬高 {stats['std'] / max(stats['robust'], 1e-12):.2f}x")

    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
