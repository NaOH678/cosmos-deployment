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

Statistics are accumulated as fixed-bin histograms plus (n, sum, sumsq) per
(setting, frame, channel) bucket, so episodes can be scanned in parallel
(``--workers``) with exact, worker-count-independent merges: std is exact,
quantiles come from 0.5 mm bins.  Raw displacement values are never kept, so
memory stays flat no matter how dense ``--windows-per-episode`` gets.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cosmos_framework.data.generator.action.pointflow_source import PointFlowSource
from cosmos_framework.data.pointflow_window import PointFlowTiming

# Below this many kept voxels PointTransformerV3 has no neighbourhood to pool, so a
# selection that leaves fewer is reported as degenerate rather than merely small.
DEGENERATE_VOXELS = 8

# 0.5 mm bins over ±0.6 m; observed displacement |max| is ~0.37 m.  Quantiles read
# from these bins are far below the noise floor of any downstream use.
DISPLACEMENT_BINS = np.linspace(-0.6, 0.6, 2401)
ABS_BINS = np.linspace(0.0, 0.6, 1201)
BIN_CENTERS = (DISPLACEMENT_BINS[:-1] + DISPLACEMENT_BINS[1:]) / 2
ABS_CENTERS = (ABS_BINS[:-1] + ABS_BINS[1:]) / 2


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
    values = displacement[valid]
    # Per-frame-index, per-channel pools: the cumulative-from-anchor target grows
    # with k, and the depth (z) channel carries systematically less variance than
    # x/y -- PointWorld normalises per (timestep, channel) for the same reason.
    # These feed POINTFLOW_DISPLACEMENT_FRAME_SCALES in its 96-value form.
    frame_values = [displacement[k][valid[k]] for k in range(displacement.shape[0])]
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
    return row, frame_values


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


def _new_accumulator(steps):
    return {
        "n": np.zeros((steps, 3)),
        "sum": np.zeros((steps, 3)),
        "sq": np.zeros((steps, 3)),
        "hist": np.zeros((steps, 3, len(DISPLACEMENT_BINS) - 1), np.int64),
        "glob": np.zeros(3),  # n, sum, sumsq over every value in the setting
        "glob_hist": np.zeros(len(DISPLACEMENT_BINS) - 1, np.int64),
        "glob_abs_hist": np.zeros(len(ABS_BINS) - 1, np.int64),
        "glob_abs_max": 0.0,
        "rows": [],
    }


def _accumulate(acc, row, frame_values):
    acc["rows"].append(row)
    nonempty = [v for v in frame_values if len(v)]
    for k, values in enumerate(frame_values):
        if not len(values):
            continue
        acc["n"][k] += len(values)
        acc["sum"][k] += values.sum(0)
        acc["sq"][k] += (values.astype(np.float64) ** 2).sum(0)
        for c in range(3):
            acc["hist"][k, c] += np.histogram(values[:, c], bins=DISPLACEMENT_BINS)[0]
    if nonempty:
        flat = np.concatenate(nonempty).reshape(-1).astype(np.float64)
        acc["glob"] += (flat.size, flat.sum(), (flat**2).sum())
        acc["glob_hist"] += np.histogram(flat, bins=DISPLACEMENT_BINS)[0]
        acc["glob_abs_hist"] += np.histogram(np.abs(flat), bins=ABS_BINS)[0]
        acc["glob_abs_max"] = max(acc["glob_abs_max"], float(np.abs(flat).max()))


def _merge_accumulators(accs, steps):
    merged = _new_accumulator(steps)
    for acc in accs:
        for key in ("n", "sum", "sq", "hist", "glob", "glob_hist", "glob_abs_hist"):
            merged[key] += acc[key]
        merged["glob_abs_max"] = max(merged["glob_abs_max"], acc["glob_abs_max"])
        merged["rows"].extend(acc["rows"])
    return merged


def _bucket_std(acc):
    n, total, sq = acc["n"], acc["sum"], acc["sq"]
    mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
    var = np.divide(sq, n, out=np.zeros_like(sq), where=n > 0) - mean**2
    return np.sqrt(np.maximum(var, 0.0))


def _hist_quantile(hist, n, centers, p):
    if n <= 0:
        return 0.0
    idx = int(np.searchsorted(np.cumsum(hist), p * n, side="left"))
    return float(centers[min(idx, len(centers) - 1)])


def scale_candidates(acc):
    """Candidate ``pointflow_displacement_scale`` values from one pooled setting.

    ``std`` matches the unit-variance noise exactly but is inflated by tracking
    outliers.  ``robust`` divides the signed q01..q99 span by its Gaussian value,
    so it equals ``std`` for clean data and shrugs off the tail.  ``quantile``
    is the convention the action normalizers use -- q01/q99 mapped to +/-1 --
    which leaves a Gaussian at std ~0.43, i.e. 2.3x below the noise.
    """
    n, total, sq = (float(v) for v in acc["glob"])
    if n == 0:
        return {}
    mean = total / n
    std = float(np.sqrt(max(sq / n - mean**2, 0.0)))
    q01 = _hist_quantile(acc["glob_hist"], n, BIN_CENTERS, 0.01)
    q50 = _hist_quantile(acc["glob_hist"], n, BIN_CENTERS, 0.5)
    q99 = _hist_quantile(acc["glob_hist"], n, BIN_CENTERS, 0.99)
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "q01": q01,
        "q50": q50,
        "q99": q99,
        "abs_p50": _hist_quantile(acc["glob_abs_hist"], n, ABS_CENTERS, 0.5),
        "abs_p99": _hist_quantile(acc["glob_abs_hist"], n, ABS_CENTERS, 0.99),
        "abs_p999": _hist_quantile(acc["glob_abs_hist"], n, ABS_CENTERS, 0.999),
        "abs_max": acc["glob_abs_max"],
        "robust": (q99 - q01) / GAUSSIAN_Q01_Q99_SPAN,
        "quantile": (q99 - q01) / 2.0,
    }


def _scan_chunk(job):
    """Worker: scan a slice of episodes, return per-setting accumulators."""
    names, by_name, settings, cfg = job
    timing = PointFlowTiming(cfg["fps"], cfg["chunk_length"], 4)
    sources = {
        label: PointFlowSource(
            cfg["manifest"],
            timing=timing,
            max_points=cfg["max_points"],
            voxel_size=cfg["voxel_size"],
            seed=cfg["seed"],
            select_motion_fraction=keep,
            select_top_n=top_n,
            min_voxel_members=cfg["min_voxel_members"],
            select_regions=cfg["regions"],
            select_region_quotas=cfg["region_quotas"],
            select_min_valid_steps=cfg["min_valid_steps"],
            select_phantom_guard=cfg["phantom_guard"],
            window_cache_root=cfg["window_cache_root"],
        )
        for label, keep, top_n in settings
    }
    accs = {label: _new_accumulator(timing.steps) for label, _, _ in settings}
    for name in names:
        episode = by_name[name]
        source_fps = float(episode["source_fps"])
        windows = sampled_windows(
            int(episode["num_frames"]), source_fps, cfg["fps"], cfg["chunk_length"], cfg["windows_per_episode"]
        )
        stride = int(round(source_fps / cfg["fps"]))
        for window in windows:
            frame_ids = window + stride * np.arange(cfg["chunk_length"] + 1, dtype=np.int64)
            for label, _, _ in settings:
                sample = sources[label].load(name, frame_ids, source_fps, (544, 736))
                row, frame_values = summarize(sample, timing, source_fps)
                row.update(episode=name, window=int(window), setting=label)
                _accumulate(accs[label], row, frame_values)
    return accs


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
    parser.add_argument(
        "--region-quotas",
        default="",
        help="comma-separated label:fraction pairs splitting the top-N budget (e.g. 2:0.40,3:0.45,4:0.15)",
    )
    parser.add_argument("--phantom-guard", action="store_true", help="demote uv-frozen 3D drifters in the ranking")
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window-cache-root",
        type=Path,
        default=None,
        help="precomputed window cache (tools/build_pointflow_window_cache.py); skips online preparation",
    )
    parser.add_argument("--workers", type=int, default=16, help="parallel episode scanners; results are identical")
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
    cfg = {
        "manifest": str(args.manifest),
        "max_points": args.max_points,
        "voxel_size": args.voxel_size,
        "seed": args.seed,
        "min_voxel_members": args.min_voxel_members,
        "regions": args.regions,
        "region_quotas": args.region_quotas,
        "min_valid_steps": args.min_valid_steps,
        "phantom_guard": args.phantom_guard,
        "window_cache_root": str(args.window_cache_root) if args.window_cache_root else None,
        "fps": args.fps,
        "chunk_length": args.chunk_length,
        "windows_per_episode": args.windows_per_episode,
    }
    print(f"  {len(names)} episodes x {args.windows_per_episode} windows x {len(settings)} settings\n")

    workers = max(1, min(args.workers, len(names)))
    jobs = [(names[i::workers], by_name, settings, cfg) for i in range(workers)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            partials = list(executor.map(_scan_chunk, jobs))
    else:
        partials = [_scan_chunk(jobs[0])]
    merged = {}
    for part in partials:
        for label, acc in part.items():
            merged.setdefault(label, []).append(acc)
    accs = {label: _merge_accumulators(chunks, timing.steps) for label, chunks in merged.items()}
    rows = sorted(
        (row for acc in accs.values() for row in acc["rows"]), key=lambda r: (r["episode"], r["window"], r["setting"])
    )
    for name in names:
        done = sum(1 for r in rows if r["episode"] == name)
        print(f"  {name}: {done} window-settings done", flush=True)

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
        stats = scale_candidates(accs[label])
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

    print("\n  per-frame scale(逐帧×逐通道统计,对应 POINTFLOW_DISPLACEMENT_FRAME_SCALES 的 96 值形式):")
    frame_scale_dump = {}
    frame_scale_env = {}
    for label, keep, top_n in settings:
        acc = accs[label]
        stds = _bucket_std(acc)
        steps = stds.shape[0]
        robusts = np.zeros_like(stds)
        counts = acc["n"][:, 0].astype(int).tolist()
        for k in range(steps):
            for c in range(3):
                n = acc["n"][k, c]
                q01 = _hist_quantile(acc["hist"][k, c], n, BIN_CENTERS, 0.01)
                q99 = _hist_quantile(acc["hist"][k, c], n, BIN_CENTERS, 0.99)
                robusts[k, c] = (q99 - q01) / GAUSSIAN_Q01_Q99_SPAN if n else 0.0
        frame_scale_dump[label] = {"std": stds.round(6).tolist(), "robust": robusts.round(6).tolist(), "n": counts}
        # Canonical file the model reads via POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE.
        frame_scale_env[label] = {
            "steps": steps,
            "channels": 3,
            "order": "frame-major x,y,z",
            "stat": "std",
            "selection": label,
            "scales": [round(float(v), 6) for v in stds.reshape(-1)],
        }
        print(f"\n    select={label}")
        for start in range(0, steps, 8):
            stop = min(start + 8, steps)
            print(f"      帧 {start:>2}-{stop - 1:>2}  std_x " + " ".join(f"{v:.5f}" for v in stds[start:stop, 0]))
            print(f"      {'':>8}  std_y " + " ".join(f"{v:.5f}" for v in stds[start:stop, 1]))
            print(f"      {'':>8}  std_z " + " ".join(f"{v:.5f}" for v in stds[start:stop, 2]))
        # 帧主序、通道次序 (k0x,k0y,k0z,k1x,...) 的 96 值 env 串
        print(f"      env(std):    {','.join(f'{v:.5f}' for v in stds.reshape(-1))}")
        print(f"      env(robust): {','.join(f'{v:.5f}' for v in robusts.reshape(-1))}")

    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\n  wrote {args.output}")
        frame_path = args.output.with_name(args.output.stem + "_frame_scales.json")
        frame_path.write_text(json.dumps(frame_scale_dump, indent=2) + "\n")
        print(f"  wrote {frame_path}")
        for label, doc in frame_scale_env.items():
            suffix = "" if len(frame_scale_env) == 1 else f"_{label.replace(' ', '_')}"
            env_path = args.output.with_name(args.output.stem + f"_frame_scales_env{suffix}.json")
            env_path.write_text(json.dumps(doc, indent=2) + "\n")
            print(f"  wrote {env_path}  →  POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE={env_path.resolve()}")


if __name__ == "__main__":
    main()
