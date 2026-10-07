# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the per-window PointFlow selection cache.

Precomputes ``prepare_window`` for every window on the training lattice
(start frames 0..last_start at sample_stride) with the exact selection
configuration used by training, and writes one npz per episode plus a
cache_manifest.json.  Pure CPU; parallel across episodes.  Each episode is
computed in a single streaming pass (rolling frame buffer + one forward video
decode), see ``cosmos_framework.data.pointflow_window_cache.build_episode_archive``.

Example (sandwich, stratified 500-point selection):

    PYTHONPATH=. python tools/build_pointflow_window_cache.py \
      --manifest pointflow_outputs/sandwich_924_20260928/manifest.json \
      --output /data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache/pointflow_windows \
      --episode-allowlist examples/pointflow_sandwich_all_101_episodes.txt \
      --top-n 500 --region-quotas 2:0.40,3:0.45,4:0.15 --min-voxel-members 3 \
      --min-valid-steps 16 --phantom-guard --workers 32
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cosmos_framework.data.generator.action.pointflow_source import window_seed
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.data.pointflow_window_cache import build_episode_archive, write_manifest


def _episode_windows(job):
    ep_path, out_path, kwargs, fps, chunk_length, sample_stride, overwrite, base_seed = job
    n, failures = build_episode_archive(
        ep_path,
        out_path,
        fps=fps,
        chunk_length=chunk_length,
        sample_stride=sample_stride,
        seed_fn=lambda start: window_seed(base_seed, ep_path.name, start),
        overwrite=overwrite,
        **kwargs,
    )
    return ep_path.name, n, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="pointflow manifest.json (labeled paths)")
    parser.add_argument("--output", type=Path, required=True, help="cache root directory")
    parser.add_argument("--episode-allowlist", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--motion-fraction", type=float, default=0.0)
    parser.add_argument("--region-quotas", default="", help="e.g. 2:0.40,3:0.45,4:0.15")
    parser.add_argument("--regions", default="", help="comma-separated region labels; empty = all")
    parser.add_argument("--min-voxel-members", type=int, default=0)
    parser.add_argument("--min-valid-steps", type=int, default=0)
    parser.add_argument("--supervise-cluster-n", type=int, default=0)
    parser.add_argument("--phantom-guard", action="store_true")
    parser.add_argument("--phantom-guard-disp-mm", type=float, default=30.0)
    parser.add_argument("--phantom-guard-uv-px", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    quotas = tuple(
        (int(label), float(fraction))
        for label, fraction in (part.split(":") for part in args.region_quotas.split(",") if part.strip())
    )
    regions = tuple(int(part) for part in args.regions.split(",") if part.strip())
    timing = PointFlowTiming(fps=args.fps, steps=args.chunk_length, steps_per_token=4)

    document = json.loads(args.manifest.read_text())
    sources = {row["name"]: row["pointflow_source"] for row in document["episodes"]}
    allowlist = {line.strip() for line in args.episode_allowlist.read_text().splitlines() if line.strip()}
    episodes = sorted(name for name in allowlist if sources.get(name) is not None)
    missing = sorted(allowlist - set(episodes))
    if missing:
        raise ValueError(f"Episodes without a labeled pointflow source: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "fps": args.fps,
        "chunk_length": args.chunk_length,
        "sample_stride": args.sample_stride,
        "max_points": args.max_points,
        "voxel_size": args.voxel_size,
        "seed": args.seed,
        "select_motion_fraction": args.motion_fraction,
        "select_top_n": args.top_n,
        "min_voxel_members": args.min_voxel_members,
        "supervise_cluster_n": args.supervise_cluster_n,
        "select_regions": regions,
        "select_region_quotas": quotas,
        "select_min_valid_steps": args.min_valid_steps,
        "select_phantom_guard": args.phantom_guard,
        "phantom_guard_disp_mm": args.phantom_guard_disp_mm,
        "phantom_guard_uv_px": args.phantom_guard_uv_px,
    }
    write_manifest(args.output, config)

    jobs = []
    for name in episodes:
        ep_path = Path(sources[name]["path"])
        if not ep_path.is_absolute():
            ep_path = (args.manifest.parent / ep_path).resolve()
        jobs.append(
            (
                ep_path,
                args.output / f"{name}.npz",
                dict(
                    max_points=args.max_points,
                    voxel_size=args.voxel_size,
                    timing=timing,
                    allow_empty=True,
                    select_motion_fraction=args.motion_fraction,
                    select_top_n=args.top_n,
                    min_voxel_members=args.min_voxel_members,
                    supervise_cluster_n=args.supervise_cluster_n,
                    select_regions=regions,
                    select_region_quotas=quotas,
                    select_min_valid_steps=args.min_valid_steps,
                    select_phantom_guard=args.phantom_guard,
                    phantom_guard_disp_mm=args.phantom_guard_disp_mm,
                    phantom_guard_uv_px=args.phantom_guard_uv_px,
                ),
                args.fps,
                args.chunk_length,
                args.sample_stride,
                args.overwrite,
                args.seed,
            )
        )

    print(f"{len(jobs)} episodes over {args.workers} workers (CPU only)")
    started = time.monotonic()
    done, failed_windows = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for name, n, failures in executor.map(_episode_windows, jobs):
            done += 1
            failed_windows.extend((name, start, error) for start, error in failures)
            state = "cached" if n >= 0 else "skipped"
            elapsed = time.monotonic() - started
            print(f"[{done:02d}/{len(jobs)}] {name}: {state} {max(n, 0)} windows  [{elapsed:.0f}s]", flush=True)
    if failed_windows:
        print(f"WARNING: {len(failed_windows)} windows failed (online fallback will cover them):")
        for name, start, error in failed_windows[:10]:
            print(f"  {name}@{start}: {error}")
    report = {
        "episodes": done,
        "failed_windows": len(failed_windows),
        "seconds": round(time.monotonic() - started, 1),
    }
    (args.output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}  ({done} episodes, {report['seconds']}s)")


if __name__ == "__main__":
    main()
