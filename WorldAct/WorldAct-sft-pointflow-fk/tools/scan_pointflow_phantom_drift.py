#!/usr/bin/env python3
"""Quantify along-ray phantom drift in labeled PointFlow deliveries.

A real moving point shifts in both 3D and image space; a depth-drifting point
slides along the camera ray: large |displacement| with near-zero uv motion.
Motion-ranked selection (select_top_n) harvests exactly that tail, so this
scans, per episode over sampled windows:

  * how many points are "phantom" (uv move < UV_PX yet displacement > DISP_MM);
  * how many of the top-N selected points are phantom (the training exposure).

Uses the real training path (prepare_window) for selection, then reads uv/valid
for the selected ids only.

    .venv/bin/python tools/scan_pointflow_phantom_drift.py \
        --manifest pointflow_outputs/manifest_sandwich_labeled_20260921.json \
        --allowlist examples/pointflow_sandwich_labeled_29_episodes.txt \
        --windows-per-episode 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_framework.data.pointflow_window import PointFlowTiming, prepare_window, read_frame

DISP_MM = 30.0
UV_PX = 2.0


def scan_window(episode: Path, start: int, args, timing) -> dict | None:
    try:
        sample = prepare_window(
            episode,
            start_frame=start,
            max_points=8192,
            voxel_size=0.02,
            seed=args.seed,
            timing=timing,
            select_top_n=args.select_top_n,
            min_voxel_members=args.min_voxel_members,
            select_min_valid_steps=args.select_min_valid_steps,
            select_region_quotas=args.region_quotas,
            select_phantom_guard=args.phantom_guard,
        )
    except ValueError:
        return None
    ids = np.asarray(sample["point_ids"])
    if not len(ids):
        return None
    target = np.asarray(sample["target_displacement"])  # [steps,N,3] anchor-relative
    tvalid = np.asarray(sample["target_valid"])
    frame_ids = np.asarray(sample["raw_frame_ids"])
    rows = np.searchsorted(np.load(episode / "frame_indices.npy"), frame_ids)

    # Actual per-point displacement: anchor-relative vector at the last valid step.
    last = np.where(tvalid.any(0), len(tvalid) - 1 - np.argmax(tvalid[::-1], axis=0), -1)
    real_mm = np.array([np.linalg.norm(target[last[i], i]) * 1000 if last[i] >= 0 else np.nan for i in range(len(ids))])
    uv0 = read_frame(episode / "uv_px.npy", int(rows[0])).reshape(-1, 2)[ids]
    uv1 = read_frame(episode / "uv_px.npy", int(rows[-1])).reshape(-1, 2)[ids]
    uv_px = np.linalg.norm(uv1 - uv0, axis=1)

    labels_path = episode / "region_labels.npy"
    labels = np.load(labels_path)[ids] if labels_path.is_file() else np.zeros(len(ids), dtype=np.uint8)
    phantom = (real_mm > DISP_MM) & (uv_px < UV_PX)
    per_label = {}
    for label in sorted(np.unique(labels).tolist()):
        m = labels == label
        per_label[int(label)] = dict(n=int(m.sum()), phantom=int((phantom & m).sum()))
    return dict(
        n=len(ids),
        phantom=int(phantom.sum()),
        disp_mm_p50=float(np.nanpercentile(real_mm, 50)),
        per_label=per_label,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--allowlist", type=Path, required=True)
    p.add_argument("--windows-per-episode", type=int, default=12)
    p.add_argument("--select-top-n", type=int, default=300)
    p.add_argument("--min-voxel-members", type=int, default=3)
    p.add_argument("--select-min-valid-steps", type=int, default=16)
    p.add_argument("--region-quotas", default="", help="e.g. 2:0.40,3:0.45,4:0.15")
    p.add_argument("--phantom-guard", action="store_true", help="demote phantoms during selection (the guard itself)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    # prepare_window takes quota pairs, not the CLI string.
    args.region_quotas = tuple(
        (int(label), float(fraction))
        for label, fraction in (part.split(":") for part in args.region_quotas.split(",") if part.strip())
    )

    manifest = json.loads(args.manifest.read_text())
    entries = {r["name"]: r["pointflow_source"] for r in manifest["episodes"]}
    names = [l.strip() for l in args.allowlist.read_text().splitlines() if l.strip() and not l.startswith("#")]
    timing = PointFlowTiming()

    report = {}
    for name in names:
        entry = entries.get(name)
        if entry is None:
            print(f"{name}: no pointflow source, skipped")
            continue
        episode = (args.manifest.parent / entry["path"]).resolve()
        frame_ids = np.load(episode / "frame_indices.npy")
        stamps = np.load(episode / "timestamps_sec.npy")
        source_stride = int(round((1.0 / float(np.median(np.diff(stamps)))) / timing.fps))
        last_start = len(frame_ids) - 1 - timing.steps * source_stride
        starts = np.unique(np.linspace(0, max(last_start, 0), args.windows_per_episode, dtype=int))
        rows = [r for r in (scan_window(episode, int(s), args, timing) for s in starts) if r]
        if not rows:
            print(f"{name}: no scannable window")
            continue
        phantom = sum(r["phantom"] for r in rows)
        total = sum(r["n"] for r in rows)
        by_label: dict[int, dict] = {}
        for r in rows:
            for label, lr in r["per_label"].items():
                agg = by_label.setdefault(label, dict(n=0, phantom=0))
                agg["n"] += lr["n"]
                agg["phantom"] += lr["phantom"]
        report[name] = dict(windows=len(rows), selected=total, phantom=phantom, per_label=by_label)
        label_str = " ".join(f"L{k}:{v['phantom']}/{v['n']}" for k, v in sorted(by_label.items()))
        print(
            f"{name}: phantom-in-top{args.select_top_n} {phantom}/{total} ({100 * phantom / total:.1f}%)  {label_str}"
        )

    out = args.output or Path(f"pointflow_outputs/phantom_drift_scan_{len(names)}ep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    sel = sum(r["selected"] for r in report.values())
    ph = sum(r["phantom"] for r in report.values())
    print(f"\nTOTAL: {ph}/{sel} selected points are phantom ({100 * ph / max(sel, 1):.1f}%) -> {out}")


if __name__ == "__main__":
    main()
