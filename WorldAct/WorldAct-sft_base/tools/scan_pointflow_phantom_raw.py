#!/usr/bin/env python3
"""Population-level phantom-drift scan over raw labeled track arrays (no training path).

Reads position/uv/valid [T,N,*] directly and flags "phantom steps" between
sampled frames ~267 ms apart: |Δxyz| > STEP_DISP_MM while |Δuv| < STEP_UV_PX --
motion along the camera ray that the image does not show. Reports per episode,
per semantic label: step-level phantom fraction and per-track phantom load.
Cross-checks the training-path scan (scan_pointflow_phantom_drift.py), which
measures exposure after selection; this measures the raw population.

    .venv/bin/python tools/scan_pointflow_phantom_raw.py \
        --allowlist examples/pointflow_sandwich_labeled_29_episodes.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_framework.data.pointflow_window import read_frame

FRAME_STRIDE = 8  # source frames (~267 ms at 30 fps); thresholds scale with it
STEP_DISP_MM = 40.0
STEP_UV_PX = 2.0
LABELED_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pf_out/sandwich/labeled")
LABELS = {1: "fingertip", 2: "hand", 3: "lettuce", 4: "bread/cheese"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--allowlist", type=Path, required=True)
    p.add_argument("--labeled-root", type=Path, default=LABELED_ROOT)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    names = [l.strip() for l in args.allowlist.read_text().splitlines() if l.strip() and not l.startswith("#")]
    report = {}
    for name in names:
        episode = args.labeled_root / name
        frame_ids = np.load(episode / "frame_indices.npy")
        labels = np.load(episode / "region_labels.npy").reshape(-1)
        rows = np.arange(0, len(frame_ids), FRAME_STRIDE)

        d3s, d2s, oks = [], [], []
        for a, b in zip(rows[:-1], rows[1:]):
            pa = read_frame(episode / "position.npy", int(a)).reshape(-1, 3)
            pb = read_frame(episode / "position.npy", int(b)).reshape(-1, 3)
            ua = read_frame(episode / "uv_px.npy", int(a)).reshape(-1, 2)
            ub = read_frame(episode / "uv_px.npy", int(b)).reshape(-1, 2)
            va = read_frame(episode / "valid.npy", int(a)).reshape(-1)
            vb = read_frame(episode / "valid.npy", int(b)).reshape(-1)
            ok = va & vb & np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
            ok &= (pa[:, 2] > 0) & (pb[:, 2] > 0) & np.isfinite(ua).all(1) & np.isfinite(ub).all(1)
            d3s.append(np.linalg.norm((pb - pa).astype(np.float64), axis=1) * 1000)
            d2s.append(np.linalg.norm((ub - ua).astype(np.float64), axis=1))
            oks.append(ok)

        d3, d2, ok = np.stack(d3s), np.stack(d2s), np.stack(oks)
        phantom = ok & (d3 > STEP_DISP_MM) & (d2 < STEP_UV_PX)
        per_label = {}
        for label in sorted(np.unique(labels).tolist()):
            m = labels == label
            pl, tl = int(phantom[:, m].sum()), int(ok[:, m].sum())
            valid_counts = ok[:, m].sum(0)
            long_enough = valid_counts > 10
            track_frac = phantom[:, m].sum(0)[long_enough] / np.maximum(valid_counts[long_enough], 1)
            per_label[LABELS.get(label, str(label))] = dict(
                phantom_step_pct=round(100 * pl / max(tl, 1), 3),
                tracks_gt10pct_phantom=int((track_frac > 0.1).sum()),
                tracks_total=int(long_enough.sum()),
            )
        report[name] = dict(
            frames=int(len(frame_ids)),
            points=int(len(labels)),
            phantom_step_pct=round(100 * int(phantom.sum()) / max(int(ok.sum()), 1), 3),
            per_label=per_label,
        )
        parts = " ".join(f"{k}:{v['phantom_step_pct']}%" for k, v in per_label.items())
        print(f"{name}: phantom {report[name]['phantom_step_pct']}% of point-steps  ({parts})", flush=True)

    out = args.output or Path("pointflow_outputs/phantom_drift_raw_scan.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
