#!/usr/bin/env python3
"""Diff two window-latent caches element-wise, per window. Usage:

    python tools/latent_diff_check.py <a.npy> <b.npy>

Prints overall stats and the worst windows, to distinguish a handful of blown-up
elements from a wholesale misalignment.
"""

import sys

import numpy as np

a = np.load(sys.argv[1]).astype(np.int64)
b = np.load(sys.argv[2]).astype(np.int64)
if a.shape != b.shape:
    print(f"SHAPE MISMATCH: {a.shape} vs {b.shape}")
    sys.exit(1)
d = np.abs(a - b)
print(f"shape {a.shape}, elements {d.size}")
print(f"identical: {(d == 0).mean():.4f}  <=2 ulp: {(d <= 2).mean():.4f}  >100: {(d > 100).mean():.6f}")
per_window = d.max(axis=(1, 2, 3, 4))
bad = np.flatnonzero(per_window > 100)
print(f"windows with maxdiff>100: {len(bad)}/{len(per_window)}")
print(f"first bad windows: {bad[:10].tolist()}")
print(
    f"per-window maxdiff p50={np.percentile(per_window, 50):.0f} p99={np.percentile(per_window, 99):.0f} max={per_window.max()}"
)
if len(bad):
    w = int(bad[0])
    print(f"window {w}: a bits sample {a[w].ravel()[:4].tolist()}, b bits sample {b[w].ravel()[:4].tolist()}")
# also check cross-window shift: is a[w] closer to b[w-1] or b[w+1]?
shift = min(5, len(a) - 1)
fwd = np.abs(a[shift:] - b[:-shift]).max()
print(f"maxdiff if a[w] vs b[w-{shift}] (shift check): {fwd}")
