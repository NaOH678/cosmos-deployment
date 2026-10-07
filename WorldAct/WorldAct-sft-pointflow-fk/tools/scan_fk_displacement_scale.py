#!/usr/bin/env python3
"""Measure ``fk_displacement_scale`` from the training windows.

The FK counterpart of ``scan_pointflow_selection.py``, and the reason the number
in the experiment config is what it is.  Run it before changing that value, and
after changing which episodes train: the scale is a property of the *data
selection*, not of the model.

Why it has to be set at all: the RF forward process is
``x_sigma = sigma * eps + (1 - sigma) * target`` with ``eps`` unit-variance
Gaussian, so ``target`` has to be unit-scale too or the two terms are not
comparable.  Every other modality in this pipeline already satisfies that -- the
video VAE latent measures std 0.872, the action is normalised by
``quantile_rot``.  FK stores its labels in metres (``displacement = camera[1:] -
camera[0]``), so on its own it is the one modality that does not, and the branch
ends up training almost entirely on the noise term.

A std rather than a min/max span, for the reason the pointflow tool gives: the
noise is unit-variance Gaussian, so the matching statistic is the std.

Prints the pooled per-element std (the value to use) and, as a check that the
window enumeration here matches the one training actually uses, the mean
per-keypoint displacement norm per window -- that is the same quantity the
trainer logs as ``fk_zero_ade_mm``.  If the two disagree, this scan is
enumerating different windows and its std should not be trusted.

    python tools/scan_fk_displacement_scale.py
    python tools/scan_fk_displacement_scale.py --episodes examples/other.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_framework.data.fk_camera_extrinsic import base_to_camera  # noqa: E402

DEFAULT_ROOT = "/data/shichaojian/raw_data/sandwich_fk21"
DEFAULT_EPISODES = Path("examples/pointflow_sandwich_10_episodes.txt")

# Must match the experiment config's action dataset, or this measures a
# different set of windows than the one the branch trains on.
CHUNK_LENGTH = 32
SOURCE_STRIDE = 2
SAMPLE_STRIDE = 1
HAND_INDEX = 1  # right hand; the left is not annotated as observed here


def windows(positions: np.ndarray):
    """Every training window of one episode, as ``[chunk+1, 21, 3]`` slices."""
    last_start = len(positions) - 1 - CHUNK_LENGTH * SOURCE_STRIDE
    if last_start < 0:
        return
    for start in range(0, last_start + 1, SAMPLE_STRIDE):
        ids = start + np.arange(CHUNK_LENGTH + 1) * SOURCE_STRIDE
        camera = base_to_camera(positions[ids])
        yield camera[1:] - camera[0]


def main() -> None:
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="FK annotation tree (one dir per episode)")
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES, help="episode allowlist")
    args = parser.parse_args()

    episodes = [line.strip() for line in args.episodes.read_text().split() if line.strip()]
    root = Path(args.root)

    # Streaming, one window at a time.  The obvious version -- collect every
    # window into a list and concatenate -- holds the whole pooled array twice at
    # the peak, and that does not scale: the 10-episode set is 26.5M elements
    # (~0.2 GB) but the 101-episode set is ~268M (~4 GB peak), which is enough to
    # matter on a shared node for no reason.  Each window is small, so the mean
    # and M2 are folded in per window with Chan's parallel combination, which is
    # the numerically stable way to do it in one pass -- the naive
    # ``sum(x**2)/n - mean**2`` loses digits to cancellation at this length.
    count = 0
    mean = 0.0
    m2 = 0.0
    peak = 0.0
    norms: list[float] = []
    for episode in episodes:
        path = root / episode / "annotations" / "wuji_fk21.npz"
        if not path.is_file():
            raise SystemExit(f"missing annotation: {path}")
        with np.load(path, allow_pickle=True) as store:
            # A different frame or unit silently rescales every number below.
            if str(store["coordinate_frame"]) != "Link_Base" or str(store["units"]) != "metre":
                raise SystemExit(
                    f"{episode}: expected Link_Base metres, got {store['coordinate_frame']!r} {store['units']!r}"
                )
            positions = np.asarray(store["positions"])[:, HAND_INDEX]
        for displacement in windows(positions):
            flat = displacement.reshape(-1).astype(np.float64)
            n_b = flat.size
            mean_b = float(flat.mean())
            m2_b = float(((flat - mean_b) ** 2).sum())
            delta = mean_b - mean
            total = count + n_b
            m2 += m2_b + delta * delta * count * n_b / total
            mean += delta * n_b / total
            count = total
            peak = max(peak, float(np.abs(flat).max()))
            norms.append(float(np.linalg.norm(displacement, axis=-1).mean()))

    if count == 0:
        raise SystemExit("no windows; check the allowlist and the chunk length")

    norm = float(np.mean(norms))
    print(f"episodes        {len(episodes)}")
    print(f"elements        {count}")
    print(f"per-element std {np.sqrt(m2 / count):.6f} m      <- fk_displacement_scale")
    print(f"per-element max {peak:.4f} m")
    print()
    print(f"mean per-keypoint displacement per window {norm * 1000:.1f} mm")
    print("  compare against train/fk_zero_ade_mm in the training log; the same")
    print("  quantity computed over a different window set would not match.")


if __name__ == "__main__":
    main()
