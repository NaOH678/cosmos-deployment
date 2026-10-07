"""Pick a replacement fixed-eval window, scored on the labels alone.

Why this exists.  ``fk_eval_cases.fixed_cases`` draws its candidates from
``np.linspace(0, len-1, 12)`` -- twelve evenly spaced indices over the whole
split -- and takes the highest-motion one.  On the 10-episode allowlist the val
split is only two episodes, so twelve samples over 2658 windows is coarse enough
to miss everything, and it picked a window whose last frame *is* the last frame
of its episode: the hand is leaving the scene, which is not the motion the model
is being asked to predict anywhere else.  That window is ``val_00``, and it is
the case that plateaus first.

This scans **every** window of the split instead, and ranks by a property the
plots are actually read for: how far the hand is spread along the camera's z
axis.  A window where the fingers are fanned out in depth shows whether the 21
keypoints keep their shape; a closed hand hides exactly that.

The scores come from the FK annotation ``.npz`` directly -- no video decode, no
dataset instantiation, no GPU -- and the camera transform is ``base_to_camera``,
the same one the training labels go through, so "z" here means the z of
``comparison.png``.

    PYTHONPATH=. <venv>/bin/python tools/pick_fk_eval_case.py --split val
    PYTHONPATH=. <venv>/bin/python tools/pick_fk_eval_case.py --split val --emit <index>

``--emit`` prints the ``fixed_cases.json`` entry for that index, ready to seed a
new run's manifest before its first launch (see the module docstring of
``fk_eval_cases`` for why the manifest, and not a config, is the hook).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cosmos_framework.data.fk_camera_extrinsic import base_to_camera

ANNOTATION_ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/sandwich_fk21")
ALLOWLIST = Path(__file__).resolve().parent.parent / "examples/pointflow_sandwich_10_episodes.txt"
CHUNK_LENGTH = 32
FPS = 15.0
SPLIT_SEED = 42
SPLIT_VAL_RATIO = 0.2
RIGHT_HAND = 1  # side_names == ['left', 'right']; only the right hand is observed


def _episodes(split: str):
    """The split's episodes, in the order the dataset enumerates them.

    Mirrors ``SingleRightHandRawDataset``: a seeded ``randperm`` over the
    allowlist, the first ``round(n * ratio)`` being val.  Reproduced rather than
    imported because importing the dataset drags in torchcodec, the VAE-latent
    cache and a torch device -- none of which this scoring needs, and all of
    which can fail for reasons unrelated to picking a window.
    """
    names = [line.strip() for line in ALLOWLIST.read_text().splitlines() if line.strip()]
    n_val = int(round(len(names) * SPLIT_VAL_RATIO))
    # torch.randperm(n, generator=Generator().manual_seed(s)) is Fisher-Yates on
    # the identity permutation with that MT19937 stream; numpy cannot reproduce it
    # bit for bit, so the permutation is checked against the two identities an
    # existing manifest already recorded rather than assumed.
    import torch

    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    chosen = order[:n_val] if split == "val" else order[n_val:]
    return [names[i] for i in chosen]


def _positions(episode: str):
    npz = np.load(ANNOTATION_ROOT / episode / "annotations/wuji_fk21.npz", allow_pickle=True)
    positions = np.asarray(npz["positions"], np.float64)[:, RIGHT_HAND]  # [F, 21, 3] base frame
    return np.asarray(base_to_camera(positions), np.float64)  # [F, 21, 3] camera frame


def score_split(split: str):
    """Every window of the split, with its z-spread, motion and position."""
    rows, cumulative = [], 0
    for episode in _episodes(split):
        camera = _positions(episode)
        frames = len(camera)
        last_start = frames - 1 - CHUNK_LENGTH * 2  # source is 30 fps, sampled at 15
        if last_start < 0:
            continue
        for start in range(last_start + 1):
            ids = start + np.arange(CHUNK_LENGTH + 1) * 2
            window = camera[ids]  # [33, 21, 3]
            z = window[:, :, 2]
            rows.append(
                {
                    "split": split,
                    "index": cumulative + start,
                    "episode": episode,
                    "start_frame": int(start),
                    # Fingers fanned along the camera's z axis: the spread of the
                    # 21 keypoints in z, averaged over the 32 predicted frames.
                    "z_spread_mm": float((z.max(axis=1) - z.min(axis=1)).mean() * 1000.0),
                    "motion_mm": float(np.linalg.norm(window[1:] - window[0], axis=-1).mean() * 1000.0),
                    "position": start / last_start,
                    "frames_in_episode": frames,
                    "frames_after": int(frames - 1 - ids[-1]),
                }
            )
        cumulative += last_start + 1
    return rows


def _identity(row, count_index: int):
    """The manifest entry ``fixed_cases`` will recompute and compare against."""
    ids = [int(row["start_frame"]) + 2 * i for i in range(CHUNK_LENGTH + 1)]
    identity = {
        "split": row["split"],
        "index": int(row["index"]),
        "episode": row["episode"],
        "start_frame": int(row["start_frame"]),
        "raw_frame_ids": ids,
        "point_ids": list(range(21)),
    }
    identity["case_id"] = f"{row['split']}_{count_index:02d}"
    identity["seed"] = int.from_bytes(
        hashlib.sha256(
            f"{SPLIT_SEED}:{row['split']}:{row['episode']}:{row['start_frame']}".encode()
        ).digest()[:4],
        "little",
    )
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--min-position", type=float, default=0.40, help="earliest window start, as a fraction of the episode")
    parser.add_argument("--max-position", type=float, default=0.85, help="latest window start (keeps clear of the episode tail)")
    parser.add_argument("--min-motion-mm", type=float, default=None, help="default: the median motion of the filtered set")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--emit", type=int, default=None, help="print the manifest entry for this index")
    parser.add_argument("--case-index", type=int, default=0, help="case number for --emit (val_00 -> 0)")
    args = parser.parse_args()

    rows = score_split(args.split)
    print(f"{args.split}: {len(rows)} windows, episodes {sorted({r['episode'] for r in rows})}")
    for episode in sorted({r["episode"] for r in rows}):
        per = [r for r in rows if r["episode"] == episode]
        print(f"  {episode}: indices {per[0]['index']}..{per[-1]['index']}  ({len(per)} windows)")

    if args.emit is not None:
        match = [r for r in rows if r["index"] == args.emit]
        if not match:
            print(f"ERROR: index {args.emit} is not in the {args.split} split")
            return 1
        print(json.dumps(_identity(match[0], args.case_index), indent=2))
        return 0

    kept = [r for r in rows if args.min_position <= r["position"] <= args.max_position]
    if not kept:
        print("ERROR: no window survived the position filter")
        return 1
    floor = args.min_motion_mm
    if floor is None:
        floor = float(np.median([r["motion_mm"] for r in kept]))
    kept = [r for r in kept if r["motion_mm"] >= floor]
    kept.sort(key=lambda r: -r["z_spread_mm"])
    # Consecutive windows overlap by 32 of their 33 frames, so the raw ranking
    # returns the same half-second ten times.  Keep only the best per
    # non-overlapping stretch, or the list cannot be chosen from.
    spaced: list[dict] = []
    for row in kept:
        if all(
            row["episode"] != other["episode"]
            or abs(row["start_frame"] - other["start_frame"]) > CHUNK_LENGTH * 2
            for other in spaced
        ):
            spaced.append(row)
    print(
        f"\nposition in [{args.min_position}, {args.max_position}], motion >= {floor:.1f}mm"
        f" -> {len(kept)} candidates, {len(spaced)} non-overlapping"
        f"\nranked by z-spread (higher = fingers more fanned out in depth):\n"
    )
    kept = spaced
    print(f"{'rank':>4} {'index':>6} {'episode':>30} {'start':>6} {'pos':>5} {'z_spread':>9} {'motion':>8} {'tail':>5}")
    for rank, row in enumerate(kept[: args.top], 1):
        print(
            f"{rank:>4} {row['index']:>6} {row['episode']:>30} {row['start_frame']:>6} "
            f"{row['position']:>5.2f} {row['z_spread_mm']:>9.1f} {row['motion_mm']:>8.1f} {row['frames_after']:>5}"
        )
    print("\n('tail' = raw frames remaining after this window ends; 0 would be the episode's last frame)")
    print(f"emit one with:  --emit <index> --case-index 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
