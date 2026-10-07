#!/usr/bin/env python3
"""Fit action-only q01/q99 scaling on all training action_valid frames, without reading RGB/latents."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.action.datasets.bench2dex_dataset import Bench2DexDataset
from cosmos_framework.utils.bench2dex_contract import JOINT_NAMES
from cosmos_framework.utils.bench2dex_normalization import METHOD, SCHEMA, action_quantile_parameters


def summarize(values):
    return {
        "count": len(values),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_stats(cache_root, split_seed=42, split_val_ratio=0.1, chunk_length=32, sample_stride=1, min_scale=0.05):
    if not np.isfinite(min_scale) or min_scale <= 0:
        raise ValueError("Require min_scale > 0")
    root = Path(cache_root)
    dataset = Bench2DexDataset(
        cache_root=str(root),
        split="train",
        split_seed=split_seed,
        split_val_ratio=split_val_ratio,
        chunk_length=chunk_length,
        sample_stride=sample_stride,
    )
    actions, states, fingerprints = [], [], {}
    for i, episode in enumerate(dataset._episodes):
        state, action = dataset._load_arrays(i)
        fingerprints[episode.name] = hashlib.sha256(
            (root / "episodes" / f"{episode.name}.npz").read_bytes()
        ).hexdigest()
        with np.load(root / "episodes" / f"{episode.name}.npz", allow_pickle=False) as data:
            valid = np.asarray(data["action_valid"], dtype=bool)
        if not valid.any():
            continue
        # Same sampling as Bench2Dex finalize_sim_batch: no overlapping-window weights.
        # The converted cache already excludes homing frames.
        actions.append(action[valid])
        states.append(state[valid])
    action_values, state_values = (np.concatenate(x).astype(np.float64) for x in (actions, states))
    action_stats, state_stats = summarize(action_values), summarize(state_values)
    offset, scale = action_quantile_parameters(action_stats, min_scale)
    return {
        "schema": SCHEMA,
        "method": METHOD,
        "source": "training episodes only; absolute radians; statistics, no clipping applied",
        "units": "radian",
        "joint_names": list(JOINT_NAMES),
        "fit_split": "train",
        "split_seed": split_seed,
        "split_val_ratio": split_val_ratio,
        "chunk_length": chunk_length,
        "sample_stride": sample_stride,
        "fps": 20,
        "frame_weighting": "all_action_valid_frames_once",
        "state_transform": "same_as_action",
        "min_scale": min_scale,
        "forward_clamp": None,
        "offset": offset.tolist(),
        "scale": scale.tolist(),
        "action": action_stats,
        "state": state_stats,
        "train_episodes": [e.name for e in dataset._episodes],
        "train_windows": len(dataset),
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "episode_sha256": fingerprints,
        "normalized_abs_max": {
            "action": float(np.max(np.abs((action_values - offset) / scale))),
            "state": float(np.max(np.abs((state_values - offset) / scale))),
        },
        "normalized_fraction_abs_gt_5": {
            "action": float(np.mean(np.abs((action_values - offset) / scale) > 5)),
            "state": float(np.mean(np.abs((state_values - offset) / scale) > 5)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-val-ratio", type=float, default=0.1)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--min-scale", type=float, default=0.05)
    args = vars(parser.parse_args())
    output = Path(args.pop("output"))
    stats = compute_stats(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(stats, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": len(stats["train_episodes"]),
                "windows": stats["train_windows"],
                "normalized_abs_max": stats["normalized_abs_max"],
            }
        )
    )


if __name__ == "__main__":
    main()
