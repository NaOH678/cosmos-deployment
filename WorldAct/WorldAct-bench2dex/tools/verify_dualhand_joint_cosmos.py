#!/usr/bin/env python3
"""Verify a prepared dual-hand joint Cosmos dataset without loading training code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=98)
    parser.add_argument("--expected-total-frames", type=int, default=146272)
    parser.add_argument("--require-video-cache", action="store_true")
    return parser.parse_args()


def _read_npy_metadata(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with path.open("rb") as file:
        version = np.lib.format.read_magic(file)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
        else:
            raise ValueError(f"Unsupported NPY version {version} in {path}")
        data_offset = file.tell()
    if fortran_order:
        raise ValueError(f"Fortran-order NPY cache is not supported: {path}")
    expected_size = data_offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Truncated NPY cache {path}: expected {expected_size} bytes, got {actual_size}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def main() -> None:
    args = _parse_args()
    root = args.dataset_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "cosmos_dualhand_joint_v1":
        raise ValueError(f"Unexpected manifest schema: {manifest.get('schema')!r}")
    if manifest.get("arm_action_space") != "joint":
        raise ValueError(f"Expected arm_action_space='joint', found {manifest.get('arm_action_space')!r}")
    episodes = manifest["episodes"]
    if len(episodes) != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, found {len(episodes)}")
    if int(manifest["total_frames"]) != args.expected_total_frames:
        raise ValueError(
            f"Expected {args.expected_total_frames} total frames, found {manifest['total_frames']}"
        )

    for index, row in enumerate(episodes, start=1):
        name = str(row["name"])
        num_frames = int(row["num_frames"])
        with np.load(root / "episodes" / f"{name}.npz", allow_pickle=False) as data:
            state = np.asarray(data["state"])
            action = np.asarray(data["action"])
        expected = (num_frames, 54)
        if state.shape != expected or action.shape != expected:
            raise ValueError(f"{name}: expected {expected}, got {state.shape}/{action.shape}")
        if state.dtype != np.float32 or action.dtype != np.float32:
            raise ValueError(f"{name}: expected float32 state/action")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{name}: state/action contains NaN or Inf")
        print(f"[numeric {index:03d}/{len(episodes):03d}] {name}: {num_frames}", flush=True)

    video_manifest_path = root / "video_manifest.json"
    if args.require_video_cache and not video_manifest_path.is_file():
        raise FileNotFoundError(video_manifest_path)
    if video_manifest_path.is_file():
        video_manifest = json.loads(video_manifest_path.read_text())
        if video_manifest.get("schema") != "cosmos_dualhand_joint_video_v1":
            raise ValueError(f"Unexpected video manifest schema: {video_manifest.get('schema')!r}")
        video_rows = {str(row["name"]): row for row in video_manifest["episodes"]}
        if set(video_rows) != {str(row["name"]) for row in episodes}:
            raise ValueError("Numeric and video manifests list different episodes")
        for index, episode in enumerate(episodes, start=1):
            name = str(episode["name"])
            row = video_rows[name]
            shape, dtype = _read_npy_metadata(root / str(row["path"]))
            expected = tuple(int(value) for value in row["shape"])
            if dtype != np.uint8 or shape != expected:
                raise ValueError(f"{name}: invalid video cache {dtype} {shape}, expected {expected}")
            if shape[0] != int(episode["num_frames"]) or shape[1] != 3:
                raise ValueError(f"{name}: video cache must be [num_frames,3,H,W], got {shape}")
            print(f"[video   {index:03d}/{len(episodes):03d}] {name}: {expected}", flush=True)

    print(
        f"[PASS] {len(episodes)} dual-hand episodes, {manifest['total_frames']} frames, "
        f"54D joint state/action; video_cache={video_manifest_path.is_file()}"
    )


if __name__ == "__main__":
    main()
