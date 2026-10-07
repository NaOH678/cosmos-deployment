"""Standalone PointFlow window dataset; not yet wired into Cosmos joint training."""

import bisect
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.pointflow_window import PointFlowTiming, prepare_window


class PointFlowWindowDataset(Dataset):
    """Deterministic map-style dataset with explicit episode split membership.

    Training shuffling belongs to the caller's sampler. Future labels never decide
    window membership, anchor point selection, or the sample's random seed.
    """

    def __init__(
        self, root, episodes, *, timing: PointFlowTiming, sample_stride=1, max_points=8192, voxel_size=0.02, seed=0
    ):
        if sample_stride < 1 or type(sample_stride) is not int:
            raise ValueError("sample_stride must be a positive integer")
        if not episodes or len(set(episodes)) != len(episodes):
            raise ValueError("Provide a nonempty list of unique episode names for this split")
        if max_points < 3 or not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("Invalid point budget or voxel size")
        self.root, self.episodes = Path(root), tuple(episodes)
        self.timing, self.sample_stride = timing, sample_stride
        self.max_points, self.voxel_size, self.seed = max_points, voxel_size, seed
        self.starts, self.ends = [], []
        total = 0
        for episode in self.episodes:
            ids = np.load(self.root / episode / "frame_indices.npy", allow_pickle=False)
            stamps = np.load(self.root / episode / "timestamps_sec.npy", allow_pickle=False)
            if (
                ids.ndim != 1
                or stamps.ndim != 1
                or len(ids) != len(stamps)
                or len(ids) < 2
                or not np.issubdtype(ids.dtype, np.integer)
                or not np.isfinite(stamps).all()
                or not np.all(np.diff(ids) == 1)
            ):
                raise ValueError(f"{episode}: expected consecutive integer frame IDs and finite timestamps")
            delta = np.diff(stamps)
            if delta[0] <= 0 or not np.allclose(delta, delta[0], atol=1e-6, rtol=0):
                raise ValueError(f"{episode}: nonuniform source timing")
            ratio = 1 / (delta[0] * timing.fps)
            stride = int(round(ratio))
            if stride < 1 or not np.isclose(ratio, stride, atol=1e-5, rtol=0):
                raise ValueError(f"{episode}: source FPS must be an integer multiple of Cosmos FPS")
            last = len(ids) - 1 - timing.steps * stride
            starts = ids[np.arange(0, last + 1, sample_stride)] if last >= 0 else ids[:0]
            self.starts.append(starts)
            total += len(starts)
            self.ends.append(total)
        if not total:
            raise ValueError("No complete windows for this Cosmos time configuration")

    @classmethod
    def from_cosmos(cls, root, episodes, dataset_config, tokenizer_config, **kwargs):
        return cls(
            root,
            episodes,
            timing=PointFlowTiming.from_cosmos(dataset_config, tokenizer_config),
            sample_stride=dataset_config.get("sample_stride", 1),
            **kwargs,
        )

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.ends, index)
        previous = self.ends[episode_index - 1] if episode_index else 0
        start = int(self.starts[episode_index][index - previous])
        episode = self.episodes[episode_index]
        sample_seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{episode}:{start}".encode()).digest()[:8], "little")
        window = prepare_window(
            self.root / episode,
            start,
            self.max_points,
            self.voxel_size,
            sample_seed,
            timing=self.timing,
            allow_empty=True,
        )
        return pointflow_sample(window, episode, start, sample_seed, self.timing)


def pointflow_sample(window, episode, start, sample_seed, timing):
    # Keep the model-visible inputs and the future labels in separate containers.
    input_keys = (
        "point_ids",
        "anchor_xyz",
        "anchor_uv",
        "normal",
        "color",
        "coord",
        "feat",
        "grid_coord",
        "original_to_voxel",
        "voxel_representatives",
        "coord_shift",
        "intrinsics_normalized",
        "image_size_wh",
    )
    return {
        "inputs": {key: window[key] for key in input_keys},
        "targets": {"displacement": window["target_displacement"], "valid": window["target_valid"]},
        "metadata": {
            "episode": episode,
            "start_frame": start,
            "seed": sample_seed,
            "raw_frame_ids": window["raw_frame_ids"],
            "timestamps_sec": window["timestamps_sec"],
            "target_seconds": window["target_seconds"],
            "point_block_seconds": window["point_block_seconds"],
            "timing": timing,
            "empty_anchor": len(window["point_ids"]) == 0,
            "geometry_source": "offline_full_sequence_track4world",
        },
    }


def collate_pointflow_windows(samples):
    """Pack variable point/voxel counts with independent offsets, including empty samples.

    Point IDs are local to an episode: use point_batch/metadata to disambiguate.
    Input -> voxel indices are global within this batch. Targets have [H,sum(N),3].
    No padding is needed, so target valid means label quality only.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    timing = samples[0]["metadata"]["timing"]
    if any(s["metadata"]["timing"] != timing for s in samples):
        raise ValueError("Cannot batch windows with different time contracts")
    points = np.array([len(s["inputs"]["point_ids"]) for s in samples], dtype=np.int64)
    voxels = np.array([len(s["inputs"]["coord"]) for s in samples], dtype=np.int64)
    point_bounds = np.r_[0, np.cumsum(points)]
    voxel_bounds = np.r_[0, np.cumsum(voxels)]
    inputs = {}
    for key in ("point_ids", "anchor_xyz", "anchor_uv", "normal", "color", "coord", "feat", "grid_coord"):
        inputs[key] = torch.from_numpy(np.concatenate([s["inputs"][key] for s in samples]))
    for key in ("coord_shift", "intrinsics_normalized", "image_size_wh"):
        inputs[key] = torch.from_numpy(np.stack([s["inputs"][key] for s in samples]))
    inputs["original_to_voxel"] = torch.from_numpy(
        np.concatenate([s["inputs"]["original_to_voxel"] + voxel_bounds[i] for i, s in enumerate(samples)])
    )
    inputs["voxel_representatives"] = torch.from_numpy(
        np.concatenate([s["inputs"]["voxel_representatives"] + point_bounds[i] for i, s in enumerate(samples)])
    )
    inputs["point_offsets"] = torch.from_numpy(point_bounds[1:])
    inputs["voxel_offsets"] = torch.from_numpy(voxel_bounds[1:])
    inputs["point_batch"] = torch.from_numpy(np.repeat(np.arange(len(samples)), points))
    inputs["voxel_batch"] = torch.from_numpy(np.repeat(np.arange(len(samples)), voxels))
    inputs["has_geometry"] = torch.from_numpy(points > 0)
    return {
        "inputs": inputs,
        "targets": {
            key: torch.from_numpy(np.concatenate([s["targets"][key] for s in samples], axis=1))
            for key in ("displacement", "valid")
        },
        "metadata": [s["metadata"] for s in samples],
    }
