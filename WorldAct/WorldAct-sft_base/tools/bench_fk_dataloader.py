# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Time ``SingleRightHandRawDataset.__getitem__`` and its components, on CPU.

The training step logs ``timer/dataloader_train`` = 1.05 s out of 4.53 s, so the
question this answers is *where inside a sample* that second goes: mmap read,
frame stack, resize/pad, FK annotation read, or window-latent read.

Run from the worktree root with the sibling venv:

    PYTHONPATH=. <venv>/bin/python tools/bench_fk_dataloader.py --samples 8

This is a CPU-only measurement; it does not touch a GPU and does not reproduce
the training node's page-cache state.  Read it as a *breakdown*, not as the
absolute per-sample cost on the training host.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault(
    "SINGLERIGHTHAND_RAW_ROOT",
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100",
)
os.environ.setdefault(
    "SINGLERIGHTHAND_CACHE_ROOT",
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache",
)
os.environ.setdefault(
    "FK_ANNOTATION_ROOT",
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/sandwich_fk21",
)
os.environ.setdefault("SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT", os.environ["SINGLERIGHTHAND_CACHE_ROOT"] + "/vae_window_latents")

import torch  # noqa: E402

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import (  # noqa: E402
    SingleRightHandRawDataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--allowlist", default="examples/singlerighthand_101_episodes.txt")
    args = parser.parse_args()

    dataset = SingleRightHandRawDataset(
        root=os.environ["SINGLERIGHTHAND_RAW_ROOT"],
        cache_root=os.environ["SINGLERIGHTHAND_CACHE_ROOT"],
        fps=15.0,
        chunk_length=32,
        split="train",
        split_seed=42,
        split_val_ratio=0.2,
        sample_stride=1,
        mode="wam",
        use_state=True,
        use_precomputed_video=True,
        video_cache_resolution="480",
        video_decoder="opencv",
        vae_window_latent_root=os.environ["SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT"],
        fk_root=os.environ["FK_ANNOTATION_ROOT"],
        fk_steps_per_token=4,
        episode_allowlist=args.allowlist,
    )
    print(f"dataset: {len(dataset)} windows over {len(dataset._episodes)} episodes")

    # This dev node's GPFS mount refuses mmap (OSError: [Errno 19] No such device),
    # so read the video cache eagerly instead.  Changes the absolute I/O cost, not
    # the CPU-work breakdown the script exists to measure.
    def _get_precomputed_video_no_mmap(episode_idx):
        cached = dataset._video_array_cache.get(episode_idx)
        if cached is not None:
            dataset._video_array_cache.move_to_end(episode_idx)
            return cached
        row = dataset._video_cache_rows[dataset._episodes[episode_idx].name]
        cached = np.load(dataset._cache_root / str(row["path"]), allow_pickle=False)
        dataset._video_array_cache[episode_idx] = cached
        while len(dataset._video_array_cache) > dataset._video_array_cache_size:
            dataset._video_array_cache.popitem(last=False)
        return cached

    import numpy as np

    dataset._get_precomputed_video = _get_precomputed_video_no_mmap

    # Component timers, wrapped around the methods __getitem__ calls.
    timings: dict[str, list[float]] = {}
    originals = {
        "_load_video": dataset._load_video,
        "_load_arrays": dataset._load_arrays,
        "_read_window_latent": dataset._read_window_latent,
    }

    def wrap(name, fn):
        def timed(*a, **kw):
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                timings.setdefault(name, []).append(time.perf_counter() - start)

        return timed

    dataset._load_video = wrap("load_video", originals["_load_video"])
    dataset._load_arrays = wrap("load_arrays", originals["_load_arrays"])
    dataset._read_window_latent = wrap("read_window_latent", originals["_read_window_latent"])

    if dataset._fk_source is not None:
        fk_load = dataset._fk_source.load
        dataset._fk_source.load = wrap("fk_load", fk_load)

    # Consecutive indices = one shuffle block, i.e. what a worker actually streams.
    first = int(dataset._cumulative_ends[0]) - int(dataset._valid_windows[0])
    indices = list(range(first, first + args.samples))

    total = []
    for i, idx in enumerate(indices):
        start = time.perf_counter()
        sample = dataset[idx]
        elapsed = time.perf_counter() - start
        total.append(elapsed)
        if i == 0:
            print(
                "  video tensor:",
                tuple(sample["video"].shape),
                sample["video"].dtype,
                f"{sample['video'].numel() * sample['video'].element_size() / 1e6:.1f} MB",
            )
        print(f"  sample {i:2d}  total {elapsed * 1000:7.1f} ms")

    print("\nmean per sample:")
    print(f"  __getitem__ total      {sum(total) / len(total) * 1000:7.1f} ms")
    for name, values in timings.items():
        mean = sum(values) / len(values) * 1000
        print(f"  {name:22s} {mean:7.1f} ms   (n={len(values)})")

    # The remaining slice is the frame cache bookkeeping + torch.stack + permute.
    print("\ncache sizes:")
    print("  composed_frame_cache_size:", dataset._composed_frame_cache_size)
    print("  video_array_cache_size   :", dataset._video_array_cache_size)
    print("  reader_cache_size        :", dataset._reader_cache_size)
    print("  array_cache_size         :", dataset._array_cache_size)


if __name__ == "__main__":
    main()
