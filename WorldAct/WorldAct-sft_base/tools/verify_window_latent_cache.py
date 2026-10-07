#!/usr/bin/env python3
"""Check that a cached window latent equals a fresh online encode of that window.

Independent of the generator: this reads the 33 frames from the **mp4s**, composes
and resizes them (the video cache's own path), pads with the real
``reflection_pad_to_target``, normalizes the way the model does, encodes, and
compares against what ``tools/cache_window_vae_latents.py`` stored. Reading from a
different source and re-running the whole chain means a wrong frame index, the wrong
padding mode, a wrong normalization or a broken uint16 storage round-trip all show up.

    python tools/verify_window_latent_cache.py \
        --raw-root .../raw_data/singlerighthand_sandwich_100 \
        --cache-root .../singlerighthand-sandwich-100-cosmos-cache \
        --vae-path .../models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth \
        --episode episode_0019_20260731_134304

By default it samples five windows spread across the episode's own range; pass
``--windows 0,100,200`` to pin them. Offsets past the episode's end are rejected
rather than silently re-clamped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import (
    SingleRightHandRawDataset,
    _OpenCVFrameReader,
)
from cosmos_framework.data.generator.action.transforms import reflection_pad_to_target
from tools.cache_window_vae_latents import STORAGE, window_plan
from tools.prepare_singlerighthand_video_cache import resize_to_resolution_content


def read_window_latent(path: Path, index: int, shape, storage: str) -> torch.Tensor:
    """One window out of the [windows, C, T, H, W] uint16 npy, back to bfloat16."""
    if storage != STORAGE:
        raise ValueError(f"unexpected storage {storage!r}; expected {STORAGE!r}")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (2, 0):
            file_shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            file_shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        if tuple(file_shape[1:]) != tuple(shape) or dtype != np.uint16:
            raise ValueError(f"{path}: header {file_shape} {dtype} does not match manifest {shape} uint16")
        header = stream.tell()
        item = int(np.prod(file_shape[1:])) * dtype.itemsize
        if not 0 <= index < file_shape[0]:
            raise IndexError(f"{path}: window {index} out of range 0..{file_shape[0] - 1}")
        stream.seek(header + index * item)
        raw = np.frombuffer(stream.read(item), dtype=np.uint16).reshape(file_shape[1:])
    return torch.from_numpy(raw.copy()).view(torch.bfloat16)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--vae-path", type=Path, required=True)
    p.add_argument("--episode", required=True)
    p.add_argument(
        "--windows",
        default=None,
        help="comma-separated window offsets; default: five spread over the episode's own range",
    )
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--chunk-length", type=int, default=32)
    p.add_argument("--sample-stride", type=int, default=1)
    p.add_argument("--resolution", default="480")
    p.add_argument("--atol", type=float, default=0.0, help="0 = require an exact match")
    args = p.parse_args()

    cache_root = args.cache_root.resolve()
    manifest = json.loads((cache_root / "vae_window_latents" / "window_manifest.json").read_text())
    if float(manifest["fps"]) != args.fps or int(manifest["chunk_length"]) != args.chunk_length:
        raise ValueError(
            f"cache was built for fps={manifest['fps']} chunk_length={manifest['chunk_length']}, "
            f"asked about fps={args.fps} chunk_length={args.chunk_length}"
        )
    row = next(r for r in manifest["episodes"] if r["name"] == args.episode)

    state = json.loads((cache_root / "manifest.json").read_text())
    episode = next(e for e in state["episodes"] if e["name"] == args.episode)
    expected, source_stride = window_plan(
        int(episode["num_frames"]), float(episode["source_fps"]), args.fps, args.chunk_length, args.sample_stride
    )
    if expected != row["windows"]:
        raise ValueError(f"cache has {row['windows']} windows but the dataset would enumerate {expected}")

    if args.windows is None:
        # Episodes differ in length, so a fixed default list falls off the end of a
        # short one. Spread the sample across this episode's own range instead.
        last = int(row["windows"]) - 1
        starts = sorted({0, last // 4, last // 2, (3 * last) // 4, last})
    else:
        starts = [int(v) for v in args.windows.split(",")]
        outside = [s for s in starts if not 0 <= s < int(row["windows"])]
        if outside:
            raise ValueError(
                f"{args.episode} has windows 0..{int(row['windows']) - 1}; out of range: {outside}"
            )

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    tokenizer = Wan2pt2VAEInterface(
        vae_path=str(args.vae_path),
        causal=True,
        encode_chunk_frames={"256": 68, "480": 24, "720": 8, "768": 8},
        encode_exact_durations=[args.chunk_length + 1],
        spatial_compression_factor=16,
        temporal_compression_factor=4,
    )

    video_dir = args.raw_root / args.episode / "videos"
    head = _OpenCVFrameReader(video_dir / "head.mp4")
    wrist = _OpenCVFrameReader(video_dir / "right_wrist.mp4")
    latent_path = cache_root / "vae_window_latents" / row["path"]

    print(
        f"{args.episode}: {row['windows']} windows, shape {tuple(row['shape'])}, "
        f"content {tuple(row['content_hw'])} -> canvas {tuple(row['target_hw'])}\n"
    )
    header = f"{'window':>7} {'max|diff|':>12} {'mean|diff|':>12} {'rel':>10}"
    print(header)
    print("-" * len(header))
    worst = 0.0
    for start in starts:
        indices = start * args.sample_stride + source_stride * np.arange(args.chunk_length + 1, dtype=np.int64)
        composed = SingleRightHandRawDataset._compose_views(
            head.get_frames(indices), wrist.get_frames(indices)
        )
        resized, _ = resize_to_resolution_content(composed, args.resolution)
        # _compose_views and resize_to_resolution_content both work in [T,C,H,W];
        # the tokenizer wants [B,C,T,H,W]. Skipping this permute makes the encoder
        # read the channel dim as time and trip its 4n+1 assert. This mirrors
        # tools/cache_window_vae_latents.py, which permutes right after loading the
        # TCHW frame cache.
        window = resized.permute(1, 0, 2, 3)
        target_h, target_w = row["target_hw"]
        padded = reflection_pad_to_target({"video": window}, ["video"], True, target_w, target_h)["video"]
        x = padded.unsqueeze(0).cuda().float() / 127.5 - 1.0
        with torch.inference_mode():
            online = tokenizer.encode(x.to(torch.bfloat16))[0].cpu()
        if tuple(online.shape) != tuple(row["shape"]):
            raise ValueError(
                f"online encode gave {tuple(online.shape)} but the cache says {tuple(row['shape'])}; "
                "the input layout is wrong"
            )

        cached = read_window_latent(latent_path, start, row["shape"], row["storage"]).float()
        difference = (cached - online.float()).abs()
        relative = difference.mean().item() / max(online.float().abs().mean().item(), 1e-6)
        worst = max(worst, difference.max().item())
        print(f"{start:>7} {difference.max().item():>12.6f} {difference.mean().item():>12.6f} {relative:>10.6f}")

    print(f"\n最差 max|diff| = {worst:.6f}")
    if worst > args.atol:
        raise SystemExit(
            "FAIL: cached window latent does not match a fresh online encode. "
            "Regenerate vae_window_latents/ before training."
        )
    print("OK: 缓存 == 在线路径(逐位相同)")


if __name__ == "__main__":
    main()
