#!/usr/bin/env python3
"""Measure how far the cached VAE latent deviates from the online (cache-less) encode.

NVIDIA's pipeline encodes each sample's own ``[C, T, H, W]`` window once per step.
Our latent cache instead stores a whole-episode encode at source FPS and resamples
9 latent frames out of it, so a cached run feeds the model a different video
representation than a cache-less run would. Nothing downstream notices, so the size
of that difference has to be measured directly:

  online        encode(window frames r+2k, 33 of them)          -> 9 latent frames
  cached_floor  whole-episode latent at index floor((r+8k)/4)   -> 9 latent frames
  cached_ceil   same with ceil                                  -> 9 latent frames

The script also reports how much the latent moves when the window shifts by one
15 Hz frame -- that is the natural yardstick for "is this difference meaningful".

    python tools/compare_latent_cache_vs_online.py \
        --raw-root .../raw_data/singlerighthand_sandwich_100 \
        --cache-root .../singlerighthand-sandwich-100-cosmos-cache \
        --vae-path .../models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth \
        --episode episode_0019_20260731_134304 \
        --starts 100,101,102,103
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import (
    SingleRightHandRawDataset,
    _OpenCVFrameReader,
)
from tools.prepare_singlerighthand_video_cache import resize_to_resolution_content

TEMPORAL_DOWNSAMPLE = 2  # 30 fps source -> 15 Hz Cosmos window
CHUNK_LENGTH = 32  # Cosmos chunk_length; the window holds chunk_length + 1 frames


def window_frames(reader_head, reader_wrist, start):
    """The dataset's observation indices and the composed/resized window."""
    indices = start + TEMPORAL_DOWNSAMPLE * np.arange(CHUNK_LENGTH + 1, dtype=np.int64)
    composed = SingleRightHandRawDataset._compose_views(
        reader_head.get_frames(indices), reader_wrist.get_frames(indices)
    )
    return indices, composed


def to_vae_input(resized, target_hw):
    """[T,C,H,W] uint8 -> [1,C,T,H,W] float, replicate-padded to the target canvas."""
    x = resized.permute(1, 0, 2, 3).unsqueeze(0).float()
    if x.max() > 1.5:
        x = x / 127.5 - 1.0
    oh, ow = x.shape[-2:]
    ph, pw = ((target_hw[0] + 15) // 16) * 16, ((target_hw[1] + 15) // 16) * 16
    return F.pad(x, (0, pw - ow, 0, ph - oh, 0, 0), mode="replicate")


def crop_padding(latent, content_hw):
    """Mirror OmniMoTModel._remove_padding_from_latent (spatial factor 16, floor)."""
    return latent[:, :, :, : content_hw[0] // 16, : content_hw[1] // 16]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--vae-path", type=Path, required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--starts", default="100,101,102,103")
    args = p.parse_args()

    starts = [int(v) for v in args.starts.split(",")]
    row = next(
        r for r in json.loads((args.cache_root / "video_manifest.json").read_text())["episodes"]
        if r["name"] == args.episode
    )
    target_h, target_w, content_h, content_w = (int(v) for v in row["image_size"])
    cached_file = torch.load(
        args.cache_root / "vae_latents" / f"{args.episode}.pt", map_location="cpu", weights_only=True
    )
    cached = cached_file["latent"].float()

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    tokenizer = Wan2pt2VAEInterface(
        vae_path=str(args.vae_path),
        causal=True,
        encode_chunk_frames={"256": 68, "480": 24, "720": 8, "768": 8},
        encode_exact_durations=[33],
        spatial_compression_factor=16,
        temporal_compression_factor=4,
    )

    video_dir = args.raw_root / args.episode / "videos"
    head = _OpenCVFrameReader(video_dir / "head.mp4")
    wrist = _OpenCVFrameReader(video_dir / "right_wrist.mp4")

    def online_latent(start):
        _, composed = window_frames(head, wrist, start)
        resized, _ = resize_to_resolution_content(composed, "480")
        x = to_vae_input(resized, (target_h, target_w))
        with torch.inference_mode():
            z = tokenizer.encode(x.to(device="cuda", dtype=torch.bfloat16)).cpu().float()
        return crop_padding(z, (content_h, content_w))

    def cached_at(start, delta):
        indices = np.floor((start + 8 * np.arange(9)) / 4.0).astype(np.int64) + delta
        indices = np.clip(indices, 0, cached.shape[2] - 1)
        return crop_padding(cached.index_select(2, torch.from_numpy(indices)), (content_h, content_w))

    print(f"episode {args.episode}  canvas {target_w}x{target_h}  content {content_w}x{content_h}")
    print(f"cached latent {tuple(cached.shape)}  online window {CHUNK_LENGTH + 1} frames -> 9 latents\n")

    reference = {}
    for start in starts:
        reference[start] = online_latent(start)
    print("reference (online) encodings done:", ", ".join(str(s) for s in starts), "\n")

    header = f"{'start':>6} {'|latent|':>9} {'floor rel':>10} {'ceil rel':>10} {'best d':>7} {'best rel':>10} {'shift1 rel':>11}"
    print(header)
    print("-" * len(header))
    rows = []
    for start in starts:
        ref = reference[start]
        scale = ref.abs().mean().item()
        variants = {}
        for delta in range(-2, 3):
            c = cached_at(start, delta)
            # compare only the 9 latent frames
            n = min(c.shape[2], ref.shape[2])
            variants[delta] = (c[:, :, :n] - ref[:, :, :n]).abs().mean().item() / max(scale, 1e-6)
        best = min(variants, key=variants.get)
        # How much does the latent move when the window advances by one 15 Hz frame?
        other = reference.get(start + 2)
        if other is None:
            other = online_latent(start + 2)
            reference[start + 2] = other
        n = min(ref.shape[2], other.shape[2])
        shift = (other[:, :, :n] - ref[:, :, :n]).abs().mean().item() / max(scale, 1e-6)
        rows.append((start, scale, variants[0], variants[1], best, variants[best], shift))
        print(
            f"{start:>6} {scale:>9.4f} {variants[0]:>10.4f} {variants[1]:>10.4f} "
            f"{best:>+7d} {variants[best]:>10.4f} {shift:>11.4f}"
        )

    arr = np.array([[r[2], r[3], r[5], r[6]] for r in rows])
    print("\n" + "=" * 70)
    print(f"floor 相对差   平均 {arr[:,0].mean():.4f}   最大 {arr[:,0].max():.4f}")
    print(f"ceil  相对差   平均 {arr[:,1].mean():.4f}   最大 {arr[:,1].max():.4f}")
    print(f"最优 delta 相对差 平均 {arr[:,2].mean():.4f}")
    print(f"窗口平移一帧的差  平均 {arr[:,3].mean():.4f}   <- 判断尺度")
    print(f"\n最优 delta 分布: {sorted(set(r[4] for r in rows))}")
    print("\n读法:若 floor/ceil 的相对差与 '窗口平移一帧' 同量级,说明缓存的重采样")
    print("      改变了等效时间对齐;若远小于它,说明差异在实践中可忽略。")


if __name__ == "__main__":
    main()
