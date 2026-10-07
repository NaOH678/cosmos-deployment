#!/usr/bin/env python3
"""Re-encode a video prefix and compare it with the cached VAE latent, numerically.

The latent cache is the one link in the PointFlow chain that cannot be checked
without the VAE: its file records ``original_size``/``padded_size``/``padded_frames``,
which proves the *shape* contract matches the video cache, but says nothing about
whether the same pixels went in. If the cache was built from a different compose or
a different canvas, every sample trains against a latent that does not describe its
video, and nothing downstream notices.

The Wan2.2 VAE is causal in time, so latent frame ``t`` depends only on source
frames ``<= 4t + 1``: a prefix encode is bit-equivalent to the prefix of the full
encode, so ``--frames 33`` checks the first 9 latent frames in seconds. Pass the
episode's full frame count to check all of them -- the generator replicates the last
frame out to ``1 + 4k``, and this repeats that so the lengths line up.

    python tools/verify_vae_latent_cache.py \
        --raw-root .../raw_data/singlerighthand_sandwich_100 \
        --cache-root .../singlerighthand-sandwich-100-cosmos-cache \
        --vae-path .../models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth \
        --episode episode_0019_20260731_134304 \
        --frames 33          # prefix; pass the episode frame count for all of it
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--vae-path", type=Path, required=True)
    p.add_argument("--episode", required=True)
    p.add_argument(
        "--frames",
        type=int,
        default=33,
        help="how many source frames to read; pass the episode's full frame count to check the whole "
        "latent, or a small prefix (33 -> 9 latent frames) for a quick check",
    )
    p.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args()

    video_dir = args.raw_root / args.episode / "videos"
    head = _OpenCVFrameReader(video_dir / "head.mp4").get_frames(np.arange(args.frames))
    wrist = _OpenCVFrameReader(video_dir / "right_wrist.mp4").get_frames(np.arange(args.frames))
    composed = SingleRightHandRawDataset._compose_views(head, wrist)
    resized, image_size = resize_to_resolution_content(composed, "480")
    print(f"composed {tuple(composed.shape)} -> resized {tuple(resized.shape)}, image_size={image_size}")

    # Same padding the latent cache generator applied: replicate out to the canvas
    # the video pipeline targets, and repeat the last frame out to 1 + 4k in time.
    # The target comes from the cache manifest -- padding only to the next multiple
    # of 16 would give 720, while the cache used the 736 canvas the VAE sees. The
    # dataset hands over [T,C,H,W]; Wan2.2 wants [B,C,T,H,W].
    row = next(
        r for r in json.loads((args.cache_root / "video_manifest.json").read_text())["episodes"]
        if r["name"] == args.episode
    )
    target_h, target_w, content_h, content_w = (int(v) for v in row["image_size"])
    if (content_h, content_w) != tuple(resized.shape[-2:]):
        raise SystemExit(
            f"FAIL: resized content {tuple(resized.shape[-2:])} does not match the cache manifest "
            f"{(content_h, content_w)}; the video cache and this checker disagree about the canvas"
        )

    x = resized.permute(1, 0, 2, 3).unsqueeze(0).float()
    if x.max() > 1.5:
        x = x / 127.5 - 1.0
    original_h, original_w = x.shape[-2:]
    padded_h, padded_w = ((target_h + 15) // 16) * 16, ((target_w + 15) // 16) * 16
    x = F.pad(x, (0, padded_w - original_w, 0, padded_h - original_h, 0, 0), mode="replicate")
    original_t = x.shape[2]
    target_t = 1 + 4 * ((original_t - 1 + 3) // 4)
    if target_t != original_t:
        x = torch.cat([x, x[:, :, -1:].expand(-1, -1, target_t - original_t, -1, -1)], dim=2)
    print(f"padded to {tuple(x.shape)} (H,W {original_h}x{original_w} -> {padded_h}x{padded_w})")

    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    tokenizer = Wan2pt2VAEInterface(vae_path=str(args.vae_path), causal=True)
    with torch.inference_mode():
        fresh = tokenizer.encode(x.to(device="cuda", dtype=torch.bfloat16)).cpu()
    print(f"fresh encode {tuple(fresh.shape)}")

    cached_file = torch.load(args.cache_root / "vae_latents" / f"{args.episode}.pt", map_location="cpu", weights_only=True)
    cached = cached_file["latent"]
    print(f"cached latent {tuple(cached.shape)} (original_size={cached_file['original_size']}, "
          f"padded_size={cached_file['padded_size']})")

    if tuple(cached.shape[:-3]) != tuple(fresh.shape[:-3]):
        raise SystemExit(f"FAIL: channel/batch mismatch {cached.shape} vs {fresh.shape}")
    if tuple(cached.shape[-2:]) != tuple(fresh.shape[-2:]):
        raise SystemExit(
            f"FAIL: latent spatial size differs: cached {tuple(cached.shape[-2:])} vs fresh "
            f"{tuple(fresh.shape[-2:])} -- the cache was built for a different canvas"
        )

    overlap = min(cached.shape[2], fresh.shape[2])
    a, b = cached[:, :, :overlap].float(), fresh[:, :, :overlap].float()
    difference = (a - b).abs()
    scale = b.abs().mean() + 1e-6
    print(f"\ncompared {overlap} latent frames")
    print(f"  mean |cached| {a.abs().mean():.4f}   mean |fresh| {b.abs().mean():.4f}")
    print(f"  mean |diff|   {difference.mean():.6f}   max |diff| {difference.max():.6f}")
    print(f"  relative      {difference.mean() / scale:.6f}")

    if difference.mean() > args.atol:
        raise SystemExit(
            "FAIL: the cached latent does not match a fresh encode of the same video. "
            "Regenerate vae_latents/ before training."
        )
    print("\nOK: the latent cache describes the same pixels as the video cache.")


if __name__ == "__main__":
    main()
