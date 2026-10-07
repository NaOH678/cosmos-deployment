#!/usr/bin/env python3
"""Encode preprocessed video tensors with the Cosmos Wan2.2 VAE.

Input tensors are ``[T,3,H,W]`` uint8 or float and are saved as one latent
file per input. This is the offline building block used by the PointFlow
training cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help=".pt tensor [T,3,H,W] or [B,3,T,H,W]")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vae-path", type=Path, required=True)
    p.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), default=None)
    args = p.parse_args()
    x = torch.load(args.input, map_location="cpu", weights_only=True)
    if x.ndim == 4:
        # Dataset tensors are [T,C,H,W]; Wan2.2 expects [B,C,T,H,W].
        x = x.permute(1, 0, 2, 3).unsqueeze(0)
    elif x.ndim == 5 and x.shape[2] != 3 and x.shape[1] == 3:
        # Accept [B,C,T,H,W] directly; reject ambiguous layouts below.
        pass
    if x.ndim != 5:
        raise ValueError(f"expected [T,3,H,W] or [B,3,T,H,W], got {tuple(x.shape)}")
    original_t = x.shape[2]
    target_t = 1 if original_t == 1 else 1 + 4 * ((original_t - 1 + 3) // 4)
    if target_t != original_t:
        x = torch.cat([x, x[:, :, -1:].expand(-1, -1, target_t - original_t, -1, -1)], dim=2)
    original_h, original_w = x.shape[-2:]
    padded_h, padded_w = args.image_size or (original_h, original_w)
    if padded_h < original_h or padded_w < original_w:
        raise ValueError("--image-size must contain the full content size")
    padded_h = ((padded_h + 15) // 16) * 16
    padded_w = ((padded_w + 15) // 16) * 16
    if (padded_h, padded_w) != (original_h, original_w):
        x = F.pad(x, (0, padded_w - original_w, 0, padded_h - original_h, 0, 0), mode="replicate")
    x = x.float()
    if x.max() > 1.5:
        x = x / 127.5 - 1.0
    tok = Wan2pt2VAEInterface(vae_path=str(args.vae_path), causal=True)
    with torch.inference_mode():
        z = tok.encode(x.to(device="cuda", dtype=torch.bfloat16)).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent": z,
            "original_frames": original_t,
            "padded_frames": target_t,
            "original_size": [original_h, original_w],
            "padded_size": [padded_h, padded_w],
            "vae": str(args.vae_path),
            "shape": list(z.shape),
        },
        args.output,
    )
    print({"output": str(args.output), "latent_shape": list(z.shape)})


if __name__ == "__main__":
    main()
