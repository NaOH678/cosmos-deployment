"""Normalize collated offline latents without changing their values."""

import torch


def unpack_cached_vision_latents(cached, videos):
    if isinstance(cached, torch.Tensor):
        if cached.ndim == 6:
            items = list(cached.unbind(0))
        elif cached.ndim == 5:
            items = list(cached.split(1, dim=0))
        else:
            raise ValueError("Invalid collated latent tensor shape")
    elif isinstance(cached, (list, tuple)):
        items = list(cached)
    else:
        raise TypeError("Invalid cached latent batch")
    if len(items) != len(videos):
        raise ValueError("Cached latent/video sample count mismatch")
    result = []
    for item, video in zip(items, videos, strict=True):
        while isinstance(item, torch.Tensor) and item.ndim > 5 and item.shape[0] == 1:
            item = item.squeeze(0)
        if not isinstance(item, torch.Tensor) or item.ndim != 5 or item.shape[0] != 1:
            raise ValueError("Expected each cached latent to be [1,C,T,H,W]")
        if (
            item.dtype != torch.bfloat16
            or item.shape[1] != 48
            or item.shape[2] != (video.shape[2] - 1) // 4 + 1
            or tuple(item.shape[-2:]) != tuple(x // 16 for x in video.shape[-2:])
        ):
            raise ValueError("Cached latent shape/dtype differs from Wan video input")
        result.append(item.to(device=video.device))
    return result
