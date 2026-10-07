# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GPU-side, temporally consistent augmentation for training video clips."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import kornia.augmentation as K
import torch
import torch.nn.functional as F
from kornia.augmentation.container import VideoSequential

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _flatten_video_tree(value: Any, clips: list[torch.Tensor]) -> tuple[str, Any]:
    if value is None:
        return ("none", None)
    if isinstance(value, torch.Tensor):
        if value.ndim == 4:
            clips.append(value)
            return ("clip", None)
        if value.ndim == 5:
            clips.extend(value.unbind(0))
            return ("batch", value.shape[0])
        raise ValueError(f"Expected video shape [C,T,H,W] or [B,C,T,H,W], got {tuple(value.shape)}")
    if isinstance(value, list):
        return ("list", [_flatten_video_tree(item, clips) for item in value])
    if isinstance(value, tuple):
        return ("tuple", [_flatten_video_tree(item, clips) for item in value])
    raise TypeError(f"Unsupported video batch entry type: {type(value).__name__}")


def _rebuild_video_tree(spec: tuple[str, Any], clips: Iterator[torch.Tensor]) -> Any:
    kind, payload = spec
    if kind == "none":
        return None
    if kind == "clip":
        return next(clips)
    if kind == "batch":
        return torch.stack([next(clips) for _ in range(payload)])
    if kind == "list":
        return [_rebuild_video_tree(child, clips) for child in payload]
    if kind == "tuple":
        return tuple(_rebuild_video_tree(child, clips) for child in payload)
    raise AssertionError(f"Unknown video tree kind: {kind}")


def _metadata_for_sample(image_sizes: Any, sample_index: int, sample_count: int) -> Any:
    if isinstance(image_sizes, torch.Tensor):
        if image_sizes.ndim >= 2 and image_sizes.shape[0] == sample_count:
            return image_sizes[sample_index]
        return image_sizes
    if isinstance(image_sizes, (list, tuple)) and len(image_sizes) == sample_count:
        return image_sizes[sample_index]
    return image_sizes


def _parse_image_size(image_size: Any) -> tuple[int, int, int, int]:
    while isinstance(image_size, (list, tuple)):
        if not image_size:
            raise ValueError("Expected a non-empty image_size entry")
        image_size = image_size[0]
    if not isinstance(image_size, torch.Tensor):
        raise TypeError(f"Expected image_size tensor, got {type(image_size).__name__}")
    values = image_size.reshape(-1)
    if values.numel() < 4:
        raise ValueError(f"Expected image_size to contain four values, got shape {tuple(image_size.shape)}")
    return tuple(int(value.item()) for value in values[:4])  # type: ignore[return-value]


class GPUVideoAugmentation(Callback):
    """Apply random crop/rescale and color jitter after a batch reaches its GPU.

    Each clip receives independent random parameters, while all frames in that
    clip share the same parameters. Input and output remain uint8 so the model's
    existing CUDA normalization path remains unchanged.
    """

    def __init__(
        self,
        enabled: bool | str = False,
        crop_ratio: float = 0.95,
        brightness: float = 0.3,
        contrast: float = 0.4,
        saturation: float = 0.5,
        hue: float = 0.08,
        chunk_size: int = 8,
        require_cuda: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = _as_bool(enabled)
        if not 0.0 < crop_ratio <= 1.0:
            raise ValueError(f"crop_ratio must be in (0, 1], got {crop_ratio}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.crop_ratio = crop_ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.chunk_size = chunk_size
        self.require_cuda = require_cuda
        self._pipelines: dict[tuple[int, int, torch.device], VideoSequential] = {}

    def on_train_start(self, model: Any, iteration: int = 0) -> None:
        del model, iteration
        if self.enabled:
            log.info(
                "GPU video augmentation enabled: "
                f"crop_ratio={self.crop_ratio}, brightness={self.brightness}, contrast={self.contrast}, "
                f"saturation={self.saturation}, hue={self.hue}, chunk_size={self.chunk_size}",
                rank0_only=True,
            )

    def _pipeline(self, height: int, width: int, device: torch.device) -> VideoSequential:
        key = (height, width, device)
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            crop_h = max(1, int(height * self.crop_ratio))
            crop_w = max(1, int(width * self.crop_ratio))
            crop_scale = (crop_h * crop_w) / (height * width)
            crop_aspect_ratio = crop_w / crop_h
            pipeline = VideoSequential(
                # The resample path fuses crop and resize into one batched
                # warp. RandomCrop(slice) loops over every flattened frame
                # when samples have different crop offsets.
                K.RandomResizedCrop(
                    (height, width),
                    scale=(crop_scale, crop_scale),
                    ratio=(crop_aspect_ratio, crop_aspect_ratio),
                    resample="BILINEAR",
                    align_corners=False,
                    cropping_mode="resample",
                    p=1.0,
                ),
                K.ColorJiggle(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                    p=1.0,
                ),
                data_format="BCTHW",
                same_on_frame=True,
            ).to(device)
            self._pipelines[key] = pipeline
        return pipeline

    def _augment_chunk(
        self,
        clips: list[torch.Tensor],
        target_h: int,
        target_w: int,
        content_h: int,
        content_w: int,
    ) -> list[torch.Tensor]:
        batch = torch.stack([clip[..., :content_h, :content_w] for clip in clips])
        if self.require_cuda and not batch.is_cuda:
            raise RuntimeError(
                "GPUVideoAugmentation received a CPU tensor; the callback must run after device transfer"
            )
        if batch.dtype != torch.uint8:
            raise TypeError(f"GPUVideoAugmentation expects uint8 video, got {batch.dtype}")

        augmented = self._pipeline(content_h, content_w, batch.device)(batch.float().div_(255.0))
        padding_right = target_w - content_w
        padding_bottom = target_h - content_h
        if padding_right or padding_bottom:
            batch_size, channels, frames = augmented.shape[:3]
            flattened = augmented.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, content_h, content_w)
            mode = "replicate" if padding_right >= content_w or padding_bottom >= content_h else "reflect"
            flattened = F.pad(flattened, (0, padding_right, 0, padding_bottom), mode=mode)
            augmented = flattened.reshape(batch_size, frames, channels, target_h, target_w).permute(0, 2, 1, 3, 4)

        augmented = augmented.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        return list(augmented.unbind(0))

    @torch.no_grad()
    def on_training_step_start(self, model: Any, data: dict[str, Any], iteration: int = 0) -> None:
        del iteration
        if not self.enabled:
            return

        input_key = getattr(model, "input_video_key", "video")
        videos = data.get(input_key)
        if videos is None:
            return
        if not isinstance(videos, list):
            raise TypeError(f"Expected data[{input_key!r}] to be a list, got {type(videos).__name__}")
        image_sizes = data.get("image_size")
        if image_sizes is None:
            raise KeyError("GPUVideoAugmentation requires data['image_size'] to exclude reflection padding")

        flat_clips: list[torch.Tensor] = []
        sample_specs: list[tuple[tuple[str, Any], int, int]] = []
        layouts: list[tuple[int, int, int, int]] = []
        sample_count = len(videos)
        for sample_index, item in enumerate(videos):
            start = len(flat_clips)
            spec = _flatten_video_tree(item, flat_clips)
            end = len(flat_clips)
            layout = _parse_image_size(_metadata_for_sample(image_sizes, sample_index, sample_count))
            for clip in flat_clips[start:end]:
                target_h, target_w, content_h, content_w = layout
                if clip.shape[-2:] != (target_h, target_w):
                    raise ValueError(
                        f"Video canvas {tuple(clip.shape[-2:])} does not match image_size {(target_h, target_w)}"
                    )
                if content_h > target_h or content_w > target_w:
                    raise ValueError(f"Invalid image_size content {(content_h, content_w)} for {(target_h, target_w)}")
                layouts.append(layout)
            sample_specs.append((spec, start, end))

        groups: dict[tuple[int, int, int, int, torch.device], list[int]] = {}
        for index, (clip, layout) in enumerate(zip(flat_clips, layouts, strict=True)):
            groups.setdefault((*layout, clip.device), []).append(index)

        augmented_clips: list[torch.Tensor | None] = [None] * len(flat_clips)
        for (target_h, target_w, content_h, content_w, _device), indices in groups.items():
            for offset in range(0, len(indices), self.chunk_size):
                chunk_indices = indices[offset : offset + self.chunk_size]
                outputs = self._augment_chunk(
                    [flat_clips[index] for index in chunk_indices],
                    target_h,
                    target_w,
                    content_h,
                    content_w,
                )
                for index, output in zip(chunk_indices, outputs, strict=True):
                    augmented_clips[index] = output

        if any(clip is None for clip in augmented_clips):
            raise AssertionError("Not all video clips were augmented")
        completed_clips = [clip for clip in augmented_clips if clip is not None]
        for sample_index, (spec, start, end) in enumerate(sample_specs):
            videos[sample_index] = _rebuild_video_tree(spec, iter(completed_clips[start:end]))


__all__ = ["GPUVideoAugmentation"]
