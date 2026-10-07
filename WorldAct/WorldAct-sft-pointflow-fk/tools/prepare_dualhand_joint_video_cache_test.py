# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import torch

from cosmos_framework.data.generator.action.transforms import reflection_pad_to_target
from tools.prepare_dualhand_joint_video_cache import (
    _read_npy_metadata,
    _write_array_bytes,
    _write_npy_header,
    compose_dualhand_views,
    resize_to_resolution_content,
)


def test_streamed_npy_is_loadable_without_writable_memmap(tmp_path) -> None:
    path = tmp_path / "frames.npy"
    chunks = [
        np.full((2, 3, 4, 5), 11, dtype=np.uint8),
        np.full((1, 3, 4, 5), 29, dtype=np.uint8),
    ]
    with path.open("wb") as file:
        _write_npy_header(file, (3, 3, 4, 5))
        for chunk in chunks:
            _write_array_bytes(file, chunk)

    np.testing.assert_array_equal(np.load(path, allow_pickle=False), np.concatenate(chunks))
    assert _read_npy_metadata(path) == ((3, 3, 4, 5), np.dtype(np.uint8))


def test_compose_dualhand_views_uses_head_top_and_wrists_bottom() -> None:
    head = torch.full((2, 3, 4, 8), 10, dtype=torch.uint8)
    left = torch.full((2, 3, 4, 8), 20, dtype=torch.uint8)
    right = torch.full((2, 3, 4, 8), 30, dtype=torch.uint8)

    composed = compose_dualhand_views(head, left, right)

    assert composed.shape == (2, 3, 6, 8)
    torch.testing.assert_close(composed[:, :, :4], head)
    assert torch.all(composed[:, :, 4:, :4] == 20)
    assert torch.all(composed[:, :, 4:, 4:] == 30)


def test_resize_to_resolution_content_defers_only_padding() -> None:
    video = torch.randint(0, 256, (2, 3, 661, 640), dtype=torch.uint8)

    resized, image_size = resize_to_resolution_content(video, "480")
    target_h, target_w, content_h, content_w = image_size

    assert resized.dtype == torch.uint8
    assert resized.shape == (2, 3, content_h, content_w)
    online = reflection_pad_to_target(
        {"video": video.clone()}, ["video"], keep_aspect_ratio=True, target_w=target_w, target_h=target_h
    )
    cached = reflection_pad_to_target(
        {"video": resized}, ["video"], keep_aspect_ratio=True, target_w=target_w, target_h=target_h
    )
    torch.testing.assert_close(cached["video"], online["video"], rtol=0, atol=0)
    torch.testing.assert_close(cached["image_size"], online["image_size"])
