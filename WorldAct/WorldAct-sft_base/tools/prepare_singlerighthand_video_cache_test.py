# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.data.generator.action.transforms import reflection_pad_to_target
from tools.prepare_singlerighthand_video_cache import resize_to_resolution_content


def test_resize_to_resolution_content_defers_only_padding() -> None:
    video = torch.randint(0, 256, (2, 3, 842, 640), dtype=torch.uint8)

    resized, image_size = resize_to_resolution_content(video, "480")

    assert resized.dtype == torch.uint8
    assert resized.shape == (2, 3, 716, 544)
    assert image_size == [736, 544, 716, 544]

    online = reflection_pad_to_target(
        {"video": video.clone()}, ["video"], keep_aspect_ratio=True, target_w=544, target_h=736
    )
    cached = reflection_pad_to_target(
        {"video": resized}, ["video"], keep_aspect_ratio=True, target_w=544, target_h=736
    )
    torch.testing.assert_close(cached["video"], online["video"], rtol=0, atol=0)
    torch.testing.assert_close(cached["image_size"], online["image_size"])
