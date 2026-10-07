# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from kornia import augmentation as K

from cosmos_framework.callbacks.gpu_video_augmentation import GPUVideoAugmentation


def _video_batch() -> dict:
    frame = torch.randint(0, 256, (3, 1, 8, 10), dtype=torch.uint8)
    content = frame.expand(-1, 4, -1, -1).clone()
    clip = F.pad(content, (0, 2, 0, 2), mode="reflect")
    return {
        "video": [[clip.clone()], [clip.clone()]],
        "image_size": [[torch.tensor([[10.0, 12.0, 8.0, 10.0]])] for _ in range(2)],
    }


def test_disabled_callback_is_noop() -> None:
    data = _video_batch()
    original = data["video"][0][0].clone()

    GPUVideoAugmentation(enabled=False, require_cuda=False).on_training_step_start(
        SimpleNamespace(input_video_key="video"), data
    )

    assert torch.equal(data["video"][0][0], original)


def test_augmentation_preserves_temporal_consistency_and_padding() -> None:
    torch.manual_seed(7)
    data = _video_batch()
    callback = GPUVideoAugmentation(enabled=True, chunk_size=2, require_cuda=False)

    callback.on_training_step_start(SimpleNamespace(input_video_key="video"), data)

    for sample in data["video"]:
        output = sample[0]
        assert output.shape == (3, 4, 10, 12)
        assert output.dtype == torch.uint8
        assert torch.equal(output[:, 0], output[:, 1])
        assert torch.equal(output[:, 1], output[:, 2])
        assert torch.equal(output[:, 2], output[:, 3])
        assert torch.equal(output[..., 8, :10], output[..., 6, :10])
        assert torch.equal(output[..., :8, 10], output[..., :8, 8])


def test_crop_and_resize_use_one_vectorized_resample() -> None:
    callback = GPUVideoAugmentation(enabled=True, require_cuda=False)

    pipeline = callback._pipeline(8, 10, torch.device("cpu"))

    assert len(pipeline) == 2
    assert isinstance(pipeline[0], K.RandomResizedCrop)
    assert pipeline[0].flags["cropping_mode"] == "resample"


def test_augmentation_rejects_non_uint8_input() -> None:
    data = _video_batch()
    data["video"][0][0] = data["video"][0][0].float()
    callback = GPUVideoAugmentation(enabled=True, require_cuda=False)

    with pytest.raises(TypeError, match="expects uint8"):
        callback.on_training_step_start(SimpleNamespace(input_video_key="video"), data)
