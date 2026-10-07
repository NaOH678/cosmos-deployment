"""FK must survive the outer training loader, including worker collation and packing."""

from collections import deque

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cosmos_framework.data.fk_batch import build_fk_batch
from cosmos_framework.data.fk_batch_test import fk_sample
from cosmos_framework.data.fk_window import FKTiming
from cosmos_framework.data.generator.joint_dataloader import JointDataLoader, custom_collate_fn


def _sample(n):
    sample = {"video": torch.zeros(3, 33, 16, 16)}
    if n:
        sample["fk"] = fk_sample(n)
        sample["fk"]["inputs"]["anchor_uv"] = np.zeros((n, 2), dtype=np.float32)
    return sample


def _loader():
    loader = object.__new__(JointDataLoader)
    loader.tokenizer_spatial_compression_factor = 8
    loader.tokenizer_temporal_compression_factor = 4
    loader.patch_spatial = 2
    loader.uniae_chunk_frames = None
    loader.sound_latent_fps = 0
    return loader


@pytest.mark.parametrize("num_workers", [0, 2])
def test_worker_collation_split_and_repack_preserve_fk(num_workers):
    # Includes missing keys, explicit None, and different keypoint counts. The
    # FKTiming dataclass reproduces the first real training prewarm failure.
    samples = [_sample(0), _sample(21), {**_sample(0), "fk": None}, _sample(17)]
    loader = _loader()
    loader.buffers = [deque()]
    worker_kwargs = {"multiprocessing_context": "spawn", "timeout": 45} if num_workers else {}
    loader.dataloaders = [
        iter(DataLoader(samples, batch_size=4, num_workers=num_workers, collate_fn=custom_collate_fn, **worker_kwargs))
    ]
    packed = {}
    for _ in samples:
        sample = loader._get_next_sample(0)
        loader._update_output_batch(packed, sample)
    assert [s is not None for s in packed["fk"]] == [False, True, False, True]
    assert isinstance(packed["fk"][1]["metadata"]["timing"], FKTiming)
    batch = build_fk_batch(packed["fk"], batch_size=4)
    assert batch.inputs["point_offsets"].tolist() == [0, 21, 21, 38]
    assert batch.inputs["anchor_uv"].shape == (38, 2)
    assert batch.has_fk.tolist() == [False, True, False, True]
    with pytest.raises(StopIteration):
        loader._get_next_sample(0)


def test_column_batch_keeps_fk_payloads():
    values = [fk_sample(), None]
    assert custom_collate_fn({"fk": values})["fk"] is values


@pytest.mark.parametrize("counts", [(0, 21, 0), (21, 0, 0), (0, 0)])
def test_packing_backfills_across_separately_collated_batches(counts):
    loader = _loader()
    packed = {}
    for n in counts:
        sample = _sample(n)
        sample["video"] = [sample["video"]]
        loader._update_output_batch(packed, sample)
    if any(counts):
        assert [s is not None for s in packed["fk"]] == [bool(n) for n in counts]
    else:
        assert "fk" not in packed


def test_outer_packing_budget_includes_fk_anchor_and_motion_tokens():
    loader = _loader()
    sample = {"video": [torch.zeros(3, 33, 16, 16)]}
    video_tokens = loader._compute_num_tokens_per_sample(sample)
    assert video_tokens == 9
    assert loader._compute_num_tokens_per_sample({**sample, "fk": fk_sample()}) == video_tokens + 189


def test_cached_video_budget_preserves_full_temporal_extent_after_collation():
    loader = _loader()
    loader.buffers = [deque()]
    samples = [{"video": torch.zeros(3, 1, 16, 16), "cached_video_num_frames": frames} for frames in (33, 17)]
    loader.dataloaders = [iter(DataLoader(samples, batch_size=2, collate_fn=custom_collate_fn))]
    for frames in (33, 17):
        sample = loader._get_next_sample(0)
        full = {"video": [torch.zeros(3, frames, 16, 16)]}
        assert loader._compute_num_tokens_per_sample(sample) == loader._compute_num_tokens_per_sample(full)
