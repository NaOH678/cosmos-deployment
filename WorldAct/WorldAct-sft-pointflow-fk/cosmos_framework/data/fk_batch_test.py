"""Ragged-batch contract, recycled-buffer isolation, and validation for FK."""

import numpy as np
import pytest
import torch

from cosmos_framework.data.fk_batch import FKNoised, build_fk_batch, fk_token_upper_bound
from cosmos_framework.data.fk_window import FKTiming

KEYPOINTS = 21
STEPS = 32
TIMING = FKTiming()


def fk_sample(n=KEYPOINTS, *, anchor_fill=0.25, ids_offset=0, timing=TIMING):
    # Distinct constants per field: with all-zero fixtures a recycled buffer shared
    # by two fields is invisible, because it copies one zero block over another.
    return {
        "inputs": {
            "anchor_xyz": np.full((n, 3), anchor_fill, dtype=np.float32),
            "point_ids": np.arange(n, dtype=np.int64) + ids_offset,
        },
        "targets": {
            "displacement": np.full((STEPS, n, 3), 0.5, dtype=np.float32),
            "valid": np.ones((STEPS, n), dtype=bool),
        },
        "metadata": {"timing": timing, "episode": "ep"},
    }


def test_batch_keeps_the_ragged_offsets_contract():
    batch = build_fk_batch([fk_sample(), fk_sample()], batch_size=2)
    assert batch.inputs["anchor_xyz"].shape == (2 * KEYPOINTS, 3)
    assert batch.displacement.shape == (STEPS, 2 * KEYPOINTS, 3)
    assert batch.valid.shape == (STEPS, 2 * KEYPOINTS)
    assert batch.inputs["point_offsets"].tolist() == [KEYPOINTS, 2 * KEYPOINTS]
    assert batch.inputs["point_batch"].shape == (2 * KEYPOINTS,)
    assert batch.has_fk.tolist() == [True, True]
    assert batch.labeled.tolist() == [True, True]


def test_none_slots_are_padded_not_dropped():
    batch = build_fk_batch([fk_sample(), None, fk_sample()], batch_size=3)
    assert batch.inputs["anchor_xyz"].shape == (2 * KEYPOINTS, 3)
    assert batch.inputs["point_offsets"].tolist() == [KEYPOINTS, KEYPOINTS, 2 * KEYPOINTS]
    assert batch.inputs["has_geometry"].tolist() == [True, False, True]
    assert batch.labeled.tolist() == [True, False, True]
    # The middle slot contributes no points, so its batch index never appears.
    assert set(batch.inputs["point_batch"].tolist()) == {0, 2}


def test_all_absent_returns_none():
    assert build_fk_batch([None, None], batch_size=2) is None
    assert build_fk_batch(None) is None


def test_fields_never_share_a_recycled_buffer():
    """The PointFlow pool once handed ``anchor_xyz`` and ``normal`` one storage.

    Both are ``[sum(N),3]`` float32, so a shape-keyed pool silently overwrote the
    camera-frame coordinates with surface normals.  FK's ``anchor_xyz`` has that
    exact shape, so this asserts the namespacing actually separates them.
    """
    batch = build_fk_batch([fk_sample(), fk_sample()], batch_size=2)
    pointers = {
        "anchor_xyz": batch.inputs["anchor_xyz"].data_ptr(),
        "displacement": batch.displacement.data_ptr(),
        "valid": batch.valid.data_ptr(),
        "point_ids": batch.inputs["point_ids"].data_ptr(),
        "point_batch": batch.inputs["point_batch"].data_ptr(),
    }
    assert len(set(pointers.values())) == len(pointers), pointers


def test_recycled_buffers_keep_two_batches_isolated():
    # Two samples, because a single-sample build skips concatenation entirely and
    # would pass this without ever touching the pool.
    a = build_fk_batch([fk_sample(anchor_fill=0.25), fk_sample(anchor_fill=0.25)], batch_size=2)
    pointer = a.inputs["anchor_xyz"].data_ptr()
    b = build_fk_batch([fk_sample(anchor_fill=0.75), fk_sample(anchor_fill=0.75)], batch_size=2)
    # Same shape reuses the buffer (that is the point of the pool)...
    assert b.inputs["anchor_xyz"].data_ptr() == pointer
    # ...so the second build must have overwritten it, not appended to it.
    assert torch.allclose(b.inputs["anchor_xyz"], torch.full((2 * KEYPOINTS, 3), 0.75))


def test_token_upper_bound_matches_the_design():
    # 21 keypoints x (1 anchor + 32/4 motion blocks) = 189, the v1 budget.
    assert fk_token_upper_bound(fk_sample()) == 189
    assert fk_token_upper_bound(None) == 0


def test_mixed_timing_is_rejected():
    other = FKTiming(fps=15.0, steps=32, steps_per_token=8)
    with pytest.raises(ValueError, match="Mixed FK timing"):
        build_fk_batch([fk_sample(), fk_sample(timing=other)], batch_size=2)


def test_target_shape_must_match_the_anchor_count():
    broken = fk_sample()
    broken["targets"]["displacement"] = np.zeros((STEPS, KEYPOINTS + 1, 3), np.float32)
    with pytest.raises(ValueError, match="target shape does not match"):
        build_fk_batch([broken], batch_size=1)


def test_nonfinite_targets_are_rejected():
    broken = fk_sample()
    broken["targets"]["displacement"] = np.full((STEPS, KEYPOINTS, 3), np.nan, np.float32)
    with pytest.raises(ValueError, match="must be finite"):
        build_fk_batch([broken], batch_size=1)


def test_wrong_slot_count_is_rejected():
    with pytest.raises(ValueError, match="one FK sample or None per original batch slot"):
        build_fk_batch([fk_sample()], batch_size=2)


def test_noised_state_validates_against_the_clean_batch():
    batch = build_fk_batch([fk_sample(), fk_sample()], batch_size=2)
    good = FKNoised(
        xt=torch.zeros_like(batch.displacement),
        epsilon=torch.zeros_like(batch.displacement),
        velocity_target=torch.zeros_like(batch.displacement),
        sigma=torch.tensor([0.5, 0.5]),
    )
    good.validate(batch)
    bad = FKNoised(good.xt, good.epsilon, good.velocity_target, torch.tensor([1.5, 0.5]))
    with pytest.raises(ValueError, match="sigma must be"):
        bad.validate(batch)
