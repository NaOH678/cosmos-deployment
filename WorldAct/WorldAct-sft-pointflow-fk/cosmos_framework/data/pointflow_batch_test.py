"""Mixed-sample alignment and actual Cosmos packer contract tests."""

from collections import deque

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.data.generator.joint_dataloader import JointDataLoader, custom_collate_fn
from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.data.pointflow_batch import PointFlowNoised, build_pointflow_batch, pointflow_token_upper_bound
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def point_sample(n):
    # Every field carries its own constant: with all-zero fixtures a recycled
    # buffer shared by two fields is invisible, because it copies one zero block
    # over another.  Distinct values turn that into a visible mismatch.
    fills = {
        "anchor_xyz": (3, 0.25),
        "anchor_uv": (2, 0.5),
        "normal": (3, 0.75),
        "color": (3, 1.0),
        "coord": (3, 1.25),
        "feat": (9, 1.5),
        "grid_coord": (3, 1.75),
    }
    inputs = {key: np.full((n, dim), value, dtype=np.float32) for key, (dim, value) in fills.items()}
    inputs.update(
        point_ids=np.arange(n) + 100,
        original_to_voxel=np.arange(n),
        voxel_representatives=np.arange(n),
        coord_shift=np.zeros(3, np.float32),
        intrinsics_normalized=np.eye(3, dtype=np.float32),
        image_size_wh=np.array([32, 32]),
    )
    return {
        "inputs": inputs,
        "targets": {"displacement": np.zeros((32, n, 3), np.float32), "valid": np.ones((32, n), bool)},
        "metadata": {
            "timing": PointFlowTiming(),
            "uv_to_video": np.eye(3, dtype=np.float32)[:2],
            "video_size_wh": [32, 32],
        },
    }


def samples():
    return [
        {
            "video": torch.zeros(3, 33, 32, 32),
            "pointflow": point,
            "sequence_plan": SequencePlan(
                has_text=True, has_vision=True, has_point=point is not None and len(point["inputs"]["point_ids"]) > 0
            ),
        }
        for point in [point_sample(3), None, point_sample(0), point_sample(2)]
    ]


def legacy_pack(raw):
    loader = object.__new__(JointDataLoader)
    loader.buffers = [deque()]
    loader.dataloaders = [iter([custom_collate_fn(raw)])]
    batch = {}
    for _ in raw:
        loader._update_output_batch(batch, loader._get_next_sample(0))
    return batch


def test_mixed_collators_and_offsets():
    raw = samples()
    del raw[1]["pointflow"]  # also cover old samples with no key at all
    for packed in [legacy_pack(raw), VFMListCollator().collate(raw)]:
        batch = build_pointflow_batch(packed["pointflow"], batch_size=4)
        assert batch.labeled.tolist() == [True, False, True, True]
        assert batch.has_point.tolist() == [True, False, False, True]
        assert batch.point_spans.tolist() == [[0, 3], [3, 3], [3, 3], [3, 5]]
        assert batch.voxel_spans.tolist() == batch.point_spans.tolist()
        assert batch.inputs["original_to_voxel"].tolist() == [0, 1, 2, 3, 4]
        assert batch.inputs["point_batch"].tolist() == [0, 0, 0, 3, 3]
        assert batch.valid.dtype == torch.bool
        assert batch.to("cpu").inputs["point_ids"].dtype == torch.int64
        noised = PointFlowNoised(
            torch.ones_like(batch.displacement),
            torch.zeros_like(batch.displacement),
            torch.zeros_like(batch.displacement),
            torch.ones(4) * 0.5,
        )
        noised.validate(batch)
        noised.sigma = torch.ones(2)
        with pytest.raises(ValueError, match="sigma"):
            noised.validate(batch)


def test_real_packer_preserves_data_without_inserting_learned_tokens():
    raw = samples()
    point = build_pointflow_batch([s["pointflow"] for s in raw])
    clean = GenerationDataClean(
        batch_size=4, is_image_batch=False, x0_tokens_vision=[torch.zeros(1, 4, 9, 2, 2) for _ in raw], pointflow=point
    )
    plans = [s["sequence_plan"] for s in raw]
    packed = pack_input_sequence(
        plans,
        [[1, 2]] * 4,
        clean,
        torch.ones(4) * 0.5,
        {"eos_token_id": 3, "start_of_generation": 4, "end_of_generation": 5},
    )
    assert packed.pointflow_data is point
    assert packed.point is None
    assert len(packed.sample_lens) == 4
    plans[1].has_point = True
    with pytest.raises(ValueError, match="slots"):
        pack_input_sequence(
            plans,
            [[1, 2]] * 4,
            clean,
            torch.ones(4) * 0.5,
            {"eos_token_id": 3, "start_of_generation": 4, "end_of_generation": 5},
        )


def test_empty_and_upper_bound():
    assert build_pointflow_batch([None, None]) is None
    empty = build_pointflow_batch([None, point_sample(0)])
    assert empty.displacement.shape == (32, 0, 3)
    assert empty.inputs["point_offsets"].tolist() == [0, 0]
    assert pointflow_token_upper_bound(point_sample(3)) == 27
    bad = point_sample(2)
    bad["inputs"]["original_to_voxel"][0] = 2
    with pytest.raises(ValueError, match="bounds"):
        build_pointflow_batch([bad])


def test_upper_bound_cluster_mode_caps_at_env(monkeypatch):
    monkeypatch.setenv("POINTFLOW_TOKEN_MODE", "cluster")
    monkeypatch.setenv("POINTFLOW_CLUSTER_TOKEN_CAP", "2")
    assert pointflow_token_upper_bound(point_sample(3)) == 18  # min(3, 2) * 9
    monkeypatch.setenv("POINTFLOW_TOKEN_MODE", "per_point")
    assert pointflow_token_upper_bound(point_sample(3)) == 27  # uncapped


def test_future_mask_does_not_change_topology_or_budget():
    sample = point_sample(3)
    before = build_pointflow_batch([sample])
    # Snapshot first: build_pointflow_batch recycles its concat buffers across
    # calls, so the live ``before`` tensors are overwritten by the next build and
    # comparing them directly would pass no matter what the second build did.
    snapshot = {key: value.clone() for key, value in before.inputs.items()}
    budget = pointflow_token_upper_bound(sample)
    sample["targets"]["valid"][:] = False
    after = build_pointflow_batch([sample])
    for key, value in snapshot.items():
        torch.testing.assert_close(value, after.inputs[key])
    assert after.has_point.tolist() == [True]
    assert pointflow_token_upper_bound(sample) == budget


def test_each_field_gets_its_own_buffer():
    """anchor_xyz, normal and coord are all [sum(N), 3] float32.

    A buffer pool keyed on shape alone handed them the same memory, so the later
    concatenation overwrote the camera-frame anchor coordinates with surface
    normals -- silently, and only for batches with more than one geometry sample.
    """
    first, second = point_sample(3), point_sample(2)
    batch = build_pointflow_batch([first, second])
    concatenated = (
        "point_ids",
        "anchor_xyz",
        "anchor_uv",
        "normal",
        "color",
        "coord",
        "feat",
        "grid_coord",
        "original_to_voxel",
        "voxel_representatives",
    )
    for key in concatenated:
        # The two mapping fields are rebased onto the packed batch, so compare
        # against the per-sample offsets the batch itself reports.
        starts = {
            "original_to_voxel": [int(span[0]) for span in batch.voxel_spans],
            "voxel_representatives": [int(span[0]) for span in batch.point_spans],
        }.get(key, [0, 0])
        expected = np.concatenate(
            [
                first["inputs"][key].astype(np.float64) + starts[0],
                second["inputs"][key].astype(np.float64) + starts[1],
            ]
        )
        np.testing.assert_allclose(batch.inputs[key].numpy().astype(np.float64), expected, err_msg=key)
    # Values alone cannot separate original_to_voxel from voxel_representatives in
    # this fixture, so also reject two names backed by one storage.
    storages = {}
    for key in concatenated + ("displacement", "valid"):
        value = batch.inputs.get(key)
        value = batch.displacement if key == "displacement" else batch.valid if key == "valid" else value
        if value.numel():
            storages.setdefault(value.untyped_storage().data_ptr(), []).append(key)
    assert [names for names in storages.values() if len(names) > 1] == []


def test_legacy_accumulate_preserves_missing_slots_across_inner_batches():
    loader = object.__new__(JointDataLoader)
    batch = {}
    loader._update_output_batch(batch, {"video": [torch.zeros(3, 33, 32, 32)]})
    loader._update_output_batch(batch, {"video": [torch.zeros(3, 33, 32, 32)], "pointflow": point_sample(2)})
    loader._update_output_batch(batch, {"video": [torch.zeros(3, 33, 32, 32)]})
    assert len(batch["pointflow"]) == 3
    assert batch["pointflow"][0] is None and batch["pointflow"][2] is None
    assert build_pointflow_batch(batch["pointflow"]).inputs["point_batch"].tolist() == [1, 1]
