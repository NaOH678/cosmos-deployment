from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from cosmos_framework.callbacks.fk_eval import FKEvalCallback
from cosmos_framework.callbacks.joint_pointflow_fk_eval import attach_fk_overlay, sample_joint_case


def test_one_joint_call_returns_all_modalities_and_checks_frames():
    batch = {key: [{"metadata": {"raw_frame_ids": np.arange(33)}}] for key in ("pointflow", "fk")}
    result = {key: [torch.zeros(1)] for key in ("vision", "action", "pointflow", "fk")}
    model = SimpleNamespace(generate_samples_from_batch=Mock(return_value=result))
    assert sample_joint_case(model, batch, seed=42, steps=4) is result
    model.generate_samples_from_batch.assert_called_once_with(
        batch, guidance=1.0, seed=[42], n_sample=1, has_negative_prompt=False, num_steps=4
    )
    batch["fk"][0]["metadata"]["raw_frame_ids"] = np.arange(33) + 1
    with pytest.raises(ValueError, match="identical"):
        sample_joint_case(model, batch, seed=42, steps=4)
    assert model.generate_samples_from_batch.call_count == 1


def test_reused_fk_callback_never_samples():
    model = Mock()
    cb = FKEvalCallback(reuse_pointflow_joint=True)
    cb.on_validation_step_end(model, {}, {}, None, iteration=0)
    model.sample_fk.assert_not_called()


def test_fk_overlay_has_anchor_and_future_frames():
    xyz = np.tile([0.0, 0.0, 1.0], (21, 1))
    fk = {
        "inputs": {"anchor_xyz": xyz},
        "targets": {"displacement": np.zeros((32, 21, 3)), "valid": np.ones((32, 21), bool)},
    }
    record = {"frames_bgr": [np.zeros((240, 320, 3), np.uint8)] * 33}
    pf = {
        "inputs": {"image_size_wh": np.array([640, 448])},
        "metadata": {"uv_to_video": np.array([[1.0, 0.0, 0.0], [0.0, 480 / 448, 362.0]])},
    }
    attach_fk_overlay(record, fk, fk["targets"]["displacement"], pf, dream_hw=(842, 640))
    assert record["fk_pred_uv"].shape == (33, 21, 2)
    np.testing.assert_array_equal(record["fk_pred_uv"], record["fk_gt_uv"])
    assert record["fk_pred_valid"].all()


def test_shared_fk_writer_persists_metrics(tmp_path):
    import json

    from cosmos_framework.callbacks.joint_pointflow_fk_eval import make_fk_writer

    config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    writer = make_fk_writer(config)
    fk = {
        "inputs": {"anchor_xyz": np.zeros((21, 3))},
        "targets": {"displacement": np.ones((32, 21, 3)), "valid": np.ones((32, 21), bool)},
    }
    identity = {"case_id": "val_00", "episode": "episode", "raw_frame_ids": list(range(33))}
    writer._save_case(identity, {"fk": [fk]}, torch.zeros(32, 21, 3), None, 0)
    metrics = json.loads((tmp_path / "fk_eval/step_0000000/val_00/metrics.json").read_text())
    assert metrics["sampling_mode"] == "joint"
    assert metrics["all_ade_mm"] > 0


def test_simulation_overlay_uses_recorded_intrinsics_for_both_hands():
    fk = {
        "inputs": {"anchor_xyz": np.tile([0.0, 0.0, 1.0], (42, 1))},
        "targets": {"displacement": np.zeros((32, 42, 3)), "valid": np.ones((32, 42), bool)},
        "metadata": {
            "intrinsics_px": np.array([[100.0, 0, 200], [0, 100, 120], [0, 0, 1]]),
            "image_size_wh": [640, 480],
        },
    }
    record = {"frames_bgr": [np.zeros((480, 640, 3), np.uint8)] * 33}
    attach_fk_overlay(record, fk, fk["targets"]["displacement"], {})
    np.testing.assert_allclose(record["fk_pred_uv"], np.broadcast_to([200, 120], (33, 42, 2)))
