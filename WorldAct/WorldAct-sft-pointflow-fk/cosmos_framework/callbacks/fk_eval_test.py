"""``FKEvalCallback._record`` must survive tensors, not just host arrays.

The two sides of this callback live on different devices.  ``cpu_batch`` is built
on the host by ``fixed_cases``; ``prediction`` and ``reference`` come out of
``model.sample_fk`` on CUDA.  The first launch that reached validation died on

    TypeError: can't convert cuda:0 device type tensor to numpy.

because ``_record`` called a bare ``np.asarray`` on the prediction.

CUDA is not reachable from this node, so the device half of that cannot be
reproduced here.  What *is* reproduced is the same defect in the same place: a
tensor that ``np.asarray`` refuses.  ``requires_grad=True`` on a CPU tensor raises
"Can't call numpy() on Tensor that requires grad" from the identical call, so a
regression that reinstates the bare conversion fails this test.  The helper is
``detach().cpu().numpy()``, which fixes both.

Run:  PYTHONPATH=. <venv>/bin/python cosmos_framework/callbacks/fk_eval_test.py
"""

import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from cosmos_framework.callbacks.fk_eval import FKEvalCallback, _numpy

HORIZON, KEYPOINTS = 32, 21


def make_batch(as_tensor: bool, requires_grad: bool):
    """A minimal stand-in for a fixed eval case's ``cpu_batch``."""
    rng = np.random.default_rng(0)
    displacement = rng.normal(size=(HORIZON, KEYPOINTS, 3)).astype(np.float32)
    valid = np.ones((HORIZON, KEYPOINTS), bool)
    anchor = rng.normal(size=(KEYPOINTS, 3)).astype(np.float32)
    if as_tensor:
        displacement = torch.from_numpy(displacement)
        anchor = torch.from_numpy(anchor)
        if requires_grad:
            displacement = displacement.clone().requires_grad_(True)
    sample = {
        "targets": {"displacement": displacement, "valid": valid},
        "inputs": {"anchor_xyz": anchor},
    }
    return {"fk": [sample]}, np.asarray(valid)


def make_prediction(as_tensor: bool, requires_grad: bool):
    rng = np.random.default_rng(1)
    value = rng.normal(size=(HORIZON, KEYPOINTS, 3)).astype(np.float32)
    if not as_tensor:
        return value
    tensor = torch.from_numpy(value)
    return tensor.clone().requires_grad_(True) if requires_grad else tensor


IDENTITY = {
    "case_id": "val_00",
    "episode": "episode_0013_20260731_133649",
    "raw_frame_ids": list(range(HORIZON)),
}


def call_record(batch, prediction, reference=None):
    callback = FKEvalCallback()
    return callback._record(IDENTITY, batch, prediction, reference, iteration=0)


def main():
    # Joint mode must replace the primary artifacts, including the reference;
    # simply enabling the old extra joint arm would leave conditional outputs here.
    batch, _ = make_batch(as_tensor=False, requires_grad=False)
    prediction = make_prediction(True, False)
    vision = [torch.zeros(1, 2, 2, 2)]
    model = Mock()
    model.sample_fk.return_value = SimpleNamespace(fk=prediction, vision=vision, finite=True)
    with tempfile.TemporaryDirectory() as directory:
        cb = FKEvalCallback(
            joint_only=True,
            make_figures=False,
            make_video=False,
        )
        cb.config = SimpleNamespace(job=SimpleNamespace(path_local=directory))
        cb._fixed = ([dict(IDENTITY, seed=42)], [batch])
        with (
            patch("cosmos_framework.callbacks.fk_eval.misc.to", side_effect=lambda value, **kw: value),
            patch("cosmos_framework.callbacks.fk_eval.evaluation_rng", side_effect=lambda seed: nullcontext()),
            patch("cosmos_framework.callbacks.fk_eval.distributed.is_rank0", return_value=True),
        ):
            cb.on_validation_step_end(model, {}, {}, None, iteration=500)
        assert model.sample_fk.call_count == 2
        for call, steps in zip(model.sample_fk.call_args_list, [4, 16], strict=True):
            assert call.kwargs["generate_video"] and call.kwargs["joint_action"]
            assert call.kwargs["sampler"] == "unipc" and call.kwargs["steps"] == steps
        output = Path(directory) / "fk_eval/step_0000500/val_00"
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["sampling_mode"] == "joint" and "unipc16_all_ade_mm" in metrics
        with np.load(output / "prediction.npz", allow_pickle=True) as saved:
            np.testing.assert_array_equal(saved["prediction"], prediction.numpy())
            assert saved["vision"].shape == (1, 1, 2, 2, 2)
        assert not (output / "joint_action").exists()
        # Nonzero ranks still sample, but must not write files or run a duplicate arm.
        cb.on_validation_start(model, None, iteration=1000)
        with (
            patch("cosmos_framework.callbacks.fk_eval.misc.to", side_effect=lambda value, **kw: value),
            patch("cosmos_framework.callbacks.fk_eval.evaluation_rng", side_effect=lambda seed: nullcontext()),
            patch("cosmos_framework.callbacks.fk_eval.distributed.is_rank0", return_value=False),
        ):
            cb.on_validation_step_end(model, {}, {}, None, iteration=1000)
        assert model.sample_fk.call_count == 3
        assert not (Path(directory) / "fk_eval/step_0001000").exists()
    print("✅ 联合主产物、联合参考、原路径与全 rank 采样检查通过")

    # Baseline: plain host arrays.
    batch, _ = make_batch(as_tensor=False, requires_grad=False)
    record = call_record(batch, make_prediction(False, False))
    assert isinstance(record["prediction"], np.ndarray)
    baseline = record["metrics"]["all_ade_mm"]
    print(f"✅ 主机数组: all_ade_mm = {baseline:.3f} mm")

    # Host tensors (no grad): the shape the collator actually produces.
    batch, _ = make_batch(as_tensor=True, requires_grad=False)
    record = call_record(batch, make_prediction(True, False))
    assert isinstance(record["prediction"], np.ndarray), type(record["prediction"])
    assert np.isclose(record["metrics"]["all_ade_mm"], baseline), record["metrics"]["all_ade_mm"]
    print("✅ 主机 tensor: 同样通过，数值一致")

    # Tensors that need detaching -- the stand-in for the CUDA case.
    batch, _ = make_batch(as_tensor=True, requires_grad=True)
    record = call_record(
        batch,
        make_prediction(True, requires_grad=True),
        reference=make_prediction(True, requires_grad=True),
    )
    assert isinstance(record["prediction"], np.ndarray)
    assert isinstance(record["reference"], np.ndarray)
    assert np.isclose(record["metrics"]["all_ade_mm"], baseline)
    assert "sampling_difference_mm" in record["metrics"]
    print("✅ 需 detach 的 tensor（CUDA 情形的替身）: 通过，含 reference 分支")

    # Teeth: the pre-fix conversion raises on exactly this input, so the test above
    # is not passing for a reason unrelated to the fix.
    try:
        np.asarray(make_prediction(True, requires_grad=True), np.float64)
    except RuntimeError as error:
        assert "requires grad" in str(error), error
        print(f"🔬 裸 np.asarray（对照）-> RuntimeError: {str(error)[:58]}")
    else:
        raise AssertionError("裸 np.asarray 应该失败，否则这组断言没有牙齿")

    # ``_numpy`` passes None-like values through untouched for the reference=None path.
    assert _numpy(np.zeros(3)).shape == (3,)
    assert isinstance(_numpy(torch.zeros(3)), np.ndarray)

    print("\nPASS")


if __name__ == "__main__":
    main()
