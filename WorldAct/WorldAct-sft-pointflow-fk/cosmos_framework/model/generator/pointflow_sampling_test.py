"""Sampling sign, determinism and GT-isolation regressions on CPU."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.pointflow_sampling import sample_displacement, shifted_sigmas, training_shift


def test_euler_sign_and_scale():
    target = torch.full((4, 3, 3), 0.25)
    observed = []

    def velocity(x, sigma):
        observed.append(sigma[0].item())
        return (x - target) / sigma[0]

    result = sample_displacement(
        velocity, steps=16, seed=42, horizon=4, num_points=3, batch_size=1, device="cpu", scale=2.0
    )
    torch.testing.assert_close(result, target * 2)
    assert observed[0] == 1 and observed[-1] == 1 / 16


def test_seed_reproducible_and_rng_unchanged():
    before = torch.random.get_rng_state().clone()
    kwargs = dict(steps=16, seed=123, horizon=4, num_points=3, batch_size=1, device="cpu")
    a = sample_displacement(lambda x, s: torch.zeros_like(x), **kwargs)
    b = sample_displacement(lambda x, s: torch.zeros_like(x), **{**kwargs, "steps": 32})
    torch.testing.assert_close(a, b)
    assert torch.equal(before, torch.random.get_rng_state())
    c = sample_displacement(lambda x, s: torch.zeros_like(x), **{**kwargs, "seed": 124})
    assert not torch.equal(a, c)


def test_bad_velocity_rejected():
    with pytest.raises(ValueError, match="shape"):
        sample_displacement(lambda x, s: x[0], steps=2, seed=1, horizon=4, num_points=3, batch_size=1, device="cpu")


def test_shift_none_keeps_the_uniform_schedule_exactly():
    """The pre-shift behaviour, reproduced inline, must stay bit-identical.

    Anything else silently changes every existing PointFlow sample.
    """

    def velocity(x, sigma):
        return x * 0.5 + sigma[0]

    kwargs = dict(steps=16, seed=7, horizon=5, num_points=9, batch_size=1, device="cpu", scale=1.0)
    generator = torch.Generator(device="cpu").manual_seed(7)
    old_state = torch.randn((5, 9, 3), generator=generator, dtype=torch.float32)
    for index in range(16):
        sigma = torch.full((1,), 1.0 - index / 16, dtype=torch.float32)
        old_state = old_state - velocity(old_state, sigma).float() / 16
    torch.testing.assert_close(sample_displacement(velocity, shift=None, **kwargs), old_state)


def test_shift_moves_sigma_nodes_onto_the_training_distribution():
    """The branch is trained on waver+shift=5, which is concentrated at high sigma.

    Sampling a uniform grid spends 3 of 16 steps below sigma 0.25 where the branch
    has under 3% of its training signal, and the last node is 0.0625 -- the worst
    place to be, because recovering epsilon there divides by sigma.  The shifted
    grid must put nothing down there.
    """
    uniform = shifted_sigmas(16, None, "cpu")
    assert uniform[-1] == 0
    assert int((uniform[:-1] < 0.25).sum()) == 3

    shifted = shifted_sigmas(16, 5.0, "cpu")
    assert shifted[0] == 1 and shifted[-1] == 0
    assert int((shifted[:-1] < 0.25).sum()) == 0
    assert int((shifted[:-1] > 0.75).sum()) == 10


def test_shift_keeps_euler_exact_for_a_constant_velocity():
    """Warping the nodes must not break the integrator: constant v lands on init - v."""
    target = torch.randn(5, 9, 3)
    constant = torch.randn(5, 9, 3)
    for shift in (None, 5.0):
        generator = torch.Generator(device="cpu").manual_seed(7)
        initial = torch.randn((5, 9, 3), generator=generator)
        result = sample_displacement(
            lambda state, sigma: constant.expand_as(state),
            steps=16,
            seed=7,
            horizon=5,
            num_points=9,
            batch_size=1,
            device="cpu",
            shift=shift,
        )
        torch.testing.assert_close(result, initial - constant, atol=1e-5, rtol=0)
    assert target.shape == constant.shape  # keep the fixture honest


def test_training_shift_resolves_by_resolution():
    assert training_shift(5, "480") == 5.0
    assert training_shift({"256": 3, "480": 5, "720": 10}, "480") == 5.0
    with pytest.raises(ValueError, match="not found in shift dict"):
        training_shift({"256": 3}, "480")


def test_model_sampling_excludes_future_labels():
    # Exercise the actual model method without importing/loading the full VFM.
    tree = ast.parse(Path(__file__).with_name("omni_mot_model.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniMoTModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_pointflow")
    namespace = {
        "torch": torch,
        "log": SimpleNamespace(info=lambda message: None),
        "build_sequence_plans_from_data_batch": lambda **kw: ["fixed"],
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<sample_pointflow>", "exec"), namespace)

    @dataclass
    class Data:
        displacement: torch.Tensor
        valid: torch.Tensor
        labeled: torch.Tensor
        inputs: dict
        timing: object

    class Model:
        training = False
        input_video_key, input_image_key = "video", "images"
        # `resolution` lives on the top-level config, as set_up_scheduler_and_sampler reads it.
        config = SimpleNamespace(
            resolution="480",
            rectified_flow_training_config=SimpleNamespace(pointflow_displacement_scale=1.0, shift={"480": 5}),
            rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
        )

        def sampler(self, velocity, state, num_steps, shift, seed):
            return torch.zeros_like(state)

        def _load_and_tokenize_text_data(self, batch, step):
            return [[1, 2]]

        def get_data_and_condition(self, batch, iteration):
            data = Data(
                batch["future"],
                torch.ones((4, 3), dtype=torch.bool),
                torch.ones(1, dtype=torch.bool),
                {"point_ids": torch.arange(3), "anchor_xyz": torch.zeros((3, 3))},
                SimpleNamespace(steps=4),
            )
            return SimpleNamespace(pointflow=data, batch_size=1)

        def _pack_input_sequence(self, plans, text, clean, ts):
            assert torch.count_nonzero(ts) == 0
            return SimpleNamespace(pointflow_data=clean.pointflow, to_cuda=lambda: None)

        def denoise(self, data_batch_packed):
            data = data_batch_packed.pointflow_data
            assert torch.count_nonzero(data.displacement) == 0
            assert not data.valid.any() and not data.labeled.any()
            return {"preds_pointflow": torch.zeros_like(data_batch_packed.pointflow_noised.xt)}

    run = namespace["sample_pointflow"]
    first = run(Model(), {"future": torch.zeros((4, 3, 3))}, steps=4, seed=8)
    second = run(Model(), {"future": torch.full((4, 3, 3), 999.0)}, steps=4, seed=8)
    torch.testing.assert_close(first, second)
