import pytest
import torch

from cosmos_framework.data.pointflow_batch import build_pointflow_batch
from cosmos_framework.data.pointflow_batch_test import point_sample
from cosmos_framework.model.generator.pointflow_training import pointflow_add_noise, pointflow_loss


def test_noise_and_valid_masked_loss():
    clean = build_pointflow_batch([point_sample(2), point_sample(1)])
    sigma = torch.tensor([0.0, 1.0])
    epsilon = torch.ones_like(clean.displacement)
    state = pointflow_add_noise(clean, sigma, epsilon=epsilon)
    assert torch.equal(state.xt[:, :2], clean.displacement[:, :2])
    assert torch.equal(state.xt[:, 2:], epsilon[:, 2:])
    prediction = state.velocity_target.clone()
    prediction[:, 0] += 1
    loss, metrics = pointflow_loss(prediction, state, clean)
    assert loss.item() == pytest.approx(1 / 4)
    assert metrics["pointflow_valid_count"].tolist() == [64, 32]


def test_invalid_sigma_and_empty_valid_are_safe():
    clean = build_pointflow_batch([point_sample(0)])
    with pytest.raises(ValueError, match="sigma"):
        pointflow_add_noise(clean, torch.tensor([1.1]))
    state = pointflow_add_noise(clean, torch.tensor([0.5]), epsilon=torch.empty_like(clean.displacement))
    loss, metrics = pointflow_loss(torch.empty_like(clean.displacement), state, clean)
    assert loss.item() == 0
    assert metrics["pointflow_supervised_samples"].item() == 0
    assert metrics["pointflow_ade_mm"].item() == 0
    assert metrics["pointflow_zero_ade_mm"].item() == 0


def test_ade_is_zero_for_perfect_prediction_and_baseline_for_zero():
    clean = build_pointflow_batch([point_sample(2), point_sample(1)])
    state = pointflow_add_noise(clean, torch.tensor([0.3, 0.7]))
    valid = clean.valid
    baseline = clean.displacement[valid].norm(dim=-1).mean().item() * 1000
    _, metrics = pointflow_loss(state.velocity_target.clone(), state, clean)
    assert metrics["pointflow_ade_mm"].item() == pytest.approx(0, abs=1e-3)
    assert metrics["pointflow_zero_ade_mm"].item() == pytest.approx(baseline, rel=1e-5)
    # A zero *velocity* is not the stationary baseline: it leaves x0_hat = x_t,
    # which the freshly added noise dominates, so its ADE is far worse.
    _, zeros = pointflow_loss(torch.zeros_like(clean.displacement), state, clean)
    assert zeros["pointflow_ade_mm"].item() > baseline


def test_ade_tracks_the_displacement_scale():
    clean = build_pointflow_batch([point_sample(2)])
    state = pointflow_add_noise(clean, torch.tensor([0.5]), scale=2.0)
    _, metrics = pointflow_loss(state.velocity_target.clone(), state, clean, scale=2.0)
    assert metrics["pointflow_ade_mm"].item() == pytest.approx(0, abs=1e-3)
    valid = clean.valid
    assert metrics["pointflow_zero_ade_mm"].item() == pytest.approx(
        clean.displacement[valid].norm(dim=-1).mean().item() * 1000, rel=1e-5
    )
    with pytest.raises(ValueError, match="scale"):
        pointflow_loss(state.velocity_target.clone(), state, clean, scale=0.0)


def test_sigma_bins_separate_the_two_noise_regimes():
    """One batch can hold both a low-sigma and a high-sigma sample; keep them apart."""
    clean = build_pointflow_batch([point_sample(1), point_sample(1)])
    state = pointflow_add_noise(clean, torch.tensor([0.1, 0.9]))
    prediction = state.velocity_target.clone()
    prediction[:, :1] += 1.0  # corrupt only the sigma=0.1 sample
    _, metrics = pointflow_loss(prediction, state, clean)

    assert metrics["pointflow_loss_sigma_0_0.3"].item() == pytest.approx(1.0)
    assert metrics["pointflow_loss_sigma_0.9_1"].item() == pytest.approx(0.0)
    assert metrics["pointflow_loss_sigma_0_0.3_n"].item() == 32
    assert metrics["pointflow_loss_sigma_0.9_1_n"].item() == 32


def test_empty_sigma_bins_are_omitted_not_zeroed():
    """A placeholder zero would halve the average: the callback divides per-key steps."""
    clean = build_pointflow_batch([point_sample(1), point_sample(1)])
    state = pointflow_add_noise(clean, torch.tensor([0.1, 0.9]))
    _, metrics = pointflow_loss(state.velocity_target.clone(), state, clean)

    assert {k for k in metrics if k.startswith("pointflow_loss_sigma_")} == {
        "pointflow_loss_sigma_0_0.3",
        "pointflow_loss_sigma_0_0.3_n",
        "pointflow_loss_sigma_0.9_1",
        "pointflow_loss_sigma_0.9_1_n",
    }


def test_sigma_bins_are_empty_when_nothing_is_valid():
    clean = build_pointflow_batch([point_sample(0)])
    state = pointflow_add_noise(clean, torch.tensor([0.5]), epsilon=torch.empty_like(clean.displacement))
    _, metrics = pointflow_loss(torch.empty_like(clean.displacement), state, clean)
    assert not [k for k in metrics if k.startswith("pointflow_loss_sigma_")]


def test_independent_pointflow_schedule_wiring():
    """training_step must forward the independent sigma draw into the noise path."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).with_name("omni_mot_model.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniMoTModel")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "training_step")
    calls = [
        n
        for n in ast.walk(step)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_add_noise_to_input"
    ]
    assert calls, "training_step does not call _add_noise_to_input"
    assert any(kw.arg == "sigmas_pointflow" for call in calls for kw in call.keywords)
    add_noise = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_add_noise_to_input")
    params = [a.arg for a in add_noise.args.args] + [a.arg for a in add_noise.args.kwonlyargs]
    assert "sigmas_pointflow" in params


def test_pointflow_noise_level_matches_video_marginal():
    """The independent draw must come from the *vision* RF with the resolution shift."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    tree = ast.parse(Path(__file__).with_name("omni_mot_model.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniMoTModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_train_noise_level_pointflow")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<pointflow_sigma>", "exec"), namespace)

    class VideoRF:
        def __init__(self):
            self.seen_shifts = None

        def sample_train_time(self, batch_size, iteration=None, shifts=None):
            self.seen_shifts = shifts
            return torch.full((batch_size,), 0.37)

    rf = VideoRF()
    model = SimpleNamespace(
        config=SimpleNamespace(
            rectified_flow_training_config=SimpleNamespace(shift={"256": 3, "480": 5}),
            resolution="480",
        ),
        rectified_flow_video=rf,
        tensor_kwargs_fp32={"dtype": torch.float32},
    )
    sigmas = namespace["_get_train_noise_level_pointflow"](model, batch_size=4)
    assert sigmas.shape == (4, 1)
    assert torch.all(sigmas == 0.37)
    assert torch.equal(rf.seen_shifts, torch.full((4,), 5.0))  # resolution "480" -> shift 5


def test_joint_sampling_wiring():
    """Route B: the main inference loop must carry the point state end to end.

    Guards the three integration points: noise init in _prepare_inference_data,
    per-step state injection in _get_velocity, and the final split in
    generate_samples_from_batch.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).with_name("omni_mot_model.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniMoTModel")

    prep = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_inference_data")
    prep_src = ast.get_source_segment(Path(__file__).with_name("omni_mot_model.py").read_text(), prep)
    assert "noise_pointflow_list" in prep_src, "no pointflow noise init in _prepare_inference_data"
    assert "displacement=torch.zeros_like" in prep_src, "GT targets must be zeroed before inference packing"

    vel = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_velocity")
    vel_src = ast.get_source_segment(Path(__file__).with_name("omni_mot_model.py").read_text(), vel)
    assert "noise_x_pointflow" in vel_src, "flat state is not split for pointflow"
    assert "pointflow_noised" in vel_src, "packed sequence never receives the point state"
    assert "preds_pointflow" in vel_src, "point velocity is not read from denoise output"

    gen = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate_samples_from_batch")
    gen_src = ast.get_source_segment(Path(__file__).with_name("omni_mot_model.py").read_text(), gen)
    assert 'result["pointflow"]' in gen_src, "joint sampling does not return pointflow"
