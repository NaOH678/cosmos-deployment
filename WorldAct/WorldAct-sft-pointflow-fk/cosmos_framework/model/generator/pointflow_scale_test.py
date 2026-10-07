import numpy as np
import pytest
import torch

from cosmos_framework.data.pointflow_batch import build_pointflow_batch
from cosmos_framework.data.pointflow_batch_test import point_sample
from cosmos_framework.model.generator.pointflow_sampling import sample_displacement
from cosmos_framework.model.generator.pointflow_scale import (
    ENV_FRAME_SCALES,
    ENV_FRAME_SCALES_FILE,
    broadcast_scale,
    frame_scales_from_env,
    load_frame_scales_file,
    parse_frame_scales,
    validate_scale,
)
from cosmos_framework.model.generator.pointflow_training import pointflow_add_noise, pointflow_loss


def test_parse_frame_scales_validates_length_and_positivity():
    assert parse_frame_scales("0.01, 0.02,0.03", 3) == (0.01, 0.02, 0.03)
    assert len(parse_frame_scales(",".join(["0.01"] * 9), 3)) == 9  # per-frame per-channel
    with pytest.raises(ValueError, match="3 .* or 9"):
        parse_frame_scales("0.01,0.02", 3)
    with pytest.raises(ValueError, match="positive"):
        parse_frame_scales("0.01,-0.02,0.03", 3)
    with pytest.raises(ValueError, match="positive"):
        parse_frame_scales("0.01,nan,0.03", 3)


def test_frame_scales_from_env(monkeypatch):
    monkeypatch.delenv(ENV_FRAME_SCALES, raising=False)
    monkeypatch.delenv(ENV_FRAME_SCALES_FILE, raising=False)
    assert frame_scales_from_env(2) is None
    monkeypatch.setenv(ENV_FRAME_SCALES, "0.01,0.02")
    assert frame_scales_from_env(2) == (0.01, 0.02)


def test_broadcast_scale_passthrough_or_frame_tensor(monkeypatch):
    monkeypatch.delenv(ENV_FRAME_SCALES, raising=False)
    monkeypatch.delenv(ENV_FRAME_SCALES_FILE, raising=False)
    assert broadcast_scale(0.0528, 2, "cpu") == 0.0528
    monkeypatch.setenv(ENV_FRAME_SCALES, "0.01,0.02")
    scale = broadcast_scale(0.0528, 2, "cpu")
    assert isinstance(scale, torch.Tensor)
    assert scale.shape == (2, 1, 1)
    assert scale[:, 0, 0].tolist() == pytest.approx([0.01, 0.02])


def test_broadcast_scale_per_channel_form(monkeypatch):
    # 2 frames x 3 channels, frame-major: k0x,k0y,k0z,k1x,k1y,k1z
    monkeypatch.setenv(ENV_FRAME_SCALES, "0.01,0.02,0.03,0.04,0.05,0.06")
    scale = broadcast_scale(0.0528, 2, "cpu")
    assert scale.shape == (2, 1, 3)
    torch.testing.assert_close(scale[:, 0, :], torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]))


def test_frame_scales_file_roundtrip_and_precedence(monkeypatch, tmp_path):
    import json

    env_file = tmp_path / "scales_env.json"
    env_file.write_text(
        json.dumps(
            {
                "steps": 2,
                "channels": 3,
                "order": "frame-major x,y,z",
                "stat": "std",
                "selection": "top500",
                "scales": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            }
        )
    )
    assert load_frame_scales_file(env_file, 2) == (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    # The file env wins over the comma-string env.
    monkeypatch.setenv(ENV_FRAME_SCALES, "9.9,9.9")
    monkeypatch.setenv(ENV_FRAME_SCALES_FILE, str(env_file))
    scale = broadcast_scale(0.0528, 2, "cpu")
    assert scale.shape == (2, 1, 3)
    assert scale[0, 0, 0].item() == pytest.approx(0.01)
    # Declared steps must match the run's POINTFLOW_STEPS.
    monkeypatch.setenv(ENV_FRAME_SCALES_FILE, str(env_file))
    with pytest.raises(ValueError, match="steps"):
        load_frame_scales_file(env_file, 4)
    # A missing 'scales' key is a clear error, not a silent fallback.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"steps": 2}))
    with pytest.raises(ValueError, match="scales"):
        load_frame_scales_file(bad, 2)


def test_validate_scale_accepts_scalar_or_tensor():
    validate_scale(1.0)
    validate_scale(torch.ones(3, 1, 1))
    for bad in (0.0, -1.0, float("nan"), torch.zeros(2, 1, 1), torch.tensor([1.0, -1.0])):
        with pytest.raises(ValueError, match="scale"):
            validate_scale(bad)


def scaled_sample(n, frames=32):
    sample = point_sample(n)
    ramp = np.linspace(0.001, 0.1, frames, dtype=np.float32)
    sample["targets"]["displacement"] = np.broadcast_to(ramp[:, None, None], (frames, n, 3)).copy()
    return sample


def test_add_noise_divides_per_frame(monkeypatch):
    clean = build_pointflow_batch([scaled_sample(2)])
    scale = torch.linspace(0.01, 0.05, 32)[:, None, None]
    state = pointflow_add_noise(clean, torch.tensor([0.0]), scale=scale, epsilon=torch.zeros_like(clean.displacement))
    torch.testing.assert_close(state.xt, clean.displacement / scale)
    torch.testing.assert_close(state.velocity_target, -clean.displacement / scale)
    # A perfect velocity prediction is exact in metres under the same vector.
    _, metrics = pointflow_loss(state.velocity_target.clone(), state, clean, scale=scale)
    assert metrics["pointflow_ade_mm"].item() == pytest.approx(0, abs=1e-3)
    # Per-channel form [H,1,3]: same contract, channels scaled independently.
    scale3 = torch.linspace(0.01, 0.05, 32 * 3).view(32, 1, 3)
    state3 = pointflow_add_noise(clean, torch.tensor([0.0]), scale=scale3, epsilon=torch.zeros_like(clean.displacement))
    torch.testing.assert_close(state3.xt, clean.displacement / scale3)


def test_sample_displacement_multiplies_per_frame():
    frames = 4
    scale = torch.tensor([0.01, 0.02, 0.03, 0.04])[:, None, None]
    result = sample_displacement(
        lambda x, s: torch.zeros_like(x),
        steps=2,
        seed=7,
        horizon=frames,
        num_points=3,
        batch_size=1,
        device="cpu",
        scale=scale,
    )
    reference = sample_displacement(
        lambda x, s: torch.zeros_like(x), steps=2, seed=7, horizon=frames, num_points=3, batch_size=1, device="cpu"
    )
    torch.testing.assert_close(result, reference * scale)
