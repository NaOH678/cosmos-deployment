"""CPU analytical checks for RTC VJP and fixed observation coordinates."""

import math

import pytest
import torch

from cosmos_framework.inference.robot_policy.rtc_vjp import guided_velocity, prefix_weights


def test_vjp_matches_analytic_nondiagonal_denoiser_jacobian():
    a = torch.tensor([[0.2, 0.4], [-0.3, 0.5]], dtype=torch.float64)
    x = torch.tensor([0.4, -0.7], dtype=torch.float64)
    y = torch.tensor([0.8, 0.2], dtype=torch.float64)
    w = torch.tensor([1.0, 0.3], dtype=torch.float64)
    sigma = 0.6
    result, metrics = guided_velocity(lambda z: a @ z, x, sigma, y, w)
    v = a @ x
    error = (y - (x - sigma * v)) * w
    jac = torch.eye(2, dtype=x.dtype) - sigma * a
    gain = ((1 - sigma) ** 2 + sigma**2) / (sigma * (1 - sigma))
    torch.testing.assert_close(result, v - gain * jac.T @ error)
    assert not torch.allclose(result, v - gain * error)  # Reject identity approximation.
    assert metrics["jacobian_effect_norm"] > 0
    assert not result.requires_grad


def test_full_vjp_includes_free_video_coordinates_but_keeps_condition_fixed():
    # 0=condition, 1=generated vision, 2=action. Action depends on both 0 and1.
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.2, 0.1], [0.5, 0.7, 0.3]], dtype=torch.float64)
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    w = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    free = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    result, metrics = guided_velocity(lambda z: a @ z, x, 0.5, y, w, free_mask=free)
    assert result[0] == 0
    assert result[1] != (a @ x)[1]  # Joint-video path cannot be silently detached.
    jac = (torch.eye(3, dtype=x.dtype) - 0.5 * a) @ torch.diag(free)
    error = (y - (x - 0.5 * a @ x)) * w
    torch.testing.assert_close(result, ((a @ x) - 2 * jac.T @ error) * free)


def test_exp_weights_match_reference_example():
    w = prefix_weights(2, 6, 10, dtype=torch.float64)
    linear = torch.tensor([1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0], dtype=torch.float64)
    torch.testing.assert_close(w, linear * torch.expm1(linear) / math.expm1(1.0))
    assert torch.equal(prefix_weights(0, 0, 32), torch.zeros(32))


def test_noise_endpoint_caps_guidance_and_detached_model_is_rejected():
    x = torch.ones(2)
    result, metrics = guided_velocity(lambda z: 0.2 * z, x, 1.0, x, torch.ones(2))
    assert metrics["gain"] == 10.0
    with pytest.raises(RuntimeError, match="detached"):
        guided_velocity(lambda z: z.detach(), x, 0.5, x, torch.ones(2))


def test_vjp_matches_nonlinear_finite_difference_directional_derivative():
    x = torch.tensor([0.3, 0.8], dtype=torch.float64)
    y = torch.tensor([-0.2, 0.5], dtype=torch.float64)
    w = torch.tensor([1.0, 0.2], dtype=torch.float64)

    def v(z):
        return torch.stack((z[0] * z[1], z[0].sin() + z[1] ** 2))

    s = 0.4
    result, _ = guided_velocity(v, x, s, y, w)
    error = (y - (x - s * v(x))) * w
    columns = []
    for i in range(2):
        dx = torch.zeros_like(x)
        dx[i] = 1e-6
        columns.append(((x + dx - s * v(x + dx)) - (x - dx - s * v(x - dx))) / (2e-6))
    j = torch.stack(columns, dim=1)
    gain = ((1 - s) ** 2 + s * s) / (s * (1 - s))
    torch.testing.assert_close(result, v(x) - gain * j.T @ error, atol=1e-9, rtol=1e-8)


def test_zero_video_output_cotangent_can_skip_decoder_without_losing_input_vjp():
    x = torch.tensor([0.2, 0.7, 0.4], dtype=torch.float64)
    target = torch.tensor([0.0, 0.0, 0.9], dtype=torch.float64)
    weights = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    def velocity(z):
        return torch.stack((z[0] ** 2 + z[1], z[2].sin(), z[0] * z[1] + z[2] ** 2))

    def pruned(z):
        out = velocity(z)
        return torch.cat((out[:2].detach(), out[2:]))

    full, _ = guided_velocity(velocity, x, 0.6, target, weights)
    optimized, _ = guided_velocity(pruned, x, 0.6, target, weights)
    torch.testing.assert_close(full, optimized, rtol=0, atol=0)
    assert optimized[0] != velocity(x)[0]
