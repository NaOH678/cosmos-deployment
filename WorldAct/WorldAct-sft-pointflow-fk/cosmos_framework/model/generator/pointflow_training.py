"""PointFlow rectified-flow convention matches Cosmos: v = epsilon - clean."""

import torch

from cosmos_framework.data.pointflow_batch import PointFlowNoised
from cosmos_framework.model.generator.pointflow_scale import validate_scale


def pointflow_add_noise(clean, sigmas, *, scale=1.0, generator=None, epsilon=None):
    validate_scale(scale)
    b = len(clean.labeled)
    if sigmas.shape not in ((b,), (b, 1)):
        raise ValueError("PointFlow currently requires one shared video sigma per sample, not diffusion forcing")
    sigma = sigmas.reshape(b).to(clean.displacement.device, dtype=torch.float32)
    target = clean.displacement.float() / scale
    if epsilon is None:
        epsilon = torch.randn(target.shape, device=target.device, dtype=torch.float32, generator=generator)
    if epsilon.shape != target.shape:
        raise ValueError("PointFlow noise shape must match [H,sum(N),3]")
    per_point_sigma = sigma[clean.inputs["point_batch"]][None, :, None]
    state = PointFlowNoised(
        per_point_sigma * epsilon + (1 - per_point_sigma) * target, epsilon, epsilon - target, sigma
    )
    state.validate(clean)
    return state


# The training sigma is drawn once per sample and shared by every point and timestep of
# that sample, so a batch of two covers two sigma values.  The scalar loss averages over
# the whole training distribution -- ``waver`` with shift=5, median 0.833, under 3% below
# 0.25 -- while the sampler's decisive steps live at the bottom of the range.  These bins
# exist to tell those two regimes apart: if the high bins keep falling while the low bins
# rise, the branch is being optimised for something the sampled trajectory does not depend
# on, which is exactly what a loss that falls while the eval error climbs looks like.
POINTFLOW_SIGMA_BINS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.0))


def sigma_binned_loss(error, state, clean, valid):
    """Mean squared velocity error per sigma bin, for the bins that are populated.

    Only non-empty bins are emitted.  The wandb callback divides every loss key by the
    number of steps *that key* appeared in (``wandb_log.py:143``), so writing a
    placeholder zero for an empty bin would silently dilute its average; the key names
    must contain ``"loss"`` for the same reason, or the callback drops them.
    """
    sigma = state.sigma[clean.inputs["point_batch"]].float()[None].expand_as(valid)[valid]
    metrics = {}
    for low, high in POINTFLOW_SIGMA_BINS:
        inside = (sigma >= low) & (sigma <= high) if high >= 1.0 else (sigma >= low) & (sigma < high)
        if not inside.any():
            continue
        stem = f"pointflow_loss_sigma_{low:g}_{high:g}"
        metrics[stem] = error[inside].mean()
        metrics[f"{stem}_n"] = error.new_tensor(float(inside.sum()))
    return metrics


def pointflow_loss(prediction, state, clean, scale=1.0):
    """Mean XYZ MSE per valid observation, then mean over supervised samples."""
    state.validate(clean)
    if prediction.shape != clean.displacement.shape:
        raise ValueError("PointFlow prediction shape does not match supervision")
    valid = clean.valid
    sample_ids = clean.inputs["point_batch"][None].expand_as(valid)[valid]
    error = (prediction.float()[valid] - state.velocity_target.float()[valid]).square().mean(-1)
    counts = torch.bincount(sample_ids, minlength=len(clean.labeled))
    sums = prediction.new_zeros(len(clean.labeled), dtype=torch.float32).index_add(0, sample_ids, error)
    per_sample = sums / counts.clamp_min(1)
    active = counts > 0
    loss = per_sample.sum() / active.sum().clamp_min(1)
    # index_add remains connected even for an empty valid set.
    metrics = {
        "pointflow_loss_per_sample": per_sample.detach(),
        "pointflow_valid_count": counts,
        "pointflow_supervised_samples": active.sum(),
    }
    metrics.update(pointflow_ade(prediction, state, clean, scale))
    metrics.update(sigma_binned_loss(error, state, clean, valid))
    return loss, metrics


@torch.no_grad()
def pointflow_ade(prediction, state, clean, scale=1.0):
    """Millimetre error of the one-step clean estimate ``x0_hat = x_t - sigma * v_hat``.

    The masked MSE above is dominated by the unit-variance noise term -- even a
    prediction of zero scores 1.0 -- so this distance to the metric target is the
    readable progress signal. ``pointflow_zero_ade_mm`` is the baseline of a
    stationary trajectory that predicts no displacement at all, i.e. the mean
    displacement magnitude of the supervised points. Note that a *velocity* of
    zero is not that baseline: it leaves ``x0_hat = x_t``, which is dominated by
    the noise just added.
    """
    validate_scale(scale)
    valid = clean.valid
    if not valid.any():
        zeros = prediction.new_zeros(())
        return {"pointflow_ade_mm": zeros, "pointflow_zero_ade_mm": zeros}
    sigma = state.sigma[clean.inputs["point_batch"]].float().unsqueeze(-1)
    estimate = (state.xt.float() - sigma * prediction.float()) * scale  # model space -> metres
    target = clean.displacement.float()
    return {
        "pointflow_ade_mm": (estimate - target)[valid].norm(dim=-1).mean() * 1000.0,
        "pointflow_zero_ade_mm": target[valid].norm(dim=-1).mean() * 1000.0,
    }
