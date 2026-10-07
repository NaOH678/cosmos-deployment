"""FK rectified-flow convention matches Cosmos: v = epsilon - clean.

The counterpart of ``pointflow_training.py``.  The maths is the same by design --
same noising, same masked MSE, same one-step clean estimate -- so that a
difference in results is a difference in the modality, not in the training rule.
"""

import math

import torch

from cosmos_framework.data.fk_batch import FKNoised


def fk_add_noise(clean, sigmas, *, scale=1.0, generator=None, epsilon=None):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FK displacement scale must be finite and positive")
    b = len(clean.labeled)
    if sigmas.shape not in ((b,), (b, 1)):
        raise ValueError("FK currently requires one shared video sigma per sample, not diffusion forcing")
    sigma = sigmas.reshape(b).to(clean.displacement.device, dtype=torch.float32)
    target = clean.displacement.float() / scale
    if epsilon is None:
        epsilon = torch.randn(target.shape, device=target.device, dtype=torch.float32, generator=generator)
    if epsilon.shape != target.shape:
        raise ValueError("FK noise shape must match [H,sum(N),3]")
    # One sigma per sample, shared by every keypoint and timestep of that sample:
    # a per-point sigma would ask the model to solve a different denoising problem
    # at each joint of one hand, which is not the process the sampler runs.
    per_point_sigma = sigma[clean.inputs["point_batch"]][None, :, None]
    state = FKNoised(per_point_sigma * epsilon + (1 - per_point_sigma) * target, epsilon, epsilon - target, sigma)
    state.validate(clean)
    return state


# Same reasoning as the PointFlow bins: the scalar loss averages over a training
# sigma distribution whose mass sits high, while the sampler's decisive steps are
# at the bottom of the range.  If the high bins keep falling while the low bins
# rise, the branch is being optimised for something the sampled trajectory does
# not depend on -- which is exactly a loss that falls while the eval error climbs.
FK_SIGMA_BINS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.0))


def sigma_binned_loss(error, state, clean, valid):
    """Mean squared velocity error per sigma bin, for the bins that are populated.

    Only non-empty bins are emitted: the wandb callback divides every loss key by
    the number of steps *that key* appeared in, so writing a placeholder zero for
    an empty bin would silently dilute its average.  The key names must contain
    ``"loss"`` for the same reason, or the callback drops them.
    """
    sigma = state.sigma[clean.inputs["point_batch"]].float()[None].expand_as(valid)[valid]
    metrics = {}
    for low, high in FK_SIGMA_BINS:
        inside = (sigma >= low) & (sigma <= high) if high >= 1.0 else (sigma >= low) & (sigma < high)
        if not inside.any():
            continue
        stem = f"fk_loss_sigma_{low:g}_{high:g}"
        metrics[stem] = error[inside].mean()
        metrics[f"{stem}_n"] = error.new_tensor(float(inside.sum()))
    return metrics


FK_LOSS_OBJECTIVES = ("mse", "charbonnier", "l1")


def _objective_error(squared, objective, charbonnier_eps):
    """Per-keypoint value the loss averages, from the per-keypoint *squared* error.

    MSE's minimiser is the conditional **mean**; L1's is the conditional
    **median**.  The motion a hand makes given a fixed observation is
    right-skewed -- most of the time it barely moves, occasionally it moves a
    lot -- and for a right-skewed conditional the median sits *above* the mean.
    So an MSE-trained branch under-predicts exactly the large motions, which is
    the magnitude compression this modality was measured to have (pred/gt falls
    from ~1.1 to ~0.83 as the true displacement grows).

    That matters here more than it usually would, because the number this model
    is judged by -- ADE -- is an L1 quantity.  Training on MSE while reporting
    ADE optimises a different functional than the one being read.

    ``charbonnier`` is the smooth L1, ``sqrt(x^2 + eps^2) - eps``: L2 inside
    ``eps`` and L1 outside it, differentiable everywhere, with a gradient at zero
    (plain L1 has none).  ``eps`` is in the units of the *squared* error, which is
    model space -- see ``fk_displacement_scale``.

    Not a guaranteed improvement: it only changes the answer where the
    conditional is skewed, and L1 objectives are higher-variance than MSE.  It is
    a switch so the two can be compared, not a correction that is known to help.
    """
    if objective == "mse":
        return squared
    if not math.isfinite(charbonnier_eps) or charbonnier_eps <= 0:
        raise ValueError(f"charbonnier_eps must be finite and positive, got {charbonnier_eps!r}")
    if objective == "l1":
        return squared.clamp_min(0).sqrt()
    if objective == "charbonnier":
        return (squared + charbonnier_eps**2).sqrt() - charbonnier_eps
    raise ValueError(f"unknown FK loss objective {objective!r}; expected one of {FK_LOSS_OBJECTIVES}")


def fk_loss(prediction, state, clean, scale=1.0, *, objective="mse", charbonnier_eps=0.01):
    """Mean XYZ per-observation error, then mean over supervised samples.

    ``objective`` selects what "error" means: ``mse`` (the original, and the
    default) or a smoother/L1 variant.  The value is *always* also reported as
    ``fk_mse_loss`` and the sigma bins stay on the squared error, so a run with a
    different objective is still numerically comparable to an MSE one -- changing
    the objective must not move the yardstick it is being measured against.
    """
    state.validate(clean)
    if prediction.shape != clean.displacement.shape:
        raise ValueError("FK prediction shape does not match supervision")
    valid = clean.valid
    sample_ids = clean.inputs["point_batch"][None].expand_as(valid)[valid]
    squared = (prediction.float()[valid] - state.velocity_target.float()[valid]).square().mean(-1)
    error = _objective_error(squared, objective, charbonnier_eps)
    counts = torch.bincount(sample_ids, minlength=len(clean.labeled))
    sums = prediction.new_zeros(len(clean.labeled), dtype=torch.float32).index_add(0, sample_ids, error)
    per_sample = sums / counts.clamp_min(1)
    active = counts > 0
    loss = per_sample.sum() / active.sum().clamp_min(1)
    # index_add remains connected even for an empty valid set.
    metrics = {
        "fk_loss_per_sample": per_sample.detach(),
        "fk_valid_count": counts,
        "fk_supervised_samples": active.sum(),
        # Always the squared error, whatever the objective: this is the number a
        # Charbonnier run is compared against an MSE run with.
        "fk_mse_loss": squared.mean().detach(),
        "fk_objective": error.new_tensor(float(FK_LOSS_OBJECTIVES.index(objective))),
    }
    metrics.update(fk_ade(prediction, state, clean, scale))
    metrics.update(sigma_binned_loss(squared, state, clean, valid))
    return loss, metrics


@torch.no_grad()
def fk_ade(prediction, state, clean, scale=1.0):
    """Millimetre error of the one-step clean estimate ``x0_hat = x_t - sigma * v_hat``.

    The masked MSE above is dominated by the unit-variance noise term -- a
    prediction of zero already scores 1.0 -- so this distance to the metric
    target is the readable progress signal, and it is also the number the
    design's verdict is stated in (``all_ade_mm / zero_all_ade_mm < 1``).

    ``fk_zero_ade_mm`` is the stationary-trajectory baseline: the mean
    displacement magnitude of the supervised keypoints.  A *velocity* of zero is
    not that baseline -- it leaves ``x0_hat = x_t``, which the noise just added
    dominates.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FK displacement scale must be finite and positive")
    valid = clean.valid
    if not valid.any():
        zeros = prediction.new_zeros(())
        return {"fk_ade_mm": zeros, "fk_zero_ade_mm": zeros}
    sigma = state.sigma[clean.inputs["point_batch"]].float().unsqueeze(-1)
    estimate = (state.xt.float() - sigma * prediction.float()) * scale  # model space -> metres
    target = clean.displacement.float()
    return {
        "fk_ade_mm": (estimate - target)[valid].norm(dim=-1).mean() * 1000.0,
        "fk_zero_ade_mm": target[valid].norm(dim=-1).mean() * 1000.0,
    }
