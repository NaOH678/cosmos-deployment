"""Label-free Euler sampling for the Cosmos v = epsilon - clean convention."""

import torch

from cosmos_framework.model.generator.pointflow_scale import validate_scale


def training_shift(shift_config, resolution):
    """Resolve the sigma shift the branch was trained under, for one resolution.

    Mirrors ``OmniMoTModel.set_up_scheduler_and_sampler``: the config may hold a
    single int or a resolution-keyed mapping, and this recipe uses the mapping.
    """
    if isinstance(shift_config, int):
        return float(shift_config)
    shift_dict = dict(shift_config)
    if resolution not in shift_dict:
        raise ValueError(
            f"Resolution '{resolution}' not found in shift dict. Available resolutions: {list(shift_dict.keys())}"
        )
    return float(shift_dict[resolution])


def shifted_sigmas(steps: int, shift: float | None, device) -> torch.Tensor:
    """`steps + 1` sigma nodes from 1 to 0, warped exactly as unipc warps them.

    The point branch is trained on the *video* sigma, which this recipe draws from
    ``waver`` with ``shift=5`` -- a distribution concentrated near 0.83 with under 3%
    of samples below 0.25.  A plain uniform grid instead spends 3 of 16 steps below
    0.25, and the last node is 0.0625, where the branch has essentially no training
    signal *and* where recovering epsilon from ``x_sigma`` divides by sigma, so an
    error there is amplified by 1/sigma.  That combination is what made the sampled
    trajectory fly off in straight lines while the one-step estimate looked fine:
    the one-step number averages over sigma, the sampler has to walk all of it.

    ``shift=None`` reproduces the previous uniform schedule exactly.
    """
    t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    if not shift:
        return t
    return shift * t / (1 + (shift - 1) * t)


@torch.no_grad()
def sample_displacement(
    velocity_fn, *, steps, seed, horizon, num_points, batch_size, device, scale=1.0, shift: float | None = None
):
    """Integrate from sigma=1 to 0. The API deliberately accepts no GT labels.

    ``shift`` must match the value the branch was trained with; leaving it out
    samples outside the training distribution.  See :func:`shifted_sigmas`.
    """
    if steps < 1 or horizon < 1 or num_points < 1 or batch_size < 1:
        raise ValueError("PointFlow sampling sizes must be positive")
    validate_scale(scale)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    state = torch.randn((horizon, num_points, 3), generator=generator, device=device, dtype=torch.float32)
    sigmas = shifted_sigmas(steps, shift, device)
    for index in range(steps):
        sigma = sigmas[index].expand(batch_size)
        velocity = velocity_fn(state, sigma)
        if velocity.shape != state.shape:
            raise ValueError("PointFlow sampling velocity has the wrong shape")
        # Step by the actual node spacing: with a shift the nodes are not uniform.
        state = state + velocity.float() * (sigmas[index + 1] - sigmas[index])
    if not torch.isfinite(state).all():
        raise FloatingPointError("PointFlow sampling produced non-finite displacement")
    return state * scale
