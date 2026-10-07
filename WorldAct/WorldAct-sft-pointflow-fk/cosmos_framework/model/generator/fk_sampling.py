"""Label-free Euler sampling for the Cosmos v = epsilon - clean convention.

The FK counterpart of ``pointflow_sampling.py``.  Reimplemented rather than
imported because this worktree contains no PointFlow module at all -- v1 trains
FK alone -- and the shared maths is short enough that the dependency would cost
more than it saves.  The shift handling is copied exactly: it is the part that is
easy to get subtly wrong and hard to notice.
"""

import math
from dataclasses import dataclass

import torch

# Seed stride between modalities in the joint arm.  ``arch_invariant_rand`` draws
# from ``np.random.RandomState(seed)``, whose stream depends on the seed alone --
# so two modalities handed the same seed get the same leading values, whatever
# their shapes.  A stride far wider than any batch keeps the streams disjoint.
MODALITY_SEED_STRIDE = 1_000_003


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
    """``steps + 1`` sigma nodes from 1 to 0, warped exactly as unipc warps them.

    The branch is trained on the *video* sigma, drawn with ``shift=5`` -- a
    distribution concentrated near 0.83 with under 3% of samples below 0.25.  A
    plain uniform grid instead spends 3 of 16 steps below 0.25, and its last node
    is 0.0625, where the branch has essentially no training signal *and* where
    recovering epsilon from ``x_sigma`` divides by sigma, so an error there is
    amplified by 1/sigma.  Sampling unshifted walks out of the training
    distribution while the one-step estimate still looks fine -- the one-step
    number averages over sigma, the sampler has to walk all of it.

    ``shift=None`` reproduces a uniform schedule.
    """
    t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    if not shift:
        return t
    return shift * t / (1 + (shift - 1) * t)


def element_count(shape) -> int:
    """Number of elements in one flattened token (``[C,T,H,W]`` vision, ``[T,D]`` action)."""
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


@dataclass(frozen=True)
class JointLayout:
    """Per-sample flat-state boundaries: ``[vision_i | action_i | pointflow_i | FK_i]``.

    ``vision_sizes[i]`` / ``action_sizes[i]`` / ``pointflow_sizes[i]`` /
    ``fk_sizes[i]`` are element counts, and ``spans[i] == (start, end)`` is sample
    ``i``'s slice of the FK tensor's *keypoint* axis.  FK is last -- the fully
    fused four-modality order -- which makes the ``* scale`` slice trivial and
    keeps the FK tail contiguous per sample.  PointFlow sits between action and FK
    because in the four-modality run it is denoised in the same loop, and the two
    extra modalities each keep the segment shape they had on their own.

    ``action_sizes`` and ``pointflow_sizes`` are all zeros for an arm that leaves
    that modality out, so the narrower layouts are this one with empty middle
    segments rather than separate code paths.

    ``action_sizes`` is all zeros for the arm that leaves the action clean, so the
    two-segment layout is this one with an empty middle segment rather than a
    second code path -- and the sizes are carried on the layout instead of being
    re-derived per call site, which is where a ``(end - start) * 3 * horizon``
    drift would hide.

    Exists as a named type because the reassembly is the easiest thing here to get
    silently wrong: a sample occupies a *contiguous run of the keypoint axis*, and
    its flat layout is step-major, so recombining with a flatten-and-reshape
    scrambles steps across samples.  The correct recombination is
    ``torch.cat([piece_i.view(horizon, n_i, 3) for i], dim=1)``.
    """

    vision_sizes: list[int]
    action_sizes: list[int]
    pointflow_sizes: list[int]
    fk_sizes: list[int]
    spans: list[tuple[int, int]]

    def __len__(self) -> int:
        return len(self.vision_sizes)

    def sizes(self, index: int) -> tuple[int, int, int, int]:
        """``(vision, action, pointflow, FK)`` element counts, in flat order."""
        return (
            self.vision_sizes[index],
            self.action_sizes[index],
            self.pointflow_sizes[index],
            self.fk_sizes[index],
        )

    def segment_sizes(self, index: int, *, with_action: bool, with_pointflow: bool = False) -> tuple[int, ...]:
        """The sizes of the segments a caller is actually building, in flat order.

        An arm that leaves a modality out has no piece for it at all rather than an
        empty one, so the sizes handed to :func:`flatten_pieces` must list exactly
        the pieces being passed -- an extra number would fail the ``zip(strict=True)``
        rather than the layout.  FK is always present here: it is the modality this
        joint arm exists to generate.
        """
        vision, action, pointflow, fk = self.sizes(index)
        sizes = [vision]
        if with_action:
            sizes.append(action)
        if with_pointflow:
            sizes.append(pointflow)
        sizes.append(fk)
        return tuple(sizes)


def joint_layout(vision_shapes, point_spans, horizon, action_shapes=None, pointflow_shapes=None):
    """Build the joint arm's per-sample flat layout.

    ``action_shapes`` omitted gives the arm that generates the video and denoises FK
    while still reading a *clean* action, which is what the previously measured
    joint numbers are.  Passing the packed action token shapes adds the action
    segment, so the future action is denoised alongside it.

    ``pointflow_shapes`` adds the PointFlow segment, placed between action and FK.
    Pass it whenever the packed sequence carries PointFlow: in the fused run all
    four modalities denoise in this one loop, and a PointFlow payload that is
    packed but has no segment here would be silently absent from the flat state.
    """
    vision_sizes = [element_count(shape) for shape in vision_shapes]
    action_sizes = (
        [0] * len(vision_sizes) if action_shapes is None else [element_count(shape) for shape in action_shapes]
    )
    pointflow_sizes = (
        [0] * len(vision_sizes) if pointflow_shapes is None else [element_count(shape) for shape in pointflow_shapes]
    )
    spans = [(int(start), int(end)) for start, end in point_spans]
    if len(vision_sizes) != len(spans):
        raise ValueError(f"{len(vision_sizes)} vision items but {len(spans)} FK spans")
    if len(action_sizes) != len(vision_sizes):
        raise ValueError(f"{len(action_sizes)} action items but {len(vision_sizes)} vision items")
    if len(pointflow_sizes) != len(vision_sizes):
        raise ValueError(f"{len(pointflow_sizes)} pointflow items but {len(vision_sizes)} vision items")
    if int(horizon) < 1:
        raise ValueError(f"FK horizon must be positive, got {horizon}")
    for i, (start, end) in enumerate(spans):
        if end <= start:
            raise ValueError(f"sample {i} has an empty FK span ({start}, {end})")
    return JointLayout(
        vision_sizes=vision_sizes,
        action_sizes=action_sizes,
        pointflow_sizes=pointflow_sizes,
        fk_sizes=[(end - start) * 3 * int(horizon) for start, end in spans],
        spans=spans,
    )


def flatten_pieces(pieces, expected_sizes, *, what):
    """Flatten one sample's modality pieces, checking each against its layout size.

    ``torch.cat`` on dim 0 takes pieces of any length, so a mis-shaped piece gives
    a joint vector of the wrong *length* rather than an error -- and the split that
    reads it back slices at the layout's boundaries, so every segment after the
    short one is silently shifted.  The check costs a ``numel`` and turns that into
    a message naming the segment.
    """
    flattened = []
    for index, (piece, expected) in enumerate(zip(pieces, expected_sizes, strict=True)):
        got = int(piece.numel())
        if got != int(expected):
            raise ValueError(f"{what}: segment {index} flattens to {got} elements, the layout says {expected}")
        flattened.append(piece.reshape(-1))
    return flattened


def reassemble_fk(pieces, horizon: int):
    """``[horizon, n_i, 3]`` per sample -> ``[horizon, sum(n_i), 3]``.

    ``cat`` on the keypoint axis, never a flatten-and-reshape -- see
    :func:`joint_layout`.
    """
    return torch.cat(list(pieces), dim=1)


@torch.no_grad()
def sample_displacement(
    velocity_fn, *, steps, seed, horizon, num_points, batch_size, device, scale=1.0, shift: float | None = None
):
    """Integrate from sigma=1 to 0.  The API deliberately accepts no GT labels.

    ``shift`` must match the value the branch was trained with; leaving it out
    samples outside the training distribution.  See :func:`shifted_sigmas`.
    """
    if steps < 1 or horizon < 1 or num_points < 1 or batch_size < 1:
        raise ValueError("FK sampling sizes must be positive")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FK displacement scale must be positive and finite")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    state = torch.randn((horizon, num_points, 3), generator=generator, device=device, dtype=torch.float32)
    sigmas = shifted_sigmas(steps, shift, device)
    for index in range(steps):
        sigma = sigmas[index].expand(batch_size)
        velocity = velocity_fn(state, sigma)
        if velocity.shape != state.shape:
            raise ValueError("FK sampling velocity has the wrong shape")
        # Step by the actual node spacing: with a shift the nodes are not uniform.
        state = state + velocity.float() * (sigmas[index + 1] - sigmas[index])
    if not torch.isfinite(state).all():
        raise FloatingPointError("FK sampling produced non-finite displacement")
    return state * scale
