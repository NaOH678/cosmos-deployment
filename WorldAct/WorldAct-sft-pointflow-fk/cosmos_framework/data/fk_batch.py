"""Training data contract for mixed, ragged FK samples (no learned modules).

Mirrors ``pointflow_batch.py`` so the sequence packer, the attention plumbing and
the eval callbacks can treat both modalities the same way.  The geometry is far
simpler -- 21 keypoints per sample against PointFlow's thousands of voxels -- but
the ragged ``offsets`` / ``batch`` contract is kept deliberately: the code that
consumes it is shared, and a fixed-width special case would have to be unwound
again the moment a variant used a different keypoint set.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from cosmos_framework.data.fk_window import FKTiming

# Namespaces FK's recycled buffers away from PointFlow's.  The pool in
# pointflow_batch.py is keyed on (name, shape, dtype) and once handed ``anchor_xyz``
# and ``normal`` -- both [sum(N),3] float32 -- the same storage, so the second
# concatenation overwrote the camera-frame coordinates with surface normals and
# destroyed a run.  FK's ``anchor_xyz`` has exactly that shape, so sharing the pool
# would reintroduce the bug the moment both modalities are loaded together.
_REUSED_BUFFERS: dict[tuple, torch.Tensor] = {}


@dataclass
class FKBatch:
    inputs: dict[str, torch.Tensor]
    displacement: torch.Tensor  # [H,sum(N),3], camera metres
    valid: torch.Tensor  # [H,sum(N)], supervision only
    labeled: torch.Tensor  # [B], distinguishes absent labels from empty anchors
    metadata: list[dict | None]
    timing: FKTiming

    @property
    def has_fk(self):
        return self.inputs["has_geometry"]

    @property
    def point_spans(self):
        ends = self.inputs["point_offsets"]
        return torch.stack((torch.cat((ends.new_zeros(1), ends[:-1])), ends), -1)

    def to(self, device):
        # Never cast IDs/masks or metric geometry to the model's BF16 dtype.
        return replace(
            self,
            inputs={k: v.to(device) for k, v in self.inputs.items()},
            displacement=self.displacement.to(device),
            valid=self.valid.to(device),
            labeled=self.labeled.to(device),
        )

    def to_cuda(self):
        moved = self.to("cuda")
        self.inputs, self.displacement, self.valid, self.labeled = (
            moved.inputs,
            moved.displacement,
            moved.valid,
            moved.labeled,
        )


@dataclass
class FKNoised:
    """Explicit caller-supplied RF state, mirroring ``PointFlowNoised``."""

    xt: torch.Tensor
    epsilon: torch.Tensor
    velocity_target: torch.Tensor
    sigma: torch.Tensor  # [B], one diffusion time per original sample

    def validate(self, clean: FKBatch):
        for value in (self.xt, self.epsilon, self.velocity_target):
            if value.shape != clean.displacement.shape or not torch.isfinite(value).all():
                raise ValueError("FK noised state must have finite shape [H,sum(N),3]")
        if (
            self.sigma.shape != clean.labeled.shape
            or not torch.isfinite(self.sigma).all()
            or torch.any((self.sigma < 0) | (self.sigma > 1))
        ):
            raise ValueError("FK sigma must be [B] in [0,1]")

    def to(self, device):
        return FKNoised(*(v.to(device) for v in (self.xt, self.epsilon, self.velocity_target, self.sigma)))

    def to_cuda(self):
        self.xt, self.epsilon, self.velocity_target, self.sigma = (
            value.cuda() for value in (self.xt, self.epsilon, self.velocity_target, self.sigma)
        )


def fk_token_upper_bound(sample):
    """One anchor plus H/q motion tokens per keypoint; the count is fixed at 21."""
    if sample is None:
        return 0
    timing = sample["metadata"]["timing"]
    return len(sample["inputs"]["point_ids"]) * (1 + timing.steps // timing.steps_per_token)


def _reused_buffer(name, shape, dtype):
    """A process-lifetime buffer for one named concatenated FK tensor.

    Same reasoning as the PointFlow pool: on this cluster the per-step allocation
    of a ``[H, sum(N), 3]`` tensor is what costs, not the copy.  ``name`` must be
    unique per logical field -- FK prefixes every key with ``fk_`` so an FK field
    can never be handed PointFlow's storage (or another FK field's).
    """
    if not name.startswith("fk_"):
        raise ValueError(f"FK buffer names must be namespaced, got {name!r}")
    key = (name, tuple(shape), dtype)
    buffer = _REUSED_BUFFERS.get(key)
    if buffer is None or buffer.device.type != "cpu":
        buffer = torch.empty(shape, dtype=dtype)
        _REUSED_BUFFERS[key] = buffer
    return buffer


def _concatenate(parts, dim, name):
    if len(parts) == 1:
        return parts[0]
    shape = list(parts[0].shape)
    shape[dim] = sum(part.shape[dim] for part in parts)
    return torch.cat(parts, dim=dim, out=_reused_buffer(name, shape, parts[0].dtype))


def assert_distinct_buffers(fields):
    """Reject two logical fields that share storage (see ``_reused_buffer``)."""
    owners = {}
    for name, value in fields:
        if not value.numel():
            continue
        storage = value.untyped_storage().data_ptr()
        if storage in owners:
            raise ValueError(f"FK fields {owners[storage]} and {name} share one buffer")
        owners[storage] = name


def build_fk_batch(samples, *, batch_size=None, device="cpu"):
    if samples is None:
        return None
    if not isinstance(samples, list) or (batch_size is not None and len(samples) != batch_size):
        raise ValueError("Expected one FK sample or None per original batch slot")
    reference = next((s for s in samples if s is not None), None)
    if reference is None:
        return None
    timing = reference["metadata"]["timing"]
    inputs = {}
    counts = [0 if s is None else len(s["inputs"]["point_ids"]) for s in samples]
    bounds = np.r_[0, np.cumsum(counts)]

    for s, n in zip(samples, counts, strict=True):
        if s is None:
            continue
        if s["metadata"]["timing"] != timing:
            raise ValueError("Mixed FK timing contracts")
        if tuple(s["targets"]["displacement"].shape) != (timing.steps, n, 3) or tuple(s["targets"]["valid"].shape) != (
            timing.steps,
            n,
        ):
            raise ValueError("FK target shape does not match anchor IDs")

    concatenated = []
    for key in ("point_ids", "anchor_xyz"):
        inputs[key] = _concatenate(
            [torch.as_tensor(s["inputs"][key]) for s in samples if s is not None], 0, "fk_" + key
        )
        concatenated.append((key, inputs[key]))
    projected = ["anchor_uv" in s["inputs"] for s in samples if s is not None]
    if any(projected):
        if not all(projected):
            raise ValueError("Cannot mix projected and unprojected FK samples")
        inputs["anchor_uv"] = _concatenate(
            [torch.as_tensor(s["inputs"]["anchor_uv"]) for s in samples if s is not None], 0, "fk_anchor_uv"
        )
    inputs["point_offsets"] = torch.tensor(bounds[1:], dtype=torch.long)
    inputs["point_batch"] = torch.repeat_interleave(torch.arange(len(samples)), torch.tensor(counts))
    inputs["has_geometry"] = torch.tensor(counts) > 0

    displacement = _concatenate(
        [torch.as_tensor(s["targets"]["displacement"]) for s in samples if s is not None], 1, "fk_displacement"
    ).float()
    valid = _concatenate(
        [torch.as_tensor(s["targets"]["valid"]) for s in samples if s is not None], 1, "fk_valid"
    ).bool()
    concatenated.extend((("displacement", displacement), ("valid", valid)))

    # An empty batch has nothing to check; extremes propagate NaN and +/-inf, which
    # keeps the finiteness guarantee without materializing a full-size bool mask.
    if displacement.numel() and not (torch.isfinite(displacement.amin()) and torch.isfinite(displacement.amax())):
        raise ValueError("FK target must be finite, with invalid labels zero-filled")
    assert_distinct_buffers(concatenated)

    return FKBatch(
        inputs,
        displacement,
        valid,
        torch.tensor([s is not None for s in samples]),
        [None if s is None else s["metadata"] for s in samples],
        timing,
    ).to(device)
