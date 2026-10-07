"""Training data contract for mixed, ragged PointFlow samples (no learned modules)."""

import os
from dataclasses import dataclass, replace

import numpy as np
import torch

from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.pointflow_profiling import phase as pointflow_phase


@dataclass
class PointFlowBatch:
    inputs: dict[str, torch.Tensor]
    displacement: torch.Tensor  # [H,sum(N),3], camera meters
    valid: torch.Tensor  # [H,sum(N)], supervision only
    labeled: torch.Tensor  # [B], distinguishes absent labels from empty anchors
    metadata: list[dict | None]
    timing: PointFlowTiming

    @property
    def has_point(self):
        return self.inputs["has_geometry"]

    @property
    def point_spans(self):
        ends = self.inputs["point_offsets"]
        return torch.stack((torch.cat((ends.new_zeros(1), ends[:-1])), ends), -1)

    @property
    def voxel_spans(self):
        ends = self.inputs["voxel_offsets"]
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


def pointflow_token_upper_bound(sample):
    """Reserve one anchor + H/q motion tokens per input point.

    Per-point tokens make N tokens where the cluster mode makes V, and N >= V,
    so the point count is a safe upper bound for both token modes.  With dense
    selections (thousands of points) the N-based bound inflates the packing
    budget one to two orders of magnitude and collapses the batch, so under
    POINTFLOW_TOKEN_MODE=cluster the reservation is capped at
    POINTFLOW_CLUSTER_TOKEN_CAP (default 1024, comfortably above the stage-3
    cluster count of any window we have scanned).
    """
    if sample is None:
        return 0
    n = len(sample["inputs"]["point_ids"])
    if os.environ.get("POINTFLOW_TOKEN_MODE", "cluster") == "cluster":
        n = min(n, int(os.environ.get("POINTFLOW_CLUSTER_TOKEN_CAP", "1024")))
    timing = sample["metadata"]["timing"]
    return n * (1 + timing.steps // timing.steps_per_token)


_REUSED_BUFFERS: dict[tuple, torch.Tensor] = {}


def _reused_buffer(name, shape, dtype):
    """A process-lifetime buffer for one named concatenated PointFlow tensor.

    ``torch.cat`` allocates a fresh ``[steps, sum(N), 3]`` tensor every step.  On
    this cluster that *allocation*, not the copy, is what costs: the same call
    measures 2.6 ms against idle memory and ~230 ms once the host is under memory
    pressure, because each new page can trigger reclaim (wall time == cpu time,
    the process on-CPU inside the kernel).  Copying into a warm buffer keeps
    those pages resident and removes the allocation from the step entirely.

    The buffer is keyed by field name, not only by shape: ``anchor_xyz`` and
    ``normal`` are both ``[sum(N), 3]`` float32, and keying on shape alone handed
    them the *same* buffer, so the later concatenation silently overwrote the
    camera-frame coordinates with surface normals.

    Each buffer is handed out to one caller at a time: the batch is consumed
    within the step that builds it, and the next step overwrites it.  Callers
    that need to keep a tensor past that point must clone it.
    """
    key = (name, tuple(shape), dtype)
    buffer = _REUSED_BUFFERS.get(key)
    if buffer is None or buffer.device.type != "cpu":
        buffer = torch.empty(shape, dtype=dtype)
        _REUSED_BUFFERS[key] = buffer
    return buffer


def _concatenate(parts, dim, name):
    """``torch.cat(parts, dim)`` without allocating when the shape repeats.

    ``name`` must be unique per logical field for the lifetime of the process.
    """
    if len(parts) == 1:
        return parts[0]
    shape = list(parts[0].shape)
    shape[dim] = sum(part.shape[dim] for part in parts)
    return torch.cat(parts, dim=dim, out=_reused_buffer(name, shape, parts[0].dtype))


def assert_distinct_buffers(fields):
    """Reject two logical fields that share storage.

    A recycled buffer handed to two fields turns one modality's data into
    another's: ``anchor_xyz`` and ``normal`` are both ``[sum(N), 3]`` float32, so
    a shape-keyed pool gave them the same memory and the second concatenation
    overwrote the camera-frame coordinates with surface normals.  Identity is not
    enough to catch it -- two tensors can be distinct views of one storage.
    """
    owners = {}
    for name, value in fields:
        if not value.numel():
            continue
        storage = value.untyped_storage().data_ptr()
        if storage in owners:
            raise ValueError(f"PointFlow fields {owners[storage]} and {name} share one buffer")
        owners[storage] = name


def build_pointflow_batch(samples, *, batch_size=None, device="cpu"):
    if samples is None:
        return None
    if not isinstance(samples, list) or (batch_size is not None and len(samples) != batch_size):
        raise ValueError("Expected one PointFlow sample or None per original batch slot")
    reference = next((s for s in samples if s is not None), None)
    if reference is None:
        return None
    timing = reference["metadata"]["timing"]
    inputs = {}
    counts = [0 if s is None else len(s["inputs"]["point_ids"]) for s in samples]
    voxels = [0 if s is None else len(s["inputs"]["coord"]) for s in samples]
    point_bounds, voxel_bounds = np.r_[0, np.cumsum(counts)], np.r_[0, np.cumsum(voxels)]
    with pointflow_phase("pf_bb_validate"):
        for s, n in zip(samples, counts, strict=True):
            if s is None:
                continue
            v = len(s["inputs"]["coord"])
            mapping = torch.as_tensor(s["inputs"]["original_to_voxel"])
            representatives = torch.as_tensor(s["inputs"]["voxel_representatives"])
            if mapping.shape != (n,) or representatives.shape != (v,):
                raise ValueError("Invalid PointFlow point/voxel mapping shape")
            if n and (v == 0 or torch.any((mapping < 0) | (mapping >= v))):
                raise ValueError("PointFlow mapping crosses sample voxel bounds")
            if v and (n == 0 or torch.any((representatives < 0) | (representatives >= n))):
                raise ValueError("PointFlow representatives cross sample point bounds")
            if s["metadata"]["timing"] != timing:
                raise ValueError("Mixed PointFlow timing contracts")
            if tuple(s["targets"]["displacement"].shape) != (timing.steps, n, 3) or tuple(
                s["targets"]["valid"].shape
            ) != (timing.steps, n):
                raise ValueError("PointFlow target shape does not match anchor IDs")
    with pointflow_phase("pf_bb_inputs"):
        point_keys = ("point_ids", "anchor_xyz", "anchor_uv", "normal", "color")
        voxel_keys = ("coord", "feat", "grid_coord")
        concatenated = []
        for key in point_keys + voxel_keys:
            inputs[key] = _concatenate([torch.as_tensor(s["inputs"][key]) for s in samples if s is not None], 0, key)
            concatenated.append((key, inputs[key]))
        for key in ("coord_shift", "intrinsics_normalized", "image_size_wh"):
            prototype = torch.as_tensor(reference["inputs"][key])
            inputs[key] = torch.stack(
                [torch.ones_like(prototype) if s is None else torch.as_tensor(s["inputs"][key]) for s in samples]
            )
        for key, bounds in (("original_to_voxel", voxel_bounds), ("voxel_representatives", point_bounds)):
            inputs[key] = _concatenate(
                [torch.as_tensor(s["inputs"][key]) + int(bounds[i]) for i, s in enumerate(samples) if s is not None],
                0,
                key,
            ).long()
            concatenated.append((key, inputs[key]))
        for prefix, sizes, bounds in (("point", counts, point_bounds), ("voxel", voxels, voxel_bounds)):
            inputs[prefix + "_offsets"] = torch.tensor(bounds[1:], dtype=torch.long)
            inputs[prefix + "_batch"] = torch.repeat_interleave(torch.arange(len(samples)), torch.tensor(sizes))
        inputs["has_geometry"] = torch.tensor(counts) > 0
        inputs["uv_to_video"] = torch.stack(
            [
                torch.eye(3)[:2] if s is None else torch.as_tensor(s["metadata"]["uv_to_video"], dtype=torch.float32)
                for s in samples
            ]
        )
        inputs["video_size_wh"] = torch.stack(
            [
                torch.ones(2, dtype=torch.long)
                if s is None
                else torch.as_tensor(s["metadata"]["video_size_wh"], dtype=torch.long)
                for s in samples
            ]
        )
    with pointflow_phase("pf_bb_targets"):
        displacement = _concatenate(
            [torch.as_tensor(s["targets"]["displacement"]) for s in samples if s is not None], 1, "displacement"
        ).float()
        valid = _concatenate(
            [torch.as_tensor(s["targets"]["valid"]) for s in samples if s is not None], 1, "valid"
        ).bool()
        concatenated.extend((("displacement", displacement), ("valid", valid)))
        # Extremes propagate NaN and +/-inf, so this keeps the finiteness guarantee
        # without materializing a full-size bool mask every step. An empty batch has
        # nothing to check, matching the vacuous truth of the previous ``.all()``.
        if displacement.numel() and not (torch.isfinite(displacement.amin()) and torch.isfinite(displacement.amax())):
            raise ValueError("PointFlow target must be finite, with invalid labels zero-filled")
    with pointflow_phase("pf_bb_alias_check"):
        assert_distinct_buffers(concatenated)
    with pointflow_phase("pf_bb_to_device"):
        return PointFlowBatch(
            inputs,
            displacement,
            valid,
            torch.tensor([s is not None for s in samples]),
            [None if s is None else s["metadata"] for s in samples],
            timing,
        ).to(device)


@dataclass
class PointFlowNoised:
    """Explicit caller-supplied RF state; task 8 owns sampling/noise/loss policy."""

    xt: torch.Tensor
    epsilon: torch.Tensor
    velocity_target: torch.Tensor
    sigma: torch.Tensor  # [B], one diffusion time per original sample

    def validate(self, clean: PointFlowBatch):
        for value in (self.xt, self.epsilon, self.velocity_target):
            if value.shape != clean.displacement.shape or not torch.isfinite(value).all():
                raise ValueError("PointFlow noised state must have finite shape [H,sum(N),3]")
        if (
            self.sigma.shape != clean.labeled.shape
            or not torch.isfinite(self.sigma).all()
            or torch.any((self.sigma < 0) | (self.sigma > 1))
        ):
            raise ValueError("PointFlow sigma must be [B] in [0,1]")

    def to(self, device):
        return PointFlowNoised(*(v.to(device) for v in (self.xt, self.epsilon, self.velocity_target, self.sigma)))

    def to_cuda(self):
        self.xt, self.epsilon, self.velocity_target, self.sigma = (
            value.cuda() for value in (self.xt, self.epsilon, self.velocity_target, self.sigma)
        )
