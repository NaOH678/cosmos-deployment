"""Scalar or per-frame displacement scale for the PointFlow branch.

The branch works in model space: metres are divided by the scale before noising
and multiplied back after decoding.  A single scalar under-weights early frames
because the target is the cumulative displacement from the anchor frame -- its
magnitude grows from millimetres at t=0 to tens of centimetres at t=H-1, so one
global scale leaves early-frame targets at ~1e-4 model units, drowned by the
unit-variance noise and barely penalised by the loss.  PointWorld solves the
same problem with per-timestep normalisation ("the displacement magnitude of
step 1 and step 10 differ too much for shared statistics",
docs/pointworld_implementation_20260929.md); ``POINTFLOW_DISPLACEMENT_FRAME_SCALES``
ports that idea here as one scale per frame index, measured by
``tools/scan_pointflow_selection.py`` over the training selection.  Two forms
are accepted: ``steps`` values (per-frame, channels pooled) or ``steps * 3``
values (per-frame per-channel, frame-major x,y,z -- PointWorld normalises per
(timestep, channel); depth carries systematically less variance than x/y).

The vector lives in the environment, not the checkpoint -- exactly like the
scalar ``pointflow_displacement_scale`` config it refines.  Resuming a run
requires the same env, and switching between scalar and vector changes the
target parameterisation, so checkpoints do not transfer across that switch.

Sources, in precedence order:

1. ``POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE`` -- path to the JSON the scan
   tool writes alongside its row dump (``<stem>_frame_scales_env.json``):
   ``{"steps": 32, "channels": 3, "order": "frame-major x,y,z", "scales": [...]}``.
2. ``POINTFLOW_DISPLACEMENT_FRAME_SCALES`` -- the same values as one
   comma-separated string (kept for quick ad-hoc overrides).
"""

import json
import math
import os
from pathlib import Path

import torch

ENV_FRAME_SCALES = "POINTFLOW_DISPLACEMENT_FRAME_SCALES"
ENV_FRAME_SCALES_FILE = "POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE"


def _check_frame_scales(values: tuple[float, ...], steps: int, origin: str) -> tuple[float, ...]:
    if len(values) not in (steps, steps * 3):
        raise ValueError(
            f"{origin} needs {steps} (per-frame) or {steps * 3} "
            f"(per-frame per-channel, frame-major x,y,z) values, got {len(values)}"
        )
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"{origin} entries must be finite and positive")
    return values


def parse_frame_scales(raw: str, steps: int) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    return _check_frame_scales(values, steps, ENV_FRAME_SCALES)


def load_frame_scales_file(path: str | Path, steps: int) -> tuple[float, ...]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "scales" not in data:
        raise ValueError(f"{ENV_FRAME_SCALES_FILE} file {path} must be a JSON object with a 'scales' list")
    declared = data.get("steps")
    if declared is not None and int(declared) != steps:
        raise ValueError(f"{path} declares steps={declared}, this run uses POINTFLOW_STEPS={steps}")
    return _check_frame_scales(tuple(float(v) for v in data["scales"]), steps, str(path))


def frame_scales_from_env(steps: int) -> tuple[float, ...] | None:
    path = os.environ.get(ENV_FRAME_SCALES_FILE, "").strip()
    if path:
        return load_frame_scales_file(path, steps)
    raw = os.environ.get(ENV_FRAME_SCALES, "").strip()
    return parse_frame_scales(raw, steps) if raw else None


def broadcast_scale(scalar: float, steps: int, device) -> float | torch.Tensor:
    """Scale ready to broadcast against a [H, N, 3] displacement tensor.

    Returns the scalar untouched when no per-frame env is set, otherwise a
    float32 tensor on ``device``: [H, 1, 1] for the 32-value per-frame form,
    [H, 1, 3] for the 96-value per-frame per-channel form (frame-major x,y,z).
    """
    frames = frame_scales_from_env(steps)
    if frames is None:
        return float(scalar)
    channels = len(frames) // steps
    return torch.tensor(frames, dtype=torch.float32, device=device).view(steps, 1, channels)


def validate_scale(scale: float | torch.Tensor) -> None:
    if isinstance(scale, torch.Tensor):
        if scale.numel() == 0 or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("PointFlow displacement scale must be finite and positive")
    elif not math.isfinite(scale) or scale <= 0:
        raise ValueError("PointFlow displacement scale must be finite and positive")
