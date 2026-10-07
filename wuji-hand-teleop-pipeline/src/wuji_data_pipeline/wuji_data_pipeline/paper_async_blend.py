"""Opt-in post-hoc asynchronous blending on one model timeline.

The paper describes weighted overlap but does not prescribe a curve. This
implementation supports linear or smoothstep weights over the remaining overlap.
It does not implement RTC, phase matching, or additional motion limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .deployment_protocol import ScheduledAction, interpolate_action_pair


@dataclass(frozen=True)
class BlendPlan:
    actions: tuple[Mapping[str, Any], ...]
    start_at: float
    new_weights: tuple[float, ...]
    new_sample_indices: tuple[float, ...]
    overlap_end_at: float | None
    protected_sequence: int | None


def sample_prediction(actions, *, origin: float, at: float, rate_hz: float):
    """Row zero is the FIRST future action, at origin + 1 / rate."""
    index = (at - origin) * rate_hz - 1.0
    if index < -1e-7 or index > len(actions) - 1 + 1e-7:
        raise ValueError("cannot extrapolate beyond prediction support")
    index = min(len(actions) - 1.0, max(0.0, index))
    left = int(math.floor(index))
    right = min(left + 1, len(actions) - 1)
    if left == right or index - left < 1e-9:
        return actions[left]
    return interpolate_action_pair(actions[left], actions[right], fraction=index - left)


def build_overlap_plan(
    old_queue: Sequence[ScheduledAction],
    new_actions: Sequence[Mapping[str, Any]],
    *,
    new_origin: float,
    rate_hz: float,
    activation_at: float,
    weight_curve: str = "linear",
) -> BlendPlan:
    """Preserve the queued head; blend moving old/new trajectories thereafter.

    Called only at the old head's deadline by the publisher. Old queue contains
    the actual previously blended plan, never an unexecuted raw model chunk.
    With no old tail (underrun), resume at the current prediction timestamp;
    this intentionally provides no invented old-motion continuation.
    """
    if (
        not new_actions
        or not all(math.isfinite(v) for v in (new_origin, rate_hz, activation_at))
        or rate_hz <= 0
    ):
        raise ValueError("invalid prediction timing")
    if weight_curve not in ("linear", "smoothstep"):
        raise ValueError("unsupported overlap weight curve")
    interval = 1.0 / rate_hz
    new_first = new_origin + interval
    new_last = new_origin + len(new_actions) * interval
    old = tuple(old_queue)
    if old:
        if any(
            abs((b.due_at - a.due_at) - interval) > 1e-6 for a, b in zip(old, old[1:])
        ):
            raise ValueError("old queue is not on the current model clock")
        start = old[0].due_at
        if activation_at < start - 1e-9 or activation_at - start >= interval:
            raise ValueError("protected endpoint is not due or was missed")
        if new_first > start + interval + 1e-7:
            raise ValueError("new prediction has no support after protected endpoint")
        overlap_end = min(old[-1].due_at, new_last)
    else:
        start = max(activation_at, new_first)
        overlap_end = None
    if start > new_last + 1e-7:
        raise ValueError("new prediction exhausted before activation")
    count = int(math.floor((new_last - start) * rate_hz + 1e-7)) + 1
    output, weights, indices = [], [], []
    for i in range(count):
        at = start + i * interval
        sample_index = (at - new_origin) * rate_hz - 1
        if old and i == 0:
            action, weight = old[0].action, 0.0
        else:
            new = sample_prediction(
                new_actions, origin=new_origin, at=at, rate_hz=rate_hz
            )
            if (
                old
                and i < len(old)
                and at <= overlap_end + 1e-7
                and overlap_end > start
            ):
                weight = min(1.0, max(0.0, (at - start) / (overlap_end - start)))
                if weight_curve == "smoothstep":
                    weight = weight * weight * (3.0 - 2.0 * weight)
                action = interpolate_action_pair(old[i].action, new, fraction=weight)
            else:
                action, weight = new, 1.0
        output.append(action)
        weights.append(weight)
        indices.append(sample_index)
    return BlendPlan(
        tuple(output),
        start,
        tuple(weights),
        tuple(indices),
        overlap_end,
        old[0].sequence if old else None,
    )
