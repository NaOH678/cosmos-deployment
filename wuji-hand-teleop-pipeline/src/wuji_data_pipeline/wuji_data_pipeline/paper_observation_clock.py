"""Map captured camera timestamps to the client's monotonic action clock.

Only real, rate-one ROS clocks sharing the camera timestamp domain are supported.
The node must reject use_sim_time and acquire source_clock_ros/captured_monotonic
as an adjacent pair while constructing the observation snapshot. No clock sample
from another request or from a later network-log event may be substituted.

The oldest camera is a conservative reference, not a claim that cameras and robot
state were acquired simultaneously. The paper does not specify this fusion rule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite timestamp")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite timestamp") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite timestamp")
    return result


def observation_origin(
    observation: Mapping[str, Any],
    *,
    source_clock_ros: float,
    captured_monotonic: float,
    previous_clock_pair: Mapping[str, float] | None = None,
    max_age_s: float = 1.0,
    max_clock_offset_change_s: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return oldest-camera monotonic origin and explicit clock diagnostics.

    ``max_age_s`` matches the existing deployment snapshot's one-second age gate.
    If supplied, previous_clock_pair is the prior successful result's
    ``clock_pair``. Reversed clocks are rejected. An offset-jump tolerance can be
    supplied by the caller; otherwise offset change is reported without inventing
    a drift threshold. Camera timestamps cannot be zero, future or stale.
    """
    ros_now = _finite(source_clock_ros, "source_clock_ros")
    mono_now = _finite(captured_monotonic, "captured_monotonic")
    max_age = _finite(max_age_s, "max_age_s")
    if ros_now <= 0 or mono_now < 0 or max_age <= 0:
        raise ValueError("Clock samples and max_age_s must be positive")
    offset_change = None
    if previous_clock_pair is not None:
        previous_ros = _finite(
            previous_clock_pair["source_clock_ros"], "previous source_clock_ros"
        )
        previous_mono = _finite(
            previous_clock_pair["captured_monotonic"], "previous captured_monotonic"
        )
        ros_elapsed = ros_now - previous_ros
        mono_elapsed = mono_now - previous_mono
        if ros_elapsed < 0 or mono_elapsed < 0:
            raise ValueError("Observation clock moved backwards")
        offset_change = ros_elapsed - mono_elapsed
    if max_clock_offset_change_s is not None:
        tolerance = _finite(max_clock_offset_change_s, "max_clock_offset_change_s")
        if tolerance < 0:
            raise ValueError("Clock offset tolerance must be nonnegative")
        if offset_change is not None and abs(offset_change) > tolerance:
            raise ValueError("ROS/monotonic clock offset jumped")

    sources = observation.get("source_timestamps")
    if not isinstance(sources, Mapping):
        raise ValueError("Observation has no source_timestamps")
    images = observation.get("images")
    names = (
        tuple(images)
        if isinstance(images, Mapping) and images
        else ("head", "right_wrist")
    )
    cameras = {}
    ages = {}
    for name in names:
        key = f"camera_{name}"
        if key not in sources:
            raise ValueError(f"Missing camera timestamp: {key}")
        stamp = _finite(sources[key], key)
        if stamp <= 0:
            raise ValueError(f"Missing or zero camera timestamp: {key}")
        age = ros_now - stamp
        if age < 0:
            raise ValueError(f"Camera timestamp is in the future: {key}")
        if age > max_age:
            raise ValueError(f"Camera timestamp is stale: {key}")
        cameras[key] = stamp
        ages[key] = age
    oldest_key = min(cameras, key=cameras.get)
    age = ages[oldest_key]
    origin = mono_now - age
    if origin < 0:
        raise ValueError("Camera timestamp predates monotonic clock origin")
    metadata = {
        "reference_rule": "oldest_camera_conservative_not_synchronized",
        "reference_camera": oldest_key,
        "observation_origin_monotonic": origin,
        "camera_age_s": ages,
        "camera_skew_s": max(cameras.values()) - min(cameras.values()),
        "camera_origin_monotonic": {
            key: mono_now - value for key, value in ages.items()
        },
        "source_timestamps": dict(sources),
        "clock_pair": {"source_clock_ros": ros_now, "captured_monotonic": mono_now},
        "clock_offset_change_s": offset_change,
        "clock_offset_jump_check_enabled": max_clock_offset_change_s is not None,
    }
    return origin, metadata
