"""Versioned wire helpers and bounded action plans for cloud deployment.

Protocol-v2 pickle messages can travel over the retained ZMQ replay transport
or the authenticated HTTP policy transport.  This module keeps image encoding
and action scheduling independent from ROS 2 and robot hardware so cloud
adapters can reuse it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import importlib
import math
import threading
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
try:
    from scipy.interpolate import PchipInterpolator
except ImportError:  # Reported explicitly when pchip_slerp is selected.
    PchipInterpolator = None
try:
    from scipy.signal import butter, sosfiltfilt
except ImportError:  # Reported explicitly when Butterworth smoothing is selected.
    butter = None
    sosfiltfilt = None

from .replay_core import (
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    quaternion_slerp_xyzw,
)
from .schema import normalize_quaternion_xyzw


PROTOCOL_VERSION = 2
IMAGE_CODECS = ("raw", "jpeg")


def encode_color_image(
    image: np.ndarray,
    *,
    codec: str,
    jpeg_quality: int,
    timestamp: Optional[float] = None,
) -> dict[str, Any]:
    """Encode one BGR uint8 image without changing camera semantics."""

    frame = np.ascontiguousarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"color image must be HxWx3 uint8, got {frame.shape}")
    normalized_codec = str(codec).strip().lower()
    if normalized_codec not in IMAGE_CODECS:
        raise ValueError(
            f"image codec must be one of {IMAGE_CODECS}, got {codec!r}"
        )
    result: dict[str, Any] = {
        "codec": normalized_codec,
        "shape": list(frame.shape),
        "color_space": "bgr8",
    }
    if timestamp is not None:
        result["timestamp"] = float(timestamp)
    if normalized_codec == "raw":
        result["color"] = frame
        return result
    quality = int(jpeg_quality)
    if not 1 <= quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode JPEG image")
    result["data"] = encoded.tobytes()
    return result


def decode_color_image(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode a protocol-v2 image payload into a contiguous BGR image."""

    codec = str(payload.get("codec", "raw")).strip().lower()
    if codec == "raw":
        if "color" not in payload:
            raise ValueError("raw image payload is missing color")
        frame = np.ascontiguousarray(payload["color"], dtype=np.uint8)
    elif codec == "jpeg":
        data = payload.get("data")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("JPEG image payload is missing bytes")
        encoded = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("failed to decode JPEG image")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
    else:
        raise ValueError(f"unsupported image codec: {codec!r}")
    shape = payload.get("shape")
    if shape is not None and tuple(int(value) for value in shape) != frame.shape:
        raise ValueError(
            f"decoded image shape {frame.shape} does not match declared {shape}"
        )
    return frame


def decode_observation_images(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an observation and decode its images for a model adapter."""

    decoded = dict(observation)
    decoded_images = {}
    images = observation.get("images", {})
    if not isinstance(images, Mapping):
        raise ValueError("observation images must be a mapping")
    for name, payload in images.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"camera {name!r} payload must be a mapping")
        camera = dict(payload)
        camera["color"] = decode_color_image(payload)
        decoded_images[str(name)] = camera
    decoded["images"] = decoded_images
    return decoded


def import_policy_factory(specification: str):
    """Import ``module:attribute`` used by the generic cloud server."""

    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("policy adapter must use module:attribute syntax")
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def extract_action_chunk(
    response: Mapping[str, Any],
    *,
    default_rate_hz: float,
) -> tuple[list[Mapping[str, Any]], float]:
    """Accept protocol-v2 chunks and legacy single-action responses."""

    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    raw_chunk = response.get("action_chunk")
    if raw_chunk is None:
        actions: Sequence[Any] = (response,)
    else:
        if not isinstance(raw_chunk, Sequence) or isinstance(
            raw_chunk, (str, bytes, bytearray)
        ):
            raise ValueError("action_chunk must be a sequence of mappings")
        actions = raw_chunk
    normalized = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise ValueError(f"action_chunk[{index}] is not a mapping")
        normalized.append(action)
    if not normalized:
        raise ValueError("policy response contains an empty action chunk")
    rate_hz = float(response.get("action_rate_hz", default_rate_hz))
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("action_rate_hz must be positive")
    return normalized, rate_hz


@dataclass(frozen=True)
class ScheduledAction:
    sequence: int
    due_at: float
    response_received_at: float
    action: Mapping[str, Any]


@dataclass(frozen=True)
class InstallResult:
    accepted: int
    latency_dropped: int
    input_count: int


@dataclass(frozen=True)
class PendingActionChunk:
    """A validated policy response waiting behind the active action plan."""

    actions: tuple[Mapping[str, Any], ...]
    action_rate_hz: float
    observation_created_at: float
    response_received_at: float
    align_to_observation: bool
    request_id: int = 0
    generation: int = 0
    wire_chunk_id: Optional[int] = None
    global_action_start: Optional[int] = None
    native_spans: tuple[Mapping[str, int], ...] = ()
    rtc_prefix_steps: Optional[int] = None


def validate_splice_kinematics(
    pose_samples: Mapping[str, Sequence[np.ndarray]],
    hand_samples: Mapping[str, Sequence[np.ndarray]],
    *,
    rate_hz: float,
    max_speed_m_s: float,
    max_acceleration_m_s2: float,
    max_rotation_deg_s: float,
    max_hand_deg_s: float,
) -> dict[str, float]:
    """Bound an EEF splice in model-time, including its preceding waypoint.

    These are command-space limits, not measured robot dynamics. Cartesian
    acceleration is a finite difference of segment velocities; linear/slerp
    interpolation is not a continuously acceleration-limited controller.
    """
    limits = (rate_hz, max_speed_m_s, max_acceleration_m_s2,
              max_rotation_deg_s, max_hand_deg_s)
    if any(not np.isfinite(v) or v <= 0 for v in limits):
        raise ValueError("splice kinematic limits must be finite and positive")
    metrics = dict(speed_m_s=0.0, acceleration_m_s2=0.0,
                   rotation_deg_s=0.0, hand_deg_s=0.0)
    for side, values in pose_samples.items():
        poses = np.asarray(values, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
            raise ValueError(f"invalid {side} splice poses")
        velocities = np.diff(poses[:, :3], axis=0) * rate_hz
        if len(velocities):
            metrics['speed_m_s'] = max(metrics['speed_m_s'], float(np.linalg.norm(velocities, axis=1).max()))
        if len(velocities) > 1:
            metrics['acceleration_m_s2'] = max(metrics['acceleration_m_s2'], float(np.linalg.norm(np.diff(velocities, axis=0) * rate_hz, axis=1).max()))
        quats = poses[:, 3:7]
        norms = np.linalg.norm(quats, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError(f"invalid {side} splice quaternion")
        quats = quats / norms[:, None]
        if len(quats) > 1:
            dots = np.abs(np.sum(quats[1:] * quats[:-1], axis=1))
            angular_speed = np.degrees(2 * np.arccos(np.clip(dots, 0, 1))) * rate_hz
            metrics['rotation_deg_s'] = max(metrics['rotation_deg_s'], float(angular_speed.max()))
    for side, values in hand_samples.items():
        hands = np.asarray(values, dtype=np.float64)
        if hands.ndim != 2 or hands.shape[1] != 20 or not np.isfinite(hands).all():
            raise ValueError(f"invalid {side} splice hands")
        if len(hands) > 1:
            metrics['hand_deg_s'] = max(metrics['hand_deg_s'], float(np.abs(np.diff(hands, axis=0)).max() * rate_hz))
    for name, limit in zip(metrics, limits[1:]):
        if metrics[name] > limit + 1e-8:
            raise ValueError(f"splice {name}={metrics[name]:.6f} exceeds {limit:.6f}")
    return metrics


def calculate_prefetch_lead(
    latency_samples_ms: Sequence[float],
    *,
    action_rate_hz: float,
    safety_actions: int,
    minimum_actions: int,
    maximum_actions: int,
    initial_actions: int,
    minimum_samples: int,
) -> tuple[int, float]:
    """Return a bounded P99 RTT lead and the measured nearest-rank P99."""

    if action_rate_hz <= 0.0:
        raise ValueError("action_rate_hz must be positive")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    if not 0 < minimum_actions <= initial_actions <= maximum_actions:
        raise ValueError("prefetch lead bounds and initial value are inconsistent")
    if safety_actions < 0:
        raise ValueError("safety_actions must be non-negative")
    samples = [float(value) for value in latency_samples_ms]
    if any(not np.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("latency samples must be finite and non-negative")
    if not samples:
        return int(initial_actions), 0.0
    ordered = sorted(samples)
    p99_index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    p99_ms = ordered[p99_index]
    if len(ordered) < minimum_samples:
        return int(initial_actions), p99_ms
    latency_actions = math.ceil(p99_ms * float(action_rate_hz) / 1000.0)
    lead = latency_actions + int(safety_actions)
    return max(minimum_actions, min(maximum_actions, lead)), p99_ms


def select_pending_action_window(
    pending: PendingActionChunk,
    *,
    activation_at: float,
    horizon: int,
    align_to_observation: Optional[bool] = None,
    fixed_skip_actions: Optional[int] = None,
) -> tuple[list[Mapping[str, Any]], int]:
    """Select a time-aligned fixed-horizon window from a pending chunk."""

    if horizon <= 0:
        raise ValueError("action horizon must be positive")
    if pending.action_rate_hz <= 0.0:
        raise ValueError("pending action rate must be positive")
    if len(pending.actions) < horizon:
        raise ValueError(
            f"pending chunk has {len(pending.actions)} actions, fewer than "
            f"the requested horizon {horizon}"
        )
    skip_actions = 0
    should_align = (
        pending.align_to_observation
        if align_to_observation is None
        else bool(align_to_observation)
    )
    if pending.rtc_prefix_steps is not None:
        if should_align or fixed_skip_actions is not None:
            raise ValueError("RTC prefix cannot be combined with time/fixed alignment")
        if type(pending.rtc_prefix_steps) is not int or pending.rtc_prefix_steps < 0:
            raise ValueError("invalid RTC prefix length")
        skip_actions = pending.rtc_prefix_steps
    if should_align:
        elapsed = max(
            0.0, float(activation_at) - pending.observation_created_at
        )
        # Select the first model action whose nominal timestamp is not older
        # than the activation point.  The epsilon avoids an extra skip when a
        # mathematically exact 30 Hz boundary is represented slightly above
        # its ideal floating-point value.
        skip_actions = max(
            0,
            math.ceil(elapsed * pending.action_rate_hz - 1e-9),
        )
    if fixed_skip_actions is not None:
        if should_align or isinstance(fixed_skip_actions, bool) or not isinstance(fixed_skip_actions, int) or fixed_skip_actions < 0:
            raise ValueError("fixed skip requires a nonnegative integer and time alignment disabled")
        if pending.align_to_observation:
            skip_actions = fixed_skip_actions
    maximum_skip = len(pending.actions) - horizon
    if skip_actions > maximum_skip:
        raise TimeoutError(
            "pending policy chunk no longer contains a full aligned horizon: "
            f"skip={skip_actions}, maximum={maximum_skip}"
        )
    return (
        list(pending.actions[skip_actions : skip_actions + horizon]),
        skip_actions,
    )


def blend_action_prefix(
    actions: Sequence[Mapping[str, Any]],
    *,
    anchor_poses: Mapping[str, np.ndarray],
    anchor_hands: Mapping[str, np.ndarray],
    anchor_zsp: Mapping[str, Optional[np.ndarray]],
    blend_steps: int,
    sides: Sequence[str] = ("left", "right"),
    action_mode: str = "eef",
    hand_blend_steps: Optional[int] = None,
) -> list[Mapping[str, Any]]:
    """Ease an aligned action prefix away from the last published target.

    The target advances through ``actions`` during the bridge; it is not held
    at action zero.  Therefore blending reduces the spatial discontinuity
    without adding a second temporal delay after alignment. Optional EEF-only
    ``hand_blend_steps`` shortens the hand transition independently; omitting it
    preserves the existing coupled arm/hand behavior.
    """

    mode = str(action_mode).strip().lower()
    if mode not in ("eef", "joint"):
        raise ValueError("action_mode must be eef or joint")
    count = min(max(0, int(blend_steps)), len(actions))
    if hand_blend_steps is not None:
        if mode != "eef":
            raise ValueError("independent hand blending only supports EEF actions")
        if not 0 <= int(hand_blend_steps) <= int(blend_steps):
            raise ValueError("hand blend steps must be between zero and arm blend steps")
    hand_count = count if hand_blend_steps is None else min(int(hand_blend_steps), len(actions))
    if count == 0:
        return list(actions)
    for side in sides:
        pose = np.asarray(anchor_poses[side], dtype=np.float32).reshape(-1)
        hand = np.asarray(anchor_hands[side], dtype=np.float32).reshape(-1)
        if pose.shape != (7,) or hand.shape != (20,):
            raise ValueError(f"invalid {side} boundary blend anchor")

    blended: list[Mapping[str, Any]] = []
    for index, original in enumerate(actions):
        if index >= count:
            blended.append(original)
            continue
        # Cubic smoothstep starts and ends the short bridge gently.  The final
        # bridge action is exactly the corresponding aligned model action.
        fraction = float(index + 1) / float(count)
        alpha = fraction * fraction * (3.0 - 2.0 * fraction)
        action = dict(original)
        for side in sides:
            arm_key = f"arm_action_{side}"
            hand_key = f"hand_action_{side}"
            target_arm = dict(original[arm_key])
            target_hand = np.asarray(
                original[hand_key], dtype=np.float32
            ).reshape(20)
            if mode == "joint":
                target_joint = np.asarray(
                    target_arm.get("joint_pos"), dtype=np.float32
                ).reshape(7)
                anchor_joint = np.asarray(
                    anchor_poses[side], dtype=np.float32
                ).reshape(7)
                target_arm["joint_pos"] = (
                    (1.0 - alpha) * anchor_joint
                    + alpha * target_joint
                ).astype(np.float32).tolist()
                hand = (
                    (1.0 - alpha)
                    * np.asarray(
                        anchor_hands[side], dtype=np.float32
                    ).reshape(20)
                    + alpha * target_hand
                )
                action[arm_key] = target_arm
                action[hand_key] = hand.astype(np.float32).tolist()
                continue
            target_position = np.asarray(
                target_arm["ee_pos"], dtype=np.float32
            ).reshape(3)
            target_quaternion = np.asarray(
                target_arm["ee_quat"], dtype=np.float32
            ).reshape(4)
            anchor_pose = np.asarray(anchor_poses[side], dtype=np.float32)
            position = (
                (1.0 - alpha) * anchor_pose[:3]
                + alpha * target_position
            )
            quaternion = quaternion_slerp_xyzw(
                anchor_pose[3:7], target_quaternion, alpha
            )
            target_arm["ee_pos"] = position.astype(np.float32).tolist()
            target_arm["ee_quat"] = quaternion.astype(np.float32).tolist()
            target_zsp = target_arm.get("zsp")
            previous_zsp = anchor_zsp.get(side)
            if target_zsp is not None and previous_zsp is not None:
                zsp = (
                    (1.0 - alpha)
                    * np.asarray(previous_zsp, dtype=np.float32).reshape(3)
                    + alpha
                    * np.asarray(target_zsp, dtype=np.float32).reshape(3)
                )
                norm = float(np.linalg.norm(zsp))
                if norm <= 1e-8:
                    zsp = np.asarray(target_zsp, dtype=np.float32).reshape(3)
                    norm = float(np.linalg.norm(zsp))
                if norm > 1e-8:
                    zsp = zsp / norm
                target_arm["zsp"] = zsp.astype(np.float32).tolist()
            hand_alpha = alpha
            if hand_blend_steps is not None:
                hand_fraction = min(1.0, (index + 1) / max(1, hand_count))
                hand_alpha = hand_fraction * hand_fraction * (3.0 - 2.0 * hand_fraction)
            hand = (
                (1.0 - hand_alpha)
                * np.asarray(anchor_hands[side], dtype=np.float32).reshape(20)
                + hand_alpha * target_hand
            )
            action[arm_key] = target_arm
            # At/after the hand endpoint preserve the original target exactly.
            # Omitting the option retains the historical arithmetic/output.
            action[hand_key] = (
                original[hand_key]
                if hand_blend_steps is not None and index + 1 >= hand_count
                else hand.astype(np.float32).tolist()
            )
        blended.append(action)
    return blended


def quaternion_to_rotvec_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return the shortest rotation vector represented by an XYZW quaternion."""

    value = normalize_quaternion_xyzw(quaternion).astype(np.float64)
    if value[3] < 0.0:
        value = -value
    vector_norm = float(np.linalg.norm(value[:3]))
    if vector_norm <= 1e-12:
        return (2.0 * value[:3]).astype(np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(value[3]))
    return (value[:3] * (angle / vector_norm)).astype(np.float64)


def quaternion_from_rotvec_xyzw(rotation_vector: np.ndarray) -> np.ndarray:
    """Return a normalized XYZW quaternion from a rotation vector."""

    value = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    if angle <= 1e-12:
        quaternion = np.concatenate([0.5 * value, [1.0]])
    else:
        quaternion = np.concatenate(
            [value * (math.sin(0.5 * angle) / angle), [math.cos(0.5 * angle)]]
        )
    return normalize_quaternion_xyzw(quaternion)


def quaternion_relative_rotvec_xyzw(
    reference: np.ndarray, value: np.ndarray
) -> np.ndarray:
    """Express ``value`` as a shortest-path rotation vector from ``reference``."""

    relative = quaternion_multiply_xyzw(
        quaternion_conjugate_xyzw(reference), value
    )
    return quaternion_to_rotvec_xyzw(relative)


def smooth_action_chunk_butterworth(
    actions: Sequence[Mapping[str, Any]],
    *,
    rate_hz: float,
    cutoff_hz: float,
    order: int = 3,
    sides: Sequence[str] = ("left", "right"),
) -> list[Mapping[str, Any]]:
    """Zero-phase low-pass one complete model action chunk.

    PCHIP only makes the curve between policy waypoints continuous; it still
    passes through every noisy 30 Hz waypoint.  This filter removes that
    waypoint-scale motion before the existing interpolation and boundary
    splice run.  ``sosfiltfilt`` is appropriate here because the complete
    future chunk is already available, so zero-phase filtering adds no online
    control delay.

    Cartesian position and hand joints are filtered directly.  Quaternion
    samples are unwrapped in the first sample's local rotation-vector chart,
    filtered there, and mapped back to unit quaternions.  ZSP is deliberately
    left unchanged because it is an IK posture hint rather than a commanded
    Cartesian trajectory.
    """

    if not actions:
        raise ValueError("cannot smooth an empty action chunk")
    if butter is None or sosfiltfilt is None:
        raise RuntimeError(
            "Butterworth action smoothing requires scipy.signal"
        )
    sample_rate = float(rate_hz)
    cutoff = float(cutoff_hz)
    filter_order = int(order)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("action smoothing rate_hz must be positive")
    if not np.isfinite(cutoff) or not 0.0 < cutoff < 0.5 * sample_rate:
        raise ValueError(
            "action smoothing cutoff_hz must be between zero and Nyquist"
        )
    if not 1 <= filter_order <= 8:
        raise ValueError("action smoothing order must be in [1, 8]")

    coefficients = butter(
        filter_order,
        cutoff,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    )
    filtered_by_side: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for side in sides:
        arm_key = f"arm_action_{side}"
        hand_key = f"hand_action_{side}"
        positions = np.stack(
            [
                np.asarray(action[arm_key]["ee_pos"], dtype=np.float64)
                .reshape(3)
                for action in actions
            ]
        )
        hands = np.stack(
            [
                np.asarray(action[hand_key], dtype=np.float64).reshape(20)
                for action in actions
            ]
        )
        quaternions = np.stack(
            [
                normalize_quaternion_xyzw(action[arm_key]["ee_quat"])
                .astype(np.float64)
                for action in actions
            ]
        )
        if not all(
            np.all(np.isfinite(value))
            for value in (positions, hands, quaternions)
        ):
            raise ValueError(f"non-finite {side} action smoothing waypoint")

        reference = quaternions[0]
        rotation_vectors = np.stack(
            [
                quaternion_relative_rotvec_xyzw(reference, quaternion)
                for quaternion in quaternions
            ]
        )
        try:
            filtered_positions = sosfiltfilt(
                coefficients, positions, axis=0
            )
            filtered_hands = sosfiltfilt(coefficients, hands, axis=0)
            filtered_rotations = sosfiltfilt(
                coefficients, rotation_vectors, axis=0
            )
        except ValueError as exc:
            raise ValueError(
                f"action chunk with {len(actions)} samples is too short for "
                f"order-{filter_order} Butterworth smoothing"
            ) from exc
        filtered_quaternions = np.stack(
            [
                normalize_quaternion_xyzw(
                    quaternion_multiply_xyzw(
                        reference,
                        quaternion_from_rotvec_xyzw(rotation_vector),
                    )
                )
                for rotation_vector in filtered_rotations
            ]
        )
        filtered_by_side[side] = (
            filtered_positions,
            filtered_quaternions,
            filtered_hands,
        )

    smoothed: list[Mapping[str, Any]] = []
    for index, original in enumerate(actions):
        action = dict(original)
        for side in sides:
            arm_key = f"arm_action_{side}"
            hand_key = f"hand_action_{side}"
            positions, quaternions, hands = filtered_by_side[side]
            arm = dict(original[arm_key])
            arm["ee_pos"] = positions[index].astype(np.float32).tolist()
            arm["ee_quat"] = quaternions[index].astype(np.float32).tolist()
            action[arm_key] = arm
            action[hand_key] = hands[index].astype(np.float32).tolist()
        smoothed.append(action)
    return smoothed


def _cubic_hermite(
    start: np.ndarray,
    end: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    duration_s: float,
    fraction: float,
) -> np.ndarray:
    """Sample a cubic Hermite segment with derivatives expressed per second."""

    t = float(np.clip(fraction, 0.0, 1.0))
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00 * start
        + h10 * float(duration_s) * start_velocity
        + h01 * end
        + h11 * float(duration_s) * end_velocity
    )


class VelocityContinuousBoundaryInterpolator:
    """C1 bridge from the executed command into a time-aligned action chunk.

    Cartesian translation and hand joints use cubic Hermite interpolation.
    Orientation uses the same construction in the anchor quaternion's local
    rotation-vector chart.  The bridge consumes the first ``blend_steps``
    model periods, reaches the corresponding raw model action exactly, and
    matches the raw trajectory's incoming derivative at that endpoint.
    """

    def __init__(
        self,
        actions: Sequence[Mapping[str, Any]],
        *,
        anchor_poses: Mapping[str, np.ndarray],
        anchor_hands: Mapping[str, np.ndarray],
        anchor_zsp: Mapping[str, Optional[np.ndarray]],
        previous_poses: Mapping[str, Optional[np.ndarray]],
        previous_hands: Mapping[str, Optional[np.ndarray]],
        previous_dt_s: float,
        blend_steps: int,
        rate_hz: float,
        sides: Sequence[str] = ("left", "right"),
    ) -> None:
        if not actions:
            raise ValueError("cannot bridge an empty action chunk")
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("boundary bridge rate_hz must be positive")
        self._actions = tuple(actions)
        self._sides = tuple(sides)
        self._count = min(max(0, int(blend_steps)), len(self._actions))
        if self._count <= 0:
            raise ValueError("velocity-continuous boundary requires blend_steps > 0")
        self._rate_hz = float(rate_hz)
        self._duration_s = self._count / self._rate_hz
        self._end_index = self._count - 1
        self._anchor_pose: dict[str, np.ndarray] = {}
        self._anchor_hand: dict[str, np.ndarray] = {}
        self._anchor_zsp: dict[str, Optional[np.ndarray]] = {}
        self._end_pose: dict[str, np.ndarray] = {}
        self._end_hand: dict[str, np.ndarray] = {}
        self._end_zsp: dict[str, Optional[np.ndarray]] = {}
        self._start_position_velocity: dict[str, np.ndarray] = {}
        self._end_position_velocity: dict[str, np.ndarray] = {}
        self._start_rotation_velocity: dict[str, np.ndarray] = {}
        self._end_rotation_velocity: dict[str, np.ndarray] = {}
        self._start_hand_velocity: dict[str, np.ndarray] = {}
        self._end_hand_velocity: dict[str, np.ndarray] = {}
        self._end_rotation_vector: dict[str, np.ndarray] = {}

        history_dt = float(previous_dt_s)
        history_valid = np.isfinite(history_dt) and history_dt > 1e-6
        indices = np.arange(len(self._actions), dtype=np.float64)
        for side in self._sides:
            arm_key = f"arm_action_{side}"
            hand_key = f"hand_action_{side}"
            anchor_pose = np.asarray(
                anchor_poses[side], dtype=np.float64
            ).reshape(-1)
            anchor_hand = np.asarray(
                anchor_hands[side], dtype=np.float64
            ).reshape(-1)
            if anchor_pose.shape != (7,) or anchor_hand.shape != (20,):
                raise ValueError(f"invalid {side} velocity bridge anchor")
            positions = np.stack(
                [
                    np.asarray(
                        action[arm_key]["ee_pos"], dtype=np.float64
                    ).reshape(3)
                    for action in self._actions
                ]
            )
            hands = np.stack(
                [
                    np.asarray(action[hand_key], dtype=np.float64).reshape(20)
                    for action in self._actions
                ]
            )
            quaternions = np.stack(
                [
                    normalize_quaternion_xyzw(action[arm_key]["ee_quat"])
                    for action in self._actions
                ]
            ).astype(np.float64)
            if not (
                np.all(np.isfinite(positions))
                and np.all(np.isfinite(hands))
                and np.all(np.isfinite(quaternions))
            ):
                raise ValueError(f"non-finite {side} velocity bridge waypoint")

            end_position = positions[self._end_index]
            end_hand = hands[self._end_index]
            end_quaternion = quaternions[self._end_index]
            if len(self._actions) >= 3 and PchipInterpolator is not None:
                end_position_velocity = np.asarray(
                    PchipInterpolator(indices, positions, axis=0).derivative()(
                        self._end_index
                    ),
                    dtype=np.float64,
                ) * self._rate_hz
                end_hand_velocity = np.asarray(
                    PchipInterpolator(indices, hands, axis=0).derivative()(
                        self._end_index
                    ),
                    dtype=np.float64,
                ) * self._rate_hz
            else:
                next_index = min(self._end_index + 1, len(self._actions) - 1)
                previous_index = max(0, self._end_index - 1)
                index_span = max(1, next_index - previous_index)
                end_position_velocity = (
                    positions[next_index] - positions[previous_index]
                ) * (self._rate_hz / index_span)
                end_hand_velocity = (
                    hands[next_index] - hands[previous_index]
                ) * (self._rate_hz / index_span)

            previous_pose = previous_poses.get(side)
            previous_hand = previous_hands.get(side)
            if history_valid and previous_pose is not None:
                previous_pose_value = np.asarray(
                    previous_pose, dtype=np.float64
                ).reshape(7)
                start_position_velocity = (
                    anchor_pose[:3] - previous_pose_value[:3]
                ) / history_dt
                previous_rotation = quaternion_relative_rotvec_xyzw(
                    anchor_pose[3:7], previous_pose_value[3:7]
                )
                start_rotation_velocity = -previous_rotation / history_dt
            else:
                start_position_velocity = np.zeros(3, dtype=np.float64)
                start_rotation_velocity = np.zeros(3, dtype=np.float64)
            if history_valid and previous_hand is not None:
                start_hand_velocity = (
                    anchor_hand
                    - np.asarray(previous_hand, dtype=np.float64).reshape(20)
                ) / history_dt
            else:
                start_hand_velocity = np.zeros(20, dtype=np.float64)

            end_rotation_vector = quaternion_relative_rotvec_xyzw(
                anchor_pose[3:7], end_quaternion
            )
            next_index = min(self._end_index + 1, len(self._actions) - 1)
            if next_index > self._end_index:
                next_rotation_vector = quaternion_relative_rotvec_xyzw(
                    anchor_pose[3:7], quaternions[next_index]
                )
                end_rotation_velocity = (
                    next_rotation_vector - end_rotation_vector
                ) * self._rate_hz
            elif self._end_index > 0:
                previous_rotation_vector = quaternion_relative_rotvec_xyzw(
                    anchor_pose[3:7], quaternions[self._end_index - 1]
                )
                end_rotation_velocity = (
                    end_rotation_vector - previous_rotation_vector
                ) * self._rate_hz
            else:
                end_rotation_velocity = np.zeros(3, dtype=np.float64)

            target_zsp = self._actions[self._end_index][arm_key].get("zsp")
            self._anchor_pose[side] = anchor_pose
            self._anchor_hand[side] = anchor_hand
            previous_zsp = anchor_zsp.get(side)
            self._anchor_zsp[side] = (
                None
                if previous_zsp is None
                else np.asarray(previous_zsp, dtype=np.float64).reshape(3)
            )
            self._end_pose[side] = np.concatenate(
                [end_position, end_quaternion]
            )
            self._end_hand[side] = end_hand
            self._end_zsp[side] = (
                None
                if target_zsp is None
                else np.asarray(target_zsp, dtype=np.float64).reshape(3)
            )
            self._start_position_velocity[side] = start_position_velocity
            self._end_position_velocity[side] = end_position_velocity
            self._start_rotation_velocity[side] = start_rotation_velocity
            self._end_rotation_velocity[side] = end_rotation_velocity
            self._start_hand_velocity[side] = start_hand_velocity
            self._end_hand_velocity[side] = end_hand_velocity
            self._end_rotation_vector[side] = end_rotation_vector

    @property
    def duration_s(self) -> float:
        return self._duration_s

    @property
    def blend_steps(self) -> int:
        return self._count

    def velocity_metrics(self) -> dict[str, dict[str, float]]:
        return {
            side: {
                "start_position_m_s": float(
                    np.linalg.norm(self._start_position_velocity[side])
                ),
                "end_position_m_s": float(
                    np.linalg.norm(self._end_position_velocity[side])
                ),
                "start_rotation_deg_s": float(
                    np.degrees(np.linalg.norm(self._start_rotation_velocity[side]))
                ),
                "end_rotation_deg_s": float(
                    np.degrees(np.linalg.norm(self._end_rotation_velocity[side]))
                ),
                "start_hand_max_deg_s": float(
                    np.max(np.abs(self._start_hand_velocity[side]), initial=0.0)
                ),
                "end_hand_max_deg_s": float(
                    np.max(np.abs(self._end_hand_velocity[side]), initial=0.0)
                ),
            }
            for side in self._sides
        }

    def sample(self, elapsed_s: float) -> Mapping[str, Any]:
        fraction = float(
            np.clip(float(elapsed_s) / self._duration_s, 0.0, 1.0)
        )
        action = dict(self._actions[self._end_index])
        smoothstep = fraction * fraction * (3.0 - 2.0 * fraction)
        for side in self._sides:
            arm_key = f"arm_action_{side}"
            hand_key = f"hand_action_{side}"
            anchor_pose = self._anchor_pose[side]
            position = _cubic_hermite(
                anchor_pose[:3],
                self._end_pose[side][:3],
                self._start_position_velocity[side],
                self._end_position_velocity[side],
                self._duration_s,
                fraction,
            )
            rotation_vector = _cubic_hermite(
                np.zeros(3, dtype=np.float64),
                self._end_rotation_vector[side],
                self._start_rotation_velocity[side],
                self._end_rotation_velocity[side],
                self._duration_s,
                fraction,
            )
            quaternion = quaternion_multiply_xyzw(
                anchor_pose[3:7],
                quaternion_from_rotvec_xyzw(rotation_vector),
            )
            hand = _cubic_hermite(
                self._anchor_hand[side],
                self._end_hand[side],
                self._start_hand_velocity[side],
                self._end_hand_velocity[side],
                self._duration_s,
                fraction,
            )
            arm = dict(action[arm_key])
            arm["ee_pos"] = position.astype(np.float32).tolist()
            arm["ee_quat"] = quaternion.astype(np.float32).tolist()
            start_zsp = self._anchor_zsp[side]
            end_zsp = self._end_zsp[side]
            if start_zsp is not None and end_zsp is not None:
                zsp = (1.0 - smoothstep) * start_zsp + smoothstep * end_zsp
                norm = float(np.linalg.norm(zsp))
                if norm > 1e-8:
                    zsp /= norm
                arm["zsp"] = zsp.astype(np.float32).tolist()
            action[arm_key] = arm
            action[hand_key] = hand.astype(np.float32).tolist()
        return action


def bridge_early_action_window(
    actions: Sequence[Mapping[str, Any]],
    *,
    previous_action: Mapping[str, Any],
    rate_hz: float,
    max_bridge_steps: int,
    arm_sides: Sequence[str],
    hand_sides: Sequence[str],
    limits: Mapping[str, float],
) -> tuple[list[Mapping[str, Any]], int, dict[str, float]]:
    """Choose the shortest passing sampled bridge after a committed head.

    Action 0 is immutable. A K-period bridge rejoins the exact original action
    K; only indices 1..K-1 change. Guard the complete modified neighborhood and
    its exit. This samples a Hermite curve at model rate; downstream linear /
    slerp interpolation does not promise continuous acceleration or robot safety.
    """
    if len(actions) < 3:
        raise ValueError("early bridge requires at least three actions")
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("early bridge rate must be finite and positive")
    if max_bridge_steps < 0:
        raise ValueError("early bridge maximum must be nonnegative")
    sides = tuple(dict.fromkeys((*arm_sides, *hand_sides)))

    def pose(action, side):
        arm = action[f"arm_action_{side}"]
        return np.asarray([*arm["ee_pos"], *arm["ee_quat"]], dtype=np.float64)

    def hand(action, side):
        return np.asarray(action[f"hand_action_{side}"], dtype=np.float64)

    anchor_poses = {side: pose(actions[0], side) for side in sides}
    anchor_hands = {side: hand(actions[0], side) for side in sides}
    previous_poses = {side: pose(previous_action, side) for side in sides}
    previous_hands = {side: hand(previous_action, side) for side in sides}
    anchor_zsp = {side: actions[0][f"arm_action_{side}"].get("zsp") for side in sides}
    last_error = None
    for steps in [0, *range(2, min(int(max_bridge_steps), len(actions) - 2) + 1)]:
        candidate = list(actions)
        if steps:
            bridge = VelocityContinuousBoundaryInterpolator(
                actions[1:], anchor_poses=anchor_poses, anchor_hands=anchor_hands,
                anchor_zsp=anchor_zsp, previous_poses=previous_poses,
                previous_hands=previous_hands, previous_dt_s=1.0 / rate_hz,
                blend_steps=steps, rate_hz=rate_hz, sides=sides,
            )
            for index in range(1, steps):
                candidate[index] = bridge.sample(index / rate_hz)
            # Keep the original action object at the endpoint, avoiding even
            # roundoff / normalization changes to an otherwise fixed target.
            candidate[steps] = actions[steps]
        samples = [previous_action, *candidate[:max(3, steps + 3)]]
        try:
            metrics = validate_splice_kinematics(
                {side: [pose(a, side) for a in samples] for side in arm_sides},
                {side: [hand(a, side) for a in samples] for side in hand_sides},
                rate_hz=rate_hz, **limits,
            )
        except ValueError as exc:
            last_error = exc
            continue
        return candidate, steps, metrics
    raise ValueError(f"no bounded early splice bridge: {last_error}")


def interpolate_action_pair(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    *,
    fraction: float,
    sides: Sequence[str] = ("left", "right"),
    action_mode: str = "eef",
) -> Mapping[str, Any]:
    """Sample a position-only Cartesian or joint trajectory segment.

    This follows the variable-degree spline convention used by the ROS 2
    joint trajectory controller: position-only waypoints are interpolated
    linearly in time.  Cartesian orientation is the corresponding shortest-
    path quaternion SLERP. Joint-mode arm and hand values remain radians.
    No endpoint velocity or acceleration is invented.
    """

    mode = str(action_mode).strip().lower()
    if mode not in ("eef", "joint"):
        raise ValueError("action_mode must be eef or joint")
    alpha = float(np.clip(fraction, 0.0, 1.0))
    action = dict(start)
    for side in sides:
        arm_key = f"arm_action_{side}"
        hand_key = f"hand_action_{side}"
        joint_key = f"arm_joint_action_{side}"
        if arm_key not in start or arm_key not in end:
            raise ValueError(f"missing paired {side} arm action")
        if hand_key not in start or hand_key not in end:
            raise ValueError(f"missing paired {side} hand action")
        start_arm = start[arm_key]
        end_arm = end[arm_key]
        if not isinstance(start_arm, Mapping) or not isinstance(end_arm, Mapping):
            raise ValueError(f"invalid paired {side} arm action")

        start_hand = np.asarray(start[hand_key], dtype=np.float32).reshape(20)
        end_hand = np.asarray(end[hand_key], dtype=np.float32).reshape(20)
        if mode == "joint":
            start_joint_array = np.asarray(
                start_arm.get("joint_pos"), dtype=np.float32
            ).reshape(7)
            end_joint_array = np.asarray(
                end_arm.get("joint_pos"), dtype=np.float32
            ).reshape(7)
            if not all(
                np.all(np.isfinite(values))
                for values in (
                    start_joint_array,
                    end_joint_array,
                    start_hand,
                    end_hand,
                )
            ):
                raise ValueError(f"non-finite {side} joint interpolation waypoint")
            arm = dict(start_arm)
            arm["joint_pos"] = (
                (1.0 - alpha) * start_joint_array
                + alpha * end_joint_array
            ).astype(np.float32).tolist()
            action[arm_key] = arm
            action[hand_key] = (
                (1.0 - alpha) * start_hand + alpha * end_hand
            ).astype(np.float32).tolist()
            continue

        start_position = np.asarray(
            start_arm.get("ee_pos"), dtype=np.float32
        ).reshape(3)
        end_position = np.asarray(
            end_arm.get("ee_pos"), dtype=np.float32
        ).reshape(3)
        start_quaternion = np.asarray(
            start_arm.get("ee_quat"), dtype=np.float32
        ).reshape(4)
        end_quaternion = np.asarray(
            end_arm.get("ee_quat"), dtype=np.float32
        ).reshape(4)
        if not all(
            np.all(np.isfinite(values))
            for values in (
                start_position,
                end_position,
                start_quaternion,
                end_quaternion,
                start_hand,
                end_hand,
            )
        ):
            raise ValueError(f"non-finite {side} interpolation waypoint")

        arm = dict(start_arm)
        arm["ee_pos"] = (
            (1.0 - alpha) * start_position + alpha * end_position
        ).astype(np.float32).tolist()
        arm["ee_quat"] = quaternion_slerp_xyzw(
            start_quaternion, end_quaternion, alpha
        ).astype(np.float32).tolist()

        start_zsp = start_arm.get("zsp")
        end_zsp = end_arm.get("zsp")
        if start_zsp is not None and end_zsp is not None:
            zsp = (
                (1.0 - alpha)
                * np.asarray(start_zsp, dtype=np.float32).reshape(3)
                + alpha * np.asarray(end_zsp, dtype=np.float32).reshape(3)
            )
            norm = float(np.linalg.norm(zsp))
            if not np.all(np.isfinite(zsp)) or norm <= 1e-8:
                raise ValueError(f"invalid interpolated {side} ZSP")
            arm["zsp"] = (zsp / norm).astype(np.float32).tolist()
        elif alpha >= 1.0:
            arm["zsp"] = end_zsp

        action[arm_key] = arm
        action[hand_key] = (
            (1.0 - alpha) * start_hand + alpha * end_hand
        ).astype(np.float32).tolist()
        start_joints = start.get(joint_key)
        end_joints = end.get(joint_key)
        if (start_joints is None) != (end_joints is None):
            raise ValueError(f"incomplete paired {side} joint action")
        if start_joints is not None:
            start_joint_array = np.asarray(
                start_joints, dtype=np.float32
            ).reshape(7)
            end_joint_array = np.asarray(
                end_joints, dtype=np.float32
            ).reshape(7)
            if not (
                np.all(np.isfinite(start_joint_array))
                and np.all(np.isfinite(end_joint_array))
            ):
                raise ValueError(f"non-finite {side} joint waypoint")
            action[joint_key] = (
                (1.0 - alpha) * start_joint_array
                + alpha * end_joint_array
            ).astype(np.float32).tolist()
    return action


class PchipActionInterpolator:
    """Shape-preserving C1 sampler for one validated model action chunk.

    Translation and hand joint positions use SciPy's PCHIP implementation.
    Orientation remains shortest-path SLERP and ZSP remains normalized linear
    interpolation because they do not live in an ordinary Euclidean vector
    space.  A two-point chunk falls back to the position-only linear rule.
    """

    def __init__(
        self,
        actions: Sequence[Mapping[str, Any]],
        *,
        sides: Sequence[str] = ("left", "right"),
        action_mode: str = "eef",
    ) -> None:
        if not actions:
            raise ValueError("cannot interpolate an empty action chunk")
        self._actions = tuple(actions)
        self._sides = tuple(sides)
        self._action_mode = str(action_mode).strip().lower()
        if self._action_mode not in ("eef", "joint"):
            raise ValueError("action_mode must be eef or joint")
        self._position = {}
        self._hand = {}
        self._joint = {}
        self._pchip_enabled = len(self._actions) >= 3
        if self._pchip_enabled and PchipInterpolator is None:
            raise RuntimeError(
                "pchip_slerp requires scipy.interpolate.PchipInterpolator"
            )

        sample_indices = np.arange(len(self._actions), dtype=np.float64)
        for side in self._sides:
            arm_key = f"arm_action_{side}"
            hand_key = f"hand_action_{side}"
            joint_key = f"arm_joint_action_{side}"
            if self._action_mode == "joint":
                joints = np.stack(
                    [
                        np.asarray(
                            action[arm_key]["joint_pos"], dtype=np.float64
                        ).reshape(7)
                        for action in self._actions
                    ]
                )
                hands = np.stack(
                    [
                        np.asarray(
                            action[hand_key], dtype=np.float64
                        ).reshape(20)
                        for action in self._actions
                    ]
                )
                if not np.all(np.isfinite(joints)) or not np.all(
                    np.isfinite(hands)
                ):
                    raise ValueError(f"non-finite {side} joint PCHIP waypoint")
                if self._pchip_enabled:
                    self._joint[side] = PchipInterpolator(
                        sample_indices,
                        joints,
                        axis=0,
                        extrapolate=False,
                    )
                    self._hand[side] = PchipInterpolator(
                        sample_indices,
                        hands,
                        axis=0,
                        extrapolate=False,
                    )
                continue
            positions = np.stack(
                [
                    np.asarray(action[arm_key]["ee_pos"], dtype=np.float64)
                    .reshape(3)
                    for action in self._actions
                ]
            )
            hands = np.stack(
                [
                    np.asarray(action[hand_key], dtype=np.float64).reshape(20)
                    for action in self._actions
                ]
            )
            if not np.all(np.isfinite(positions)) or not np.all(
                np.isfinite(hands)
            ):
                raise ValueError(f"non-finite {side} PCHIP waypoint")
            if self._pchip_enabled:
                self._position[side] = PchipInterpolator(
                    sample_indices,
                    positions,
                    axis=0,
                    extrapolate=False,
                )
                self._hand[side] = PchipInterpolator(
                    sample_indices,
                    hands,
                    axis=0,
                    extrapolate=False,
                )
                if all(joint_key in action for action in self._actions):
                    joints = np.stack(
                        [
                            np.asarray(
                                action[joint_key], dtype=np.float64
                            ).reshape(7)
                            for action in self._actions
                        ]
                    )
                    if not np.all(np.isfinite(joints)):
                        raise ValueError(
                            f"non-finite {side} joint PCHIP waypoint"
                        )
                    self._joint[side] = PchipInterpolator(
                        sample_indices,
                        joints,
                        axis=0,
                        extrapolate=False,
                    )

    @property
    def action_count(self) -> int:
        return len(self._actions)

    @property
    def pchip_enabled(self) -> bool:
        return self._pchip_enabled

    def sample(self, frame_index: float) -> Mapping[str, Any]:
        index = float(np.clip(frame_index, 0.0, len(self._actions) - 1))
        lower = int(math.floor(index))
        upper = min(lower + 1, len(self._actions) - 1)
        fraction = index - lower
        action = interpolate_action_pair(
            self._actions[lower],
            self._actions[upper],
            fraction=fraction,
            sides=self._sides,
            action_mode=self._action_mode,
        )
        if not self._pchip_enabled:
            return action

        action = dict(action)
        for side in self._sides:
            arm_key = f"arm_action_{side}"
            hand_key = f"hand_action_{side}"
            joint_key = f"arm_joint_action_{side}"
            if self._action_mode == "joint":
                joints = np.asarray(
                    self._joint[side](index), dtype=np.float32
                ).reshape(7)
                hand = np.asarray(
                    self._hand[side](index), dtype=np.float32
                ).reshape(20)
                if not np.all(np.isfinite(joints)) or not np.all(
                    np.isfinite(hand)
                ):
                    raise ValueError(
                        f"non-finite interpolated {side} joint PCHIP action"
                    )
                arm = dict(action[arm_key])
                arm["joint_pos"] = joints.tolist()
                action[arm_key] = arm
                action[hand_key] = hand.tolist()
                continue
            position = np.asarray(
                self._position[side](index), dtype=np.float32
            ).reshape(3)
            hand = np.asarray(
                self._hand[side](index), dtype=np.float32
            ).reshape(20)
            if not np.all(np.isfinite(position)) or not np.all(
                np.isfinite(hand)
            ):
                raise ValueError(f"non-finite interpolated {side} PCHIP action")
            arm = dict(action[arm_key])
            arm["ee_pos"] = position.tolist()
            action[arm_key] = arm
            action[hand_key] = hand.tolist()
            if side in self._joint:
                joints = np.asarray(
                    self._joint[side](index), dtype=np.float32
                ).reshape(7)
                if not np.all(np.isfinite(joints)):
                    raise ValueError(
                        f"non-finite interpolated {side} joint PCHIP action"
                    )
                action[joint_key] = joints.tolist()
        return action


class LatestActionPlan:
    """Thread-safe open-loop action chunk broker.

    This follows openpi's ``ActionChunkBroker`` semantics: install one chunk,
    consume its actions in order, and request another inference only after the
    selected open-loop horizon is exhausted. Normal installation rejects an
    unconsumed chunk. The opt-in controller can explicitly compare-and-replace
    by the first unconsumed sequence, after validating the splice and retaining
    that already committed interpolation endpoint.

    The action clock is phase locked to the chunk's first dispatch time.  A
    polling timer may observe a deadline slightly late, but that small error is
    not added to every following action period.  This is important when, for
    example, a 100/120 Hz ROS timer dispatches a 30 Hz action stream.

    If the executor is at least one complete action period late, the remaining
    deadlines are shifted forward once.  This preserves every action while
    preventing a stalled executor from replaying its backlog at the faster ROS
    polling rate.  ``max_schedule_lag_s`` remains an accepted compatibility
    argument for older callers; actions are never dropped based on it.
    """

    def __init__(
        self,
        *,
        max_actions: int,
        max_schedule_lag_s: Optional[float] = None,
    ) -> None:
        if max_actions <= 0:
            raise ValueError("open-loop horizon must be positive")
        if max_schedule_lag_s is not None and max_schedule_lag_s <= 0.0:
            raise ValueError("max_schedule_lag_s must be positive")
        self._max_actions = int(max_actions)
        self._actions: deque[ScheduledAction] = deque()
        self._lock = threading.Lock()
        self._sequence = 0
        self._rate_hz = 0.0
        self._next_due_at = 0.0
        self._latency_dropped = 0
        self._execution_dropped = 0
        self._schedule_realignments = 0
        self._dispatched_actions = 0
        self._last_dispatch_lag_s = 0.0
        self._max_dispatch_lag_s = 0.0
        self._dispatch_times: deque[float] = deque(maxlen=120)
        self._last_response_at = 0.0

    def snapshot(self) -> tuple[ScheduledAction, ...]:
        """Atomically snapshot unconsumed endpoints for prefix-conditioned inference."""
        with self._lock:
            return tuple(self._actions)

    def install(
        self,
        actions: Sequence[Mapping[str, Any]],
        *,
        observation_created_at: float,
        received_at: float,
        rate_hz: float,
        schedule_start_at: Optional[float] = None,
        replace_from_sequence: Optional[int] = None,
    ) -> InstallResult:
        """Install a plan, optionally replacing a caller-validated future.

        ``replace_from_sequence`` is a compare-and-replace token: it must equal
        the current head under the broker lock. The caller is responsible for
        retaining the committed head action/deadline and checking continuity.
        Ordinary callers omit it and cannot overwrite unconsumed actions.
        """
        if rate_hz <= 0.0:
            raise ValueError("action rate must be positive")
        input_count = len(actions)
        del observation_created_at  # The official broker starts at chunk index 0.
        latency_dropped = 0
        usable = list(actions[: self._max_actions])
        interval = 1.0 / float(rate_hz)
        schedule_start = float(
            received_at if schedule_start_at is None else schedule_start_at
        )
        scheduled = []
        with self._lock:
            if replace_from_sequence is not None:
                if not self._actions or self._actions[0].sequence != replace_from_sequence:
                    raise RuntimeError("splice anchor changed before install")
            elif self._actions:
                raise RuntimeError(
                    "cannot replace an unconsumed open-loop action chunk"
                )
            for index, action in enumerate(usable):
                self._sequence += 1
                scheduled.append(
                    ScheduledAction(
                        sequence=self._sequence,
                        due_at=schedule_start + index * interval,
                        response_received_at=float(received_at),
                        action=action,
                    )
                )
            self._actions = deque(scheduled)
            self._rate_hz = float(rate_hz)
            self._next_due_at = schedule_start if scheduled else 0.0
            self._last_response_at = float(received_at)
        return InstallResult(
            accepted=len(scheduled),
            latency_dropped=latency_dropped,
            input_count=input_count,
        )

    def pop_due(self, now: float) -> Optional[ScheduledAction]:
        current = float(now)
        with self._lock:
            if not self._actions or current < self._next_due_at:
                return None
            candidate = self._actions.popleft()
            dispatch_lag = max(0.0, current - candidate.due_at)
            self._dispatched_actions += 1
            self._dispatch_times.append(current)
            self._last_dispatch_lag_s = dispatch_lag
            self._max_dispatch_lag_s = max(
                self._max_dispatch_lag_s, dispatch_lag
            )
            if self._actions:
                interval = 1.0 / self._rate_hz
                if dispatch_lag >= interval:
                    # A real executor stall must not be followed by a 120 Hz
                    # catch-up burst.  Keep all remaining actions, but move
                    # their common time base so the next one is a full action
                    # period after this dispatch.
                    next_due_at = current + interval
                    shift = next_due_at - self._actions[0].due_at
                    self._actions = deque(
                        replace(action, due_at=action.due_at + shift)
                        for action in self._actions
                    )
                    self._schedule_realignments += 1
                # For ordinary timer quantization, retain the original phase.
                # This prevents 1-8 ms of callback lateness from accumulating
                # on every nominal 33.3 ms action period.
                self._next_due_at = self._actions[0].due_at
            else:
                self._next_due_at = 0.0
            return candidate

    def peek_next(self) -> Optional[ScheduledAction]:
        """Return the next scheduled model waypoint without consuming it."""

        with self._lock:
            return self._actions[0] if self._actions else None

    def clear(self) -> None:
        with self._lock:
            self._actions.clear()
            self._next_due_at = 0.0

    def remaining(self) -> int:
        with self._lock:
            return len(self._actions)

    def status(self) -> dict[str, Any]:
        with self._lock:
            rolling_dispatch_rate_hz = 0.0
            last_dispatch_interval_ms = 0.0
            if len(self._dispatch_times) >= 2:
                duration = self._dispatch_times[-1] - self._dispatch_times[0]
                if duration > 0.0:
                    rolling_dispatch_rate_hz = (
                        len(self._dispatch_times) - 1
                    ) / duration
                last_dispatch_interval_ms = (
                    self._dispatch_times[-1] - self._dispatch_times[-2]
                ) * 1000.0
            return {
                "buffered_actions": len(self._actions),
                "action_rate_hz": self._rate_hz,
                "dispatched_actions": self._dispatched_actions,
                "rolling_dispatch_rate_hz": rolling_dispatch_rate_hz,
                "last_dispatch_interval_ms": last_dispatch_interval_ms,
                "latency_dropped_actions": self._latency_dropped,
                "execution_dropped_actions": self._execution_dropped,
                "schedule_realignments": self._schedule_realignments,
                "last_dispatch_lag_ms": self._last_dispatch_lag_s * 1000.0,
                "max_dispatch_lag_ms": self._max_dispatch_lag_s * 1000.0,
                "last_response_monotonic": self._last_response_at,
            }
