# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Protocol-v2 request validation and sanitized error envelopes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig


class ProtocolError(Exception):
    """A stable, client-visible protocol or adapter error."""

    def __init__(self, code: str, message: str, *, fatal_session: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fatal_session = fatal_session


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("COSMOS_INVALID_SCHEMA", f"{field} must be a mapping")
    return value


def _require_identity(request: Mapping[str, Any]) -> tuple[str, int]:
    session_id = request.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ProtocolError("COSMOS_INVALID_IDENTITY", "session_id must be a non-empty string", fatal_session=True)
    request_id = request.get("request_id")
    if type(request_id) is not int or request_id < 0:
        raise ProtocolError("COSMOS_INVALID_IDENTITY", "request_id must be a non-negative integer")
    return session_id, request_id


def _require_protocol_version(request: Mapping[str, Any], expected: int) -> None:
    if type(request.get("protocol_version")) is not int or request["protocol_version"] != expected:
        raise ProtocolError(
            "COSMOS_PROTOCOL_VERSION",
            f"protocol_version must be {expected}",
            fatal_session=True,
        )


def _finite_array(value: Any, field: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ProtocolError("COSMOS_INVALID_STATE", f"{field} must be a numpy array")
    if value.shape != shape:
        raise ProtocolError("COSMOS_INVALID_STATE", f"{field} must have shape {shape}")
    if value.dtype != np.float32:
        raise ProtocolError("COSMOS_INVALID_STATE", f"{field} must have dtype float32")
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ProtocolError("COSMOS_INVALID_STATE", f"{field} contains non-finite values")
    return array


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except Exception:
            return False
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes, bytearray))
        and not isinstance(right, (str, bytes, bytearray))
    ):
        return len(left) == len(right) and all(_values_equal(a, b) for a, b in zip(left, right, strict=True))
    try:
        return bool(left == right)
    except Exception:
        return False


def _validate_robot_layout(request: Mapping[str, Any], config: RobotPolicyConfig) -> None:
    robot_layout = _require_mapping(request.get("robot_layout"), "robot_layout")
    expected = config.deployment.robot_layout
    if expected is not None and not _values_equal(robot_layout, expected):
        raise ProtocolError("COSMOS_PROFILE_MISMATCH", "robot_layout does not match the deployment manifest")


def _validate_action_profile(request: Mapping[str, Any], config: RobotPolicyConfig) -> None:
    expected = config.deployment.action_space
    for field in ("arm_command_mode", "action_space"):
        value = request.get(field)
        # Protocol-v2 EEF clients predate these explicit fields. Keep them
        # compatible, while requiring the safety-critical joint declaration.
        if value is None and expected == "eef":
            continue
        if value != expected:
            raise ProtocolError("COSMOS_PROFILE_MISMATCH", f"{field} must be {expected!r}")


def validate_hello(request: Mapping[str, Any], config: RobotPolicyConfig) -> tuple[str, int]:
    """Validate a hello request without invoking model inference."""

    _require_protocol_version(request, config.deployment.protocol_version)
    if request.get("message_type") != "hello":
        raise ProtocolError("COSMOS_MESSAGE_TYPE", "message_type must be hello")
    identity = _require_identity(request)
    _validate_robot_layout(request, config)
    _validate_action_profile(request, config)

    camera_names = request.get("camera_names")
    if not isinstance(camera_names, list) or not all(isinstance(name, str) for name in camera_names):
        raise ProtocolError("COSMOS_INVALID_CAMERAS", "camera_names must be a list of strings")
    if camera_names != config.deployment.camera_names:
        raise ProtocolError("COSMOS_PROFILE_MISMATCH", "camera_names do not match the deployment manifest")
    return identity


def _validate_arm_state(request: Mapping[str, Any], side: str, min_quaternion_norm: float) -> None:
    key = f"arm_state_{side}"
    state = _require_mapping(request.get(key), key)
    _finite_array(state.get("joint_pos"), f"{key}.joint_pos", (7,))
    _finite_array(state.get("joint_vel"), f"{key}.joint_vel", (7,))
    _finite_array(state.get("joint_torque"), f"{key}.joint_torque", (7,))
    ee_pos = _finite_array(state.get("ee_pos"), f"{key}.ee_pos", (3,))
    ee_quat = _finite_array(state.get("ee_quat"), f"{key}.ee_quat", (4,))
    eef = _finite_array(state.get("eef"), f"{key}.eef", (7,))
    if float(np.linalg.norm(ee_quat)) < min_quaternion_norm or float(np.linalg.norm(eef[3:])) < min_quaternion_norm:
        raise ProtocolError("COSMOS_INVALID_STATE", f"{key} contains a zero quaternion")
    if not np.allclose(eef[:3], ee_pos, atol=1e-5, rtol=1e-5):
        raise ProtocolError("COSMOS_INVALID_STATE", f"{key}.eef position disagrees with ee_pos")
    normalized_ee = ee_quat / np.linalg.norm(ee_quat)
    normalized_eef = eef[3:] / np.linalg.norm(eef[3:])
    if not (
        np.allclose(normalized_ee, normalized_eef, atol=1e-5, rtol=1e-5)
        or np.allclose(normalized_ee, -normalized_eef, atol=1e-5, rtol=1e-5)
    ):
        raise ProtocolError("COSMOS_INVALID_STATE", f"{key}.eef quaternion disagrees with ee_quat")


def _validate_hand_state(request: Mapping[str, Any], side: str) -> None:
    key = f"hand_state_{side}"
    state = _require_mapping(request.get(key), key)
    _finite_array(state.get("joint_pos"), f"{key}.joint_pos", (20,))
    _finite_array(state.get("joint_vel"), f"{key}.joint_vel", (20,))
    _finite_array(state.get("joint_torque"), f"{key}.joint_torque", (20,))


def _validate_images(request: Mapping[str, Any], config: RobotPolicyConfig) -> None:
    images = _require_mapping(request.get("images"), "images")
    required = config.deployment.camera_names
    if not set(required).issubset(images):
        missing = sorted(set(required) - set(images))
        raise ProtocolError("COSMOS_MISSING_CAMERA", f"missing required cameras: {', '.join(missing)}")

    source_timestamps = _require_mapping(request.get("source_timestamps"), "source_timestamps")
    for camera_name in required:
        payload = _require_mapping(images[camera_name], f"images.{camera_name}")
        if payload.get("codec") != config.image_preprocessing.codec:
            raise ProtocolError("COSMOS_INVALID_IMAGE", f"images.{camera_name}.codec must be jpeg")
        if payload.get("color_space") != config.image_preprocessing.input_color_space:
            raise ProtocolError("COSMOS_INVALID_IMAGE", f"images.{camera_name}.color_space must be bgr8")
        shape = payload.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[2] != 3
        ):
            raise ProtocolError("COSMOS_INVALID_IMAGE", f"images.{camera_name}.shape must be [height, width, 3]")
        if not _is_finite_number(payload.get("timestamp")):
            raise ProtocolError("COSMOS_INVALID_IMAGE", f"images.{camera_name}.timestamp must be finite")
        data = payload.get("data")
        if not isinstance(data, bytes) or not data:
            raise ProtocolError("COSMOS_INVALID_IMAGE", f"images.{camera_name}.data must be non-empty bytes")

        timestamp_key = f"camera_{camera_name}"
        source_timestamp = source_timestamps.get(timestamp_key)
        if not _is_finite_number(source_timestamp):
            raise ProtocolError("COSMOS_INVALID_IMAGE", f"source_timestamps.{timestamp_key} must be finite")
        max_skew = config.timestamps.max_image_source_skew_s
        if max_skew is not None and abs(float(payload["timestamp"]) - float(source_timestamp)) > max_skew:
            raise ProtocolError("COSMOS_TIMESTAMP_SKEW", f"camera timestamp skew is too large for {camera_name}")


def validate_observation(request: Mapping[str, Any], config: RobotPolicyConfig) -> tuple[str, int]:
    """Validate one formal protocol-v2 observation request."""

    _require_protocol_version(request, config.deployment.protocol_version)
    if type(request.get("schema_version")) is not int or request["schema_version"] != config.deployment.schema_version:
        raise ProtocolError(
            "COSMOS_SCHEMA_VERSION",
            f"schema_version must be {config.deployment.schema_version}",
            fatal_session=True,
        )
    if request.get("message_type") != "observation":
        raise ProtocolError("COSMOS_MESSAGE_TYPE", "message_type must be observation")
    identity = _require_identity(request)
    _validate_robot_layout(request, config)

    if not _is_finite_number(request.get("timestamp")) or not _is_finite_number(request.get("client_monotonic")):
        raise ProtocolError("COSMOS_INVALID_TIMESTAMP", "timestamp and client_monotonic must be finite")
    if request.get("arms") != ["left", "right"]:
        raise ProtocolError("COSMOS_PROFILE_MISMATCH", "arms must be ['left', 'right']")
    if request.get("active_hand_sides") != config.deployment.active_hand_sides:
        raise ProtocolError("COSMOS_PROFILE_MISMATCH", "active_hand_sides do not match the deployment manifest")
    if request.get("zero_filled_hand_sides") != config.deployment.zero_filled_hand_sides:
        raise ProtocolError("COSMOS_PROFILE_MISMATCH", "zero_filled_hand_sides do not match the deployment manifest")

    min_norm = float(config.safety.quaternion_min_norm)
    for side in ("left", "right"):
        _validate_arm_state(request, side, min_norm)
        _validate_hand_state(request, side)
    _validate_images(request, config)
    return identity


def error_envelope(request: Mapping[str, Any] | None, error: ProtocolError) -> dict[str, Any]:
    """Build the stable pickle error envelope required by protocol-v2."""

    return {
        "protocol_version": 2,
        "session_id": request.get("session_id") if request is not None else None,
        "request_id": request.get("request_id") if request is not None else None,
        "error": True,
        "error_code": error.code,
        "message": error.message,
        "fatal_session": error.fatal_session,
    }


__all__ = [
    "ProtocolError",
    "error_envelope",
    "validate_hello",
    "validate_observation",
]
