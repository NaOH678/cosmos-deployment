"""Pure helpers for optional, frame-aligned teleoperation diagnostics.

Nothing in this module participates in robot control.  It only converts the
already-published OpenVR and MANUS observations into fixed-shape arrays that
can be stored beside a training frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


TRACKER_DIAGNOSTIC_COLUMNS = 26
DEFAULT_TRACKER_ROLES = (
    "chest",
    "left_wrist",
    "right_wrist",
    "left_arm",
    "right_arm",
)

# Keep this list byte-for-byte equivalent to controller.wujihand_node's
# semantic MANUS -> MediaPipe conversion.  It is duplicated here deliberately
# so the recorder never imports or calls the hand control implementation.
MEDIAPIPE_MANUS_JOINTS = (
    ("Hand", "Invalid"),
    ("Thumb", "MCP"), ("Thumb", "PIP"),
    ("Thumb", "DIP"), ("Thumb", "TIP"),
    ("Index", "PIP"), ("Index", "IP"),
    ("Index", "DIP"), ("Index", "TIP"),
    ("Middle", "PIP"), ("Middle", "IP"),
    ("Middle", "DIP"), ("Middle", "TIP"),
    ("Ring", "PIP"), ("Ring", "IP"),
    ("Ring", "DIP"), ("Ring", "TIP"),
    ("Pinky", "PIP"), ("Pinky", "IP"),
    ("Pinky", "DIP"), ("Pinky", "TIP"),
)

MANUS_CHAIN_TYPE_CODES = (
    "Hand",
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Pinky",
)
MANUS_JOINT_TYPE_CODES = (
    "Invalid",
    "Metacarpal",
    "MCP",
    "PIP",
    "IP",
    "DIP",
    "TIP",
)
MANUS_ERGONOMICS_TYPE_CODES = (
    "Invalid",
    "ThumbMCPSpread",
    "ThumbMCPStretch",
    "ThumbPIPStretch",
    "ThumbDIPStretch",
    "IndexMCPStretch",
    "IndexPIPStretch",
    "IndexDIPStretch",
    "MiddleSpread",
    "MiddleMCPStretch",
    "MiddlePIPStretch",
    "MiddleDIPStretch",
    "RingSpread",
    "RingMCPStretch",
    "RingPIPStretch",
    "RingDIPStretch",
    "PinkySpread",
    "PinkyMCPStretch",
    "PinkyPIPStretch",
    "PinkyDIPStretch",
)


def _type_code(value: Any, labels: tuple[str, ...]) -> int:
    normalized = str(value).strip().lower()
    for index, label in enumerate(labels):
        if normalized == label.lower():
            return index
    return -1


@dataclass(frozen=True)
class OptionalDatasetSpec:
    """Description of one optional LMDB array stored for every frame."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    fill_value: Any = 0

    def empty(self) -> np.ndarray:
        return np.full(self.shape, self.fill_value, dtype=np.dtype(self.dtype))

    def metadata(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape_per_frame": list(self.shape),
            "dtype": np.dtype(self.dtype).str,
            "fill_value": self.fill_value,
        }


def teleop_diagnostic_specs(
    sides: Iterable[str] = ("left", "right"),
    tracker_role_count: int = len(DEFAULT_TRACKER_ROLES),
    manus_node_count: int = 25,
    manus_sensor_count: int = 5,
    manus_ergonomics_count: int = 20,
) -> dict[str, OptionalDatasetSpec]:
    """Return the fixed optional schema used by the recorder.

    All paths have the same leading dimension as ``action``.  Invalid or
    unavailable sources are represented by their fill value and an explicit
    validity/availability array.
    """
    tracker = {
        "teleop_tracker_raw_pose": OptionalDatasetSpec(
            "/teleop/tracker/raw_pose", (tracker_role_count, 7), "float32"
        ),
        "teleop_tracker_corrected_pose": OptionalDatasetSpec(
            "/teleop/tracker/corrected_pose", (tracker_role_count, 7), "float32"
        ),
        "teleop_tracker_linear_velocity": OptionalDatasetSpec(
            "/teleop/tracker/linear_velocity", (tracker_role_count, 3), "float32"
        ),
        "teleop_tracker_angular_velocity": OptionalDatasetSpec(
            "/teleop/tracker/angular_velocity", (tracker_role_count, 3), "float32"
        ),
        "teleop_tracker_detected": OptionalDatasetSpec(
            "/teleop/tracker/detected", (tracker_role_count,), "uint8"
        ),
        "teleop_tracker_connected": OptionalDatasetSpec(
            "/teleop/tracker/connected", (tracker_role_count,), "uint8"
        ),
        "teleop_tracker_valid": OptionalDatasetSpec(
            "/teleop/tracker/valid", (tracker_role_count,), "uint8"
        ),
        "teleop_tracker_corrected_valid": OptionalDatasetSpec(
            "/teleop/tracker/corrected_valid", (tracker_role_count,), "uint8"
        ),
        "teleop_tracker_tracking_result": OptionalDatasetSpec(
            "/teleop/tracker/tracking_result", (tracker_role_count,), "int32", -1
        ),
        "teleop_tracker_device_index": OptionalDatasetSpec(
            "/teleop/tracker/device_index", (tracker_role_count,), "int32", -1
        ),
        "teleop_tracker_available": OptionalDatasetSpec(
            "/teleop/tracker/available", (1,), "uint8"
        ),
        "teleop_tracker_source_timestamp": OptionalDatasetSpec(
            "/teleop/tracker/source_timestamp", (1,), "float64"
        ),
        "teleop_tracker_alignment_error_s": OptionalDatasetSpec(
            "/teleop/tracker/alignment_error_s", (1,), "float32"
        ),
    }

    manus: dict[str, OptionalDatasetSpec] = {}
    for side in sides:
        prefix = f"teleop_manus_{side}"
        path = f"/teleop/manus/{side}"
        manus.update(
            {
                f"{prefix}_raw_node_pose": OptionalDatasetSpec(
                    f"{path}/raw_node_pose", (manus_node_count, 7), "float32"
                ),
                f"{prefix}_node_id": OptionalDatasetSpec(
                    f"{path}/node_id", (manus_node_count,), "int32", -1
                ),
                f"{prefix}_parent_node_id": OptionalDatasetSpec(
                    f"{path}/parent_node_id", (manus_node_count,), "int32", -1
                ),
                f"{prefix}_chain_type_code": OptionalDatasetSpec(
                    f"{path}/chain_type_code", (manus_node_count,), "int16", -1
                ),
                f"{prefix}_joint_type_code": OptionalDatasetSpec(
                    f"{path}/joint_type_code", (manus_node_count,), "int16", -1
                ),
                f"{prefix}_node_valid": OptionalDatasetSpec(
                    f"{path}/node_valid", (manus_node_count,), "uint8"
                ),
                f"{prefix}_keypoints_21": OptionalDatasetSpec(
                    f"{path}/keypoints_21", (21, 3), "float32"
                ),
                f"{prefix}_keypoints_valid": OptionalDatasetSpec(
                    f"{path}/keypoints_valid", (1,), "uint8"
                ),
                f"{prefix}_ergonomics_value": OptionalDatasetSpec(
                    f"{path}/ergonomics_value",
                    (manus_ergonomics_count,),
                    "float32",
                ),
                f"{prefix}_ergonomics_type_code": OptionalDatasetSpec(
                    f"{path}/ergonomics_type_code",
                    (manus_ergonomics_count,),
                    "int16",
                    -1,
                ),
                f"{prefix}_ergonomics_valid": OptionalDatasetSpec(
                    f"{path}/ergonomics_valid",
                    (manus_ergonomics_count,),
                    "uint8",
                ),
                f"{prefix}_raw_sensor_orientation": OptionalDatasetSpec(
                    f"{path}/raw_sensor_orientation", (4,), "float32"
                ),
                f"{prefix}_raw_sensor_pose": OptionalDatasetSpec(
                    f"{path}/raw_sensor_pose", (manus_sensor_count, 7), "float32"
                ),
                f"{prefix}_raw_sensor_valid": OptionalDatasetSpec(
                    f"{path}/raw_sensor_valid", (manus_sensor_count,), "uint8"
                ),
                f"{prefix}_glove_id": OptionalDatasetSpec(
                    f"{path}/glove_id", (1,), "int32", -1
                ),
                f"{prefix}_reported_node_count": OptionalDatasetSpec(
                    f"{path}/reported_node_count", (1,), "int32"
                ),
                f"{prefix}_reported_sensor_count": OptionalDatasetSpec(
                    f"{path}/reported_sensor_count", (1,), "int32"
                ),
                f"{prefix}_available": OptionalDatasetSpec(
                    f"{path}/available", (1,), "uint8"
                ),
                f"{prefix}_source_timestamp": OptionalDatasetSpec(
                    f"{path}/source_timestamp", (1,), "float64"
                ),
                f"{prefix}_alignment_error_s": OptionalDatasetSpec(
                    f"{path}/alignment_error_s", (1,), "float32"
                ),
            }
        )
    tracker.update(manus)
    return tracker


def empty_optional_frame(
    specs: Mapping[str, OptionalDatasetSpec],
) -> dict[str, np.ndarray]:
    return {name: spec.empty() for name, spec in specs.items()}


def decode_tracker_diagnostic(
    data: Any,
    role_count: int = len(DEFAULT_TRACKER_ROLES),
) -> dict[str, np.ndarray]:
    """Decode the fixed OpenVR diagnostic Float64MultiArray payload."""
    flat = np.asarray(data, dtype=np.float64).reshape(-1)
    expected = role_count * TRACKER_DIAGNOSTIC_COLUMNS
    if flat.size != expected:
        raise ValueError(
            f"tracker diagnostic payload must contain {expected} values, got {flat.size}"
        )
    rows = flat.reshape(role_count, TRACKER_DIAGNOSTIC_COLUMNS)
    if not np.all(np.isfinite(rows)):
        raise ValueError("tracker diagnostic payload contains non-finite values")
    return {
        "teleop_tracker_raw_pose": rows[:, 0:7].astype(np.float32),
        "teleop_tracker_corrected_pose": rows[:, 7:14].astype(np.float32),
        "teleop_tracker_linear_velocity": rows[:, 14:17].astype(np.float32),
        "teleop_tracker_angular_velocity": rows[:, 17:20].astype(np.float32),
        "teleop_tracker_detected": (rows[:, 20] > 0.5).astype(np.uint8),
        "teleop_tracker_connected": (rows[:, 21] > 0.5).astype(np.uint8),
        "teleop_tracker_valid": (rows[:, 22] > 0.5).astype(np.uint8),
        "teleop_tracker_corrected_valid": (rows[:, 25] > 0.5).astype(np.uint8),
        "teleop_tracker_tracking_result": rows[:, 23].astype(np.int32),
        "teleop_tracker_device_index": rows[:, 24].astype(np.int32),
        "teleop_tracker_available": np.asarray([1], dtype=np.uint8),
    }


def _pose_array(pose: Any) -> np.ndarray:
    return np.asarray(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float32,
    )


def decode_manus_message(
    message: Any,
    side: str,
    node_count: int = 25,
    sensor_count: int = 5,
    ergonomics_count: int = 20,
) -> dict[str, np.ndarray]:
    """Convert one existing ManusGlove message without touching control code."""
    normalized_side = str(side).strip().lower()
    if normalized_side not in ("left", "right"):
        raise ValueError(f"unsupported MANUS side: {side!r}")
    prefix = f"teleop_manus_{normalized_side}"

    raw_nodes = sorted(
        list(getattr(message, "raw_nodes", [])),
        key=lambda item: int(getattr(item, "node_id", -1)),
    )
    node_pose = np.zeros((node_count, 7), dtype=np.float32)
    node_id = np.full(node_count, -1, dtype=np.int32)
    parent_node_id = np.full(node_count, -1, dtype=np.int32)
    chain_type_code = np.full(node_count, -1, dtype=np.int16)
    joint_type_code = np.full(node_count, -1, dtype=np.int16)
    node_valid = np.zeros(node_count, dtype=np.uint8)
    semantic_positions: dict[tuple[str, str], np.ndarray] = {}
    for node in raw_nodes:
        pose = _pose_array(node.pose)
        if not np.all(np.isfinite(pose)):
            continue
        chain = str(getattr(node, "chain_type", ""))
        joint = str(getattr(node, "joint_type", ""))
        semantic_positions[(chain.strip().lower(), joint.strip().lower())] = pose[:3]
    for index, node in enumerate(raw_nodes[:node_count]):
        pose = _pose_array(node.pose)
        if not np.all(np.isfinite(pose)):
            continue
        node_pose[index] = pose
        node_id[index] = int(getattr(node, "node_id", -1))
        parent_node_id[index] = int(getattr(node, "parent_node_id", -1))
        chain = str(getattr(node, "chain_type", ""))
        joint = str(getattr(node, "joint_type", ""))
        chain_type_code[index] = _type_code(chain, MANUS_CHAIN_TYPE_CODES)
        joint_type_code[index] = _type_code(joint, MANUS_JOINT_TYPE_CODES)
        node_valid[index] = 1

    keypoints = np.zeros((21, 3), dtype=np.float32)
    semantic_keys = tuple(
        (chain.lower(), joint.lower()) for chain, joint in MEDIAPIPE_MANUS_JOINTS
    )
    keypoints_valid = all(key in semantic_positions for key in semantic_keys)
    if keypoints_valid:
        keypoints = np.stack(
            [semantic_positions[key] for key in semantic_keys]
        ).astype(np.float32)

    ergonomics_value = np.zeros(ergonomics_count, dtype=np.float32)
    ergonomics_type_code = np.full(ergonomics_count, -1, dtype=np.int16)
    ergonomics_valid = np.zeros(ergonomics_count, dtype=np.uint8)
    for index, item in enumerate(list(getattr(message, "ergonomics", []))[:ergonomics_count]):
        value = float(getattr(item, "value", 0.0))
        if not np.isfinite(value):
            continue
        ergonomics_value[index] = value
        ergonomics_type_code[index] = _type_code(
            getattr(item, "type", ""), MANUS_ERGONOMICS_TYPE_CODES
        )
        ergonomics_valid[index] = 1

    raw_sensor_orientation = np.zeros(4, dtype=np.float32)
    orientation = getattr(message, "raw_sensor_orientation", None)
    if orientation is not None:
        candidate = np.asarray(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float32,
        )
        if np.all(np.isfinite(candidate)):
            raw_sensor_orientation = candidate

    raw_sensor_pose = np.zeros((sensor_count, 7), dtype=np.float32)
    raw_sensor_valid = np.zeros(sensor_count, dtype=np.uint8)
    for index, pose in enumerate(list(getattr(message, "raw_sensor", []))[:sensor_count]):
        candidate = _pose_array(pose)
        if not np.all(np.isfinite(candidate)):
            continue
        raw_sensor_pose[index] = candidate
        raw_sensor_valid[index] = 1

    return {
        f"{prefix}_raw_node_pose": node_pose,
        f"{prefix}_node_id": node_id,
        f"{prefix}_parent_node_id": parent_node_id,
        f"{prefix}_chain_type_code": chain_type_code,
        f"{prefix}_joint_type_code": joint_type_code,
        f"{prefix}_node_valid": node_valid,
        f"{prefix}_keypoints_21": keypoints,
        f"{prefix}_keypoints_valid": np.asarray([keypoints_valid], dtype=np.uint8),
        f"{prefix}_ergonomics_value": ergonomics_value,
        f"{prefix}_ergonomics_type_code": ergonomics_type_code,
        f"{prefix}_ergonomics_valid": ergonomics_valid,
        f"{prefix}_raw_sensor_orientation": raw_sensor_orientation,
        f"{prefix}_raw_sensor_pose": raw_sensor_pose,
        f"{prefix}_raw_sensor_valid": raw_sensor_valid,
        f"{prefix}_glove_id": np.asarray(
            [int(getattr(message, "glove_id", -1))], dtype=np.int32
        ),
        f"{prefix}_reported_node_count": np.asarray(
            [int(getattr(message, "raw_node_count", len(raw_nodes)))], dtype=np.int32
        ),
        f"{prefix}_reported_sensor_count": np.asarray(
            [int(getattr(message, "raw_sensor_count", len(getattr(message, "raw_sensor", []))))],
            dtype=np.int32,
        ),
        f"{prefix}_available": np.asarray([1], dtype=np.uint8),
    }
