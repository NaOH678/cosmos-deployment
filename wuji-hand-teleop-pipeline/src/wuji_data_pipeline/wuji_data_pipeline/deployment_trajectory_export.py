"""Tabular exports and common math for deployment trajectory analysis."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def quaternion_error_degrees(
    actual: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Return shortest-path quaternion angle errors for paired Nx4 arrays."""
    if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 4:
        raise ValueError("quaternion arrays must both have shape Nx4")
    result = np.full(actual.shape[0], np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(actual), axis=1) & np.all(
        np.isfinite(target), axis=1
    )
    if not np.any(valid):
        return result
    qa = actual[valid]
    qt = target[valid]
    qa_norm = np.linalg.norm(qa, axis=1)
    qt_norm = np.linalg.norm(qt, axis=1)
    good = (qa_norm > 1e-12) & (qt_norm > 1e-12)
    values = np.full(qa.shape[0], np.nan, dtype=np.float64)
    if np.any(good):
        dots = np.sum(
            qa[good] / qa_norm[good, None]
            * (qt[good] / qt_norm[good, None]),
            axis=1,
        )
        values[good] = np.degrees(
            2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
        )
    result[np.flatnonzero(valid)] = values
    return result


def _snapshots(
    events: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    result = [event for event in events if event.get("event") == "snapshot"]
    if not result:
        raise ValueError("state trace contains no snapshot events")
    return result


def _pose_stream(
    snapshot: Mapping[str, Any], key: str
) -> Optional[tuple[Mapping[str, Any], np.ndarray, np.ndarray]]:
    streams = snapshot.get("streams", {})
    stream = streams.get(key) if isinstance(streams, Mapping) else None
    if not isinstance(stream, Mapping):
        return None
    data = stream.get("data")
    if not isinstance(data, Mapping):
        return None
    position = np.asarray(
        data.get("position_m", []), dtype=np.float64
    ).reshape(-1)
    quaternion = np.asarray(
        data.get("quaternion_xyzw", []), dtype=np.float64
    ).reshape(-1)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
    ):
        return None
    return stream, position, quaternion


def _joint_stream(
    snapshot: Mapping[str, Any],
    key: str,
    *,
    scale: float = 1.0,
) -> Optional[tuple[Mapping[str, Any], np.ndarray]]:
    streams = snapshot.get("streams", {})
    stream = streams.get(key) if isinstance(streams, Mapping) else None
    if not isinstance(stream, Mapping):
        return None
    data = stream.get("data")
    if not isinstance(data, Mapping):
        return None
    position = np.asarray(
        data.get("position", []), dtype=np.float64
    ).reshape(-1)
    if position.shape != (7,) or not np.all(np.isfinite(position)):
        return None
    return stream, position * float(scale)


def _snapshot_metadata(
    snapshot: Mapping[str, Any], side: str, t0_ns: int
) -> dict[str, Any]:
    return {
        "side": side,
        "sample_sequence": snapshot.get("sample_sequence", ""),
        "time_s": (int(snapshot["monotonic_ns"]) - t0_ns) * 1e-9,
        "lifecycle": snapshot.get("lifecycle", ""),
        "ready": snapshot.get("ready", False),
        "complete": snapshot.get("complete", False),
    }


def _stream_metadata(
    row: dict[str, Any], prefix: str, stream: Mapping[str, Any]
) -> None:
    row[f"{prefix}_source_stamp_ns"] = stream.get("source_stamp_ns", "")
    row[f"{prefix}_age_ms"] = stream.get("age_ms", "")


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        quaternion_error_degrees(
            first.reshape(1, 4), second.reshape(1, 4)
        )[0]
    )


def create_eef_comparison_csv(
    output: str | Path,
    state_events: Sequence[Mapping[str, Any]],
    sides: Sequence[str],
) -> Path:
    """Export external, controller, and measured EEF on one time axis."""
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshots = _snapshots(state_events)
    t0_ns = int(snapshots[0]["monotonic_ns"])
    metadata_fields = [
        "side",
        "sample_sequence",
        "time_s",
        "lifecycle",
        "ready",
        "complete",
    ]
    stream_fields = [
        value
        for prefix in ("input", "controller", "actual")
        for value in (
            f"{prefix}_source_stamp_ns",
            f"{prefix}_age_ms",
            *(f"{prefix}_{axis}_m" for axis in ("x", "y", "z")),
            *(f"{prefix}_{axis}" for axis in ("qx", "qy", "qz", "qw")),
        )
    ]
    error_fields = [
        *(f"actual_minus_input_{axis}_mm" for axis in ("x", "y", "z")),
        *(f"controller_minus_input_{axis}_mm" for axis in ("x", "y", "z")),
        *(f"actual_minus_controller_{axis}_mm" for axis in ("x", "y", "z")),
        "actual_input_position_error_mm",
        "actual_input_rotation_error_deg",
        "controller_input_position_error_mm",
        "controller_input_rotation_error_deg",
        "actual_controller_position_error_mm",
        "actual_controller_rotation_error_deg",
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=metadata_fields + stream_fields + error_fields,
        )
        writer.writeheader()
        for snapshot in snapshots:
            if not bool(snapshot.get("ready", False)):
                continue
            for side in sides:
                input_pose = _pose_stream(
                    snapshot, f"{side}.arm_external_target"
                )
                controller_pose = _pose_stream(
                    snapshot, f"{side}.arm_controller_target"
                )
                actual_pose = _pose_stream(
                    snapshot, f"{side}.arm_actual_eef"
                )
                if input_pose is None or actual_pose is None:
                    continue
                input_stream, input_position, input_quaternion = input_pose
                actual_stream, actual_position, actual_quaternion = actual_pose
                row = _snapshot_metadata(snapshot, side, t0_ns)
                for prefix, value in (
                    ("input", input_pose),
                    ("actual", actual_pose),
                ):
                    value_stream, position, quaternion = value
                    _stream_metadata(row, prefix, value_stream)
                    for index, label in enumerate(("x", "y", "z")):
                        row[f"{prefix}_{label}_m"] = float(position[index])
                    for index, label in enumerate(("qx", "qy", "qz", "qw")):
                        row[f"{prefix}_{label}"] = float(quaternion[index])
                actual_input_mm = (
                    actual_position - input_position
                ) * 1000.0
                for index, label in enumerate(("x", "y", "z")):
                    row[f"actual_minus_input_{label}_mm"] = float(
                        actual_input_mm[index]
                    )
                row["actual_input_position_error_mm"] = float(
                    np.linalg.norm(actual_input_mm)
                )
                row["actual_input_rotation_error_deg"] = _rotation_error_deg(
                    actual_quaternion, input_quaternion
                )
                if controller_pose is not None:
                    controller_stream, controller_position, controller_quat = (
                        controller_pose
                    )
                    _stream_metadata(row, "controller", controller_stream)
                    for index, label in enumerate(("x", "y", "z")):
                        row[f"controller_{label}_m"] = float(
                            controller_position[index]
                        )
                    for index, label in enumerate(("qx", "qy", "qz", "qw")):
                        row[f"controller_{label}"] = float(
                            controller_quat[index]
                        )
                    controller_input_mm = (
                        controller_position - input_position
                    ) * 1000.0
                    actual_controller_mm = (
                        actual_position - controller_position
                    ) * 1000.0
                    for index, label in enumerate(("x", "y", "z")):
                        row[f"controller_minus_input_{label}_mm"] = float(
                            controller_input_mm[index]
                        )
                        row[f"actual_minus_controller_{label}_mm"] = float(
                            actual_controller_mm[index]
                        )
                    row["controller_input_position_error_mm"] = float(
                        np.linalg.norm(controller_input_mm)
                    )
                    row["controller_input_rotation_error_deg"] = (
                        _rotation_error_deg(controller_quat, input_quaternion)
                    )
                    row["actual_controller_position_error_mm"] = float(
                        np.linalg.norm(actual_controller_mm)
                    )
                    row["actual_controller_rotation_error_deg"] = (
                        _rotation_error_deg(actual_quaternion, controller_quat)
                    )
                writer.writerow(row)
    return destination


def create_joint_comparison_csv(
    output: str | Path,
    state_events: Sequence[Mapping[str, Any]],
    sides: Sequence[str],
) -> Path:
    """Export source, controller-command, and measured arm joints in degrees."""
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshots = _snapshots(state_events)
    t0_ns = int(snapshots[0]["monotonic_ns"])
    metadata_fields = [
        "side",
        "sample_sequence",
        "time_s",
        "lifecycle",
        "ready",
        "complete",
    ]
    stream_fields = [
        value
        for prefix in ("external", "command", "actual")
        for value in (
            f"{prefix}_source_stamp_ns",
            f"{prefix}_age_ms",
            *(f"{prefix}_j{joint}_deg" for joint in range(1, 8)),
        )
    ]
    error_fields = [
        value
        for prefix in ("actual_minus_external", "actual_minus_command")
        for value in (
            *(f"{prefix}_j{joint}_deg" for joint in range(1, 8)),
            f"{prefix}_rms_deg",
            f"{prefix}_max_deg",
        )
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=metadata_fields + stream_fields + error_fields,
        )
        writer.writeheader()
        for snapshot in snapshots:
            if not bool(snapshot.get("ready", False)):
                continue
            for side in sides:
                actual = _joint_stream(
                    snapshot, f"{side}.arm_joint_state"
                )
                command = _joint_stream(
                    snapshot, f"{side}.arm_joint_command"
                )
                source = _joint_stream(
                    snapshot,
                    f"{side}.arm_external_joint_target",
                    scale=180.0 / math.pi,
                )
                if actual is None or command is None:
                    continue
                row = _snapshot_metadata(snapshot, side, t0_ns)
                for prefix, value in (
                    ("command", command),
                    ("actual", actual),
                ):
                    value_stream, joints = value
                    _stream_metadata(row, prefix, value_stream)
                    for joint, position in enumerate(joints, 1):
                        row[f"{prefix}_j{joint}_deg"] = float(position)
                actual_joints = actual[1]
                command_error = actual_joints - command[1]
                for joint, error in enumerate(command_error, 1):
                    row[f"actual_minus_command_j{joint}_deg"] = float(error)
                row["actual_minus_command_rms_deg"] = float(
                    np.sqrt(np.mean(np.square(command_error)))
                )
                row["actual_minus_command_max_deg"] = float(
                    np.max(np.abs(command_error))
                )
                if source is not None:
                    source_stream, source_joints = source
                    _stream_metadata(row, "external", source_stream)
                    for joint, position in enumerate(source_joints, 1):
                        row[f"external_j{joint}_deg"] = float(position)
                    source_error = actual_joints - source_joints
                    for joint, error in enumerate(source_error, 1):
                        row[f"actual_minus_external_j{joint}_deg"] = float(error)
                    row["actual_minus_external_rms_deg"] = float(
                        np.sqrt(np.mean(np.square(source_error)))
                    )
                    row["actual_minus_external_max_deg"] = float(
                        np.max(np.abs(source_error))
                    )
                writer.writerow(row)
    return destination
