"""Offline plots for deployment state/action and policy timing traces."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .deployment_trajectory_export import (
    create_eef_comparison_csv,
    create_joint_comparison_csv,
    quaternion_error_degrees,
)


SIDES = ("left", "right")
ARM_COMMAND_MODES = ("eef", "joint")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    events: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {source}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"trace event at {source}:{line_number} is not a mapping"
                )
            events.append(value)
    if not events:
        raise ValueError(f"trace contains no events: {source}")
    return events


def _event_wall_seconds(event: Mapping[str, Any]) -> Optional[float]:
    if "wall_time_ns" in event:
        return float(event["wall_time_ns"]) * 1e-9
    if "wall_time" in event:
        return float(event["wall_time"])
    return None


def _trace_arm_command_mode(
    state_events: Sequence[Mapping[str, Any]],
) -> str:
    for event in state_events:
        if event.get("event") not in ("trace_start", "trace_summary"):
            continue
        mode = str(event.get("arm_command_mode", "eef")).strip().lower()
        if mode in ARM_COMMAND_MODES:
            return mode
    # Schema-v1 traces predate joint Replay and therefore always carried EEF
    # external targets.
    return "eef"


def _trace_active_sides(
    state_events: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    for event in state_events:
        if event.get("event") != "trace_start":
            continue
        values = event.get("active_sides")
        if not isinstance(values, Sequence) or isinstance(values, str):
            break
        sides = tuple(str(value).strip().lower() for value in values)
        if sides and all(side in SIDES for side in sides):
            return tuple(side for side in SIDES if side in sides)
        break
    return SIDES


def _deployment_reference(
    state_events: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Return the deployment session id and trace basename from status."""
    for event in state_events:
        if event.get("event") != "deployment_status":
            continue
        status = event.get("status")
        if not isinstance(status, Mapping):
            continue
        session_id = str(status.get("session_id", "")).strip()
        diagnostics = status.get("diagnostic_trace")
        path = (
            str(diagnostics.get("path", "")).strip()
            if isinstance(diagnostics, Mapping)
            else ""
        )
        if session_id or path:
            return session_id, Path(path).name if path else ""
    return "", ""


def _deployment_trace_session_id(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, Mapping):
                    return ""
                return str(event.get("session_id", "")).strip()
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return ""
    return ""


def find_nearest_deployment_trace(
    state_trace: str | Path,
    state_events: Sequence[Mapping[str, Any]],
) -> Optional[Path]:
    directory = Path(state_trace).expanduser().resolve().parent
    session_id, trace_basename = _deployment_reference(state_events)
    if trace_basename:
        exact_path = directory / trace_basename
        if exact_path.is_file() and (
            not session_id
            or _deployment_trace_session_id(exact_path) == session_id
        ):
            return exact_path
    if session_id:
        for candidate in directory.glob(
            f"deployment_trace_*_{session_id[:8]}.jsonl"
        ):
            if _deployment_trace_session_id(candidate) == session_id:
                return candidate

    target = next(
        (
            value
            for event in state_events
            if (value := _event_wall_seconds(event)) is not None
        ),
        None,
    )
    if target is None:
        return None
    best: Optional[tuple[float, Path]] = None
    for candidate in directory.glob("deployment_trace_*.jsonl"):
        try:
            with candidate.open("r", encoding="utf-8") as stream:
                first = json.loads(stream.readline())
            wall = _event_wall_seconds(first)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if wall is None:
            continue
        difference = abs(wall - target)
        if best is None or difference < best[0]:
            best = (difference, candidate)
    # Session graph processes normally start within a few seconds.  Refuse an
    # accidental match to an older run when no deployment trace is supplied.
    if best is None or best[0] > 30.0:
        return None
    return best[1]


def _snapshots(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = [event for event in events if event.get("event") == "snapshot"]
    if not result:
        raise ValueError("state trace contains no snapshot events")
    return result


def _analysis_snapshot_selection(
    snapshots: Sequence[Mapping[str, Any]],
    sample_rate_hz: float,
    *,
    stabilization_window_s: float = 0.5,
) -> tuple[list[Mapping[str, Any]], dict[str, int | float]]:
    """Select complete steady-state windows after every READY transition."""
    ready_groups: list[list[Mapping[str, Any]]] = []
    current_group: list[Mapping[str, Any]] = []
    for snapshot in snapshots:
        if bool(snapshot.get("ready")):
            current_group.append(snapshot)
        elif current_group:
            ready_groups.append(current_group)
            current_group = []
    if current_group:
        ready_groups.append(current_group)

    stabilization_samples = max(
        1, round(float(sample_rate_hz) * stabilization_window_s)
    )
    selected: list[Mapping[str, Any]] = []
    warmup_samples = 0
    warmup_incomplete = 0
    analysis_incomplete = 0
    for group in ready_groups:
        flags = [bool(snapshot.get("complete")) for snapshot in group]
        required_window = min(stabilization_samples, len(flags))
        stable_start = next(
            (
                index
                for index in range(0, len(flags) - required_window + 1)
                if all(flags[index : index + required_window])
            ),
            None,
        )
        if stable_start is None:
            warmup_samples += len(flags)
            warmup_incomplete += flags.count(False)
            analysis_incomplete += 1
            continue
        warmup_samples += stable_start
        warmup_incomplete += flags[:stable_start].count(False)
        analysis_group = group[stable_start:]
        selected.extend(analysis_group)
        analysis_incomplete += sum(
            not bool(snapshot.get("complete"))
            for snapshot in analysis_group
        )
    return selected, {
        "ready_segment_count": len(ready_groups),
        "stabilization_window_s": float(stabilization_window_s),
        "warmup_sample_count": warmup_samples,
        "warmup_incomplete_sample_count": warmup_incomplete,
        "analysis_sample_count": len(selected),
        "analysis_incomplete_sample_count": analysis_incomplete,
    }


def _time_axis(
    snapshots: Sequence[Mapping[str, Any]], t0_ns: int
) -> np.ndarray:
    return np.asarray(
        [(int(event["monotonic_ns"]) - t0_ns) * 1e-9 for event in snapshots],
        dtype=np.float64,
    )


def _stream_matrix(
    snapshots: Sequence[Mapping[str, Any]],
    key: str,
    field: str,
    width: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    result = np.full((len(snapshots), width), np.nan, dtype=np.float64)
    for index, snapshot in enumerate(snapshots):
        stream = snapshot.get("streams", {}).get(key)
        if not isinstance(stream, Mapping):
            continue
        data = stream.get("data")
        if not isinstance(data, Mapping):
            continue
        values = np.asarray(data.get(field, []), dtype=np.float64).reshape(-1)
        if values.shape == (width,) and np.all(np.isfinite(values)):
            result[index] = values * float(scale)
    return result


def _stream_scalar(
    snapshots: Sequence[Mapping[str, Any]], key: str, field: str
) -> np.ndarray:
    result = np.full(len(snapshots), np.nan, dtype=np.float64)
    for index, snapshot in enumerate(snapshots):
        stream = snapshot.get("streams", {}).get(key)
        if isinstance(stream, Mapping):
            value = stream.get(field)
            if value is not None and math.isfinite(float(value)):
                result[index] = float(value)
    return result


def _finite_metric(values: np.ndarray, operation: str) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if operation == "mean":
        return float(np.mean(finite))
    if operation == "rms":
        return float(np.sqrt(np.mean(np.square(finite))))
    if operation == "max":
        return float(np.max(finite))
    if operation == "p99":
        return float(np.percentile(finite, 99))
    raise ValueError(f"unsupported metric {operation}")


def _tracking_metrics(
    snapshots: Sequence[Mapping[str, Any]], side: str
) -> dict[str, Optional[float]]:
    actual_pos = _stream_matrix(
        snapshots, f"{side}.arm_actual_eef", "position_m", 3
    )
    action_pos = _stream_matrix(
        snapshots, f"{side}.arm_external_target", "position_m", 3
    )
    position_error = np.linalg.norm(actual_pos - action_pos, axis=1)
    actual_quat = _stream_matrix(
        snapshots, f"{side}.arm_actual_eef", "quaternion_xyzw", 4
    )
    action_quat = _stream_matrix(
        snapshots, f"{side}.arm_external_target", "quaternion_xyzw", 4
    )
    rotation_error = quaternion_error_degrees(actual_quat, action_quat)
    controller_pos = _stream_matrix(
        snapshots, f"{side}.arm_controller_target", "position_m", 3
    )
    controller_quat = _stream_matrix(
        snapshots, f"{side}.arm_controller_target", "quaternion_xyzw", 4
    )
    controller_input_position_error = np.linalg.norm(
        controller_pos - action_pos, axis=1
    )
    actual_controller_position_error = np.linalg.norm(
        actual_pos - controller_pos, axis=1
    )
    controller_input_rotation_error = quaternion_error_degrees(
        controller_quat, action_quat
    )
    actual_controller_rotation_error = quaternion_error_degrees(
        actual_quat, controller_quat
    )
    actual_joints = _stream_matrix(
        snapshots, f"{side}.arm_joint_state", "position", 7
    )
    command_joints = _stream_matrix(
        snapshots, f"{side}.arm_joint_command", "position", 7
    )
    joint_error = np.abs(actual_joints - command_joints)
    source_joints = _stream_matrix(
        snapshots,
        f"{side}.arm_external_joint_target",
        "position",
        7,
        scale=180.0 / math.pi,
    )
    source_joint_error = np.abs(actual_joints - source_joints)
    actual_hand = _stream_matrix(
        snapshots,
        f"{side}.hand_state",
        "position",
        20,
        scale=180.0 / math.pi,
    )
    command_hand = _stream_matrix(
        snapshots,
        f"{side}.hand_command",
        "position",
        20,
        scale=180.0 / math.pi,
    )
    hand_error = np.abs(actual_hand - command_hand)
    return {
        "eef_position_error_rms_m": _finite_metric(position_error, "rms"),
        "eef_position_error_max_m": _finite_metric(position_error, "max"),
        "eef_rotation_error_rms_deg": _finite_metric(rotation_error, "rms"),
        "eef_rotation_error_max_deg": _finite_metric(rotation_error, "max"),
        "controller_input_position_error_rms_m": _finite_metric(
            controller_input_position_error, "rms"
        ),
        "controller_input_position_error_max_m": _finite_metric(
            controller_input_position_error, "max"
        ),
        "actual_controller_position_error_rms_m": _finite_metric(
            actual_controller_position_error, "rms"
        ),
        "actual_controller_position_error_max_m": _finite_metric(
            actual_controller_position_error, "max"
        ),
        "controller_input_rotation_error_rms_deg": _finite_metric(
            controller_input_rotation_error, "rms"
        ),
        "actual_controller_rotation_error_rms_deg": _finite_metric(
            actual_controller_rotation_error, "rms"
        ),
        "arm_joint_error_rms_deg": _finite_metric(joint_error, "rms"),
        "arm_joint_error_max_deg": _finite_metric(joint_error, "max"),
        "source_joint_error_rms_deg": _finite_metric(
            source_joint_error, "rms"
        ),
        "source_joint_error_max_deg": _finite_metric(
            source_joint_error, "max"
        ),
        "hand_error_rms_deg": _finite_metric(hand_error, "rms"),
        "hand_error_max_deg": _finite_metric(hand_error, "max"),
    }


def build_summary(
    state_events: Sequence[Mapping[str, Any]],
    deployment_events: Sequence[Mapping[str, Any]],
    sides: Sequence[str],
) -> dict[str, Any]:
    snapshots = _snapshots(state_events)
    arm_command_mode = _trace_arm_command_mode(state_events)
    final = next(
        (
            event
            for event in reversed(state_events)
            if event.get("event") == "trace_summary"
        ),
        None,
    )
    time_ns = np.asarray(
        [int(event["monotonic_ns"]) for event in snapshots], dtype=np.int64
    )
    intervals_ms = np.diff(time_ns).astype(np.float64) * 1e-6
    writer = dict(final.get("writer", {})) if final else {}
    dropped = int(writer.get("dropped_events", -1))
    incomplete = int(final.get("incomplete_ready_sample_count", -1)) if final else -1
    stream_stats = dict(final.get("stream_stats", {})) if final else {}
    required_stream_keys = {
        "left.arm_joint_state",
        "left.arm_actual_eef",
        "right.arm_joint_state",
        "right.arm_actual_eef",
    }
    for side in sides:
        required_stream_keys.update({
            f"{side}.arm_joint_command",
            f"{side}.hand_state",
            f"{side}.hand_command",
        })
        if arm_command_mode == "eef":
            required_stream_keys.update({
                f"{side}.arm_external_target",
                f"{side}.arm_controller_target",
            })
        else:
            required_stream_keys.add(
                f"{side}.arm_external_joint_target"
            )
    invalid_messages = sum(
        int(value.get("invalid", 0))
        for key, value in stream_stats.items()
        if key in required_stream_keys and isinstance(value, Mapping)
    )
    nonmonotonic_stamps = sum(
        int(value.get("nonmonotonic_source_stamps", 0))
        for key, value in stream_stats.items()
        if key in required_stream_keys and isinstance(value, Mapping)
    )
    sequences = np.asarray(
        [int(event.get("sample_sequence", -1)) for event in snapshots],
        dtype=np.int64,
    )
    sequence_errors = int(sequences[0] != 1) + int(
        np.count_nonzero(np.diff(sequences) != 1)
    )
    summary_count_matches = bool(
        final is not None and int(final.get("sample_count", -1)) == len(snapshots)
    )
    sample_rate_hz = float(final.get("sample_rate_hz", 120.0)) if final else 120.0
    analysis_snapshots, selection = _analysis_snapshot_selection(
        snapshots, sample_rate_hz
    )
    ready_segments = int(selection["ready_segment_count"])
    analysis_samples = int(selection["analysis_sample_count"])
    analysis_incomplete = int(
        selection["analysis_incomplete_sample_count"]
    )
    full_ready_coverage = bool(incomplete == 0)
    ready_missing = Counter(
        str(key)
        for snapshot in snapshots
        if bool(snapshot.get("ready"))
        for key in snapshot.get("missing_streams", [])
    )
    ready_stale = Counter(
        str(key)
        for snapshot in snapshots
        if bool(snapshot.get("ready"))
        for key in snapshot.get("stale_streams", [])
    )
    completeness_checks = {
        "trace_summary_missing": final is None,
        "writer_dropped_events": dropped != 0,
        "invalid_stream_messages": invalid_messages != 0,
        "nonmonotonic_source_stamps": nonmonotonic_stamps != 0,
        "snapshot_sequence_errors": sequence_errors != 0,
        "summary_count_mismatch": not summary_count_matches,
        "no_ready_segment": ready_segments == 0,
        "no_steady_state_samples": analysis_samples == 0,
        "incomplete_steady_state_samples": analysis_incomplete != 0,
    }
    completeness_failures = [
        name for name, failed in completeness_checks.items() if failed
    ]
    summary: dict[str, Any] = {
        "complete": not completeness_failures,
        "completeness_failures": completeness_failures,
        "completeness_scope": "steady_state_after_required_stream_warmup",
        "arm_command_mode": arm_command_mode,
        "full_ready_coverage": full_ready_coverage,
        "trace_summary_present": final is not None,
        "sample_count": len(snapshots),
        "summary_count_matches": summary_count_matches,
        "sample_sequence_errors": sequence_errors,
        "ready_sample_count": int(final.get("ready_sample_count", -1)) if final else -1,
        "incomplete_ready_sample_count": incomplete,
        "ready_segment_count": ready_segments,
        "stabilization_window_s": selection["stabilization_window_s"],
        "warmup_sample_count": selection["warmup_sample_count"],
        "warmup_incomplete_sample_count": selection[
            "warmup_incomplete_sample_count"
        ],
        "analysis_sample_count": analysis_samples,
        "analysis_incomplete_sample_count": analysis_incomplete,
        "invalid_stream_messages": invalid_messages,
        "nonmonotonic_source_stamps": nonmonotonic_stamps,
        "ready_missing_stream_counts": dict(ready_missing.most_common()),
        "ready_stale_stream_counts": dict(ready_stale.most_common()),
        "writer": writer,
        "sample_interval_ms": {
            "mean": _finite_metric(intervals_ms, "mean"),
            "p99": _finite_metric(intervals_ms, "p99"),
            "max": _finite_metric(intervals_ms, "max"),
        },
        "stream_stats": stream_stats,
        "sides": {
            side: _tracking_metrics(analysis_snapshots, side)
            for side in sides
        },
    }
    rtt = np.asarray(
        [
            float(event["complete_rtt_ms"])
            for event in deployment_events
            if event.get("event") == "policy_response_ready"
            and event.get("complete_rtt_ms") is not None
        ],
        dtype=np.float64,
    )
    summary["policy_complete_rtt_ms"] = {
        "count": int(rtt.size),
        "mean": _finite_metric(rtt, "mean"),
        "p99": _finite_metric(rtt, "p99"),
        "max": _finite_metric(rtt, "max"),
    }
    return summary


def _chunk_boundaries(
    deployment_events: Sequence[Mapping[str, Any]], t0_s: float
) -> list[tuple[float, int]]:
    result: list[tuple[float, int]] = []
    seen: set[int] = set()
    for event in deployment_events:
        if event.get("event") != "command_publish":
            continue
        chunk_id = int(event.get("chunk_id", 0))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        event_time = float(event.get("monotonic_time", event.get("published_at", 0.0)))
        result.append((event_time - t0_s, chunk_id))
    return result


def _draw_boundaries(axes: Iterable[Any], boundaries: Sequence[tuple[float, int]]) -> None:
    for axis in axes:
        for time_s, _chunk_id in boundaries:
            axis.axvline(time_s, color="0.75", linewidth=0.6, alpha=0.7)


def create_pdf_report(
    output: str | Path,
    state_events: Sequence[Mapping[str, Any]],
    deployment_events: Sequence[Mapping[str, Any]],
    sides: Sequence[str],
    summary: Mapping[str, Any],
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for offline plotting; install "
            "python3-matplotlib in the analysis environment"
        ) from exc

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshots = _snapshots(state_events)
    t0_ns = int(snapshots[0]["monotonic_ns"])
    t0_s = t0_ns * 1e-9
    time_s = _time_axis(snapshots, t0_ns)
    boundaries = _chunk_boundaries(deployment_events, t0_s)

    with PdfPages(destination) as pdf:
        fig, axis = plt.subplots(figsize=(11.7, 8.3))
        axis.axis("off")
        axis.set_title("Tianji + Wuji deployment trace summary", fontsize=16)
        lines = [
            f"Completeness: {'PASS' if summary['complete'] else 'FAIL'}",
            f"Arm command mode: {summary.get('arm_command_mode', 'eef')}",
            "Completeness failures: "
            + (", ".join(summary.get("completeness_failures", [])) or "none"),
            f"Snapshots: {summary['sample_count']}",
            f"Ready snapshots: {summary['ready_sample_count']}",
            "Incomplete READY snapshots: "
            f"{summary['incomplete_ready_sample_count']}",
            "Warmup incomplete snapshots: "
            f"{summary['warmup_incomplete_sample_count']}",
            "Analysis-window incomplete snapshots: "
            f"{summary['analysis_incomplete_sample_count']}",
            "Writer dropped events: "
            f"{summary.get('writer', {}).get('dropped_events', 'unknown')}",
            f"Invalid source messages: {summary['invalid_stream_messages']}",
            "Nonmonotonic source stamps: "
            f"{summary['nonmonotonic_source_stamps']}",
            "Top READY missing streams: "
            + str(
                list(summary.get("ready_missing_stream_counts", {}).items())[:3]
            ),
            "Top READY stale streams: "
            + str(
                list(summary.get("ready_stale_stream_counts", {}).items())[:3]
            ),
            f"Snapshot sequence errors: {summary['sample_sequence_errors']}",
            "Sample interval p99: "
            f"{summary['sample_interval_ms']['p99']} ms",
            "Policy RTT p99: "
            f"{summary['policy_complete_rtt_ms']['p99']} ms",
        ]
        for side in sides:
            metrics = summary["sides"][side]
            lines.append("")
            if summary.get("arm_command_mode", "eef") == "eef":
                lines.extend([
                    f"[{side}] EEF position RMS/max: "
                    f"{metrics['eef_position_error_rms_m']} / "
                    f"{metrics['eef_position_error_max_m']} m",
                    f"[{side}] EEF rotation RMS/max: "
                    f"{metrics['eef_rotation_error_rms_deg']} / "
                    f"{metrics['eef_rotation_error_max_deg']} deg",
                ])
            else:
                lines.append(
                    f"[{side}] source joint RMS/max: "
                    f"{metrics['source_joint_error_rms_deg']} / "
                    f"{metrics['source_joint_error_max_deg']} deg"
                )
            lines.extend([
                f"[{side}] controller joint RMS/max: "
                f"{metrics['arm_joint_error_rms_deg']} / "
                f"{metrics['arm_joint_error_max_deg']} deg",
                f"[{side}] hand RMS/max: "
                f"{metrics['hand_error_rms_deg']} / "
                f"{metrics['hand_error_max_deg']} deg",
            ])
        axis.text(0.04, 0.92, "\n".join(lines), va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for side in sides:
            actual_pos = _stream_matrix(
                snapshots, f"{side}.arm_actual_eef", "position_m", 3
            )
            external_pos = _stream_matrix(
                snapshots, f"{side}.arm_external_target", "position_m", 3
            )
            controller_pos = _stream_matrix(
                snapshots, f"{side}.arm_controller_target", "position_m", 3
            )
            position_error = np.linalg.norm(actual_pos - external_pos, axis=1)
            if np.any(np.isfinite(external_pos)):
                fig, axes = plt.subplots(
                    4, 1, figsize=(11.7, 8.3), sharex=True
                )
                for component, label in enumerate(("X", "Y", "Z")):
                    axes[component].plot(
                        time_s, actual_pos[:, component], label="actual", lw=1.0
                    )
                    axes[component].plot(
                        time_s,
                        external_pos[:, component],
                        label="external action",
                        lw=0.9,
                    )
                    axes[component].plot(
                        time_s,
                        controller_pos[:, component],
                        label="controller target",
                        lw=0.8,
                        ls="--",
                    )
                    axes[component].set_ylabel(f"{label} (m)")
                    axes[component].grid(True, alpha=0.25)
                axes[0].legend(loc="upper right", ncol=3, fontsize=8)
                axes[3].plot(
                    time_s, position_error * 1000.0, color="tab:red", lw=0.9
                )
                axes[3].set_ylabel("actual-action (mm)")
                axes[3].set_xlabel("Monotonic time from trace start (s)")
                axes[3].grid(True, alpha=0.25)
                _draw_boundaries(axes, boundaries)
                fig.suptitle(f"{side}: EEF state and target pipeline")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            actual_quat = _stream_matrix(
                snapshots, f"{side}.arm_actual_eef", "quaternion_xyzw", 4
            )
            external_quat = _stream_matrix(
                snapshots, f"{side}.arm_external_target", "quaternion_xyzw", 4
            )
            if np.any(np.isfinite(external_quat)):
                rotation_error = quaternion_error_degrees(
                    actual_quat, external_quat
                )
                fig, axes = plt.subplots(
                    5, 1, figsize=(11.7, 8.3), sharex=True
                )
                for component, label in enumerate(("qx", "qy", "qz", "qw")):
                    axes[component].plot(
                        time_s,
                        actual_quat[:, component],
                        label="actual",
                        lw=0.9,
                    )
                    axes[component].plot(
                        time_s,
                        external_quat[:, component],
                        label="external action",
                        lw=0.8,
                    )
                    axes[component].set_ylabel(label)
                    axes[component].grid(True, alpha=0.25)
                axes[0].legend(loc="upper right", ncol=2, fontsize=8)
                axes[4].plot(
                    time_s, rotation_error, color="tab:red", lw=0.9
                )
                axes[4].set_ylabel("error (deg)")
                axes[4].set_xlabel("Time (s)")
                axes[4].grid(True, alpha=0.25)
                _draw_boundaries(axes, boundaries)
                fig.suptitle(f"{side}: EEF orientation")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            actual_joint = _stream_matrix(
                snapshots, f"{side}.arm_joint_state", "position", 7
            )
            command_joint = _stream_matrix(
                snapshots, f"{side}.arm_joint_command", "position", 7
            )
            source_joint = _stream_matrix(
                snapshots,
                f"{side}.arm_external_joint_target",
                "position",
                7,
                scale=180.0 / math.pi,
            )
            fig, axes = plt.subplots(7, 1, figsize=(11.7, 11.0), sharex=True)
            for joint in range(7):
                axes[joint].plot(time_s, actual_joint[:, joint], label="actual", lw=0.9)
                axes[joint].plot(
                    time_s,
                    command_joint[:, joint],
                    label="controller command",
                    lw=0.8,
                )
                if np.any(np.isfinite(source_joint[:, joint])):
                    axes[joint].plot(
                        time_s,
                        source_joint[:, joint],
                        label="external joint action",
                        lw=0.7,
                        ls="--",
                    )
                axes[joint].set_ylabel(f"J{joint + 1}°")
                axes[joint].grid(True, alpha=0.25)
            axes[0].legend(loc="upper right", ncol=3, fontsize=8)
            axes[-1].set_xlabel("Time (s)")
            _draw_boundaries(axes, boundaries)
            fig.suptitle(f"{side}: arm joint action, command and state")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            actual_hand = _stream_matrix(
                snapshots,
                f"{side}.hand_state",
                "position",
                20,
                scale=180.0 / math.pi,
            )
            command_hand = _stream_matrix(
                snapshots,
                f"{side}.hand_command",
                "position",
                20,
                scale=180.0 / math.pi,
            )
            if np.any(np.isfinite(actual_hand)) or np.any(np.isfinite(command_hand)):
                fig, axes = plt.subplots(5, 4, figsize=(11.7, 11.0), sharex=True)
                for joint, axis in enumerate(axes.flat):
                    axis.plot(time_s, actual_hand[:, joint], label="actual", lw=0.8)
                    axis.plot(time_s, command_hand[:, joint], label="command", lw=0.7)
                    axis.set_title(f"H{joint + 1}", fontsize=8)
                    axis.grid(True, alpha=0.2)
                axes.flat[0].legend(loc="upper right", fontsize=6)
                for axis in axes[-1]:
                    axis.set_xlabel("Time (s)")
                _draw_boundaries(axes.flat, boundaries)
                fig.suptitle(f"{side}: 20-DoF hand state vs command (degree)")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        intervals_ms = np.diff(time_s, prepend=np.nan) * 1000.0
        fig, axes = plt.subplots(4, 1, figsize=(11.7, 8.3), sharex=False)
        axes[0].plot(time_s, intervals_ms, lw=0.8)
        axes[0].axhline(1000.0 / 120.0, color="tab:red", ls="--", lw=0.8)
        axes[0].set_ylabel("snapshot dt (ms)")
        axes[0].grid(True, alpha=0.25)
        for side in sides:
            action_age_key = (
                f"{side}.arm_external_joint_target"
                if summary.get("arm_command_mode") == "joint"
                else f"{side}.arm_external_target"
            )
            axes[1].plot(
                time_s,
                _stream_scalar(snapshots, f"{side}.arm_actual_eef", "age_ms"),
                label=f"{side} actual EEF",
                lw=0.8,
            )
            axes[1].plot(
                time_s,
                _stream_scalar(snapshots, action_age_key, "age_ms"),
                label=f"{side} action",
                lw=0.8,
            )
        axes[1].set_ylabel("source age (ms)")
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].grid(True, alpha=0.25)

        response_events = [
            event
            for event in deployment_events
            if event.get("event") == "policy_response_ready"
        ]
        response_time = np.asarray(
            [float(event.get("monotonic_time", 0.0)) - t0_s for event in response_events]
        )
        response_rtt = np.asarray(
            [float(event.get("complete_rtt_ms", np.nan)) for event in response_events]
        )
        axes[2].scatter(response_time, response_rtt, s=10, label="complete RTT")
        axes[2].scatter(
            response_time,
            [float(event.get("server_inference_ms", np.nan)) for event in response_events],
            s=10,
            label="server inference",
        )
        axes[2].set_ylabel("policy time (ms)")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.25)

        completeness = np.asarray(
            [1.0 if event.get("complete") else 0.0 for event in snapshots]
        )
        axes[3].step(time_s, completeness, where="post")
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].set_ylabel("complete")
        axes[3].set_xlabel("Time (s)")
        axes[3].grid(True, alpha=0.25)
        _draw_boundaries(axes[:2], boundaries)
        fig.suptitle("Timing, freshness and trace completeness")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return destination


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot Tianji/Wuji deployment state and executed actions"
    )
    parser.add_argument("--state-trace", required=True)
    parser.add_argument("--deployment-trace", default=None)
    parser.add_argument("--output", default=None, help="output PDF path")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--eef-csv-output", default=None)
    parser.add_argument("--joint-csv-output", default=None)
    parser.add_argument(
        "--side",
        choices=("auto", "left", "right", "both"),
        default="auto",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    state_path = Path(args.state_trace).expanduser().resolve()
    state_events = load_jsonl(state_path)
    deployment_path = (
        Path(args.deployment_trace).expanduser().resolve()
        if args.deployment_trace
        else find_nearest_deployment_trace(state_path, state_events)
    )
    deployment_events = load_jsonl(deployment_path) if deployment_path else []
    if args.side == "auto":
        sides = _trace_active_sides(state_events)
    else:
        sides = SIDES if args.side == "both" else (args.side,)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else state_path.with_name(f"{state_path.stem}_report.pdf")
    )
    summary = build_summary(state_events, deployment_events, sides)
    report = create_pdf_report(
        output, state_events, deployment_events, sides, summary
    )
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else report.with_suffix(".summary.json")
    )
    eef_csv = None
    if summary["arm_command_mode"] == "eef":
        eef_csv = create_eef_comparison_csv(
            (
                Path(args.eef_csv_output).expanduser().resolve()
                if args.eef_csv_output
                else report.with_suffix(".eef.csv")
            ),
            state_events,
            sides,
        )
    joint_csv = create_joint_comparison_csv(
        (
            Path(args.joint_csv_output).expanduser().resolve()
            if args.joint_csv_output
            else report.with_suffix(".joints.csv")
        ),
        state_events,
        sides,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"State trace:      {state_path}")
    print(f"Deployment trace: {deployment_path or '<not found>'}")
    print(f"Command mode:     {summary['arm_command_mode']}")
    print(f"Analyzed sides:   {','.join(sides)}")
    print(f"Report PDF:       {report}")
    print(f"EEF comparison:   {eef_csv or '<not applicable>'}")
    print(f"Joint comparison: {joint_csv}")
    print(f"Summary JSON:     {summary_path}")
    print(f"Completeness:     {'PASS' if summary['complete'] else 'FAIL'}")
    if summary["completeness_failures"]:
        print(
            "Failure reasons:  "
            + ", ".join(summary["completeness_failures"])
        )
        if summary["ready_missing_stream_counts"]:
            print(
                "READY missing:    "
                + ", ".join(
                    f"{key}={count}"
                    for key, count in list(
                        summary["ready_missing_stream_counts"].items()
                    )[:5]
                )
            )
        if summary["ready_stale_stream_counts"]:
            print(
                "READY stale:      "
                + ", ".join(
                    f"{key}={count}"
                    for key, count in list(
                        summary["ready_stale_stream_counts"].items()
                    )[:5]
                )
            )


if __name__ == "__main__":
    main()
