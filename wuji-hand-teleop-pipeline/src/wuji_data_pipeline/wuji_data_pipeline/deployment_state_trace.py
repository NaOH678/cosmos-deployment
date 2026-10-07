"""Read-only state/action sidecar for offline deployment analysis.

This node intentionally runs in a separate process from the Tianji controller
and deployment client.  DDS callbacks only validate and cache the latest
message.  A fixed-rate snapshot timer copies that cache into a bounded queue;
all JSON encoding and disk I/O happen on a dedicated writer thread.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Mapping, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int8, String

from .config import load_config, section
from .schema import SIDES


COMMAND_LIFECYCLES = (2, 10)  # READY, TARGET_HOLD
TRACE_SCHEMA_VERSION = 2
ARM_COMMAND_MODES = ("eef", "joint")


def _active_sides(value: str) -> tuple[str, ...]:
    normalized = str(value).strip().lower()
    if normalized == "both":
        return SIDES
    if normalized in SIDES:
        return (normalized,)
    raise ValueError("active side must be left, right, or both")


def _stamp_ns(message: Any) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _finite_vector(values: Any, expected: int, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != expected:
        raise ValueError(f"{name} must contain {expected} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _stale_limits_ms(deployment: Mapping[str, Any]) -> dict[str, float]:
    action_max_age_ms = float(
        deployment.get("state_trace_action_max_age_ms", 50.0)
    )
    return {
        "arm_joint_state": float(
            deployment.get("state_trace_arm_state_max_age_ms", 25.0)
        ),
        "arm_actual_eef": float(
            deployment.get("state_trace_eef_max_age_ms", 100.0)
        ),
        "arm_external_target": action_max_age_ms,
        "arm_external_joint_target": action_max_age_ms,
        "arm_controller_target": action_max_age_ms,
        "arm_joint_command": action_max_age_ms,
        "hand_state": float(
            deployment.get("state_trace_hand_max_age_ms", 100.0)
        ),
        "hand_command": float(
            deployment.get("state_trace_hand_max_age_ms", 100.0)
        ),
    }


@dataclass
class StreamStats:
    received: int = 0
    invalid: int = 0
    nonmonotonic_source_stamps: int = 0
    first_receive_monotonic_ns: int = 0
    last_receive_monotonic_ns: int = 0
    first_source_stamp_ns: int = 0
    last_source_stamp_ns: int = 0
    max_source_gap_ms: float = 0.0

    def observe(self, receive_ns: int, source_stamp_ns: int) -> int:
        self.received += 1
        if self.first_receive_monotonic_ns == 0:
            self.first_receive_monotonic_ns = int(receive_ns)
        self.last_receive_monotonic_ns = int(receive_ns)
        if source_stamp_ns > 0:
            if self.first_source_stamp_ns == 0:
                self.first_source_stamp_ns = int(source_stamp_ns)
            if self.last_source_stamp_ns > 0:
                gap_ns = int(source_stamp_ns) - self.last_source_stamp_ns
                if gap_ns <= 0:
                    self.nonmonotonic_source_stamps += 1
                else:
                    self.max_source_gap_ms = max(
                        self.max_source_gap_ms, gap_ns * 1e-6
                    )
            self.last_source_stamp_ns = int(source_stamp_ns)
        return self.received


class StateTraceWriter:
    """Batch JSONL writer isolated from ROS callbacks."""

    def __init__(
        self,
        directory: str | Path,
        *,
        queue_size: int,
        flush_interval_s: float,
    ) -> None:
        if queue_size <= 0 or flush_interval_s <= 0.0:
            raise ValueError("trace queue and flush interval must be positive")
        output_directory = Path(directory).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        # Microseconds prevent an immediate restart from overwriting a trace
        # created in the same wall-clock second.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = output_directory / f"deployment_state_{timestamp}.jsonl"
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=int(queue_size)
        )
        self._flush_interval_s = float(flush_interval_s)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._dropped = 0
        self._max_queue_depth = 0
        self._written = 0
        self._writer_error = ""
        self._final_event: Optional[dict[str, Any]] = None
        self._thread = threading.Thread(
            target=self._run,
            name="wuji-deployment-state-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def max_queue_depth(self) -> int:
        with self._lock:
            return self._max_queue_depth

    @property
    def writer_error(self) -> str:
        with self._lock:
            return self._writer_error

    def record(self, item: Mapping[str, Any]) -> None:
        if self._stop_event.is_set():
            return
        try:
            self._queue.put_nowait(dict(item))
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return
        depth = self._queue.qsize()
        with self._lock:
            self._max_queue_depth = max(self._max_queue_depth, depth)

    def close(
        self,
        final_event: Mapping[str, Any],
        *,
        timeout_s: float = 5.0,
    ) -> None:
        self._final_event = dict(final_event)
        self._stop_event.set()
        self._thread.join(timeout=max(0.0, float(timeout_s)))

    @staticmethod
    def _encode(item: Mapping[str, Any]) -> str:
        return json.dumps(
            item,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"

    def _run(self) -> None:
        try:
            last_flush = time.monotonic()
            with self.path.open("w", encoding="utf-8") as stream:
                while not self._stop_event.is_set() or not self._queue.empty():
                    batch: list[dict[str, Any]] = []
                    try:
                        batch.append(
                            self._queue.get(timeout=self._flush_interval_s)
                        )
                    except queue.Empty:
                        pass
                    while len(batch) < 512:
                        try:
                            batch.append(self._queue.get_nowait())
                        except queue.Empty:
                            break
                    if batch:
                        stream.writelines(self._encode(item) for item in batch)
                        with self._lock:
                            self._written += len(batch)
                    now = time.monotonic()
                    if now - last_flush >= self._flush_interval_s:
                        stream.flush()
                        last_flush = now

                final = dict(self._final_event or {})
                final.update(
                    {
                        "event": "trace_summary",
                        "wall_time_ns": time.time_ns(),
                        "monotonic_ns": time.monotonic_ns(),
                        "writer": {
                            "written_events": self._written,
                            "dropped_events": self.dropped,
                            "max_queue_depth": self.max_queue_depth,
                            "queue_capacity": self._queue.maxsize,
                        },
                    }
                )
                stream.write(self._encode(final))
                stream.flush()
        except Exception as exc:  # The sidecar must never affect robot control.
            with self._lock:
                self._writer_error = f"{type(exc).__name__}: {exc}"


def _recording_identity(config: Mapping[str, Any], directory: str | Path) -> dict[str, str]:
    """Join launcher recordings without inventing IDs for legacy trace paths."""
    deployment = section(config, "deployment")
    run_id = str(deployment.get("recording_run_id", ""))
    path = Path(directory).expanduser()
    if not run_id and path.name == "client" and path.parent.parent.name == "cosmos_runs":
        run_id = path.parent.name
    return {
        "recording_run_id": run_id,
        "source_pipeline_config": str(deployment.get(
            "recording_source_pipeline_config", config.get("_config_path", ""))),
    }


class DeploymentStateTraceNode(Node):
    """Capture synchronized 120 Hz snapshots of measured state and commands."""

    def __init__(
        self,
        config: Mapping[str, Any],
        active_side: str,
        arm_command_mode: str = "eef",
    ) -> None:
        super().__init__("wuji_deployment_state_trace")
        self._topics = section(config, "topics")
        deployment = section(config, "deployment")
        self._active_sides = _active_sides(active_side)
        self._arm_command_mode = str(arm_command_mode).strip().lower()
        if self._arm_command_mode not in ARM_COMMAND_MODES:
            raise ValueError(
                "arm command mode must be eef or joint"
            )
        self._sample_rate_hz = float(
            deployment.get("state_action_trace_rate_hz", 120.0)
        )
        if self._sample_rate_hz <= 0.0:
            raise ValueError("state_action_trace_rate_hz must be positive")
        directory = deployment.get(
            "state_action_trace_directory",
            deployment.get(
                "diagnostic_trace_directory",
                "/home/wuji/datasets/tianji_wuji/diagnostics",
            ),
        )
        self._recording_identity = _recording_identity(config, directory)
        self._writer = StateTraceWriter(
            directory,
            queue_size=int(
                deployment.get("state_action_trace_queue_size", 32768)
            ),
            flush_interval_s=float(
                deployment.get("state_action_trace_flush_interval_s", 0.5)
            ),
        )
        self._latest: dict[str, dict[str, Any]] = {}
        self._stats: dict[str, StreamStats] = {}
        self._lock = threading.Lock()
        self._sample_sequence = 0
        self._ready_samples = 0
        self._incomplete_ready_samples = 0
        self._lifecycle: Optional[int] = None
        self._closed = False

        self._stale_limits_ms = _stale_limits_ms(deployment)
        self._create_subscriptions()
        self.create_timer(1.0 / self._sample_rate_hz, self._sample)
        self._writer.record(
            {
                "event": "trace_start",
                **self._recording_identity,
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "sample_rate_hz": self._sample_rate_hz,
                "active_sides": list(self._active_sides),
                "arm_command_mode": self._arm_command_mode,
                "config_path": str(config.get("_config_path", "")),
                "pid": os.getpid(),
            }
        )
        self.get_logger().info(
            "Deployment state/action trace ready: "
            f"path={self._writer.path}, rate={self._sample_rate_hz:.1f}Hz, "
            f"active_sides={self._active_sides}, "
            f"arm_command_mode={self._arm_command_mode}"
        )

    @property
    def trace_path(self) -> Path:
        return self._writer.path

    @staticmethod
    def _best_effort_qos(depth: int = 64) -> QoSProfile:
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=int(depth),
        )

    def _create_subscriptions(self) -> None:
        qos = self._best_effort_qos()
        for side in SIDES:
            arm = self._topics["arm"][side]
            self.create_subscription(
                JointState,
                arm["state"],
                lambda msg, s=side: self._joint_callback(
                    f"{s}.arm_joint_state", msg, 7, "degree"
                ),
                qos,
            )
            self.create_subscription(
                PoseStamped,
                arm["actual_eef"],
                lambda msg, s=side: self._pose_callback(
                    f"{s}.arm_actual_eef", msg
                ),
                qos,
            )
            self.create_subscription(
                PoseStamped,
                arm["target_eef"],
                lambda msg, s=side: self._pose_callback(
                    f"{s}.arm_controller_target", msg
                ),
                qos,
            )
            self.create_subscription(
                JointState,
                arm["command"],
                lambda msg, s=side: self._joint_callback(
                    f"{s}.arm_joint_command", msg, 7, "degree"
                ),
                qos,
            )
            self.create_subscription(
                Float64MultiArray,
                arm["zsp"],
                lambda msg, s=side: self._array_callback(
                    f"{s}.arm_zsp", msg, None
                ),
                qos,
            )
            if side in self._active_sides:
                if self._arm_command_mode == "eef":
                    self.create_subscription(
                        PoseStamped,
                        arm["external_target"],
                        lambda msg, s=side: self._pose_callback(
                            f"{s}.arm_external_target", msg
                        ),
                        qos,
                    )
                    self.create_subscription(
                        Float64MultiArray,
                        arm["external_zsp"],
                        lambda msg, s=side: self._array_callback(
                            f"{s}.arm_external_zsp", msg, 3
                        ),
                        qos,
                    )
                else:
                    self.create_subscription(
                        JointState,
                        f"/{side}_arm/external_joint_target",
                        lambda msg, s=side: self._joint_callback(
                            f"{s}.arm_external_joint_target",
                            msg,
                            7,
                            "radian",
                        ),
                        qos,
                    )
                hand = self._topics["hand"][side]
                self.create_subscription(
                    JointState,
                    hand["state"],
                    lambda msg, s=side: self._joint_callback(
                        f"{s}.hand_state", msg, 20, "radian"
                    ),
                    qos,
                )
                self.create_subscription(
                    JointState,
                    hand["command"],
                    lambda msg, s=side: self._joint_callback(
                        f"{s}.hand_command", msg, 20, "radian"
                    ),
                    qos,
                )

        lifecycle_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Int8,
            "/tianji_arm/lifecycle_state",
            self._lifecycle_callback,
            lifecycle_qos,
        )
        self.create_subscription(
            String,
            "/wuji_deployment/status",
            self._status_callback,
            10,
        )

    def _observe(
        self,
        key: str,
        source_stamp_ns: int,
        receive_ns: int,
        data: Mapping[str, Any],
    ) -> None:
        with self._lock:
            stats = self._stats.setdefault(key, StreamStats())
            receive_sequence = stats.observe(receive_ns, source_stamp_ns)
            self._latest[key] = {
                "source_stamp_ns": int(source_stamp_ns),
                "received_monotonic_ns": int(receive_ns),
                "receive_sequence": int(receive_sequence),
                "data": dict(data),
            }

    def _mark_invalid(self, key: str, exc: Exception) -> None:
        with self._lock:
            self._stats.setdefault(key, StreamStats()).invalid += 1
        self.get_logger().warn(f"Invalid trace stream {key}: {exc}")

    def _joint_callback(
        self,
        key: str,
        message: JointState,
        expected: int,
        unit: str,
    ) -> None:
        receive_ns = time.monotonic_ns()
        try:
            position = _finite_vector(message.position, expected, f"{key}.position")
            velocity = [float(value) for value in message.velocity]
            effort = [float(value) for value in message.effort]
            if velocity and (
                len(velocity) != expected
                or not all(math.isfinite(value) for value in velocity)
            ):
                raise ValueError(f"{key}.velocity is invalid")
            if effort and (
                len(effort) != expected
                or not all(math.isfinite(value) for value in effort)
            ):
                raise ValueError(f"{key}.effort is invalid")
            self._observe(
                key,
                _stamp_ns(message),
                receive_ns,
                {
                    "type": "joint",
                    "unit": unit,
                    "frame_id": str(message.header.frame_id),
                    "name": list(message.name),
                    "position": position,
                    "velocity": velocity,
                    "effort": effort,
                },
            )
        except Exception as exc:
            self._mark_invalid(key, exc)

    def _pose_callback(self, key: str, message: PoseStamped) -> None:
        receive_ns = time.monotonic_ns()
        pose = message.pose
        try:
            position = _finite_vector(
                [pose.position.x, pose.position.y, pose.position.z],
                3,
                f"{key}.position",
            )
            quaternion = _finite_vector(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                4,
                f"{key}.quaternion",
            )
            norm = math.sqrt(sum(value * value for value in quaternion))
            if norm <= 1e-8:
                raise ValueError(f"{key}.quaternion has zero norm")
            quaternion = [value / norm for value in quaternion]
            self._observe(
                key,
                _stamp_ns(message),
                receive_ns,
                {
                    "type": "pose",
                    "frame_id": str(message.header.frame_id),
                    "position_m": position,
                    "quaternion_xyzw": quaternion,
                },
            )
        except Exception as exc:
            self._mark_invalid(key, exc)

    def _array_callback(
        self,
        key: str,
        message: Float64MultiArray,
        expected: Optional[int],
    ) -> None:
        receive_ns = time.monotonic_ns()
        try:
            values = [float(value) for value in message.data]
            if expected is not None and len(values) != expected:
                raise ValueError(f"{key} must contain {expected} values")
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{key} is empty or contains NaN/infinity")
            self._observe(
                key,
                0,
                receive_ns,
                {"type": "array", "values": values},
            )
        except Exception as exc:
            self._mark_invalid(key, exc)

    def _lifecycle_callback(self, message: Int8) -> None:
        self._lifecycle = int(message.data)
        self._writer.record(
            {
                "event": "lifecycle",
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "state": self._lifecycle,
            }
        )

    def _status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            status = {"raw": str(message.data)}
        self._writer.record(
            {
                "event": "deployment_status",
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "status": status,
            }
        )

    def _required_streams(self) -> list[str]:
        required = []
        for side in SIDES:
            required.extend(
                [f"{side}.arm_joint_state", f"{side}.arm_actual_eef"]
            )
        for side in self._active_sides:
            required.extend([
                f"{side}.arm_joint_command",
                f"{side}.hand_state",
                f"{side}.hand_command",
            ])
            if self._arm_command_mode == "eef":
                required.extend([
                    f"{side}.arm_external_target",
                    f"{side}.arm_controller_target",
                ])
            else:
                required.append(f"{side}.arm_external_joint_target")
        return required

    def _sample(self) -> None:
        now_monotonic_ns = time.monotonic_ns()
        now_wall_ns = time.time_ns()
        with self._lock:
            latest = {key: dict(value) for key, value in self._latest.items()}
        streams: dict[str, Any] = {}
        for key, value in latest.items():
            item = dict(value)
            item["age_ms"] = max(
                0.0,
                (now_monotonic_ns - int(item["received_monotonic_ns"])) * 1e-6,
            )
            streams[key] = item

        missing: list[str] = []
        stale: list[str] = []
        for key in self._required_streams():
            item = streams.get(key)
            if item is None:
                missing.append(key)
                continue
            suffix = key.split(".", 1)[1]
            limit_ms = self._stale_limits_ms[suffix]
            if float(item["age_ms"]) > limit_ms:
                stale.append(key)

        ready = self._lifecycle in COMMAND_LIFECYCLES
        complete = not missing and not stale
        if ready:
            self._ready_samples += 1
            if not complete:
                self._incomplete_ready_samples += 1
        self._sample_sequence += 1
        self._writer.record(
            {
                "event": "snapshot",
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "sample_sequence": self._sample_sequence,
                "wall_time_ns": now_wall_ns,
                "monotonic_ns": now_monotonic_ns,
                "ros_time_ns": self.get_clock().now().nanoseconds,
                "lifecycle": self._lifecycle,
                "ready": ready,
                "complete": complete,
                "missing_streams": missing,
                "stale_streams": stale,
                "streams": streams,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            stats = {key: asdict(value) for key, value in self._stats.items()}
        final = {
            **getattr(self, "_recording_identity", {}),
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "arm_command_mode": self._arm_command_mode,
            "sample_rate_hz": self._sample_rate_hz,
            "sample_count": self._sample_sequence,
            "ready_sample_count": self._ready_samples,
            "incomplete_ready_sample_count": self._incomplete_ready_samples,
            "stream_stats": stats,
        }
        self._writer.close(final)
        if self._writer.writer_error:
            self.get_logger().error(
                f"State/action trace writer failed: {self._writer.writer_error}"
            )
        elif self._writer.dropped:
            self.get_logger().error(
                "State/action trace is incomplete: "
                f"dropped_events={self._writer.dropped}"
            )
        else:
            self.get_logger().info(
                "State/action trace closed without queue loss: "
                f"{self._writer.path}"
            )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only Tianji/Wuji deployment state/action trace"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--active-side", choices=("left", "right", "both"), default="both"
    )
    parser.add_argument(
        "--arm-command-mode",
        choices=ARM_COMMAND_MODES,
        default="eef",
    )
    raw_arguments = list(sys.argv if argv is None else [sys.argv[0], *argv])
    return parser.parse_args(remove_ros_args(args=raw_arguments)[1:])


def main(argv=None) -> None:
    args = _parse_args(argv)
    # Lower only this sidecar's CPU priority.  The controller and deployment
    # client run in different processes and never wait for this logger.
    try:
        os.nice(10)
    except OSError:
        pass
    rclpy.init(args=None)
    node: Optional[DeploymentStateTraceNode] = None
    try:
        node = DeploymentStateTraceNode(
            load_config(args.config),
            args.active_side,
            args.arm_command_mode,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
