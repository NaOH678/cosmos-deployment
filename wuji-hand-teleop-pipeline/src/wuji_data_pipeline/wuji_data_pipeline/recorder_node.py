"""ROS 2 synchronized recorder for Tianji + Wuji teleoperation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from stereocamera.shared_frames import DEFAULT_DIRECTORY, SharedFrameReader

from .config import load_config, section
from .episode import EpisodeWriter
from .schema import RobotLayout, build_training_frame
from .sync import FiniteDifferenceVelocity, TimedRingBuffer, TimedSample
from .teleop_diagnostics import (
    DEFAULT_TRACKER_ROLES,
    MANUS_CHAIN_TYPE_CODES,
    MANUS_ERGONOMICS_TYPE_CODES,
    MANUS_JOINT_TYPE_CODES,
    decode_manus_message,
    decode_tracker_diagnostic,
    empty_optional_frame,
    teleop_diagnostic_specs,
)


SIDES = ("left", "right")
ACTIVE_HAND_CHOICES = ("both", "left", "right")
CAMERA_TRANSPORT_CHOICES = ("direct", "ros")


def active_hand_sides(active_hand: str) -> tuple[str, ...]:
    """Resolve the operator-selected physical Wuji Hand sides.

    The inactive side remains present in the 54-D dataset layout, but its
    20 hand values are deliberately zero-filled by the recorder.
    """
    normalized = str(active_hand).strip().lower()
    if normalized == "both":
        return SIDES
    if normalized in SIDES:
        return (normalized,)
    raise ValueError(
        f"active_hand must be one of {ACTIVE_HAND_CHOICES}, got {active_hand!r}"
    )


def camera_names_for_active_arm(
    camera_names: list[str] | tuple[str, ...],
    active_arm: str,
) -> list[str]:
    """Keep the head camera and only the wrist cameras used by the arm mode."""
    active_sides = active_hand_sides(active_arm)
    if active_sides == SIDES:
        return list(camera_names)
    selected = {"head", *(f"{side}_wrist" for side in active_sides)}
    return [str(name) for name in camera_names if str(name) in selected]


def zero_hand_sample(hand_dof: int = 20) -> dict[str, np.ndarray]:
    """Return the observation/target payload for an explicitly absent hand."""
    if hand_dof <= 0:
        raise ValueError("hand_dof must be positive")
    return {
        "actual_q_rad": np.zeros(hand_dof),
        "target_q_rad": np.zeros(hand_dof),
        "velocity_rad_s": np.zeros(hand_dof),
        "effort": np.zeros(hand_dof),
    }


def stamp_to_seconds(message: Any, fallback: float) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback
    seconds = int(getattr(stamp, "sec", 0))
    nanoseconds = int(getattr(stamp, "nanosec", 0))
    if seconds == 0 and nanoseconds == 0:
        return fallback
    return seconds + nanoseconds * 1e-9


def pose_message_to_array(message: PoseStamped) -> np.ndarray:
    pose = message.pose
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


def decode_compressed_image(message: CompressedImage) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to decode compressed camera data") from exc
    encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to decode compressed image")
    return frame


def decode_raw_image(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding: {message.encoding}")
    row_width = int(message.step)
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    expected = int(message.height) * row_width
    if raw.size < expected:
        raise ValueError(f"short image buffer: expected {expected}, got {raw.size}")
    rows = raw[:expected].reshape(int(message.height), row_width)
    pixels = rows[:, : int(message.width) * channels]
    image = pixels.reshape(int(message.height), int(message.width), channels)
    if encoding == "rgb8":
        image = image[:, :, ::-1]
    elif encoding == "rgba8":
        image = image[:, :, [2, 1, 0]]
    elif encoding == "bgra8":
        image = image[:, :, :3]
    elif encoding == "mono8":
        image = np.repeat(image, 3, axis=2)
    return np.ascontiguousarray(image, dtype=np.uint8)


class TeleopRecorderNode(Node):
    def __init__(
        self,
        config: Mapping[str, Any],
        require_cameras: Optional[bool] = None,
        active_hand: str = "both",
        active_arm: str = "both",
        output_dir: Optional[str] = None,
        task_name: Optional[str] = None,
        camera_transport: Optional[str] = None,
    ):
        super().__init__("wuji_teleop_recorder")
        self.config = dict(config)
        self.recording_config = section(config, "recording")
        self.topics = section(config, "topics")
        self.teleop_diagnostics_config = dict(
            config.get("teleop_diagnostics", {}) or {}
        )
        self.teleop_diagnostics_enabled = bool(
            self.teleop_diagnostics_config.get("enabled", True)
        )
        self.tracker_roles = tuple(
            self.teleop_diagnostics_config.get(
                "tracker_roles", DEFAULT_TRACKER_ROLES
            )
        )
        if not self.tracker_roles:
            raise ValueError("teleop_diagnostics.tracker_roles cannot be empty")
        if self.tracker_roles != DEFAULT_TRACKER_ROLES:
            raise ValueError(
                "teleop_diagnostics.tracker_roles must match the OpenVR "
                f"diagnostic protocol order {DEFAULT_TRACKER_ROLES}"
            )
        self.manus_node_count = int(
            self.teleop_diagnostics_config.get("manus_node_count", 25)
        )
        self.manus_sensor_count = int(
            self.teleop_diagnostics_config.get("manus_sensor_count", 5)
        )
        self.manus_ergonomics_count = int(
            self.teleop_diagnostics_config.get("manus_ergonomics_count", 20)
        )
        self.teleop_sync_tolerance_s = float(
            self.teleop_diagnostics_config.get("sync_tolerance_s", 0.10)
        )
        self.optional_specs = (
            teleop_diagnostic_specs(
                tracker_role_count=len(self.tracker_roles),
                manus_node_count=self.manus_node_count,
                manus_sensor_count=self.manus_sensor_count,
                manus_ergonomics_count=self.manus_ergonomics_count,
            )
            if self.teleop_diagnostics_enabled
            else {}
        )
        self.active_hand_sides = active_hand_sides(active_hand)
        self.zero_filled_hand_sides = tuple(
            side for side in SIDES if side not in self.active_hand_sides
        )
        self.active_arm_sides = active_hand_sides(active_arm)
        configured_require = bool(self.recording_config.get("require_cameras", True))
        self.require_cameras = configured_require if require_cameras is None else bool(require_cameras)
        configured_names = camera_names_for_active_arm(
            list(self.recording_config.get("camera_names", [])),
            active_arm,
        )
        self.configured_camera_names = (
            configured_names if self.require_cameras else []
        )
        configured_transport = str(
            self.recording_config.get("camera_transport", "direct")
        ).strip().lower()
        self.camera_transport = str(
            camera_transport or configured_transport
        ).strip().lower()
        if self.camera_transport not in CAMERA_TRANSPORT_CHOICES:
            raise ValueError(
                "camera_transport must be one of "
                f"{CAMERA_TRANSPORT_CHOICES}, got {self.camera_transport!r}"
            )
        self.camera_shared_memory_dir = str(
            self.recording_config.get(
                "camera_shared_memory_dir", DEFAULT_DIRECTORY
            )
        )
        self.auxiliary_camera_config = dict(
            self.recording_config.get("auxiliary_camera", {}) or {}
        )
        requested_auxiliary = bool(
            self.auxiliary_camera_config.get("enabled", False)
        )
        depth_camera_names = tuple(
            str(name)
            for name in self.auxiliary_camera_config.get(
                "depth_camera_names",
                ("head", "left_wrist", "right_wrist"),
            )
            if str(name) in self.configured_camera_names
        )
        infrared_stream_names = tuple(
            str(name)
            for name in self.auxiliary_camera_config.get(
                "infrared_stream_names",
                ("head_ir_left", "head_ir_right"),
            )
        )
        self.auxiliary_camera_enabled = (
            requested_auxiliary
            and self.require_cameras
            and self.camera_transport == "direct"
        )
        self.auxiliary_depth_stream_names = (
            tuple(f"{name}_depth" for name in depth_camera_names)
            if self.auxiliary_camera_enabled
            else ()
        )
        self.auxiliary_infrared_stream_names = (
            infrared_stream_names
            if self.auxiliary_camera_enabled
            else ()
        )
        self.auxiliary_stream_names = (
            self.auxiliary_depth_stream_names
            + self.auxiliary_infrared_stream_names
        )
        self.auxiliary_sync_tolerance_s = float(
            self.auxiliary_camera_config.get(
                "sync_tolerance_s", 0.06
            )
        )
        if self.auxiliary_sync_tolerance_s <= 0.0:
            raise ValueError(
                "auxiliary camera sync_tolerance_s must be positive"
            )
        self._camera_readers: dict[str, SharedFrameReader] = {}
        self._auxiliary_camera_readers: dict[
            str, SharedFrameReader
        ] = {}
        self._camera_sequences = {
            name: 0 for name in self.configured_camera_names
        }
        self._camera_generations = {
            name: 0 for name in self.configured_camera_names
        }
        self._camera_overwritten = {
            name: 0 for name in self.configured_camera_names
        }
        self._camera_capture_failures = {
            name: 0 for name in self.configured_camera_names
        }
        self._auxiliary_camera_sequences = {
            name: 0 for name in self.auxiliary_stream_names
        }
        self._auxiliary_camera_generations = {
            name: 0 for name in self.auxiliary_stream_names
        }
        self._auxiliary_camera_overwritten = {
            name: 0 for name in self.auxiliary_stream_names
        }
        self._auxiliary_camera_capture_failures = {
            name: 0 for name in self.auxiliary_stream_names
        }
        # The selected set is frozen independently for each episode.  Before
        # the first capture it mirrors the configuration so status output still
        # describes every supported camera position.
        self.camera_names = list(self.configured_camera_names)
        self.frame_rate = float(self.recording_config.get("frame_rate", 30.0))
        self.sync_tolerance_s = float(self.recording_config.get("sync_tolerance_s", 0.04))
        buffer_seconds = float(self.recording_config.get("buffer_seconds", 3.0))
        maxlen = max(128, int(buffer_seconds * 1200.0))
        self.buffers: dict[str, TimedRingBuffer] = {}
        self._maxlen = maxlen
        self._buffer_age = max(buffer_seconds, 1.0)
        self._writer: Optional[EpisodeWriter] = None
        self._pending_writer: Optional[EpisodeWriter] = None
        self._finalizing = False
        self._record_started_at: Optional[float] = None
        self._pending_duration_s = 0.0
        self._capture_has_finished = False
        self.output_dir = str(
            output_dir or self.recording_config.get(
                "output_dir", "./datasets/tianji_wuji"
            )
        )
        self.task_name = str(task_name or "").strip()
        self._record_lock = threading.Lock()
        self._last_anchor_timestamp = float("-inf")
        self._sync_skip_count = 0
        self._sync_error_sum = 0.0
        self._sync_error_max = 0.0
        self._latest_lifecycle: Optional[int] = None
        self._joint_velocity = FiniteDifferenceVelocity(max_dt_s=1.0)
        self._last_diagnostic_warning_ns = 0
        self._manus_diagnostics_subscription_error: Optional[str] = None

        self._create_scalar_subscriptions()
        self._create_camera_inputs()
        self._create_teleop_diagnostic_subscriptions()
        self.create_service(Trigger, "~/start", self._start_callback)
        self.create_service(
            Trigger, "~/toggle_capture", self._toggle_capture_callback
        )
        self.create_service(Trigger, "~/finish", self._finish_callback)
        self.create_service(Trigger, "~/save", self._save_callback)
        self.create_service(Trigger, "~/discard", self._discard_callback)
        self.create_service(Trigger, "~/stop", self._stop_callback)
        self.create_service(Trigger, "~/status", self._status_callback)
        self._status_publisher = self.create_publisher(String, "~/status_text", 10)
        # A fast timer detects every new anchor while all expensive disk work
        # remains isolated in this recorder process.
        self.create_timer(1.0 / 200.0, self._sync_timer)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            "Recorder ready: "
            f"configured_cameras={self.configured_camera_names or '<disabled>'}, "
            f"camera_transport={self.camera_transport}, "
            "camera_policy=online_subset_per_episode, "
            f"dataset_rate={self.frame_rate:.1f}Hz, "
            f"active_arms={self.active_arm_sides}, "
            f"active_hands={self.active_hand_sides}, "
            f"zero_filled_hands={self.zero_filled_hand_sides or '<none>'}, "
            f"teleop_diagnostics={self.teleop_diagnostics_enabled}, "
            f"auxiliary_camera_streams="
            f"{self.auxiliary_stream_names or '<disabled>'}, "
            f"output_dir={self.output_dir}, task={self.task_name or '<unset>'}"
        )

    def _buffer(self, name: str) -> TimedRingBuffer:
        if name not in self.buffers:
            self.buffers[name] = TimedRingBuffer(
                maxlen=self._maxlen, max_age_s=self._buffer_age
            )
        return self.buffers[name]

    def _append(self, key: str, message: Any, value: Any = None) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        timestamp = stamp_to_seconds(message, now)
        self._buffer(key).append(timestamp, message if value is None else value)

    def _create_scalar_subscriptions(self) -> None:
        arm_topics = self.topics["arm"]
        hand_topics = self.topics["hand"]
        for side in SIDES:
            arm = arm_topics[side]
            hand = hand_topics[side]
            self.create_subscription(
                JointState,
                arm["state"],
                lambda msg, s=side: self._append(f"arm_state_{s}", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                JointState,
                arm["command"],
                lambda msg, s=side: self._append(f"arm_command_{s}", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                PoseStamped,
                arm["actual_eef"],
                lambda msg, s=side: self._append(
                    f"arm_actual_eef_{s}", msg, pose_message_to_array(msg)
                ),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                PoseStamped,
                arm["target_eef"],
                lambda msg, s=side: self._append(
                    f"arm_target_eef_{s}", msg, pose_message_to_array(msg)
                ),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Float64MultiArray,
                arm["zsp"],
                lambda msg, s=side: self._append(
                    f"arm_zsp_{s}", msg, np.asarray(msg.data[:3], dtype=np.float32)
                ),
                qos_profile_sensor_data,
            )
            if side in self.active_hand_sides:
                self.create_subscription(
                    JointState,
                    hand["state"],
                    lambda msg, s=side: self._append(f"hand_state_{s}", msg),
                    qos_profile_sensor_data,
                )
                self.create_subscription(
                    JointState,
                    hand["command"],
                    lambda msg, s=side: self._append(f"hand_command_{s}", msg),
                    qos_profile_sensor_data,
                )

    def _create_camera_inputs(self) -> None:
        if not self.configured_camera_names:
            return
        if self.camera_transport == "direct":
            for name in self.configured_camera_names:
                self._camera_readers[name] = SharedFrameReader(
                    name, directory=self.camera_shared_memory_dir
                )
            for name in self.auxiliary_stream_names:
                self._auxiliary_camera_readers[name] = SharedFrameReader(
                    name, directory=self.camera_shared_memory_dir
                )
            # Drain the producer rings independently of recording state so
            # readiness and online-camera selection always use recent frames.
            self.create_timer(1.0 / 200.0, self._poll_direct_cameras)
            return
        camera_topics = self.topics["camera"]
        for name in self.configured_camera_names:
            camera = camera_topics[name]
            message_type = str(camera.get("type", "raw")).lower()
            if message_type == "compressed":
                self.create_subscription(
                    CompressedImage,
                    camera["topic"],
                    lambda msg, n=name: self._append(f"camera_{n}", msg),
                    qos_profile_sensor_data,
                )
            elif message_type == "raw":
                self.create_subscription(
                    Image,
                    camera["topic"],
                    lambda msg, n=name: self._append(f"camera_{n}", msg),
                    qos_profile_sensor_data,
                )
            else:
                raise ValueError(f"unsupported camera message type: {message_type}")

    def _poll_direct_cameras(self) -> None:
        for name, reader in self._camera_readers.items():
            last_sequence = self._camera_sequences[name]
            try:
                batch = reader.read_since(last_sequence)
            except (OSError, ValueError) as exc:
                self._warn_diagnostic(
                    f"Direct camera {name} transport error: {exc}"
                )
                continue
            if (
                batch.producer_generation
                and self._camera_generations[name]
                and batch.producer_generation != self._camera_generations[name]
            ):
                # The producer restarted. Its sequence counter starts at one.
                last_sequence = 0
                batch = reader.read_since(0)
            if batch.producer_generation:
                self._camera_generations[name] = batch.producer_generation
            self._camera_overwritten[name] += batch.overwritten
            self._camera_capture_failures[name] = batch.capture_failures
            for frame in batch.frames:
                self._buffer(f"camera_{name}").append(
                    frame.timestamp, frame.image
                )
                last_sequence = max(last_sequence, frame.sequence)
            self._camera_sequences[name] = last_sequence
        for name, reader in getattr(
            self, "_auxiliary_camera_readers", {}
        ).items():
            last_sequence = self._auxiliary_camera_sequences[name]
            try:
                batch = reader.read_since(last_sequence)
            except (OSError, ValueError) as exc:
                self._warn_diagnostic(
                    f"Auxiliary camera {name} transport error: {exc}"
                )
                continue
            if (
                batch.producer_generation
                and self._auxiliary_camera_generations[name]
                and batch.producer_generation
                != self._auxiliary_camera_generations[name]
            ):
                last_sequence = 0
                batch = reader.read_since(0)
            if batch.producer_generation:
                self._auxiliary_camera_generations[name] = (
                    batch.producer_generation
                )
            self._auxiliary_camera_overwritten[name] += batch.overwritten
            self._auxiliary_camera_capture_failures[name] = (
                batch.capture_failures
            )
            for frame in batch.frames:
                self._buffer(f"aux_camera_{name}").append(
                    frame.timestamp,
                    {
                        "image": frame.image,
                        "sequence": frame.sequence,
                        "generation": batch.producer_generation,
                    },
                )
                last_sequence = max(last_sequence, frame.sequence)
            self._auxiliary_camera_sequences[name] = last_sequence

    def _create_teleop_diagnostic_subscriptions(self) -> None:
        if not self.teleop_diagnostics_enabled:
            return
        tracker_topic = str(
            self.teleop_diagnostics_config.get(
                "tracker_topic", "/openvr/tracker_diagnostics"
            )
        )
        self.create_subscription(
            Float64MultiArray,
            tracker_topic,
            self._tracker_diagnostics_callback,
            qos_profile_sensor_data,
        )

        # MANUS messages are optional at runtime.  A missing message package
        # disables this diagnostic subscription only; arm/hand control and the
        # original recorder sources continue unchanged.
        try:
            from manus_ros2_msgs.msg import ManusGlove
        except ImportError as exc:
            self._manus_diagnostics_subscription_error = str(exc)
            self.get_logger().warning(
                f"MANUS recorder diagnostics disabled: {exc}"
            )
            return
        manus_topics = list(
            self.teleop_diagnostics_config.get(
                "manus_topics", ["/manus_glove_0", "/manus_glove_1"]
            )
        )
        for topic in manus_topics:
            self.create_subscription(
                ManusGlove,
                str(topic),
                self._manus_diagnostics_callback,
                qos_profile_sensor_data,
            )

    def _warn_diagnostic(self, message: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_diagnostic_warning_ns >= 5_000_000_000:
            self.get_logger().warning(message)
            self._last_diagnostic_warning_ns = now_ns

    def _tracker_diagnostics_callback(self, message: Float64MultiArray) -> None:
        try:
            value = decode_tracker_diagnostic(
                message.data, role_count=len(self.tracker_roles)
            )
            self._append("teleop_tracker", message, value)
        except Exception as exc:
            self._warn_diagnostic(f"Ignored malformed OpenVR diagnostics: {exc}")

    def _manus_diagnostics_callback(self, message: Any) -> None:
        side = str(getattr(message, "side", "")).strip().lower()
        if side not in SIDES:
            self._warn_diagnostic(
                f"Ignored MANUS diagnostics with unknown side: {side!r}"
            )
            return
        try:
            value = decode_manus_message(
                message,
                side=side,
                node_count=self.manus_node_count,
                sensor_count=self.manus_sensor_count,
                ergonomics_count=self.manus_ergonomics_count,
            )
            self._append(f"teleop_manus_{side}", message, value)
        except Exception as exc:
            self._warn_diagnostic(f"Ignored malformed MANUS diagnostics: {exc}")

    def _required_source_keys(self) -> list[str]:
        keys = []
        for side in SIDES:
            keys.extend(
                [
                    f"arm_state_{side}",
                    f"arm_actual_eef_{side}",
                ]
            )
            if side in self.active_arm_sides:
                keys.extend(
                    [
                        f"arm_command_{side}",
                        f"arm_target_eef_{side}",
                        f"arm_zsp_{side}",
                    ]
                )
            if side in self.active_hand_sides:
                keys.extend(
                    [
                        f"hand_state_{side}",
                        f"hand_command_{side}",
                    ]
                )
        keys.extend(f"camera_{name}" for name in self.camera_names)
        return keys

    def readiness_errors(self, max_age_s: float = 1.0) -> list[str]:
        now = self.get_clock().now().nanoseconds * 1e-9
        errors = []
        for key in self._required_source_keys():
            sample = self.buffers.get(key).latest() if key in self.buffers else None
            if sample is None:
                errors.append(f"{key}:missing")
            elif now - sample.timestamp > max_age_s:
                errors.append(f"{key}:stale({now - sample.timestamp:.2f}s)")
        return errors

    def online_camera_names(self, max_age_s: float = 1.0) -> list[str]:
        """Return configured cameras that have a fresh frame right now."""
        now = self.get_clock().now().nanoseconds * 1e-9
        online = []
        for name in self.configured_camera_names:
            buffer = self.buffers.get(f"camera_{name}")
            sample = buffer.latest() if buffer is not None else None
            if sample is not None and now - sample.timestamp <= max_age_s:
                online.append(name)
        return online

    def online_auxiliary_stream_names(
        self,
        max_age_s: float = 1.0,
    ) -> list[str]:
        now = self.get_clock().now().nanoseconds * 1e-9
        online = []
        for name in getattr(self, "auxiliary_stream_names", ()):
            buffer = self.buffers.get(f"aux_camera_{name}")
            sample = buffer.latest() if buffer is not None else None
            if sample is not None and now - sample.timestamp <= max_age_s:
                online.append(name)
        return online

    def _episode_auxiliary_config(self) -> dict[str, Any]:
        if not getattr(self, "auxiliary_camera_enabled", False):
            return {}
        config = getattr(self, "auxiliary_camera_config", {})
        return {
            "depth_stream_names": tuple(
                getattr(self, "auxiliary_depth_stream_names", ())
            ),
            "infrared_stream_names": tuple(
                getattr(self, "auxiliary_infrared_stream_names", ())
            ),
            "map_size": int(
                config.get(
                    "lmdb_map_size",
                    self.recording_config.get(
                        "lmdb_map_size", 1 << 40
                    ),
                )
            ),
            "queue_capacity": int(
                config.get("queue_capacity", 64)
            ),
            "depth_png_compression": int(
                config.get("depth_png_compression", 1)
            ),
            "video_fourcc": str(
                config.get(
                    "video_fourcc",
                    self.recording_config.get(
                        "video_fourcc", "mp4v"
                    ),
                )
            ),
            "infrared_frame_rate": float(
                config.get(
                    "infrared_frame_rate",
                    self.recording_config.get("frame_rate", 30.0),
                )
            ),
            "capture_metadata_path": str(
                Path(self.camera_shared_memory_dir)
                / "camera_metadata.json"
            ),
        }

    def start_recording(self) -> tuple[bool, str]:
        with self._record_lock:
            if self._writer is not None:
                return False, "an episode is already recording"
            if self._finalizing:
                return False, "an episode is currently being saved"
            if self.require_cameras:
                self.camera_names = self.online_camera_names()
                if not self.camera_names:
                    return (
                        False,
                        "no configured camera is publishing fresh frames; "
                        "connect at least one camera or start with cameras disabled",
                    )
            errors = self.readiness_errors()
            if errors:
                return False, "sources not ready: " + ", ".join(errors)
            pending = self._pending_writer
            new_writer = EpisodeWriter(
                output_dir=self.output_dir,
                layout=RobotLayout(),
                camera_names=self.camera_names,
                frame_rate=self.frame_rate,
                map_size=int(self.recording_config.get("lmdb_map_size", 1 << 40)),
                video_fourcc=str(self.recording_config.get("video_fourcc", "mp4v")),
                metadata={
                    "source": "wuji-hand-teleop/ros2",
                    "pipeline_config": self.config.get("_config_path"),
                    "camera_layout": self._camera_metadata(),
                    "active_arm_sides": list(self.active_arm_sides),
                    "active_hand_sides": list(self.active_hand_sides),
                    "zero_filled_hand_sides": list(self.zero_filled_hand_sides),
                    "teleop_diagnostics": self._teleop_diagnostics_metadata(),
                    "task_name": self.task_name or None,
                },
                optional_specs=self.optional_specs,
                auxiliary_camera=self._episode_auxiliary_config(),
            )
            discarded = None
            if pending is not None:
                try:
                    discarded = pending.discard()
                except Exception:
                    new_writer.discard()
                    raise
            self._pending_writer = None
            self._pending_duration_s = 0.0
            self._writer = new_writer
            self._record_started_at = time.monotonic()
            self._last_anchor_timestamp = float("-inf")
            self._sync_skip_count = 0
            self._sync_error_sum = 0.0
            self._sync_error_max = 0.0
            self._joint_velocity.reset()
            path = self._writer.episode_dir
            self.get_logger().info(f"Recording started: {path}")
            detail = f"; discarded unsaved {discarded}" if discarded else ""
            return True, f"{path}{detail}"

    def finish_recording(self) -> tuple[bool, str]:
        """Stop appending and retain the episode until save/discard is chosen."""
        with self._record_lock:
            writer = self._writer
            if writer is None:
                return False, "no episode is recording"
            if self._pending_writer is not None:
                return False, "another episode is already waiting for a decision"
            self._writer = None
            started_at = self._record_started_at
            self._record_started_at = None
            self._pending_writer = writer
            self._capture_has_finished = True
            self._pending_duration_s = (
                max(0.0, time.monotonic() - started_at)
                if started_at is not None
                else 0.0
            )
            count = writer.step_count
            path = writer.episode_dir
        self.get_logger().info(
            f"Capture finished; awaiting save/discard: {path} ({count} steps)"
        )
        return True, f"{path} ({count} steps) awaiting save"

    def save_pending(self) -> tuple[bool, str]:
        with self._record_lock:
            if self._writer is not None:
                return False, "finish the active capture before saving"
            writer = self._pending_writer
            if writer is None:
                return False, "no finished episode is waiting to be saved"
            self._pending_writer = None
            self._pending_duration_s = 0.0
            self._finalizing = True
        count = writer.step_count
        if count == 0:
            path = writer.discard()
            with self._record_lock:
                self._finalizing = False
            message = f"empty episode discarded: {path}"
            self.get_logger().error(message)
            return False, message
        mean_error = self._sync_error_sum / max(count, 1)
        try:
            path = writer.finalize(
                {
                    "sync_skip_count": self._sync_skip_count,
                    "sync_error_mean_s": mean_error,
                    "sync_error_max_s": self._sync_error_max,
                    "frame_sync_vqe": count
                    / max(count + self._sync_skip_count, 1),
                }
            )
        except Exception as exc:
            writer.close_incomplete()
            message = (
                "failed to finalize episode; .inprogress data preserved: "
                f"{exc}"
            )
            self.get_logger().error(message)
            return False, message
        finally:
            with self._record_lock:
                self._finalizing = False
        self.get_logger().info(f"Episode saved: {path} ({count} steps)")
        return True, f"{path} ({count} steps)"

    def discard_unsaved(self) -> tuple[bool, str]:
        with self._record_lock:
            if self._finalizing:
                return False, "cannot discard while an episode is being saved"
            writers = [
                writer
                for writer in (self._writer, self._pending_writer)
                if writer is not None
            ]
            self._writer = None
            self._pending_writer = None
            self._record_started_at = None
            self._pending_duration_s = 0.0
        if not writers:
            return True, "no unsaved episode"
        paths = [str(writer.discard()) for writer in writers]
        self.get_logger().warning(
            "Unsaved episode discarded: " + ", ".join(paths)
        )
        return True, ", ".join(paths)

    def stop_recording(self) -> tuple[bool, str]:
        """Compatibility API: finish an active capture and save immediately."""
        if self._writer is not None:
            ok, message = self.finish_recording()
            if not ok:
                return ok, message
        return self.save_pending()

    def _camera_metadata(self) -> dict[str, Any]:
        if self.camera_transport == "direct":
            return {
                name: {
                    "transport": "direct_shared_memory",
                    "shared_memory_dir": self.camera_shared_memory_dir,
                    "pixel_format": "bgr8",
                    "timestamp": "producer_system_and_monotonic_clock",
                }
                for name in self.camera_names
            }
        camera_topics = self.topics.get("camera", {})
        return {
            name: {
                "topic": camera_topics[name]["topic"],
                "message_type": camera_topics[name].get("type", "raw"),
            }
            for name in self.camera_names
        }

    def _teleop_diagnostics_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.teleop_diagnostics_enabled,
            "non_blocking": True,
            "alignment_anchor": self._alignment_anchor_key(),
            "sync_tolerance_s": self.teleop_sync_tolerance_s,
            "tracker_topic": self.teleop_diagnostics_config.get(
                "tracker_topic", "/openvr/tracker_diagnostics"
            ),
            "tracker_roles": list(self.tracker_roles),
            "tracker_message_columns": [
                "raw_pose[x,y,z,qx,qy,qz,qw]",
                "corrected_pose[x,y,z,qx,qy,qz,qw]",
                "linear_velocity[x,y,z]",
                "angular_velocity[x,y,z]",
                "detected",
                "connected",
                "valid",
                "tracking_result",
                "device_index",
                "corrected_valid",
            ],
            "timestamp_semantics": (
                "recorder ROS receive time; existing MANUS/OpenVR message "
                "protocols are left unchanged"
            ),
            "tracker_sampling": (
                "raw pose/state is copied from the same SDK poll used to "
                "produce each existing corrected TF; no extra SDK poll"
            ),
            "manus_topics": list(
                self.teleop_diagnostics_config.get(
                    "manus_topics", ["/manus_glove_0", "/manus_glove_1"]
                )
            ),
            "manus_node_order": "ascending node_id; inspect node_id and semantic type codes",
            "manus_keypoints_order": "MediaPipe 21, identical semantic mapping to controller",
            "manus_type_codebooks": {
                "chain_type_code": list(MANUS_CHAIN_TYPE_CODES),
                "joint_type_code": list(MANUS_JOINT_TYPE_CODES),
                "ergonomics_type_code": list(MANUS_ERGONOMICS_TYPE_CODES),
                "unknown_code": -1,
            },
            "missing_policy": "zero_or_fill_value_with_available_and_valid_equal_zero",
        }

    def _start_callback(self, _request, response):
        try:
            response.success, response.message = self.start_recording()
        except Exception as exc:
            response.success = False
            response.message = f"failed to start episode: {exc}"
            self.get_logger().error(response.message)
        TeleopRecorderNode._publish_status_if_available(self)
        return response

    def _toggle_capture_callback(self, _request, response):
        try:
            if self._writer is not None:
                response.success, response.message = self.finish_recording()
            else:
                response.success, response.message = self.start_recording()
        except Exception as exc:
            response.success = False
            response.message = f"capture toggle failed: {exc}"
            self.get_logger().error(response.message)
        TeleopRecorderNode._publish_status_if_available(self)
        return response

    def _finish_callback(self, _request, response):
        try:
            response.success, response.message = self.finish_recording()
        except Exception as exc:
            response.success = False
            response.message = f"failed to finish capture: {exc}"
            self.get_logger().error(response.message)
        TeleopRecorderNode._publish_status_if_available(self)
        return response

    def _save_callback(self, _request, response):
        try:
            response.success, response.message = self.save_pending()
        except Exception as exc:
            response.success = False
            response.message = f"failed to save episode: {exc}"
            self.get_logger().error(response.message)
        TeleopRecorderNode._publish_status_if_available(self)
        return response

    def _discard_callback(self, _request, response):
        try:
            response.success, response.message = self.discard_unsaved()
        except Exception as exc:
            response.success = False
            response.message = f"failed to discard episode: {exc}"
            self.get_logger().error(response.message)
        TeleopRecorderNode._publish_status_if_available(self)
        return response

    def _stop_callback(self, _request, response):
        try:
            response.success, response.message = self.stop_recording()
        except Exception as exc:
            response.success = False
            response.message = f"failed to stop episode: {exc}"
            self.get_logger().error(response.message)
        TeleopRecorderNode._publish_status_if_available(self)
        return response

    def _status_callback(self, _request, response):
        response.success = True
        response.message = json.dumps(self.status_dict())
        return response

    def status_dict(self) -> dict[str, Any]:
        writer = self._writer
        pending = self._pending_writer
        auxiliary_owner = writer if writer is not None else pending
        if writer is not None and self._record_started_at is not None:
            elapsed = time.monotonic() - self._record_started_at
        elif pending is not None:
            elapsed = self._pending_duration_s
        else:
            elapsed = 0.0
        return {
            "recording": writer is not None,
            "state": (
                "recording"
                if writer is not None
                else "pending_save"
                if pending is not None
                else "saving"
                if self._finalizing
                else "idle"
            ),
            "episode_dir": str(writer.episode_dir) if writer is not None else None,
            "steps": writer.step_count if writer is not None else 0,
            "elapsed_s": elapsed,
            "pending_episode_dir": (
                str(pending.episode_dir) if pending is not None else None
            ),
            "pending_steps": pending.step_count if pending is not None else 0,
            "output_dir": self.output_dir,
            "task_name": self.task_name or None,
            "capture_has_finished": self._capture_has_finished,
            "sync_skips": self._sync_skip_count,
            "cameras": self.camera_names,
            "configured_cameras": self.configured_camera_names,
            "online_cameras": self.online_camera_names(),
            "camera_transport": self.camera_transport,
            "camera_overwritten_frames": dict(self._camera_overwritten),
            "camera_capture_failures": dict(
                self._camera_capture_failures
            ),
            "auxiliary_camera": {
                "enabled": self.auxiliary_camera_enabled,
                "configured_streams": list(
                    self.auxiliary_stream_names
                ),
                "online_streams": self.online_auxiliary_stream_names(),
                "overwritten_frames": dict(
                    self._auxiliary_camera_overwritten
                ),
                "capture_failures": dict(
                    self._auxiliary_camera_capture_failures
                ),
                "writer": (
                    auxiliary_owner.auxiliary_status()
                    if auxiliary_owner is not None
                    else {"enabled": False}
                ),
            },
            "active_arm_sides": list(self.active_arm_sides),
            "active_hand_sides": list(self.active_hand_sides),
            "zero_filled_hand_sides": list(self.zero_filled_hand_sides),
            "teleop_diagnostics": {
                "enabled": self.teleop_diagnostics_enabled,
                "tracker_received": self.buffers.get("teleop_tracker") is not None
                and self.buffers["teleop_tracker"].latest() is not None,
                "manus_received": {
                    side: self.buffers.get(f"teleop_manus_{side}") is not None
                    and self.buffers[f"teleop_manus_{side}"].latest() is not None
                    for side in SIDES
                },
                "manus_subscription_error": self._manus_diagnostics_subscription_error,
            },
            "not_ready": self.readiness_errors(),
        }

    def _publish_status(self) -> None:
        message = String()
        message.data = json.dumps(self.status_dict())
        self._status_publisher.publish(message)

    def _publish_status_if_available(self) -> None:
        if getattr(self, "_status_publisher", None) is not None:
            self._publish_status()

    def _alignment_anchor_key(self) -> str:
        if self.camera_names:
            return f"camera_{self.camera_names[0]}"
        return f"arm_state_{self.active_arm_sides[0]}"

    def _anchor_sample(self) -> Optional[TimedSample]:
        anchor_key = self._alignment_anchor_key()
        buffer = self.buffers.get(anchor_key)
        if buffer is None:
            return None
        if self.camera_names:
            return buffer.latest_after(self._last_anchor_timestamp)
        # Without cameras, downsample the high-rate arm feedback to the same
        # nominal dataset rate that will be used after cameras are installed.
        latest = buffer.latest()
        if latest is None:
            return None
        if latest.timestamp - self._last_anchor_timestamp < 1.0 / self.frame_rate:
            return None
        return latest

    def _nearest(self, key: str, timestamp: float, multiplier: float = 1.0) -> Optional[TimedSample]:
        buffer = self.buffers.get(key)
        if buffer is None:
            return None
        return buffer.nearest(timestamp, self.sync_tolerance_s * multiplier)

    def _arm_frame_sample(
        self,
        side: str,
        matched: Mapping[str, TimedSample],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        state_sample = matched[f"arm_state_{side}"]
        state: JointState = state_sample.value
        arm_position = list(state.position)
        arm = {
            "joint_pos_deg": arm_position,
            "joint_vel_deg_s": self._joint_velocity.measure(
                f"arm_{side}",
                state_sample.timestamp,
                arm_position,
                state.velocity,
            ),
            "joint_effort": (
                list(state.effort) if state.effort else np.zeros(7)
            ),
            "actual_eef": matched[f"arm_actual_eef_{side}"].value,
        }
        source_times = {
            f"arm_state_{side}": state_sample.timestamp,
            f"arm_actual_eef_{side}": matched[
                f"arm_actual_eef_{side}"
            ].timestamp,
        }
        if side not in self.active_arm_sides:
            # The fixed dual-arm schema still records the parked arm's measured
            # state. Schema defaults fill command/target with the hold pose.
            return arm, source_times

        command_sample = matched[f"arm_command_{side}"]
        arm.update(
            {
                "joint_command_deg": list(command_sample.value.position),
                "target_eef": matched[f"arm_target_eef_{side}"].value,
                "zsp": matched[f"arm_zsp_{side}"].value,
            }
        )
        source_times.update(
            {
                f"arm_command_{side}": command_sample.timestamp,
                f"arm_target_eef_{side}": matched[
                    f"arm_target_eef_{side}"
                ].timestamp,
                f"arm_zsp_{side}": matched[f"arm_zsp_{side}"].timestamp,
            }
        )
        return arm, source_times

    def _aligned_teleop_diagnostics(
        self, anchor_timestamp: float
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        frame = empty_optional_frame(self.optional_specs)
        alignment: dict[str, Any] = {}
        if not self.teleop_diagnostics_enabled:
            return frame, alignment

        sources = ["tracker", *[f"manus_{side}" for side in SIDES]]
        for source in sources:
            buffer_key = f"teleop_{source}"
            buffer = self.buffers.get(buffer_key)
            sample = (
                buffer.nearest(anchor_timestamp, self.teleop_sync_tolerance_s)
                if buffer is not None
                else None
            )
            if sample is None:
                alignment[source] = {"available": False}
                continue
            for field, value in sample.value.items():
                if field in frame:
                    frame[field] = value
            error_s = abs(sample.timestamp - anchor_timestamp)
            if source == "tracker":
                prefix = "teleop_tracker"
            else:
                prefix = f"teleop_manus_{source.removeprefix('manus_')}"
            frame[f"{prefix}_source_timestamp"] = np.asarray(
                [sample.timestamp], dtype=np.float64
            )
            frame[f"{prefix}_alignment_error_s"] = np.asarray(
                [error_s], dtype=np.float32
            )
            alignment[source] = {
                "available": True,
                "source": sample.timestamp,
                "error_s": error_s,
            }
        return frame, alignment

    def _aligned_auxiliary_camera(
        self,
        anchor_timestamp: float,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, int],
    ]:
        images: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        sequences: dict[str, int] = {}
        if not getattr(self, "auxiliary_camera_enabled", False):
            return images, timestamps, sequences
        for stream_name in self.auxiliary_stream_names:
            buffer = self.buffers.get(f"aux_camera_{stream_name}")
            sample = (
                buffer.nearest(
                    anchor_timestamp,
                    self.auxiliary_sync_tolerance_s,
                )
                if buffer is not None
                else None
            )
            if sample is None or not isinstance(sample.value, Mapping):
                continue
            image = sample.value.get("image")
            if not isinstance(image, np.ndarray):
                continue
            images[stream_name] = image
            timestamps[stream_name] = sample.timestamp
            sequences[stream_name] = int(
                sample.value.get("sequence", -1)
            )
        return images, timestamps, sequences

    def _sync_timer(self) -> None:
        writer = self._writer
        if writer is None:
            return
        anchor = self._anchor_sample()
        if anchor is None:
            return
        self._last_anchor_timestamp = anchor.timestamp
        matched: dict[str, TimedSample] = {}
        for key in self._required_source_keys():
            multiplier = 2.0 if key.startswith("camera_") else (3.0 if key.startswith("hand_") else 1.0)
            sample = self._nearest(key, anchor.timestamp, multiplier)
            if sample is None:
                self._sync_skip_count += 1
                return
            matched[key] = sample

        try:
            arms: dict[str, dict[str, Any]] = {}
            hands: dict[str, dict[str, Any]] = {}
            source_times: dict[str, Any] = {"anchor": anchor.timestamp}
            for side in SIDES:
                arms[side], arm_source_times = self._arm_frame_sample(
                    side, matched
                )
                source_times.update(arm_source_times)
                if side in self.active_hand_sides:
                    hand_state: JointState = matched[f"hand_state_{side}"].value
                    hand_command: JointState = matched[f"hand_command_{side}"].value
                    hand_position = list(hand_state.position)
                    hand_velocity = self._joint_velocity.measure(
                        f"hand_{side}",
                        matched[f"hand_state_{side}"].timestamp,
                        hand_position,
                        hand_state.velocity,
                    )
                    hands[side] = {
                        "actual_q_rad": hand_position,
                        "target_q_rad": list(hand_command.position),
                        "velocity_rad_s": hand_velocity,
                        "effort": (
                            list(hand_state.effort)
                            if hand_state.effort
                            else np.zeros(20)
                        ),
                    }
                    for category in ("hand_state", "hand_command"):
                        source_times[f"{category}_{side}"] = matched[
                            f"{category}_{side}"
                        ].timestamp
                else:
                    # Preserve the agreed dual-arm/dual-hand 54-D schema while
                    # one physical hand is unavailable. Both observation and
                    # target are zero so replay cannot command the absent hand.
                    hands[side] = zero_hand_sample(RobotLayout().hand_dof)

            scalar_frame = build_training_frame(RobotLayout(), arms, hands)
            images = {}
            for name in self.camera_names:
                sample = matched[f"camera_{name}"]
                source_times[f"camera_{name}"] = sample.timestamp
                if isinstance(sample.value, CompressedImage):
                    images[name] = decode_compressed_image(sample.value)
                elif isinstance(sample.value, np.ndarray):
                    images[name] = np.ascontiguousarray(
                        sample.value, dtype=np.uint8
                    )
                else:
                    images[name] = decode_raw_image(sample.value)
            errors = [abs(value - anchor.timestamp) for key, value in source_times.items() if key != "anchor"]
            max_error = max(errors, default=0.0)
            self._sync_error_sum += max_error
            self._sync_error_max = max(self._sync_error_max, max_error)
            # Optional diagnostics are aligned only after the required-source
            # match and never participate in skip/readiness/sync-error logic.
            optional_frame = empty_optional_frame(self.optional_specs)
            teleop_alignment: dict[str, Any] = {}
            try:
                optional_frame, teleop_alignment = self._aligned_teleop_diagnostics(
                    anchor.timestamp
                )
            except Exception as exc:
                self._warn_diagnostic(
                    f"Teleop diagnostics fell back to invalid fill values: {exc}"
                )
            if self.teleop_diagnostics_enabled:
                source_times["teleop_diagnostics"] = teleop_alignment
            auxiliary_images: dict[str, np.ndarray] = {}
            auxiliary_timestamps: dict[str, float] = {}
            auxiliary_sequences: dict[str, int] = {}
            try:
                (
                    auxiliary_images,
                    auxiliary_timestamps,
                    auxiliary_sequences,
                ) = self._aligned_auxiliary_camera(anchor.timestamp)
            except Exception as exc:
                self._warn_diagnostic(
                    "Auxiliary cameras fell back to missing for this "
                    f"training frame: {exc}"
                )
            source_times["system"] = time.time()
            writer.append(
                scalar_frame,
                images,
                source_times,
                optional_frame=optional_frame,
                auxiliary_images=auxiliary_images,
                auxiliary_timestamps=auxiliary_timestamps,
                auxiliary_sequences=auxiliary_sequences,
            )
        except Exception as exc:
            self._sync_skip_count += 1
            self.get_logger().error(f"Failed to record synchronized frame: {exc}")

    def shutdown(self) -> None:
        for reader in getattr(self, "_camera_readers", {}).values():
            try:
                reader.close()
            except (Exception, KeyboardInterrupt):
                # A second launch SIGINT must not prevent the recorder's
                # existing discard/close semantics from running.
                pass
        for reader in getattr(
            self, "_auxiliary_camera_readers", {}
        ).values():
            try:
                reader.close()
            except (Exception, KeyboardInterrupt):
                pass
        try:
            ok, message = self.discard_unsaved()
        except Exception as exc:
            self.get_logger().error(
                f"Recorder shutdown could not discard unsaved data: {exc}"
            )
            return
        if ok and message != "no unsaved episode":
            self.get_logger().warning(
                f"Recorder shutdown discarded unsaved data: {message}"
            )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tianji + Wuji ROS 2 recorder")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--camera-transport",
        choices=CAMERA_TRANSPORT_CHOICES,
        default=None,
        help="direct shared-memory camera path (default) or ROS migration fallback",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument(
        "--active-hand",
        choices=ACTIVE_HAND_CHOICES,
        default="both",
        help=(
            "physical Wuji Hand side(s) connected; the other side is stored "
            "as 20 zeros while the 54-D schema is preserved"
        ),
    )
    parser.add_argument(
        "--active-arm",
        choices=ACTIVE_HAND_CHOICES,
        default="both",
        help="Tianji arm side(s) controlled during this episode",
    )
    return parser.parse_args(argv)


def main(argv=None):
    import sys

    raw_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    config = load_config(args.config)
    rclpy.init(args=raw_argv)
    node = TeleopRecorderNode(
        config,
        require_cameras=not args.no_camera,
        active_hand=args.active_hand,
        active_arm=args.active_arm,
        output_dir=args.output_dir,
        task_name=args.task_name,
        camera_transport=args.camera_transport,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
