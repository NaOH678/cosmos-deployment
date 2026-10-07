"""ROS2 node that streams OpenVR Tracker data and publishes to ROS topics and TF."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ament_index_python.packages import get_package_share_directory

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to load configuration files.") from exc

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import TransformStamped
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension
    from tf2_ros import TransformBroadcaster
    from rclpy.utilities import remove_ros_args
except ImportError as exc:
    raise ImportError("This module requires ROS 2 Python packages.") from exc

from .openvr_tracker_wrapper import OpenVRTrackerWrapper


TRACKER_DIAGNOSTIC_ROLES = (
    "chest",
    "left_wrist",
    "right_wrist",
    "left_arm",
    "right_arm",
)
TRACKER_DIAGNOSTIC_COLUMNS = 26


@dataclass
class OpenVRInputConfig:
    """Configuration for the OpenVR input streaming node."""

    config_path: Optional[str] = None

    # Tracker serial numbers
    tracker_serials: Optional[Dict[str, str]] = None
    # e.g., {'chest': 'LHR-xxx', 'right_wrist': 'LHR-yyy', 'left_wrist': 'LHR-zzz'}

    # Wrist tracker offset in tracker's local frame [x, y, z] (meters)
    # Applied before coordinate correction to map tracker position to actual wrist
    wrist_offset: Optional[list] = None

    # Publishing settings
    publish_rate_hz: float = 120.0

    # TF settings
    publish_tf: bool = True
    parent_frame: str = "world"

    # Best-effort recorder input.  This is a separate publisher and never
    # participates in the established TF/control path.
    publish_diagnostics: bool = True
    diagnostics_topic: str = "/openvr/tracker_diagnostics"
    diagnostics_rate_hz: float = 30.0

    # Chest to head offset (meters)
    chest_to_head_offset_z: float = 0.3

    @classmethod
    def from_file(cls, path: str | Path) -> "OpenVRInputConfig":
        """Load configuration from a YAML file."""
        cfg_path = Path(path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, dict):
            raise ValueError("OpenVRInputConfig expects a mapping at the root of the YAML file.")

        valid_fields = {field.name for field in fields(cls)}
        data: Dict[str, Any] = {"config_path": str(cfg_path)}
        ignored_keys: list[str] = []

        for key, value in raw.items():
            if key in valid_fields:
                data[key] = value
            else:
                ignored_keys.append(key)

        if ignored_keys:
            print(f"[OpenVRInputConfig] Ignoring unknown keys: {', '.join(sorted(ignored_keys))}")

        return cls(**data)


class OpenVRInputNode(Node):
    """ROS2 node that reads OpenVR Tracker data and publishes TF."""

    def __init__(self, config: OpenVRInputConfig):
        super().__init__("openvr_input")
        self.config = config

        if not self.config.tracker_serials:
            raise ValueError(
                "tracker_serials must be specified in config. "
                "Run list_trackers.py to get your tracker serial numbers."
            )

        self.get_logger().info(
            f"[OpenVR Input] Initializing with trackers: {list(self.config.tracker_serials.keys())}"
        )
        if self.config.wrist_offset:
            self.get_logger().info(
                f"[OpenVR Input] Wrist offset (local frame): {self.config.wrist_offset}"
            )

        # Initialize OpenVR tracker wrapper
        self.tracker_wrapper = OpenVRTrackerWrapper(
            self.config.tracker_serials,
            wrist_offset=self.config.wrist_offset
        )

        # Create TF broadcaster
        if self.config.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        else:
            self.tf_broadcaster = None

        self._diagnostics_publisher = None
        self._last_diagnostics_error_ns = 0
        self._next_diagnostics_publish_ns = 0
        self._diagnostics_period_ns = int(
            1e9 / max(self.config.diagnostics_rate_hz, 1.0)
        )
        if self.config.publish_diagnostics:
            self._diagnostics_publisher = self.create_publisher(
                Float64MultiArray,
                self.config.diagnostics_topic,
                qos_profile_sensor_data,
            )

        # Create timer for periodic publishing
        self.timer = self.create_timer(
            1.0 / max(self.config.publish_rate_hz, 1.0),
            self._publish_callback
        )

        self.get_logger().info(
            f"[OpenVR Input] Publishing TF at {self.config.publish_rate_hz:.1f} Hz"
        )
        self.get_logger().info("[OpenVR Input] Publishing TF: head, chest, right_wrist, left_wrist, left_arm, right_arm")
        if self._diagnostics_publisher is not None:
            self.get_logger().info(
                f"[OpenVR Input] Publishing recorder diagnostics: "
                f"{self.config.diagnostics_topic} at "
                f"{self.config.diagnostics_rate_hz:.1f} Hz"
            )

    def _publish_callback(self) -> None:
        """Timer callback to fetch and publish tracker data as TF."""
        if self.tf_broadcaster is None:
            return

        try:
            poses = self.tracker_wrapper.get_poses()
        except RuntimeError as e:
            self.get_logger().error(f"Failed to get tracker poses: {e}")
            return

        current_time = self.get_clock().now().to_msg()

        # Process chest -> head (with Z offset along chest's local Z axis)
        chest_pose = poses.get('chest')
        if chest_pose is not None:
            # Publish chest TF
            self._broadcast_tf(
                chest_pose,
                self.config.parent_frame,
                "chest",
                current_time
            )

            # Create head pose by adding offset along chest's local Z axis
            head_pose = chest_pose.copy()
            chest_z_axis = chest_pose[:3, 2]  # Local Z axis direction in world frame
            head_pose[:3, 3] += chest_z_axis * self.config.chest_to_head_offset_z

            # Publish head TF
            self._broadcast_tf(
                head_pose,
                self.config.parent_frame,
                "head",
                current_time
            )

        # Process right wrist
        right_wrist_pose = poses.get('right_wrist')
        if right_wrist_pose is not None:
            self._broadcast_tf(
                right_wrist_pose,
                self.config.parent_frame,
                "right_wrist",
                current_time
            )

        # Process left wrist
        left_wrist_pose = poses.get('left_wrist')
        if left_wrist_pose is not None:
            self._broadcast_tf(
                left_wrist_pose,
                self.config.parent_frame,
                "left_wrist",
                current_time
            )

        # Process left arm tracker (upper arm)
        left_arm_pose = poses.get('left_arm')
        if left_arm_pose is not None:
            self._broadcast_tf(
                left_arm_pose,
                self.config.parent_frame,
                "left_arm",
                current_time
            )

        # Process right arm tracker (upper arm)
        right_arm_pose = poses.get('right_arm')
        if right_arm_pose is not None:
            self._broadcast_tf(
                right_arm_pose,
                self.config.parent_frame,
                "right_arm",
                current_time
            )

        # Diagnostics are strictly best-effort and execute after the original
        # TF path.  An SDK/read/serialization error here cannot stop TF output.
        diagnostics_now_ns = self.get_clock().now().nanoseconds
        if (
            self._diagnostics_publisher is not None
            and diagnostics_now_ns >= self._next_diagnostics_publish_ns
        ):
            self._next_diagnostics_publish_ns = (
                diagnostics_now_ns + self._diagnostics_period_ns
            )
            try:
                self._publish_tracker_diagnostics(poses)
            except Exception as exc:
                if (
                    diagnostics_now_ns - self._last_diagnostics_error_ns
                    >= 5_000_000_000
                ):
                    self.get_logger().warning(
                        f"OpenVR recorder diagnostics unavailable: {exc}"
                    )
                    self._last_diagnostics_error_ns = diagnostics_now_ns

    def _publish_tracker_diagnostics(
        self, corrected_poses: Dict[str, Optional[np.ndarray]]
    ) -> None:
        raw_states = self.tracker_wrapper.get_raw_states()
        rows = np.zeros(
            (len(TRACKER_DIAGNOSTIC_ROLES), TRACKER_DIAGNOSTIC_COLUMNS),
            dtype=np.float64,
        )
        rows[:, 23:25] = -1.0
        for index, role in enumerate(TRACKER_DIAGNOSTIC_ROLES):
            state = raw_states.get(role)
            if state is not None:
                rows[index, 0:7] = self._matrix_to_pose(state["pose"])
                rows[index, 14:17] = state["linear_velocity"]
                rows[index, 17:20] = state["angular_velocity"]
                rows[index, 20] = 1.0
                rows[index, 21] = float(state["connected"])
                rows[index, 22] = float(state["valid"])
                rows[index, 23] = float(state["tracking_result"])
                rows[index, 24] = float(state["device_index"])
            corrected = corrected_poses.get(role)
            if corrected is not None and np.all(np.isfinite(corrected)):
                rows[index, 7:14] = self._matrix_to_pose(corrected)
                rows[index, 25] = 1.0

        message = Float64MultiArray()
        message.layout.dim = [
            MultiArrayDimension(
                label="tracker_role",
                size=len(TRACKER_DIAGNOSTIC_ROLES),
                stride=len(TRACKER_DIAGNOSTIC_ROLES) * TRACKER_DIAGNOSTIC_COLUMNS,
            ),
            MultiArrayDimension(
                label="raw7_corrected7_linear3_angular3_detected_connected_valid_result_index_corrected_valid",
                size=TRACKER_DIAGNOSTIC_COLUMNS,
                stride=TRACKER_DIAGNOSTIC_COLUMNS,
            ),
        ]
        message.data = rows.reshape(-1).tolist()
        self._diagnostics_publisher.publish(message)

    @classmethod
    def _matrix_to_pose(cls, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"tracker pose must be a finite 4x4 matrix, got {matrix.shape}")
        quaternion = cls._rotation_matrix_to_quaternion(matrix[:3, :3])
        return np.asarray(
            [*matrix[:3, 3], *quaternion],
            dtype=np.float64,
        )

    def _broadcast_tf(
        self,
        matrix: np.ndarray,
        parent_frame: str,
        child_frame: str,
        stamp
    ) -> None:
        """Broadcast a 4x4 matrix as a TF transform."""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # Extract translation
        t.transform.translation.x = float(matrix[0, 3])
        t.transform.translation.y = float(matrix[1, 3])
        t.transform.translation.z = float(matrix[2, 3])

        # Extract rotation and convert to quaternion
        rotation_matrix = matrix[:3, :3]
        quat = self._rotation_matrix_to_quaternion(rotation_matrix)
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([x, y, z, w])

    def shutdown(self) -> None:
        """Shutdown the node and cleanup resources."""
        if hasattr(self, 'tracker_wrapper') and self.tracker_wrapper:
            try:
                self.tracker_wrapper.shutdown()
            except Exception as exc:
                self.get_logger().error(f"Error while closing tracker wrapper: {exc}")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish OpenVR Tracker data to ROS2 topics and TF."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to an OpenVR input YAML configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    program_name = sys.argv[0] if sys.argv else "openvr_input"
    raw_argv = sys.argv if argv is None else [program_name, *argv]
    cli_argv = remove_ros_args(raw_argv)[1:]
    args = _parse_args(cli_argv)

    if args.config:
        config_path = Path(args.config)
    else:
        share_dir = Path(get_package_share_directory("openvr_input"))
        config_path = share_dir / "config" / "openvr_input.yaml"

    config = OpenVRInputConfig.from_file(str(config_path))

    rclpy.init(args=raw_argv)
    node = OpenVRInputNode(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
