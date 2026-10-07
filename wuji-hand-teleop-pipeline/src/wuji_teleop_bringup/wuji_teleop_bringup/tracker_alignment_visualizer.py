"""Visualize tracker arms against the Tianji init pose without robot control."""

from __future__ import annotations

import copy
import ctypes
import logging
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


SHOULDER_MID_LINK_BASE = np.array([0.0, 0.0, 1.121], dtype=float)


@contextmanager
def suppress_native_stdout():
    """Silence verbose C-library stdout while preserving stderr and ROS logs."""
    libc = ctypes.CDLL(None)
    libc.fflush(None)
    sys.stdout.flush()
    saved_stdout = os.dup(1)
    null_stdout = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_stdout, 1)
        yield
    finally:
        libc.fflush(None)
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        os.close(null_stdout)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the ROS fixed-axis XYZ rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def quaternion_to_matrix(quat_xyzw) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat_xyzw, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero-length quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=float)


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
    m = np.asarray(rotation, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                          (m[0, 2] + m[2, 0]) / s,
                          (m[2, 1] - m[1, 2]) / s])
        elif i == 1:
            s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                          (m[1, 2] + m[2, 1]) / s,
                          (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            q = np.array([(m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, 0.25 * s,
                          (m[1, 0] - m[0, 1]) / s])
    return q / np.linalg.norm(q)


def compute_chest_anchor(
    chest_rotation: np.ndarray,
    chest_position: np.ndarray,
    chest_config: dict,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place the robot shoulder midpoint over the virtual human shoulders."""
    up = np.asarray(up, dtype=float)
    up /= np.linalg.norm(up)
    shared = np.array([
        chest_config["shared"]["x"],
        chest_config["shared"]["y"],
        chest_config["shared"]["z"],
    ], dtype=float)

    side_origins = {}
    for side in ("left", "right"):
        cfg = chest_config[side]
        rotation = rpy_to_matrix(cfg["roll"], cfg["pitch"], cfg["yaw"])
        local = np.array([
            cfg["local_x"], cfg["local_y"], cfg["local_z"]
        ], dtype=float)
        side_origins[side] = shared + rotation @ local

    operator_left = chest_rotation @ (
        side_origins["left"] - side_origins["right"])
    horizontal_left = operator_left - np.dot(operator_left, up) * up
    horizontal_norm = float(np.linalg.norm(horizontal_left))
    if horizontal_norm < 0.2 * float(np.linalg.norm(operator_left)):
        raise ValueError("virtual shoulder axis is too close to vertical")
    horizontal_left /= horizontal_norm

    base_rotation = np.column_stack([
        np.cross(horizontal_left, up), horizontal_left, up
    ])
    shoulder_world = chest_position + chest_rotation @ shared
    base_position = shoulder_world - base_rotation @ SHOULDER_MID_LINK_BASE
    return base_position, base_rotation


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest angular distance between two rotation matrices."""
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class TrackerAlignmentVisualizer(Node):
    """Publish a fixed init robot, tracker skeleton, and live TCP errors."""

    def __init__(self) -> None:
        super().__init__("tracker_alignment_visualizer")
        self.declare_parameter("static_config", "")
        self.declare_parameter("arm_config", "")
        self.declare_parameter("kinematics_config", "")
        self.declare_parameter("anchor_up", "y_up")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("position_good_m", 0.05)
        self.declare_parameter("rotation_good_deg", 15.0)
        self.declare_parameter("publish_joint_previews", True)

        static_path = Path(str(self.get_parameter("static_config").value))
        arm_path = Path(str(self.get_parameter("arm_config").value))
        kinematics_path = Path(str(
            self.get_parameter("kinematics_config").value))
        if (not static_path.is_file() or not arm_path.is_file()
                or not kinematics_path.is_file()):
            raise RuntimeError(
                "configuration missing: "
                f"static={static_path}, arm={arm_path}, kine={kinematics_path}")
        with static_path.open() as stream:
            self._static_config = yaml.safe_load(stream)
        with arm_path.open() as stream:
            arm_config = yaml.safe_load(stream)

        anchor_up = str(self.get_parameter("anchor_up").value)
        if anchor_up not in ("y_up", "z_up"):
            raise ValueError("anchor_up must be y_up or z_up")
        self._up = np.array([0.0, 1.0, 0.0] if anchor_up == "y_up"
                            else [0.0, 0.0, 1.0], dtype=float)
        self._position_good_m = float(
            self.get_parameter("position_good_m").value)
        self._rotation_good_deg = float(
            self.get_parameter("rotation_good_deg").value)
        self._publish_joint_previews = bool(
            self.get_parameter("publish_joint_previews").value)
        rate = float(self.get_parameter("publish_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")

        init = arm_config["init_joints"]
        self._init_degrees = {
            "left": [float(value) for value in init["left"]],
            "right": [float(value) for value in init["right"]],
        }
        self._init_names = (
            [f"Joint{i}_L" for i in range(1, 8)]
            + [f"Joint{i}_R" for i in range(1, 8)])
        self._init_positions = [
            math.radians(value) for value in init["left"] + init["right"]
        ]
        self._kinematics = (
            self._initialize_kinematics(kinematics_path)
            if self._publish_joint_previews else {})
        self._ik_status = {"left": "waiting", "right": "waiting"}
        self._teleop_status = (
            "OFFLINE PREVIEW: no hardware commands"
            if self._publish_joint_previews
            else "VISUALIZATION: waiting for controller status")
        self._controller_online = False

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._static_broadcaster = StaticTransformBroadcaster(self)
        joint_qos = QoSProfile(depth=1)
        joint_qos.reliability = ReliabilityPolicy.RELIABLE
        joint_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._joint_publisher = None
        self._target_joint_publisher = None
        if self._publish_joint_previews:
            self._joint_publisher = self.create_publisher(
                JointState, "/tianji_alignment/joint_states", joint_qos)
            self._target_joint_publisher = self.create_publisher(
                JointState, "/tianji_alignment/target_joint_states", joint_qos)
        self.create_subscription(
            String, "/tianji_arm/teleop_status", self._status_callback, joint_qos)
        self._marker_publisher = self.create_publisher(
            MarkerArray, "/tianji_alignment/markers", 10)
        self._anchor_published = False
        self._waiting_logged = False
        self._timer = self.create_timer(1.0 / rate, self._update)
        source = ("offline init/IK preview" if self._publish_joint_previews
                  else "controller actual/raw-target topics")
        self.get_logger().info(
            f"Visualization-only mode: {source}; no robot driver or hardware "
            "command interface is used")

    def _status_callback(self, message: String) -> None:
        self._teleop_status = str(message.data)

    def _update_controller_source(self) -> None:
        online = self.count_publishers("/tianji_arm/teleop_status") > 0
        if online == self._controller_online:
            return
        self._controller_online = online
        if online:
            self.get_logger().info(
                "Controller detected; offline joint preview paused and RViz "
                "handed over to controller actual/raw-target topics")
        else:
            self._teleop_status = "OFFLINE PREVIEW: no hardware commands"
            self.get_logger().info(
                "Controller not present; offline libKine preview resumed")

    def _initialize_kinematics(self, config_path: Path) -> dict:
        """Load only libKine; this path never constructs Marvin_Robot."""
        from tianji_output._internal.fx_kine import Marvin_Kine
        logging.getLogger("debug_printer").setLevel(logging.CRITICAL)

        kinematics = {}
        for side, serial in (("left", 0), ("right", 1)):
            with suppress_native_stdout():
                kine = Marvin_Kine()
                if hasattr(kine.kine, "FX_LOG_SWITCH"):
                    kine.kine.FX_LOG_SWITCH.argtypes = [ctypes.c_int32]
                    kine.kine.FX_LOG_SWITCH(ctypes.c_int32(0))
                config = kine.load_config(config_path=str(config_path))
                initialized = kine.initial_kine(
                    robot_serial=serial,
                    robot_type=config["TYPE"][serial],
                    dh=config["DH"][serial],
                    pnva=config["PNVA"][serial],
                    j67=config["BD"][serial],
                )
                if not initialized:
                    raise RuntimeError(
                        f"offline kinematics init failed for {side}")
                kine.set_tool_kine(serial, np.eye(4).tolist())
            kinematics[side] = kine
        self.get_logger().info(
            "Offline Tianji kinematics initialized (libKine only; no robot connection)")
        return kinematics

    def _lookup_matrix(
        self, child: str, parent: str = "world"
    ) -> Optional[np.ndarray]:
        try:
            transform = self._tf_buffer.lookup_transform(
                parent, child, Time(), timeout=Duration(seconds=0.02))
        except Exception:
            return None
        t = transform.transform.translation
        q = transform.transform.rotation
        matrix = np.eye(4)
        matrix[:3, :3] = quaternion_to_matrix([q.x, q.y, q.z, q.w])
        matrix[:3, 3] = [t.x, t.y, t.z]
        return matrix

    def _publish_anchor(self) -> None:
        if self._anchor_published:
            return
        chest = self._lookup_matrix("chest")
        if chest is None:
            if not self._waiting_logged:
                self.get_logger().info("Waiting for world -> chest TF")
                self._waiting_logged = True
            return
        try:
            position, rotation = compute_chest_anchor(
                chest[:3, :3], chest[:3, 3],
                self._static_config["chest_mount"], self._up)
        except ValueError as error:
            self.get_logger().warning(f"Cannot anchor robot model: {error}")
            return

        quaternion = matrix_to_quaternion(rotation)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = "Link_Base"
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])
        ghost_transform = copy.deepcopy(transform)
        ghost_transform.child_frame_id = "ghost_Link_Base"
        self._static_broadcaster.sendTransform([transform, ghost_transform])
        self._anchor_published = True
        self.get_logger().info(
            "Robot init model anchored to the current chest pose; restart "
            "this launch to capture a new anchor")

    def _publish_init_joints(self) -> None:
        if self._joint_publisher is None:
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = self._init_names
        message.position = self._init_positions
        self._joint_publisher.publish(message)

    def _solve_tracker_ik(self, frames: dict[str, Optional[np.ndarray]]) -> list:
        """Run the same chest-frame IK inputs as teleop, without dispatching."""
        target_positions = []
        for side, serial in (("left", 0), ("right", 1)):
            chest = frames.get(f"{side}_chest")
            target_world = frames.get(f"tianji_{side}")
            arm_world = frames.get(f"{side}_arm")
            if chest is None or target_world is None or arm_world is None:
                self._ik_status[side] = "missing TF"
                target_positions.extend(
                    math.radians(value) for value in self._init_degrees[side])
                continue

            target = np.linalg.inv(chest) @ target_world
            target[:3, 3] *= 1000.0
            arm_in_chest = np.linalg.inv(chest) @ arm_world
            direction = arm_in_chest[:3, 1]
            zsp = [float(value) for value in direction] + [0.0, 0.0, 0.0]
            try:
                with suppress_native_stdout():
                    result = self._kinematics[side].ik(
                        robot_serial=serial,
                        pose_mat=target.tolist(),
                        ref_joints=self._init_degrees[side],
                        zsp_type=1,
                        zsp_para=zsp,
                        zsp_angle=0.0,
                        dgr=[5.0, 5.0, 5.0],
                    )
                if result is False:
                    reason = "no solution"
                    joints = self._init_degrees[side]
                elif bool(result.m_Output_IsOutRange):
                    reason = "out of range"
                    joints = self._init_degrees[side]
                elif bool(result.m_Output_IsJntExd):
                    reason = "joint limit"
                    joints = self._init_degrees[side]
                else:
                    reason = "OK"
                    joints = result.m_Output_RetJoint.to_list()
                self._ik_status[side] = reason
                target_positions.extend(math.radians(value) for value in joints)
            except Exception as error:
                self._ik_status[side] = f"error: {type(error).__name__}"
                target_positions.extend(
                    math.radians(value) for value in self._init_degrees[side])
        return target_positions

    def _publish_target_joints(self, frames: dict[str, Optional[np.ndarray]]) -> None:
        if self._target_joint_publisher is None:
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = self._init_names
        message.position = self._solve_tracker_ik(frames)
        self._target_joint_publisher.publish(message)

    @staticmethod
    def _point(position) -> Point:
        point = Point()
        point.x, point.y, point.z = map(float, position)
        return point

    def _base_marker(self, marker_id: int, marker_type: int, namespace: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=0.3).to_msg()
        return marker

    @staticmethod
    def _set_color(marker: Marker, rgba) -> None:
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba

    def _skeleton_markers(self, frames: dict[str, np.ndarray]) -> list[Marker]:
        required = ("chest", "left_chest", "right_chest", "left_arm",
                    "right_arm", "left_wrist", "right_wrist")
        if any(frames.get(name) is None for name in required):
            return []

        line = self._base_marker(0, Marker.LINE_LIST, "tracker_skeleton")
        line.scale.x = 0.04
        self._set_color(line, (1.0, 0.38, 0.05, 0.95))
        segments = [
            ("left_chest", "right_chest"),
            ("chest", "left_chest"), ("chest", "right_chest"),
            ("left_chest", "left_arm"), ("left_arm", "left_wrist"),
            ("right_chest", "right_arm"), ("right_arm", "right_wrist"),
        ]
        for start, end in segments:
            line.points.append(self._point(frames[start][:3, 3]))
            line.points.append(self._point(frames[end][:3, 3]))

        spheres = self._base_marker(1, Marker.SPHERE_LIST, "tracker_skeleton")
        spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.085
        self._set_color(spheres, (1.0, 0.55, 0.12, 0.95))
        for frame in required:
            spheres.points.append(self._point(frames[frame][:3, 3]))
        return [line, spheres]

    def _alignment_markers(
        self, side: str, target: Optional[np.ndarray], reference: Optional[np.ndarray]
    ) -> list[Marker]:
        if target is None or reference is None:
            return []
        index = 10 if side == "left" else 20
        distance = float(np.linalg.norm(target[:3, 3] - reference[:3, 3]))
        angle = rotation_error_deg(reference[:3, :3], target[:3, :3])
        good = distance <= self._position_good_m and angle <= self._rotation_good_deg
        near = (distance <= 2.0 * self._position_good_m
                and angle <= 2.0 * self._rotation_good_deg)
        color = ((0.15, 0.9, 0.25, 1.0) if good else
                 (1.0, 0.72, 0.05, 1.0) if near else
                 (0.95, 0.12, 0.08, 1.0))

        line = self._base_marker(index, Marker.LINE_LIST, "alignment_error")
        line.scale.x = 0.018
        self._set_color(line, color)
        line.points = [
            self._point(reference[:3, 3]), self._point(target[:3, 3])]

        reference_sphere = self._base_marker(
            index + 1, Marker.SPHERE, "alignment_reference")
        reference_sphere.pose.position = self._point(reference[:3, 3])
        reference_sphere.scale.x = reference_sphere.scale.y = 0.075
        reference_sphere.scale.z = 0.025
        self._set_color(reference_sphere, (0.15, 0.75, 1.0, 0.9))

        target_sphere = self._base_marker(
            index + 2, Marker.SPHERE, "alignment_error")
        target_sphere.pose.position = self._point(target[:3, 3])
        target_sphere.scale.x = target_sphere.scale.y = target_sphere.scale.z = 0.065
        self._set_color(target_sphere, color)

        label = self._base_marker(index + 3, Marker.TEXT_VIEW_FACING,
                                  "alignment_error")
        midpoint = 0.5 * (reference[:3, 3] + target[:3, 3])
        label.pose.position = self._point(midpoint + self._up * 0.13)
        label.scale.z = 0.07
        self._set_color(label, color)
        label.text = (
            f"{side.upper()}: {distance * 1000:.0f} mm | {angle:.1f} deg"
            f"\nIK: {self._ik_status[side]}")
        return [line, reference_sphere, target_sphere, label]

    def _status_marker(self, chest: Optional[np.ndarray]) -> Optional[Marker]:
        if chest is None:
            return None
        marker = self._base_marker(40, Marker.TEXT_VIEW_FACING, "teleop_status")
        marker.pose.position = self._point(chest[:3, 3] + self._up * 0.35)
        marker.scale.z = 0.075
        status = self._teleop_status
        if status.startswith("READY"):
            color = (0.15, 0.9, 0.25, 1.0)
        elif status.startswith("HOLD"):
            color = (1.0, 0.72, 0.05, 1.0)
        elif status.startswith("ERROR"):
            color = (0.95, 0.12, 0.08, 1.0)
        else:
            color = (0.75, 0.8, 0.85, 1.0)
        self._set_color(marker, color)
        marker.text = status
        return marker

    def _publish_markers(self) -> None:
        names = ("chest", "left_chest", "right_chest", "left_arm",
                 "right_arm", "left_wrist", "right_wrist",
                 "tianji_left", "tianji_right")
        frames = {name: self._lookup_matrix(name) for name in names}
        if self._publish_joint_previews and not self._controller_online:
            self._publish_target_joints(frames)
        markers = self._skeleton_markers(frames)
        if self._publish_joint_previews and not self._controller_online:
            left_base_tcp = self._lookup_matrix("TCP_Link_L", "Base_L")
            right_base_tcp = self._lookup_matrix("TCP_Link_R", "Base_R")
            left_reference = (
                frames["left_chest"] @ left_base_tcp
                if frames["left_chest"] is not None and left_base_tcp is not None
                else None)
            right_reference = (
                frames["right_chest"] @ right_base_tcp
                if frames["right_chest"] is not None and right_base_tcp is not None
                else None)
            markers.extend(self._alignment_markers(
                "left", frames["tianji_left"], left_reference))
            markers.extend(self._alignment_markers(
                "right", frames["tianji_right"], right_reference))
        status = self._status_marker(frames.get("chest"))
        if status is not None:
            markers.append(status)
        if markers:
            self._marker_publisher.publish(MarkerArray(markers=markers))

    def _update(self) -> None:
        self._update_controller_source()
        if self._publish_joint_previews and not self._controller_online:
            self._publish_init_joints()
        self._publish_anchor()
        if self._anchor_published:
            self._publish_markers()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackerAlignmentVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
