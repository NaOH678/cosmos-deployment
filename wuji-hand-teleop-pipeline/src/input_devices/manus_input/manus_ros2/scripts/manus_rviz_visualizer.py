#!/usr/bin/env python3
"""Convert MANUS glove skeleton messages to RViz MarkerArray messages."""

import math
from typing import Dict, Iterable, Tuple

import rclpy
from geometry_msgs.msg import Point
from manus_ros2_msgs.msg import ManusGlove
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from visualization_msgs.msg import Marker, MarkerArray


Color = Tuple[float, float, float, float]


class ManusRvizVisualizer(Node):
    """Render MANUS joints and parent-child bones as RViz markers."""

    def __init__(self) -> None:
        super().__init__("manus_rviz_visualizer")

        self.declare_parameter("frame_id", "manus_world")
        self.declare_parameter("marker_topic", "/manus/skeleton_markers")
        self.declare_parameter("joint_scale", 0.012)
        self.declare_parameter("bone_scale", 0.005)
        self.declare_parameter("max_rate_hz", 60.0)
        self.declare_parameter("separate_hands", True)
        self.declare_parameter("hand_separation", 0.28)
        self.declare_parameter("show_labels", True)

        self._frame_id = str(self.get_parameter("frame_id").value)
        marker_topic = str(self.get_parameter("marker_topic").value)
        self._joint_scale = float(self.get_parameter("joint_scale").value)
        self._bone_scale = float(self.get_parameter("bone_scale").value)
        self._separate_hands = bool(self.get_parameter("separate_hands").value)
        self._hand_separation = float(self.get_parameter("hand_separation").value)
        self._show_labels = bool(self.get_parameter("show_labels").value)

        max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self._min_period_ns = (
            int(1_000_000_000 / max_rate_hz) if max_rate_hz > 0.0 else 0
        )
        self._last_publish_ns: Dict[str, int] = {}

        marker_qos = QoSProfile(depth=5)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        self._publisher = self.create_publisher(
            MarkerArray, marker_topic, marker_qos
        )

        self._subscriptions = [
            self.create_subscription(
                ManusGlove,
                topic,
                lambda msg, source_topic=topic: self._on_glove(msg, source_topic),
                qos_profile_sensor_data,
            )
            for topic in ("/manus_glove_0", "/manus_glove_1")
        ]

        self.get_logger().info(
            "Visualizing /manus_glove_0 and /manus_glove_1 -> "
            f"{marker_topic} (fixed frame: {self._frame_id})"
        )

    @staticmethod
    def _color_for_side(side: str) -> Color:
        side_lower = side.lower()
        if side_lower == "left":
            return (0.15, 0.55, 1.0, 1.0)
        if side_lower == "right":
            return (1.0, 0.35, 0.10, 1.0)
        return (0.30, 1.0, 0.45, 1.0)

    def _x_offset(self, side: str) -> float:
        if not self._separate_hands:
            return 0.0
        if side.lower() == "left":
            return -0.5 * self._hand_separation
        if side.lower() == "right":
            return 0.5 * self._hand_separation
        return 0.0

    @staticmethod
    def _set_color(marker: Marker, color: Color) -> None:
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

    def _base_marker(self, namespace: str, marker_id: int, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        # Expire quickly when a glove stops publishing instead of leaving a
        # frozen hand in RViz indefinitely.
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 300_000_000
        return marker

    @staticmethod
    def _valid_position(node) -> bool:
        position = node.pose.position
        return all(
            math.isfinite(value)
            for value in (position.x, position.y, position.z)
        )

    @staticmethod
    def _point(node, x_offset: float) -> Point:
        point = Point()
        point.x = node.pose.position.x + x_offset
        point.y = node.pose.position.y
        point.z = node.pose.position.z
        return point

    def _build_markers(self, msg: ManusGlove) -> Iterable[Marker]:
        side = msg.side or "unknown"
        namespace = f"manus_{msg.glove_id}_{side.lower()}"
        color = self._color_for_side(side)
        x_offset = self._x_offset(side)

        valid_nodes = [node for node in msg.raw_nodes if self._valid_position(node)]
        points_by_id = {
            node.node_id: self._point(node, x_offset) for node in valid_nodes
        }

        joints = self._base_marker(namespace, 0, Marker.SPHERE_LIST)
        joints.scale.x = self._joint_scale
        joints.scale.y = self._joint_scale
        joints.scale.z = self._joint_scale
        self._set_color(joints, color)
        joints.points = list(points_by_id.values())
        yield joints

        bones = self._base_marker(namespace, 1, Marker.LINE_LIST)
        bones.scale.x = self._bone_scale
        self._set_color(bones, color)
        for node in valid_nodes:
            if node.parent_node_id == node.node_id:
                continue
            parent = points_by_id.get(node.parent_node_id)
            child = points_by_id.get(node.node_id)
            if parent is not None and child is not None:
                bones.points.extend((parent, child))
        yield bones

        if self._show_labels:
            label = self._base_marker(namespace, 2, Marker.TEXT_VIEW_FACING)
            label.pose.position.x = x_offset
            label.pose.position.y = 0.0
            label.pose.position.z = -0.035
            label.scale.z = 0.022
            self._set_color(label, color)
            label.text = f"{side} glove {msg.glove_id} ({len(valid_nodes)} joints)"
            yield label

    def _on_glove(self, msg: ManusGlove, source_topic: str) -> None:
        key = f"{source_topic}:{msg.glove_id}"
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._last_publish_ns.get(key, 0)
        if self._min_period_ns and now_ns - last_ns < self._min_period_ns:
            return
        self._last_publish_ns[key] = now_ns

        if not msg.raw_nodes:
            self.get_logger().warning(
                f"Received empty skeleton from {source_topic}",
                throttle_duration_sec=5.0,
            )
            return

        output = MarkerArray()
        output.markers = list(self._build_markers(msg))
        self._publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManusRvizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
