"""Tracker-to-Tianji alignment visualization with no robot hardware node."""

from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from wuji_teleop_bringup.tf_utils import (
    create_chest_tf_nodes,
    create_tianji_tf_nodes,
)


def _make_ghost_description(robot_description: str) -> str:
    """Tint every visual cyan while preserving the validated URDF geometry."""
    root = ET.fromstring(robot_description)
    for visual in root.findall(".//visual"):
        material = visual.find("material")
        if material is None:
            material = ET.SubElement(visual, "material")
        color = material.find("color")
        if color is None:
            color = ET.SubElement(material, "color")
        color.set("rgba", "0.05 0.85 1.0 1.0")
    return ET.tostring(root, encoding="unicode")


def generate_launch_description():
    bringup_share = Path(get_package_share_directory("wuji_teleop_bringup"))
    description_share = Path(get_package_share_directory("tianji_description"))
    openvr_share = Path(get_package_share_directory("openvr_input"))
    tianji_share = Path(get_package_share_directory("tianji_output"))

    robot_description = (
        description_share / "urdf" / "tianji_dual_arm.urdf"
    ).read_text()
    ghost_description = _make_ghost_description(robot_description)
    mapping_nodes = create_chest_tf_nodes() + create_tianji_tf_nodes()

    return LaunchDescription([
        DeclareLaunchArgument(
            "start_openvr", default_value="false",
            description="Start OpenVR input; leave false when tracker input is already running"),
        DeclareLaunchArgument(
            "publish_mapping_tf", default_value="true",
            description="Publish chest/wrist mapping TFs used by the official pipeline"),
        DeclareLaunchArgument(
            "publish_joint_previews", default_value="true",
            description=(
                "Use offline init/IK previews before the controller starts; "
                "the visualizer automatically hands over when it detects the "
                "real controller")),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument(
            "openvr_config",
            default_value=str(openvr_share / "config" / "openvr_input.yaml")),

        Node(
            package="openvr_input",
            executable="openvr_input",
            name="openvr_input",
            arguments=["-c", LaunchConfiguration("openvr_config")],
            condition=IfCondition(LaunchConfiguration("start_openvr")),
            output="screen",
        ),
        GroupAction(
            actions=mapping_nodes,
            condition=IfCondition(LaunchConfiguration("publish_mapping_tf")),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="tianji_alignment_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", "/tianji_alignment/joint_states")],
            output="screen",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace="tianji_alignment/ghost",
            name="robot_state_publisher",
            parameters=[{
                "robot_description": ghost_description,
                "frame_prefix": "ghost_",
            }],
            remappings=[
                ("joint_states", "/tianji_alignment/target_joint_states")],
            output="screen",
        ),
        Node(
            package="wuji_teleop_bringup",
            executable="tracker_alignment_visualizer",
            name="tracker_alignment_visualizer",
            parameters=[{
                "static_config": str(
                    bringup_share / "config" / "static_transforms.yaml"),
                "arm_config": str(
                    tianji_share / "config" / "tianji_chest.yaml"),
                "kinematics_config": str(
                    tianji_share / "config" / "ccs_m6.MvKDCfg"),
                "anchor_up": "y_up",
                "publish_joint_previews": ParameterValue(
                    LaunchConfiguration("publish_joint_previews"),
                    value_type=bool),
            }],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="tracker_alignment_rviz",
            arguments=["-d", str(
                description_share / "rviz" / "tracker_alignment.rviz")],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
