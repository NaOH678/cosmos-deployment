"""Launch the MANUS skeleton MarkerArray converter and RViz."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("manus_ros2")
    rviz_config = os.path.join(package_share, "rviz", "manus_skeleton.rviz")

    frame_id = LaunchConfiguration("frame_id")
    show_labels = LaunchConfiguration("show_labels")
    separate_hands = LaunchConfiguration("separate_hands")
    hand_separation = LaunchConfiguration("hand_separation")
    max_rate_hz = LaunchConfiguration("max_rate_hz")
    start_rviz = LaunchConfiguration("rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("frame_id", default_value="manus_world"),
            DeclareLaunchArgument("show_labels", default_value="true"),
            DeclareLaunchArgument("separate_hands", default_value="true"),
            DeclareLaunchArgument("hand_separation", default_value="0.28"),
            DeclareLaunchArgument("max_rate_hz", default_value="60.0"),
            DeclareLaunchArgument("rviz", default_value="true"),
            Node(
                package="manus_ros2",
                executable="manus_rviz_visualizer",
                name="manus_rviz_visualizer",
                output="screen",
                parameters=[
                    {
                        "frame_id": frame_id,
                        "show_labels": ParameterValue(show_labels, value_type=bool),
                        "separate_hands": ParameterValue(
                            separate_hands, value_type=bool
                        ),
                        "hand_separation": ParameterValue(
                            hand_separation, value_type=float
                        ),
                        "max_rate_hz": ParameterValue(max_rate_hz, value_type=float),
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="manus_rviz",
                arguments=["-d", rviz_config],
                output="screen",
                condition=IfCondition(start_rviz),
            ),
        ]
    )
