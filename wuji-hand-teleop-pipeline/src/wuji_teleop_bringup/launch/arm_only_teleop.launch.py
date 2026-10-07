"""Validated HTC Tracker -> Tianji arm-only hardware graph.

OpenVR input is intentionally external and supervised by
``start_openvr_input.sh``.  This graph launches only the documented static TFs
and Tianji controller; it contains no recorder, camera, MANUS, or Wuji Hand
nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "active_arm",
            default_value="both",
            description="Tianji arm controlled by Tracker: both, left, or right",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("wuji_teleop_bringup"),
                    "launch",
                    "vive_arm_tf.launch.py",
                ])
            ])
        ),
        Node(
            package="controller",
            executable="tianji_arm_controller",
            name="tianji_arm_controller",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "auto_enable": False,
                "active_arm": LaunchConfiguration("active_arm"),
                "control_rate": 120.0,
                "state_publish_rate": 500.0,
                "teleop_position_scale": 1.25,
                "impedance_velocity_ratio": 30,
                "impedance_acceleration_ratio": 30,
                "handoff_hold_sec": 0.2,
                "handoff_ramp_sec": 1.0,
                "recovery_max_speed_deg_s": 5.0,
                "recovery_max_accel_deg_s2": 10.0,
                "impedance_max_drift_deg": 3.0,
                "performance_metrics_enabled": True,
                "performance_publish_rate": 1.0,
                "status_snapshot_rate": 2.0,
            }],
        ),
    ])
