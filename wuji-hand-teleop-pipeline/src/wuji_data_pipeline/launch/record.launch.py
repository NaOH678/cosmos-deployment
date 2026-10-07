from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pipeline_config = LaunchConfiguration("pipeline_config")
    enable_teleop = LaunchConfiguration("enable_teleop")
    enable_camera = LaunchConfiguration("enable_camera")
    camera_transport = LaunchConfiguration("camera_transport")
    camera_config = LaunchConfiguration("camera_config")
    camera_shared_memory_dir = LaunchConfiguration(
        "camera_shared_memory_dir"
    )
    active_hand = LaunchConfiguration("active_hand")
    active_arm = LaunchConfiguration("active_arm")
    output_dir = LaunchConfiguration("output_dir")
    task_name = LaunchConfiguration("task_name")
    handoff_ramp_sec = LaunchConfiguration("handoff_ramp_sec")
    enable_left_hand = PythonExpression(
        ["'", active_hand, "' in ('both', 'left')"]
    )
    enable_right_hand = PythonExpression(
        ["'", active_hand, "' in ('both', 'right')"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pipeline_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("wuji_data_pipeline"), "config", "pipeline.yaml"]
                ),
            ),
            DeclareLaunchArgument("enable_teleop", default_value="true"),
            DeclareLaunchArgument("enable_camera", default_value="true"),
            DeclareLaunchArgument(
                "camera_transport",
                default_value="direct",
                choices=["direct", "ros"],
                description=(
                    "Stage-D direct shared-memory cameras or the previous "
                    "ROS image-topic migration fallback"
                ),
            ),
            DeclareLaunchArgument(
                "camera_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("camera"), "config", "camera_config.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "camera_shared_memory_dir",
                default_value="/dev/shm/wuji_camera_v1",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="/home/wuji/datasets/tianji_wuji",
            ),
            DeclareLaunchArgument("task_name", default_value=""),
            DeclareLaunchArgument(
                "handoff_ramp_sec",
                default_value="1.0",
                description=(
                    "Tracker-to-robot handoff ramp; the data GUI explicitly "
                    "uses 6.0 while the validated CLI default stays 1.0"
                ),
            ),
            DeclareLaunchArgument(
                "active_hand",
                default_value="both",
                choices=["both", "left", "right"],
                description=(
                    "Connected physical Wuji Hand side(s); the inactive side "
                    "is zero-filled by the recorder"
                ),
            ),
            DeclareLaunchArgument(
                "active_arm",
                default_value="both",
                choices=["both", "left", "right"],
                description="Tianji arm side(s) controlled by the Tracker",
            ),
            # TIANJI_VIVE_TELEOP_RECORD.md section 5.4: only the static TFs
            # belong in this graph. OpenVR runs in the separately supervised
            # wuji-openvr-input container started by start_openvr_input.sh.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                FindPackageShare("wuji_teleop_bringup"),
                                "launch",
                                "vive_arm_tf.launch.py",
                            ]
                        )
                    ]
                ),
                condition=IfCondition(enable_teleop),
            ),
            # Exact field-validated parameters from
            # TIANJI_VIVE_TELEOP_RECORD.md section 5.5. Do not replace this
            # node with another Tianji control path.
            Node(
                package="controller",
                executable="tianji_arm_controller",
                name="tianji_arm_controller",
                parameters=[{
                    "auto_enable": False,
                    "active_arm": active_arm,
                    "control_rate": 120.0,
                    "state_publish_rate": 500.0,
                    "teleop_position_scale": 1.25,
                    "impedance_velocity_ratio": 30,
                    "impedance_acceleration_ratio": 30,
                    "handoff_hold_sec": 0.2,
                    "handoff_ramp_sec": ParameterValue(
                        handoff_ramp_sec, value_type=float
                    ),
                    "recovery_max_speed_deg_s": 5.0,
                    "recovery_max_accel_deg_s2": 10.0,
                    "impedance_max_drift_deg": 3.0,
                    # Stage-A passive timing only. One compact JSON snapshot
                    # per second; no control parameter or command is changed.
                    "performance_metrics_enabled": True,
                    "performance_publish_rate": 1.0,
                    # Stage-B GUI status is cache-only and performs no
                    # additional Marvin SDK query.
                    "status_snapshot_rate": 2.0,
                }],
                output="screen",
                emulate_tty=True,
                condition=IfCondition(enable_teleop),
            ),
            # MANUS_WUJI_INTEGRATION_RECORD.md section 9.7: use the existing
            # per-side hand launch, which owns the single MANUS publisher.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                FindPackageShare("wuji_teleop_bringup"),
                                "launch",
                                "wuji_teleop_hand.launch.py",
                            ]
                        )
                    ]
                ),
                launch_arguments={
                    "enable_left_hand": enable_left_hand,
                    "enable_right_hand": enable_right_hand,
                    # The drivers connect during Prepare but remain
                    # de-energized. record_session enables the selected hand
                    # only after Tianji reaches READY.
                    "auto_enable": "false",
                    "command_ramp_duration": "5.0",
                }.items(),
                condition=IfCondition(enable_teleop),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [FindPackageShare("camera"), "launch", "camera_launch.py"]
                        )
                    ]
                ),
                condition=IfCondition(
                    PythonExpression([
                        "'", enable_camera, "' == 'true' and '",
                        camera_transport, "' == 'ros'",
                    ])
                ),
            ),
            Node(
                package="camera",
                executable="camera_manager",
                name="wuji_camera_manager",
                arguments=[
                    "--config",
                    camera_config,
                    "--shared-memory-dir",
                    camera_shared_memory_dir,
                    "--active-arm",
                    active_arm,
                ],
                output="screen",
                emulate_tty=True,
                condition=IfCondition(
                    PythonExpression([
                        "'", enable_camera, "' == 'true' and '",
                        camera_transport, "' == 'direct'",
                    ])
                ),
            ),
            Node(
                package="wuji_data_pipeline",
                executable="recorder_node",
                name="wuji_teleop_recorder",
                arguments=[
                    "--config",
                    pipeline_config,
                    "--active-hand",
                    active_hand,
                    "--active-arm",
                    active_arm,
                    "--output-dir",
                    output_dir,
                    "--task-name",
                    task_name,
                    "--camera-transport",
                    camera_transport,
                ],
                output="screen",
                condition=IfCondition(enable_camera),
            ),
            Node(
                package="wuji_data_pipeline",
                executable="recorder_node",
                name="wuji_teleop_recorder",
                arguments=[
                    "--config",
                    pipeline_config,
                    "--no-camera",
                    "--active-hand",
                    active_hand,
                    "--active-arm",
                    active_arm,
                    "--output-dir",
                    output_dir,
                    "--task-name",
                    task_name,
                ],
                output="screen",
                condition=UnlessCondition(enable_camera),
            ),
        ]
    )
