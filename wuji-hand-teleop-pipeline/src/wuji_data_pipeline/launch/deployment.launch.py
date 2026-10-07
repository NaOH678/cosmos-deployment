"""Robot-side Tianji + Wuji deployment graph.

No Tracker, MANUS publisher, or hand retargeter is started here.  The policy
deployment node is the sole command producer, and Tianji still requires the
operator-driven Recovery + Enable services before it accepts any command.
"""

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

from wuji_teleop_bringup.hand_defaults import (
    DRIVER_DIAGNOSTICS_RATE,
    DRIVER_FILTER_CUTOFF_FREQ,
    DRIVER_PUBLISH_RATE,
    DRIVER_RECOVERY_DURATION,
    DRIVER_RECOVERY_SETTLE_TIMEOUT,
    DRIVER_RECOVERY_TOLERANCE,
    LEFT_HAND_INITIAL_POSITION,
    LEFT_HAND_NAME,
    LEFT_HAND_SERIAL,
    RIGHT_HAND_INITIAL_POSITION,
    RIGHT_HAND_NAME,
    RIGHT_HAND_SERIAL,
)


def generate_launch_description():
    enable_camera = LaunchConfiguration("enable_camera")
    enable_state_trace = LaunchConfiguration("enable_state_trace")
    arm_command_mode = LaunchConfiguration("arm_command_mode")
    arm_hardware_mode = LaunchConfiguration("arm_hardware_mode")
    camera_transport = LaunchConfiguration("camera_transport")
    camera_config = LaunchConfiguration("camera_config")
    camera_shared_memory_dir = LaunchConfiguration(
        "camera_shared_memory_dir"
    )
    pipeline_config = LaunchConfiguration("pipeline_config")
    server = LaunchConfiguration("server")
    startup_handoff_gate_enabled = LaunchConfiguration(
        "startup_handoff_gate_enabled"
    )
    replay_first_joint_targets_json = LaunchConfiguration(
        "replay_first_joint_targets_json"
    )
    active_hand = LaunchConfiguration("active_hand")
    left_name = LaunchConfiguration("left_hand_name")
    right_name = LaunchConfiguration("right_hand_name")
    left_serial = ParameterValue(LaunchConfiguration("left_serial"), value_type=str)
    right_serial = ParameterValue(LaunchConfiguration("right_serial"), value_type=str)
    enable_left_hand = PythonExpression(
        ["'", active_hand, "' in ('both', 'left')"]
    )
    enable_right_hand = PythonExpression(
        ["'", active_hand, "' in ('both', 'right')"]
    )

    declarations = [
        DeclareLaunchArgument("enable_camera", default_value="true"),
        DeclareLaunchArgument("enable_state_trace", default_value="true"),
        DeclareLaunchArgument(
            "arm_command_mode",
            default_value="eef",
            choices=["eef", "joint"],
        ),
        DeclareLaunchArgument(
            "arm_hardware_mode",
            default_value="impedance",
            choices=["impedance", "position"],
            description=(
                "Tianji SDK actuation mode; position is reserved for "
                "explicit external replay"
            ),
        ),
        DeclareLaunchArgument(
            "camera_transport",
            default_value="direct",
            choices=["direct", "ros"],
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
            "active_hand",
            default_value="both",
            choices=["both", "left", "right"],
            description="Connected physical Wuji Hand side(s)",
        ),
        DeclareLaunchArgument(
            "pipeline_config",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("wuji_data_pipeline"),
                    "config",
                    "cosmos_protocol_v2.yaml",
                ]
            ),
        ),
        DeclareLaunchArgument(
            "server",
            default_value="",
            description=(
                "Override deployment.server (tcp://host:port or http(s)://host)"
            ),
        ),
        DeclareLaunchArgument(
            "startup_handoff_gate_enabled",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Freeze the first external arm target and wait for explicit "
                "controller handoff completion"
            ),
        ),
        DeclareLaunchArgument(
            "replay_first_joint_targets_json",
            default_value="{}",
            description=(
                "Replay-only first measured arm qpos targets in degrees; "
                "empty for model deployment"
            ),
        ),
        DeclareLaunchArgument("left_serial", default_value=LEFT_HAND_SERIAL),
        DeclareLaunchArgument("right_serial", default_value=RIGHT_HAND_SERIAL),
        DeclareLaunchArgument("left_hand_name", default_value=LEFT_HAND_NAME),
        DeclareLaunchArgument("right_hand_name", default_value=RIGHT_HAND_NAME),
    ]

    common_driver_parameters = {
        "publish_rate": DRIVER_PUBLISH_RATE,
        "filter_cutoff_freq": DRIVER_FILTER_CUTOFF_FREQ,
        "diagnostics_rate": DRIVER_DIAGNOSTICS_RATE,
        # Deployment is supervised exactly like recording: connect during
        # Prepare, energize only after Tianji reaches READY, and blend the
        # first cloud/replay command from measured hand state.
        "auto_enable": False,
        "command_ramp_duration": 1.0,
        "recovery_duration": DRIVER_RECOVERY_DURATION,
        "recovery_tolerance": DRIVER_RECOVERY_TOLERANCE,
        "recovery_settle_timeout": DRIVER_RECOVERY_SETTLE_TIMEOUT,
        "require_recovery_before_enable": True,
    }
    return LaunchDescription(
        declarations
        + [
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
                    active_hand,
                ],
                output="screen",
                condition=IfCondition(
                    PythonExpression([
                        "'", enable_camera, "' == 'true' and '",
                        camera_transport, "' == 'direct'",
                    ])
                ),
            ),
            Node(
                package="wujihand_driver",
                executable="wujihand_driver_node",
                name="wujihand_driver",
                namespace=left_name,
                parameters=[{
                    **common_driver_parameters,
                    "serial_number": left_serial,
                    "initial_position": LEFT_HAND_INITIAL_POSITION,
                }],
                output="screen",
                emulate_tty=True,
                condition=IfCondition(enable_left_hand),
            ),
            Node(
                package="wujihand_driver",
                executable="wujihand_driver_node",
                name="wujihand_driver",
                namespace=right_name,
                parameters=[{
                    **common_driver_parameters,
                    "serial_number": right_serial,
                    "initial_position": RIGHT_HAND_INITIAL_POSITION,
                }],
                output="screen",
                emulate_tty=True,
                condition=IfCondition(enable_right_hand),
            ),
            Node(
                package="controller",
                executable="tianji_arm_controller",
                name="tianji_arm_controller",
                # Keep every Tianji safety/impedance parameter identical to
                # the field-validated recording launch.  Replay changes only
                # the command source from Tracker to external policy actions.
                parameters=[{
                    "auto_enable": False,
                    "active_arm": active_hand,
                    "control_source": "external",
                    "external_command_mode": arm_command_mode,
                    "arm_hardware_mode": arm_hardware_mode,
                    "external_handoff_gate_enabled": ParameterValue(
                        startup_handoff_gate_enabled,
                        value_type=bool,
                    ),
                    "replay_first_joint_targets_json": ParameterValue(
                        replay_first_joint_targets_json,
                        value_type=str,
                    ),
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
                    "status_snapshot_rate": 2.0,
                }],
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="wuji_data_pipeline",
                executable="deployment_node",
                name="wuji_deployment",
                arguments=[
                    "--config",
                    pipeline_config,
                    "--active-hand",
                    active_hand,
                    "--arm-command-mode",
                    arm_command_mode,
                    "--server",
                    server,
                    "--camera-transport",
                    camera_transport,
                ],
                output="screen",
                condition=IfCondition(enable_camera),
            ),
            Node(
                package="wuji_data_pipeline",
                executable="deployment_node",
                name="wuji_deployment",
                arguments=[
                    "--config",
                    pipeline_config,
                    "--no-camera",
                    "--active-hand",
                    active_hand,
                    "--arm-command-mode",
                    arm_command_mode,
                    "--server",
                    server,
                ],
                output="screen",
                condition=UnlessCondition(enable_camera),
            ),
            Node(
                package="wuji_data_pipeline",
                executable="deployment_state_trace",
                name="wuji_deployment_state_trace",
                arguments=[
                    "--config",
                    pipeline_config,
                    "--active-side",
                    active_hand,
                    "--arm-command-mode",
                    arm_command_mode,
                ],
                output="screen",
                emulate_tty=True,
                condition=IfCondition(enable_state_trace),
            ),
        ]
    )
