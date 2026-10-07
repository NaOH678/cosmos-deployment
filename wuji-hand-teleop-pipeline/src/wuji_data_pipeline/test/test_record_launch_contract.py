from pathlib import Path


def test_record_launch_uses_only_the_documented_control_startup_path():
    source = (Path(__file__).parents[1] / "launch" / "record.launch.py").read_text()

    assert '"wuji_teleop.launch.py"' not in source
    assert '"vive_arm_tf.launch.py"' in source
    assert 'executable="tianji_arm_controller"' in source
    assert '"wuji_teleop_hand.launch.py"' in source
    assert '"auto_enable": False' in source
    assert 'DeclareLaunchArgument(\n                "active_arm"' in source
    assert '"active_arm": active_arm' in source
    assert '"control_rate": 120.0' in source
    assert '"state_publish_rate": 500.0' in source
    assert '"teleop_position_scale": 1.25' in source
    assert "max_initial_ik_offset_deg" not in source
    assert "max_ik_frame_jump_deg" not in source
    assert "max_joint_command_step_deg" not in source
    assert "teleop_max_target_offset_m" not in source
    assert '"impedance_velocity_ratio": 30' in source
    assert '"impedance_acceleration_ratio": 30' in source
    assert '"handoff_hold_sec": 0.2' in source
    assert '"handoff_ramp_sec",' in source
    assert 'default_value="1.0"' in source
    assert "ParameterValue(" in source
    assert 'DeclareLaunchArgument("task_name", default_value="")' in source
    assert '"--output-dir"' in source
    assert '"--task-name"' in source
    assert '"--active-arm"' in source
    assert '"recovery_max_speed_deg_s": 5.0' in source
    assert '"recovery_max_accel_deg_s2": 10.0' in source
    assert '"impedance_max_drift_deg": 3.0' in source
    assert '"status_snapshot_rate": 2.0' in source
    assert '"auto_enable": "false"' in source
    assert '"command_ramp_duration": "5.0"' in source
    assert 'default_value="direct"' in source
    assert 'executable="camera_manager"' in source
    assert '"--camera-transport"' in source
    assert "' == 'ros'" in source


def test_recording_hand_driver_starts_disabled_and_anchors_on_enable():
    root = Path(__file__).parents[2]
    hand_launch = (
        root / "wuji_teleop_bringup" / "launch" / "wuji_teleop_hand.launch.py"
    ).read_text()
    driver = (
        root
        / "wujihandros2"
        / "wujihand_driver"
        / "src"
        / "wujihand_driver_node.cpp"
    ).read_text()

    assert '"auto_enable", default_value="true"' in hand_launch
    assert '"auto_enable": auto_enable' in hand_launch
    assert 'declare_parameter("auto_enable", true)' in driver
    assert "if (auto_enable_)" in driver
    assert "if (!auto_enable_)" in driver
    assert "!external_commands_enabled_" in driver
    assert "command_stream_active_ = false;" in driver
    assert "ActualPosition" in driver
    assert '"recover_to_initial"' in driver
    assert "external_commands_enabled_" in driver
    assert "require_recovery_before_enable" in driver


def test_hand_initial_pose_is_tracked_in_the_config_template():
    root = Path(__file__).parents[2]
    template = (
        root
        / "output_devices"
        / "wujihand_output"
        / "config"
        / "wujihand_ik.yaml.template"
    ).read_text()

    assert "initial_position:" in template
    assert "recovery_duration: 5.0" in template
    assert "recovery_tolerance: 0.12" in template
    assert "recovery_settle_timeout: 2.0" in template
    assert template.count("    - ") >= 20
