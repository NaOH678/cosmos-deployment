from pathlib import Path

from wuji_data_pipeline.arm_teleop_session import (
    ARM_ONLY_REQUIRED_NODES,
    arm_only_required_nodes,
    arm_only_tf_pairs,
)


ROOT = Path(__file__).parents[2]


def test_arm_only_required_nodes_exclude_all_recording_and_hand_nodes():
    assert "/openvr_input" in ARM_ONLY_REQUIRED_NODES
    assert "/tianji_arm_controller" in ARM_ONLY_REQUIRED_NODES
    assert "/tianji_left_tf" in ARM_ONLY_REQUIRED_NODES
    assert "/tianji_right_tf" in ARM_ONLY_REQUIRED_NODES
    assert not any("recorder" in name for name in ARM_ONLY_REQUIRED_NODES)
    assert not any("manus" in name for name in ARM_ONLY_REQUIRED_NODES)
    assert not any("hand" in name for name in ARM_ONLY_REQUIRED_NODES)


def test_single_arm_preflight_requires_only_selected_tracker_tf_chain():
    left_nodes = arm_only_required_nodes("left")
    assert "/tianji_left_tf" in left_nodes
    assert "/left_chest_tf" in left_nodes
    assert "/tianji_right_tf" not in left_nodes
    assert "/right_chest_tf" not in left_nodes
    assert arm_only_tf_pairs("left") == (
        ("left_chest", "tianji_left"),
        ("left_chest", "left_arm"),
    )
    assert arm_only_required_nodes("both") == ARM_ONLY_REQUIRED_NODES


def test_arm_only_session_splits_recovery_and_enable_preflights():
    source = (
        ROOT
        / "wuji_data_pipeline"
        / "wuji_data_pipeline"
        / "arm_teleop_session.py"
    ).read_text()

    assert (
        'node.recovery_preflight_nodes(\n'
        '                    frozenset({"/tianji_arm_controller"}),'
    ) in source
    assert "tf_pairs=()," in source
    assert (
        "node.enable_preflight_nodes(\n"
        "                    arm_only_required_nodes(args.active_arm),"
    ) in source
    assert "tf_pairs=arm_only_tf_pairs(args.active_arm)," in source


def test_arm_only_launch_has_only_validated_tianji_arm_path():
    source = (
        ROOT
        / "wuji_teleop_bringup"
        / "launch"
        / "arm_only_teleop.launch.py"
    ).read_text()

    assert '"vive_arm_tf.launch.py"' in source
    assert 'executable="tianji_arm_controller"' in source
    assert 'DeclareLaunchArgument(\n            "active_arm"' in source
    assert '"active_arm": LaunchConfiguration("active_arm")' in source
    assert 'executable="openvr_input"' not in source
    assert 'executable="manus_data_publisher"' not in source
    assert 'executable="wujihand_driver_node"' not in source
    assert 'executable="recorder_node"' not in source
    for contract in (
        '"auto_enable": False',
        '"control_rate": 120.0',
        '"state_publish_rate": 500.0',
        '"teleop_position_scale": 1.25',
        '"impedance_velocity_ratio": 30',
        '"impedance_acceleration_ratio": 30',
        '"handoff_hold_sec": 0.2',
        '"handoff_ramp_sec": 1.0',
        '"recovery_max_speed_deg_s": 5.0',
        '"recovery_max_accel_deg_s2": 10.0',
        '"impedance_max_drift_deg": 3.0',
        '"status_snapshot_rate": 2.0',
    ):
        assert contract in source
    for removed_guard in (
        "max_initial_ik_offset_deg",
        "max_ik_frame_jump_deg",
        "max_joint_command_step_deg",
        "teleop_max_target_offset_m",
    ):
        assert removed_guard not in source


def test_arm_only_shell_entrypoint_runs_only_the_arm_session():
    source = (ROOT / "scripts" / "start_teleop_arm_only.sh").read_text()

    assert "arm_teleop_session" in source
    assert "record_session" not in source
    assert "replay_session" not in source
    assert "[both|left|right]" in source
    assert 'WUJI_ACTIVE_ARM="${ACTIVE_ARM}"' in source
