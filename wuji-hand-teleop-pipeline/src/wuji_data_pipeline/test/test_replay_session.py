import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wuji_data_pipeline import replay_session


def test_replay_session_defaults_to_safe_slow_speed():
    args = replay_session._parse_args(["--episode-dir", "/tmp/episode"])

    assert args.playback_rate_hz == 6.0
    assert args.action_rate_hz == 30.0
    assert args.start_hold_s == 1.0
    assert args.arm_command_mode == "eef"
    assert args.arm_hardware_mode == "position"


def test_replay_session_defaults_to_eef_position_mode():
    args = replay_session._parse_args(["--episode-dir", "/tmp/episode"])
    server, deployment = replay_session._build_commands(args, "/tmp/episode")

    assert server[server.index("--arm-command-mode") + 1] == "eef"
    assert server[server.index("--playback-rate-hz") + 1] == "6.0"
    assert server[server.index("--action-rate-hz") + 1] == "30.0"
    assert server[server.index("--start-hold-s") + 1] == "1.0"
    assert server[server.index("--chunk-size") + 1] == "30"
    assert "--no-rebase" in server
    assert "arm_command_mode:=eef" in deployment
    assert "arm_hardware_mode:=position" in deployment
    assert "replay_first_joint_targets_json:={}" in deployment
    assert "server:=tcp://127.0.0.1:5555" in deployment
    assert any(
        argument.endswith("/config/replay.yaml")
        for argument in deployment
    )


def test_replay_session_joint_switch_uses_qpos_without_rebase():
    args = replay_session._parse_args([
        "--episode-dir",
        "/tmp/episode",
        "--arm-command-mode",
        "joint",
    ])
    server, deployment = replay_session._build_commands(args, "/tmp/episode")

    assert args.arm_command_mode == "joint"
    assert server[server.index("--arm-command-mode") + 1] == "joint"
    assert "--no-rebase" in server
    assert "arm_command_mode:=joint" in deployment
    assert "arm_hardware_mode:=position" in deployment


def test_replay_session_can_select_eef_impedance_mode():
    args = replay_session._parse_args([
        "--episode-dir",
        "/tmp/episode",
        "--arm-command-mode",
        "eef",
        "--arm-hardware-mode",
        "impedance",
    ])
    server, deployment = replay_session._build_commands(args, "/tmp/episode")

    assert server[server.index("--arm-command-mode") + 1] == "eef"
    assert "--no-rebase" in server
    assert "arm_command_mode:=eef" in deployment
    assert "arm_hardware_mode:=impedance" in deployment


def test_first_qpos_targets_are_selected_from_episode_layout(monkeypatch):
    layout = replay_session.RobotLayout()
    qpos = np.zeros((2, layout.state_dim), dtype=np.float32)
    qpos[0, 0:7] = np.radians(np.arange(1.0, 8.0))
    qpos[0, 27:34] = np.radians(np.arange(11.0, 18.0))
    monkeypatch.setattr(
        replay_session,
        "load_episode",
        lambda _path: (
            {"qpos": qpos},
            {"robot_layout": layout.metadata()},
        ),
    )

    right = replay_session.load_first_arm_joint_targets(
        "/tmp/episode", "right"
    )
    both = replay_session.load_first_arm_joint_targets(
        "/tmp/episode", "both"
    )

    assert right == {"right": pytest.approx(np.arange(11.0, 18.0))}
    assert both["left"] == pytest.approx(np.arange(1.0, 8.0))
    assert both["right"] == pytest.approx(np.arange(11.0, 18.0))


def test_replay_session_can_select_joint_impedance_mode():
    args = replay_session._parse_args([
        "--episode-dir",
        "/tmp/episode",
        "--arm-command-mode",
        "joint",
        "--arm-hardware-mode",
        "impedance",
    ])
    server, deployment = replay_session._build_commands(args, "/tmp/episode")

    assert "--no-rebase" in server
    assert "arm_command_mode:=joint" in deployment
    assert "arm_hardware_mode:=impedance" in deployment


def test_replay_session_accepts_clear_and_legacy_playback_rate_names():
    clear = replay_session._parse_args([
        "--episode-dir",
        "/tmp/episode",
        "--playback-rate-hz",
        "7.5",
    ])
    legacy = replay_session._parse_args([
        "--episode-dir",
        "/tmp/episode",
        "--rate-hz",
        "7.5",
    ])

    assert clear.playback_rate_hz == 7.5
    assert legacy.playback_rate_hz == 7.5


def test_replay_config_rejects_action_rate_mismatch(tmp_path):
    path = tmp_path / "replay.yaml"
    path.write_text(
        yaml.safe_dump({
            "deployment": {
                "action_rate_hz": 30.0,
                "publish_rate_hz": 120.0,
                "open_loop_horizon": 30,
            }
        })
    )

    replay_session._validate_replay_config(str(path), 30.0)
    with pytest.raises(ValueError, match="must match"):
        replay_session._validate_replay_config(str(path), 20.0)


def test_required_replay_nodes_follow_single_hand_without_teleop_inputs():
    nodes = replay_session.required_replay_nodes("right")

    assert "/tianji_arm_controller" in nodes
    assert "/wuji_deployment" in nodes
    assert "/right_hand/wujihand_driver" in nodes
    assert "/left_hand/wujihand_driver" not in nodes
    assert "/openvr_input" not in nodes
    assert "/manus_data_publisher" not in nodes


def test_replay_preflight_requires_clean_standby_and_selected_hand_nodes():
    status = {
        side: {"state": 0, "err_code": 0, "servo_errors": ["0"] * 7}
        for side in ("left", "right")
    }
    node = SimpleNamespace(
        visible_nodes=lambda: {
            "/tianji_arm_controller",
            "/wuji_deployment",
            "/right_hand/wujihand_driver",
        },
        call=lambda *_args, **_kwargs: (True, json.dumps(status)),
    )

    assert replay_session.replay_recovery_preflight(node, "right") == []

    status["left"]["state"] = 3
    assert replay_session.replay_recovery_preflight(node, "right") == [
        "left:state=3 (required 0)"
    ]
