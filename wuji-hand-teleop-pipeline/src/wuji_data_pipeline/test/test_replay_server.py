import numpy as np
import pytest

from wuji_data_pipeline import replay_server
from wuji_data_pipeline.schema import RobotLayout


def test_replay_server_default_chunk_covers_deployment_horizon():
    args = replay_server._parse_args(["--episode-dir", "/tmp/episode"])

    assert args.chunk_size == 30
    assert args.action_rate_hz == 30.0
    assert args.rate_hz is None
    assert args.arm_command_mode == "eef"
    assert args.rebase is False


def test_replay_server_accepts_joint_source_switch():
    args = replay_server._parse_args([
        "--episode-dir",
        "/tmp/episode",
        "--arm-command-mode",
        "joint",
    ])

    assert args.arm_command_mode == "joint"


def test_replay_server_rejects_joint_mode_without_qpos(monkeypatch):
    layout = RobotLayout()
    actions = np.zeros((2, layout.action_dim), dtype=np.float32)
    monkeypatch.setattr(
        replay_server,
        "load_episode",
        lambda _path: (
            {"action": actions},
            {"frame_rate": 30.0, "robot_layout": layout.metadata()},
        ),
    )
    args = replay_server._parse_args([
        "--episode-dir",
        "/tmp/episode",
        "--arm-command-mode",
        "joint",
        "--dry-run",
    ])

    with pytest.raises(ValueError, match="requires the episode qpos"):
        replay_server.run_server(args)


def test_replay_server_accepts_playback_rate_alias():
    args = replay_server._parse_args([
        "--episode-dir",
        "/tmp/episode",
        "--playback-rate-hz",
        "6.0",
    ])

    assert args.rate_hz == 6.0
