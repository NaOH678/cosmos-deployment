import numpy as np
import pytest

from wuji_data_pipeline.replay_core import ReplayEngine
from wuji_data_pipeline.schema import RobotLayout


def _episode():
    layout = RobotLayout()
    rows = []
    for step in range(3):
        row = []
        for side_index in range(2):
            row.extend([0.4 + 0.01 * step, 0.1 * side_index, 0.5, 0, 0, 0, 1])
            row.extend([float(step)] * 20)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), {"frame_rate": 30.0, "robot_layout": layout.metadata()}


def test_replay_splits_54d_and_interpolates_both_sides():
    actions, metadata = _episode()
    engine = ReplayEngine(actions, metadata, rebase=False, start_hold_s=0.0)

    output = engine.step(10.0, {})
    output = engine.step(10.0 + 0.5 / 30.0, {})

    assert len(output["hand_action_left"]) == 20
    assert len(output["hand_action_right"]) == 20
    assert np.isclose(output["arm_action_left"]["ee_pos"][0], 0.405)
    assert np.isclose(output["hand_action_left"][0], 0.5)


def test_rebase_maps_first_recorded_pose_to_current_pose():
    actions, metadata = _episode()
    engine = ReplayEngine(actions, metadata, rebase=True, start_hold_s=1.0)
    observation = {
        "arm_state_left": {"ee_pos": [0.6, 0.2, 0.3], "ee_quat": [0, 0, 0, 1]},
        "arm_state_right": {"ee_pos": [0.6, -0.2, 0.3], "ee_quat": [0, 0, 0, 1]},
    }

    output = engine.step(5.0, observation)

    assert output["rebase_done"] is True
    assert output["in_start_hold"] is True
    assert np.allclose(output["arm_action_left"]["ee_pos"], [0.6, 0.2, 0.3])
    assert np.allclose(output["arm_action_right"]["ee_pos"], [0.6, -0.2, 0.3])


def test_empty_episode_is_rejected_before_server_start():
    layout = RobotLayout()
    with pytest.raises(ValueError, match="at least one frame"):
        ReplayEngine(
            np.empty((0, layout.action_dim), dtype=np.float32),
            {"robot_layout": layout.metadata()},
        )


def test_explicit_zero_playback_rate_is_rejected():
    actions, metadata = _episode()

    with pytest.raises(ValueError, match="replay rate must be positive"):
        ReplayEngine(actions, metadata, rate_hz=0.0)


def test_zero_filled_optional_zsp_is_omitted_from_replay_action():
    actions, metadata = _episode()
    zsp = np.zeros((actions.shape[0], 6), dtype=np.float32)
    zsp[:, 3] = 1.0
    engine = ReplayEngine(
        actions,
        metadata,
        zsp=zsp,
        rebase=False,
        start_hold_s=0.0,
    )

    action = engine.step(1.0, {})

    assert "zsp" not in action["arm_action_left"]
    assert np.allclose(
        action["arm_action_right"]["zsp"], [1.0, 0.0, 0.0]
    )


def test_replay_emits_recorded_motor_qpos_as_joint_action():
    actions, metadata = _episode()
    layout = RobotLayout()
    qpos = np.zeros((actions.shape[0], layout.state_dim), dtype=np.float32)
    qpos[:, 0:7] = np.arange(7, dtype=np.float32)
    qpos[:, 27:34] = np.arange(7, dtype=np.float32) + 10.0
    engine = ReplayEngine(
        actions,
        metadata,
        qpos=qpos,
        rebase=False,
        start_hold_s=0.0,
    )

    output = engine.step(2.0, {})

    assert np.allclose(output["arm_joint_action_left"], np.arange(7))
    assert np.allclose(
        output["arm_joint_action_right"], np.arange(7) + 10.0
    )
