import numpy as np

from wuji_data_pipeline.schema import (
    RobotLayout,
    build_training_frame,
    split_action_by_side,
)


def _samples():
    arms = {}
    hands = {}
    for index, side in enumerate(("left", "right")):
        arms[side] = {
            "joint_pos_deg": np.arange(7) + index,
            "joint_vel_deg_s": np.ones(7),
            "joint_effort": np.ones(7) * 2,
            "actual_eef": [0.4, 0.1 * index, 0.5, 0, 0, 0, 1],
            "target_eef": [0.41, 0.1 * index, 0.5, 0, 0, 0, 1],
            "joint_command_deg": np.arange(7) + index + 0.5,
            "zsp": [0, -1 if side == "left" else 1, -0.5],
        }
        hands[side] = {
            "actual_q_rad": np.linspace(0.0, 1.0, 20),
            "target_q_rad": np.linspace(0.1, 1.1, 20),
        }
    return arms, hands


def test_dual_tianji_wuji_layout_dimensions_and_units():
    layout = RobotLayout()
    arms, hands = _samples()
    frame = build_training_frame(layout, arms, hands)

    assert layout.state_dim == 54
    assert layout.action_dim == 54
    assert layout.total_eef_dim == 14
    assert frame["qpos"].shape == (54,)
    assert frame["action"].shape == (54,)
    assert frame["eef"].shape == (14,)
    assert np.isclose(frame["qpos"][1], np.radians(1.0))
    assert np.isclose(frame["action"][8], np.degrees(hands["left"]["target_q_rad"])[1])


def test_action_is_measured_eef_plus_commanded_hand():
    layout = RobotLayout()
    arms, hands = _samples()
    frame = build_training_frame(layout, arms, hands)
    split = split_action_by_side(frame["action"], layout)

    assert np.allclose(split["left"]["eef"], arms["left"]["actual_eef"])
    assert np.allclose(
        split["right"]["hand_deg"],
        np.degrees(hands["right"]["target_q_rad"]),
    )
    assert not np.allclose(frame["commanded_eef"][:7], frame["eef"][:7])


def test_layout_round_trip_uses_metadata_not_hardcoded_dimensions():
    layout = RobotLayout(sides=("right",), hand_dof=20)
    restored = RobotLayout.from_metadata({"robot_layout": layout.metadata()})
    assert restored == layout
    assert restored.action_dim == 27
