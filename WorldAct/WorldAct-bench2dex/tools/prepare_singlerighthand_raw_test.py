# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np

from tools.prepare_singlerighthand_raw import (
    build_joint_state_action,
    build_state_action,
)

ROBOT_LAYOUT = {
    "arm_dof_per_side": 7,
    "hand_dof_per_side": 20,
    "qpos_layout": [
        {"side": "left", "arm": [0, 7], "hand": [7, 27]},
        {"side": "right", "arm": [27, 34], "hand": [34, 54]},
    ],
    "action_layout": [
        {"side": "left", "eef": [0, 7], "hand": [7, 27]},
        {"side": "right", "eef": [27, 34], "hand": [34, 54]},
    ],
}


def test_build_state_action_selects_right_side_and_converts_hand_to_radians() -> None:
    raw_action = np.zeros((2, 54), dtype=np.float32)
    observed_eef = np.zeros((2, 14), dtype=np.float32)
    observed_hand_deg = np.zeros((2, 40), dtype=np.float32)

    observed_eef[:, 7:10] = [0.1, 0.2, 0.3]
    observed_eef[:, 13] = 1.0
    observed_hand_deg[:, 20:] = 180.0
    raw_action[:, 27:34] = observed_eef[:, 7:14]
    raw_action[:, 34:] = 90.0

    state, action = build_state_action(raw_action, observed_eef, observed_hand_deg)

    assert state.shape == (2, 27)
    assert action.shape == (2, 27)
    np.testing.assert_allclose(state[:, :7], observed_eef[:, 7:14])
    np.testing.assert_allclose(state[:, 7:], np.pi, rtol=1e-6)
    np.testing.assert_allclose(action[:, :7], observed_eef[:, 7:14])
    np.testing.assert_allclose(action[:, 7:], np.pi / 2, rtol=1e-6)


def test_build_joint_state_action_selects_right_joint_targets_and_uses_radians() -> None:
    qpos = np.arange(108, dtype=np.float32).reshape(2, 54) / 100.0
    raw_action = np.zeros((2, 54), dtype=np.float32)
    raw_action[:, 34:54] = 90.0
    arm_joint_command = np.arange(28, dtype=np.float32).reshape(2, 14) / 10.0

    state, action = build_joint_state_action(qpos, raw_action, arm_joint_command, ROBOT_LAYOUT)

    assert state.shape == (2, 27)
    assert action.shape == (2, 27)
    np.testing.assert_allclose(state[:, :7], qpos[:, 27:34])
    np.testing.assert_allclose(state[:, 7:], qpos[:, 34:54])
    np.testing.assert_allclose(action[:, :7], arm_joint_command[:, 7:14])
    np.testing.assert_allclose(action[:, 7:], np.pi / 2, rtol=1e-6)
