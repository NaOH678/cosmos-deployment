# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import pytest

from tools.prepare_dualhand_joint_raw import build_state_action

ROBOT_LAYOUT = {
    "sides": ["left", "right"],
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


def test_build_state_action_uses_joint_targets_and_converts_both_hands_to_radians() -> None:
    qpos = np.arange(108, dtype=np.float32).reshape(2, 54) / 100.0
    raw_action = np.zeros((2, 54), dtype=np.float32)
    raw_action[:, 7:27] = 180.0
    raw_action[:, 34:54] = 90.0
    arm_joint_command = np.arange(28, dtype=np.float32).reshape(2, 14) / 10.0

    state, action = build_state_action(qpos, raw_action, arm_joint_command, ROBOT_LAYOUT)

    assert state.shape == (2, 54)
    assert action.shape == (2, 54)
    np.testing.assert_array_equal(state, qpos)
    np.testing.assert_allclose(action[:, :7], arm_joint_command[:, :7])
    np.testing.assert_allclose(action[:, 7:27], np.pi, rtol=1e-6)
    np.testing.assert_allclose(action[:, 27:34], arm_joint_command[:, 7:14])
    np.testing.assert_allclose(action[:, 34:54], np.pi / 2, rtol=1e-6)


def test_build_state_action_rejects_non_finite_values() -> None:
    qpos = np.zeros((1, 54), dtype=np.float32)
    raw_action = np.zeros((1, 54), dtype=np.float32)
    arm_joint_command = np.zeros((1, 14), dtype=np.float32)
    arm_joint_command[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        build_state_action(qpos, raw_action, arm_joint_command, ROBOT_LAYOUT)
