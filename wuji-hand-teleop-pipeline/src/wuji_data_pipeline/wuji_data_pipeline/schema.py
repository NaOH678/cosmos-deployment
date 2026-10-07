"""Dataset layout and unit conversions.

The scalar keys intentionally match ``dexmanip_tool``.  Dimensions and
metadata are hardware-driven instead of being hard-coded to XHand's 12 DoF.

Core units also match the reference dataset:

* arm and hand observations: radians;
* end-effector position: metres;
* end-effector quaternion: xyzw;
* hand values inside ``action``: degrees.

The arm component of ``action`` is the measured end-effector pose.  That is
the slightly unusual but deliberate convention used by the reference
recorder: replay follows the trajectory that the robot actually executed,
not an unreachable upstream tracker target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SIDES = ("left", "right")

# Protocol-v2 arm action semantics are negotiated independently from the
# numeric action dimension.  Both formats happen to use seven arm scalars per
# side, so shape alone must never be used to distinguish them.
ARM_ACTION_SPACE_EEF = "eef_pose"
ARM_ACTION_SPACE_JOINT = "joint_position"
ARM_ACTION_SPACES = (
    ARM_ACTION_SPACE_EEF,
    ARM_ACTION_SPACE_JOINT,
)


def normalize_arm_action_space(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip().lower()
    if normalized not in ARM_ACTION_SPACES:
        raise ValueError(
            f"{name} must be one of {ARM_ACTION_SPACES}, got {value!r}"
        )
    return normalized


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def normalize_quaternion_xyzw(value: Any) -> np.ndarray:
    quaternion = _vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("quaternion norm is zero")
    quaternion = quaternion / norm
    # Canonical sign makes adjacent samples deterministic without changing the
    # represented rotation.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion.astype(np.float32)


def pose7(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    return np.concatenate(
        [_vector(position, 3, "position"), normalize_quaternion_xyzw(quaternion_xyzw)]
    ).astype(np.float32)


@dataclass(frozen=True)
class RobotLayout:
    sides: tuple[str, ...] = SIDES
    arm_dof: int = 7
    hand_dof: int = 20
    eef_dof: int = 7

    def __post_init__(self) -> None:
        if not self.sides or any(side not in SIDES for side in self.sides):
            raise ValueError(f"invalid sides: {self.sides}")
        if len(set(self.sides)) != len(self.sides):
            raise ValueError(f"duplicate sides: {self.sides}")
        if min(self.arm_dof, self.hand_dof, self.eef_dof) <= 0:
            raise ValueError("all DoF values must be positive")

    @property
    def n_sides(self) -> int:
        return len(self.sides)

    @property
    def per_side_state_dim(self) -> int:
        return self.arm_dof + self.hand_dof

    @property
    def per_side_action_dim(self) -> int:
        return self.eef_dof + self.hand_dof

    @property
    def state_dim(self) -> int:
        return self.n_sides * self.per_side_state_dim

    @property
    def action_dim(self) -> int:
        return self.n_sides * self.per_side_action_dim

    @property
    def total_eef_dim(self) -> int:
        return self.n_sides * self.eef_dof

    def metadata(self) -> dict[str, Any]:
        action_layout = []
        qpos_layout = []
        action_cursor = 0
        qpos_cursor = 0
        for side in self.sides:
            action_layout.append(
                {
                    "side": side,
                    "eef": [action_cursor, action_cursor + self.eef_dof],
                    "hand": [
                        action_cursor + self.eef_dof,
                        action_cursor + self.per_side_action_dim,
                    ],
                }
            )
            qpos_layout.append(
                {
                    "side": side,
                    "arm": [qpos_cursor, qpos_cursor + self.arm_dof],
                    "hand": [
                        qpos_cursor + self.arm_dof,
                        qpos_cursor + self.per_side_state_dim,
                    ],
                }
            )
            action_cursor += self.per_side_action_dim
            qpos_cursor += self.per_side_state_dim
        return {
            "arm_dof": self.arm_dof * self.n_sides,
            "hand_dof": self.hand_dof * self.n_sides,
            "arm_dof_per_side": self.arm_dof,
            "hand_dof_per_side": self.hand_dof,
            "eef_dof_per_side": self.eef_dof,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "eef_dim": self.total_eef_dim,
            "n_sides": self.n_sides,
            "sides": list(self.sides),
            "action_layout": action_layout,
            "qpos_layout": qpos_layout,
            "units": {
                "qpos.arm": "radian",
                "qpos.hand": "radian",
                "action.eef.position": "metre",
                "action.eef.quaternion": "xyzw",
                "action.hand": "degree",
            },
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "RobotLayout":
        raw = metadata.get("robot_layout", metadata)
        sides = tuple(raw.get("sides") or SIDES[: int(raw.get("n_sides", 2))])
        arm_per_side = int(raw.get("arm_dof_per_side", 7))
        hand_per_side = raw.get("hand_dof_per_side")
        if hand_per_side is None:
            total = int(raw.get("hand_dof", 20 * len(sides)))
            hand_per_side = total // max(len(sides), 1)
        return cls(
            sides=sides,
            arm_dof=arm_per_side,
            hand_dof=int(hand_per_side),
            eef_dof=int(raw.get("eef_dof_per_side", 7)),
        )


def build_training_frame(
    layout: RobotLayout,
    arms: Mapping[str, Mapping[str, Any]],
    hands: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Build one synchronized scalar frame.

    Arm samples use native Tianji degrees at the ROS boundary.  Hand samples
    use ROS-standard radians.  The returned arrays follow the reference LMDB
    units documented at module level.
    """

    qpos_parts: list[np.ndarray] = []
    qvel_parts: list[np.ndarray] = []
    effort_parts: list[np.ndarray] = []
    eef_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    hand_joint_deg_parts: list[np.ndarray] = []
    commanded_eef_parts: list[np.ndarray] = []
    arm_joint_command_parts: list[np.ndarray] = []
    zsp_parts: list[np.ndarray] = []

    for side in layout.sides:
        if side not in arms or side not in hands:
            raise ValueError(f"missing synchronized sample for {side}")
        arm = arms[side]
        hand = hands[side]

        arm_q_deg = _vector(arm["joint_pos_deg"], layout.arm_dof, f"{side}.arm_q")
        arm_dq_deg = _vector(
            arm.get("joint_vel_deg_s", np.zeros(layout.arm_dof)),
            layout.arm_dof,
            f"{side}.arm_dq",
        )
        arm_effort = _vector(
            arm.get("joint_effort", np.zeros(layout.arm_dof)),
            layout.arm_dof,
            f"{side}.arm_effort",
        )
        actual_eef = _vector(arm["actual_eef"], layout.eef_dof, f"{side}.actual_eef")
        actual_eef = np.concatenate(
            [actual_eef[:3], normalize_quaternion_xyzw(actual_eef[3:7])]
        ).astype(np.float32)

        hand_actual_rad = _vector(
            hand["actual_q_rad"], layout.hand_dof, f"{side}.hand_actual"
        )
        hand_target_rad = _vector(
            hand["target_q_rad"], layout.hand_dof, f"{side}.hand_target"
        )
        hand_velocity = _vector(
            hand.get("velocity_rad_s", np.zeros(layout.hand_dof)),
            layout.hand_dof,
            f"{side}.hand_velocity",
        )
        hand_effort = _vector(
            hand.get("effort", np.zeros(layout.hand_dof)),
            layout.hand_dof,
            f"{side}.hand_effort",
        )

        qpos_parts.extend([np.radians(arm_q_deg), hand_actual_rad])
        qvel_parts.extend([np.radians(arm_dq_deg), hand_velocity])
        effort_parts.extend([arm_effort, hand_effort])
        eef_parts.append(actual_eef)
        action_parts.extend([actual_eef, np.degrees(hand_target_rad)])
        hand_joint_deg_parts.append(np.degrees(hand_actual_rad))

        commanded_eef_parts.append(
            _vector(
                arm.get("target_eef", actual_eef),
                layout.eef_dof,
                f"{side}.target_eef",
            )
        )
        arm_joint_command_parts.append(
            np.radians(
                _vector(
                    arm.get("joint_command_deg", arm_q_deg),
                    layout.arm_dof,
                    f"{side}.arm_command",
                )
            )
        )
        zsp_parts.append(
            _vector(arm.get("zsp", np.zeros(3)), 3, f"{side}.zsp")
        )

    eef = np.concatenate(eef_parts).astype(np.float32)
    return {
        "action": np.concatenate(action_parts).astype(np.float32),
        "action_eef": eef.copy(),
        "action_bases": np.zeros(6, dtype=np.float32),
        "qpos": np.concatenate(qpos_parts).astype(np.float32),
        "qvel": np.concatenate(qvel_parts).astype(np.float32),
        "effort": np.concatenate(effort_parts).astype(np.float32),
        "eef": eef,
        "robot_base": np.zeros(6, dtype=np.float32),
        "hand_joint_deg": np.concatenate(hand_joint_deg_parts).astype(np.float32),
        "commanded_eef": np.concatenate(commanded_eef_parts).astype(np.float32),
        "arm_joint_command": np.concatenate(arm_joint_command_parts).astype(np.float32),
        "zsp": np.concatenate(zsp_parts).astype(np.float32),
    }


def split_action_by_side(
    action: Sequence[float], layout: RobotLayout
) -> dict[str, dict[str, np.ndarray]]:
    vector = _vector(action, layout.action_dim, "action")
    result: dict[str, dict[str, np.ndarray]] = {}
    cursor = 0
    for side in layout.sides:
        eef = vector[cursor : cursor + layout.eef_dof]
        cursor += layout.eef_dof
        hand = vector[cursor : cursor + layout.hand_dof]
        cursor += layout.hand_dof
        result[side] = {"eef": eef.copy(), "hand_deg": hand.copy()}
    return result
