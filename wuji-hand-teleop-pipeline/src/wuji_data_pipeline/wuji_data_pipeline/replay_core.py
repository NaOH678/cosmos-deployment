"""Metadata-driven trajectory splitting, interpolation, and rebase."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional

import numpy as np

from .schema import RobotLayout, normalize_quaternion_xyzw


def quaternion_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    value = normalize_quaternion_xyzw(q)
    return np.array([-value[0], -value[1], -value[2], value[3]], dtype=np.float32)


def quaternion_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = normalize_quaternion_xyzw(a)
    bx, by, bz, bw = normalize_quaternion_xyzw(b)
    result = np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )
    return normalize_quaternion_xyzw(result)


def quaternion_rotate_xyzw(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rotation = normalize_quaternion_xyzw(q)
    xyz = np.asarray(vector, dtype=np.float32).reshape(3)
    q_xyz = rotation[:3]
    uv = np.cross(q_xyz, xyz)
    uuv = np.cross(q_xyz, uv)
    return xyz + 2.0 * (rotation[3] * uv + uuv)


def quaternion_angle_rad(a: np.ndarray, b: np.ndarray) -> float:
    qa = normalize_quaternion_xyzw(a)
    qb = normalize_quaternion_xyzw(b)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def quaternion_slerp_xyzw(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    qa = normalize_quaternion_xyzw(a).astype(np.float64)
    qb = normalize_quaternion_xyzw(b).astype(np.float64)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    t = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(qa + t * (qb - qa))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    scale_a = math.sin((1.0 - t) * theta_0) / sin_theta_0
    scale_b = math.sin(t * theta_0) / sin_theta_0
    return normalize_quaternion_xyzw(scale_a * qa + scale_b * qb)


@dataclass(frozen=True)
class SideTrajectory:
    eef: np.ndarray
    hand_deg: np.ndarray
    zsp: Optional[np.ndarray] = None

    def sample(self, frame_index: float) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        count = self.eef.shape[0]
        if count == 0:
            raise ValueError("empty trajectory")
        index = float(np.clip(frame_index, 0.0, count - 1))
        lower = int(math.floor(index))
        upper = min(lower + 1, count - 1)
        fraction = index - lower
        position = (1.0 - fraction) * self.eef[lower, :3] + fraction * self.eef[upper, :3]
        quaternion = quaternion_slerp_xyzw(
            self.eef[lower, 3:7], self.eef[upper, 3:7], fraction
        )
        hand = (
            (1.0 - fraction) * self.hand_deg[lower]
            + fraction * self.hand_deg[upper]
        ).astype(np.float32)
        zsp = None
        if self.zsp is not None:
            sampled_zsp = (
                (1.0 - fraction) * self.zsp[lower]
                + fraction * self.zsp[upper]
            ).astype(np.float32)
            norm = float(np.linalg.norm(sampled_zsp))
            if norm > 1e-8:
                zsp = sampled_zsp / norm
            # The recorder zero-fills an unavailable side's optional ZSP to
            # preserve the fixed dual-arm dataset shape. On replay, absence
            # must remain absence: forwarding [0, 0, 0] makes interpolation
            # reject the entire paired action chunk.
        return np.concatenate([position, quaternion]).astype(np.float32), hand, zsp


def split_trajectory(
    actions: np.ndarray,
    layout: RobotLayout,
    zsp: Optional[np.ndarray] = None,
) -> dict[str, SideTrajectory]:
    sequence = np.asarray(actions, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != layout.action_dim:
        raise ValueError(
            f"action sequence must be Nx{layout.action_dim}, got {sequence.shape}"
        )
    zsp_sequence = None if zsp is None else np.asarray(zsp, dtype=np.float32)
    if zsp_sequence is not None and zsp_sequence.shape != (sequence.shape[0], 3 * layout.n_sides):
        raise ValueError(
            f"zsp sequence must be Nx{3 * layout.n_sides}, got {zsp_sequence.shape}"
        )
    result: dict[str, SideTrajectory] = {}
    cursor = 0
    for side_index, side in enumerate(layout.sides):
        eef = sequence[:, cursor : cursor + layout.eef_dof]
        cursor += layout.eef_dof
        hand = sequence[:, cursor : cursor + layout.hand_dof]
        cursor += layout.hand_dof
        side_zsp = None
        if zsp_sequence is not None:
            side_zsp = zsp_sequence[:, side_index * 3 : (side_index + 1) * 3]
        result[side] = SideTrajectory(eef=eef, hand_deg=hand, zsp=side_zsp)
    return result


def validate_trajectory(
    trajectories: Mapping[str, SideTrajectory],
    max_position_jump_m: float = 0.15,
    max_rotation_jump_rad: float = math.radians(75.0),
    max_hand_jump_deg: float = 75.0,
) -> None:
    for side, trajectory in trajectories.items():
        if trajectory.eef.shape[0] < 2:
            continue
        position_jump = np.linalg.norm(np.diff(trajectory.eef[:, :3], axis=0), axis=1)
        hand_jump = np.max(np.abs(np.diff(trajectory.hand_deg, axis=0)), axis=1)
        rotation_jump = np.array(
            [
                quaternion_angle_rad(trajectory.eef[i, 3:7], trajectory.eef[i + 1, 3:7])
                for i in range(trajectory.eef.shape[0] - 1)
            ]
        )
        if float(np.max(position_jump, initial=0.0)) > max_position_jump_m:
            raise ValueError(f"{side} trajectory contains an unsafe position jump")
        if float(np.max(rotation_jump, initial=0.0)) > max_rotation_jump_rad:
            raise ValueError(f"{side} trajectory contains an unsafe rotation jump")
        if float(np.max(hand_jump, initial=0.0)) > max_hand_jump_deg:
            raise ValueError(f"{side} trajectory contains an unsafe hand jump")


class ReplayEngine:
    """Wall-clock trajectory player shared by local and remote replay."""

    def __init__(
        self,
        actions: np.ndarray,
        metadata: Mapping[str, Any],
        zsp: Optional[np.ndarray] = None,
        qpos: Optional[np.ndarray] = None,
        rate_hz: Optional[float] = None,
        rebase: bool = False,
        start_hold_s: float = 1.0,
        loop: bool = False,
    ) -> None:
        self.layout = RobotLayout.from_metadata(metadata)
        self.actions = np.asarray(actions, dtype=np.float32)
        if self.actions.ndim != 2 or self.actions.shape[0] == 0:
            raise ValueError(
                f"action sequence must contain at least one frame, got "
                f"{self.actions.shape}"
            )
        self.trajectories = split_trajectory(self.actions, self.layout, zsp=zsp)
        validate_trajectory(self.trajectories)
        self.arm_joint_rad: Optional[dict[str, np.ndarray]] = None
        if qpos is not None:
            qpos_sequence = np.asarray(qpos, dtype=np.float32)
            expected = (self.actions.shape[0], self.layout.state_dim)
            if qpos_sequence.shape != expected:
                raise ValueError(
                    f"qpos sequence must have shape {expected}, got "
                    f"{qpos_sequence.shape}"
                )
            if not np.all(np.isfinite(qpos_sequence)):
                raise ValueError("qpos sequence contains NaN or infinity")
            self.arm_joint_rad = {}
            cursor = 0
            for side in self.layout.sides:
                self.arm_joint_rad[side] = qpos_sequence[
                    :, cursor : cursor + self.layout.arm_dof
                ].copy()
                cursor += self.layout.arm_dof + self.layout.hand_dof
        source_rate_hz = metadata.get("frame_rate", 30.0)
        self.rate_hz = float(source_rate_hz if rate_hz is None else rate_hz)
        if self.rate_hz <= 0.0:
            raise ValueError("replay rate must be positive")
        self.rebase = bool(rebase)
        self.start_hold_s = max(0.0, float(start_hold_s))
        self.loop = bool(loop)
        self._started_at: Optional[float] = None
        self._rebase_transform: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self._started_at = None
        self._rebase_transform.clear()

    def _observation_pose(self, observation: Mapping[str, Any], side: str) -> Optional[np.ndarray]:
        state = observation.get(f"arm_state_{side}")
        if not isinstance(state, Mapping):
            return None
        if "eef" in state:
            value = np.asarray(state["eef"], dtype=np.float32).reshape(-1)
            if value.shape == (7,):
                return value
        if "ee_pos" in state and "ee_quat" in state:
            return np.concatenate(
                [
                    np.asarray(state["ee_pos"], dtype=np.float32).reshape(3),
                    normalize_quaternion_xyzw(state["ee_quat"]),
                ]
            )
        return None

    def _try_start(self, now: float, observation: Mapping[str, Any]) -> bool:
        if self._started_at is not None:
            return True
        if self.rebase:
            transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for side, trajectory in self.trajectories.items():
                current = self._observation_pose(observation, side)
                if current is None:
                    return False
                first = trajectory.eef[0]
                delta_q = quaternion_multiply_xyzw(
                    current[3:7], quaternion_conjugate_xyzw(first[3:7])
                )
                transforms[side] = (current[:3].copy(), delta_q)
            self._rebase_transform = transforms
        self._started_at = float(now)
        return True

    def _apply_rebase(self, side: str, pose: np.ndarray) -> np.ndarray:
        if not self.rebase:
            return pose
        first = self.trajectories[side].eef[0]
        current_position, delta_q = self._rebase_transform[side]
        relative = pose[:3] - first[:3]
        position = current_position + quaternion_rotate_xyzw(delta_q, relative)
        quaternion = quaternion_multiply_xyzw(delta_q, pose[3:7])
        return np.concatenate([position, quaternion]).astype(np.float32)

    def step(self, now: float, observation: Mapping[str, Any]) -> dict[str, Any]:
        if not self._try_start(now, observation):
            return {
                "timestamp": float(now),
                "sides": list(self.layout.sides),
                "rebase_done": False,
                "finished": False,
                "in_start_hold": False,
            }

        elapsed = max(0.0, float(now) - float(self._started_at))
        in_hold = elapsed < self.start_hold_s
        play_elapsed = max(0.0, elapsed - self.start_hold_s)
        frame_count = self.actions.shape[0]
        raw_frame = play_elapsed * self.rate_hz
        finished = raw_frame >= frame_count - 1
        if self.loop and frame_count > 1:
            frame_index = raw_frame % (frame_count - 1)
            finished = False
        else:
            frame_index = min(raw_frame, frame_count - 1)
        if in_hold:
            frame_index = 0.0

        output: dict[str, Any] = {
            "timestamp": float(now),
            "sides": list(self.layout.sides),
            "frame_idx": int(math.floor(frame_index)),
            "frame_idx_f": float(frame_index),
            "frame_total": int(frame_count),
            "rebase_done": True,
            "in_start_hold": in_hold,
            "finished": bool(finished and not in_hold),
        }
        for side, trajectory in self.trajectories.items():
            pose, hand_deg, zsp = trajectory.sample(frame_index)
            pose = self._apply_rebase(side, pose)
            arm_action: dict[str, Any] = {
                "ee_pos": pose[:3].copy(),
                "ee_quat": pose[3:7].copy(),
            }
            if zsp is not None:
                arm_action["zsp"] = zsp.copy()
            hand_list = hand_deg.tolist()
            output[f"arm_action_{side}"] = arm_action
            output[f"hand_action_{side}"] = hand_list
            if self.arm_joint_rad is not None:
                joints = self.arm_joint_rad[side]
                lower = int(math.floor(frame_index))
                upper = min(lower + 1, joints.shape[0] - 1)
                fraction = frame_index - lower
                output[f"arm_joint_action_{side}"] = (
                    (1.0 - fraction) * joints[lower]
                    + fraction * joints[upper]
                ).astype(np.float32).tolist()
        return output
