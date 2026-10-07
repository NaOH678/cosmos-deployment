# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Model adapters and action conversion for the protocol-v2 policy service."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig
from cosmos_framework.inference.robot_policy.protocol import ProtocolError


@dataclass(frozen=True)
class AdapterOutput:
    action_chunk: list[dict[str, Any]]
    inference_ms: float


class ModelAdapter(ABC):
    """Transport-independent interface implemented by Cosmos and diagnostic policies."""

    def __init__(self, config: RobotPolicyConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether model/config/warmup checks have completed successfully."""

    def reset_session(self, session_id: str) -> None:
        """Reset model-side state. The current WAM policy is stateless across requests."""

    @abstractmethod
    def infer(self, observation: Mapping[str, Any]) -> AdapterOutput:
        """Return a complete protocol-v2 bilateral action chunk."""


def _normalize_quaternion(value: np.ndarray, min_norm: float) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ProtocolError("COSMOS_INVALID_ACTION", "model returned an invalid quaternion")
    norm = float(np.linalg.norm(quat))
    if norm < min_norm:
        raise ProtocolError("COSMOS_INVALID_ACTION", "model returned a zero quaternion")
    return (quat / norm).astype(np.float32)


def _quaternion_angle_deg(left: np.ndarray, right: np.ndarray, min_norm: float) -> float:
    q0 = _normalize_quaternion(left, min_norm).astype(np.float64)
    q1 = _normalize_quaternion(right, min_norm).astype(np.float64)
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _limit_vector_delta(current: np.ndarray, target: np.ndarray, max_norm: float) -> np.ndarray:
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_norm or norm == 0.0:
        return target
    return current + delta * (max_norm / norm)


def _slerp(current: np.ndarray, target: np.ndarray, fraction: float, min_norm: float) -> np.ndarray:
    q0 = _normalize_quaternion(current, min_norm).astype(np.float64)
    q1 = _normalize_quaternion(target, min_norm).astype(np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + fraction * (q1 - q0), min_norm)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    result = math.sin((1.0 - fraction) * theta) / sin_theta * q0 + math.sin(fraction * theta) / sin_theta * q1
    return _normalize_quaternion(result, min_norm)


def _limit_quaternion_delta(
    current: np.ndarray,
    target: np.ndarray,
    max_angle_deg: float,
    min_norm: float,
) -> np.ndarray:
    angle = _quaternion_angle_deg(current, target, min_norm)
    if angle <= max_angle_deg or angle == 0.0:
        return _normalize_quaternion(target, min_norm)
    return _slerp(current, target, max_angle_deg / angle, min_norm)


def _measured_side(observation: Mapping[str, Any], side: str) -> tuple[np.ndarray, np.ndarray]:
    arm_state = observation[f"arm_state_{side}"]
    hand_state = observation[f"hand_state_{side}"]
    eef = np.asarray(arm_state["eef"], dtype=np.float32).copy()
    eef[3:] = _normalize_quaternion(eef[3:], np.finfo(np.float32).eps)
    hand_deg = np.rad2deg(np.asarray(hand_state["joint_pos"], dtype=np.float32)).astype(np.float32)
    return eef, hand_deg


def _measured_joint_side(observation: Mapping[str, Any], side: str) -> tuple[np.ndarray, np.ndarray]:
    arm_joint_rad = np.asarray(observation[f"arm_state_{side}"]["joint_pos"], dtype=np.float32).copy()
    hand_joint_rad = np.asarray(observation[f"hand_state_{side}"]["joint_pos"], dtype=np.float32)
    if arm_joint_rad.shape != (7,) or not np.isfinite(arm_joint_rad).all():
        raise ProtocolError("COSMOS_INVALID_STATE", f"arm_state_{side}.joint_pos must be finite 7D")
    if hand_joint_rad.shape != (20,) or not np.isfinite(hand_joint_rad).all():
        raise ProtocolError("COSMOS_INVALID_STATE", f"hand_state_{side}.joint_pos must be finite 20D")
    return arm_joint_rad, np.rad2deg(hand_joint_rad).astype(np.float32)


def _hold_action(observation: Mapping[str, Any]) -> dict[str, Any]:
    left_eef, left_hand = _measured_side(observation, "left")
    right_eef, right_hand = _measured_side(observation, "right")
    return {
        "arm_action_left": {"ee_pos": left_eef[:3].tolist(), "ee_quat": left_eef[3:].tolist()},
        "hand_action_left": left_hand.tolist(),
        "arm_action_right": {"ee_pos": right_eef[:3].tolist(), "ee_quat": right_eef[3:].tolist()},
        "hand_action_right": right_hand.tolist(),
    }


def _joint_hold_action(observation: Mapping[str, Any]) -> dict[str, Any]:
    left_arm_joint, left_hand = _measured_joint_side(observation, "left")
    right_arm_joint, right_hand = _measured_joint_side(observation, "right")
    return {
        "arm_joint_action_left": left_arm_joint.tolist(),
        "hand_action_left": left_hand.tolist(),
        "arm_joint_action_right": right_arm_joint.tolist(),
        "hand_action_right": right_hand.tolist(),
    }


def build_native_right_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Build the training-time 27D state: right EEF xyzw, then hand radians."""

    right_eef, _ = _measured_side(observation, "right")
    right_hand = np.asarray(observation["hand_state_right"]["joint_pos"], dtype=np.float32)
    state = np.concatenate([right_eef, right_hand]).astype(np.float32)
    if state.shape != (27,) or not np.isfinite(state).all():
        raise ProtocolError("COSMOS_INVALID_STATE", "right policy state must be finite 27D")
    return state


def build_native_right_joint_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Build the joint-trained 27D state: right arm radians, then hand radians."""

    right_arm_joint = np.asarray(observation["arm_state_right"]["joint_pos"], dtype=np.float32)
    right_hand_joint = np.asarray(observation["hand_state_right"]["joint_pos"], dtype=np.float32)
    state = np.concatenate([right_arm_joint, right_hand_joint]).astype(np.float32)
    if state.shape != (27,) or not np.isfinite(state).all():
        raise ProtocolError("COSMOS_INVALID_STATE", "right joint policy state must be finite 27D")
    return state


def _smooth_endpoint_preserving(values: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Suppress alternating waypoint noise while preserving both trajectory endpoints."""

    sequence = np.concatenate([np.asarray(anchor, dtype=np.float64)[None], values.astype(np.float64)], axis=0)
    padded = np.pad(sequence, ((2, 2), (0, 0)), mode="edge")
    weights = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    filtered = sum(weights[index] * padded[index : index + len(sequence)] for index in range(len(weights)))

    # The symmetric filter rounds its boundaries. Add a linear correction so
    # the observed starting state and the model's final target stay exact.
    progress = np.linspace(0.0, 1.0, len(sequence), dtype=np.float64)[:, None]
    filtered += (1.0 - progress) * (sequence[0] - filtered[0])
    filtered += progress * (sequence[-1] - filtered[-1])
    return filtered[1:].astype(np.float32)


def _smooth_native_right_actions(observation: Mapping[str, Any], actions: np.ndarray, min_norm: float) -> np.ndarray:
    state = build_native_right_state(observation)
    smoothed = actions.copy()
    smoothed[:, :3] = _smooth_endpoint_preserving(actions[:, :3], state[:3])
    smoothed[:, 7:27] = _smooth_endpoint_preserving(actions[:, 7:27], state[7:27])

    # q and -q encode the same rotation but cannot be averaged together. Align
    # every xyzw quaternion to the preceding hemisphere before filtering.
    quaternion_sequence = np.empty((len(actions) + 1, 4), dtype=np.float32)
    quaternion_sequence[0] = _normalize_quaternion(state[3:7], min_norm)
    for index, quaternion in enumerate(actions[:, 3:7], start=1):
        normalized = _normalize_quaternion(quaternion, min_norm)
        if float(np.dot(normalized, quaternion_sequence[index - 1])) < 0.0:
            normalized = -normalized
        quaternion_sequence[index] = normalized
    filtered_quaternions = _smooth_endpoint_preserving(quaternion_sequence[1:], quaternion_sequence[0])
    for index, quaternion in enumerate(filtered_quaternions):
        smoothed[index, 3:7] = _normalize_quaternion(quaternion, min_norm)
    return smoothed


def _smooth_native_right_joint_actions(observation: Mapping[str, Any], actions: np.ndarray) -> np.ndarray:
    state = build_native_right_joint_state(observation)
    smoothed = actions.copy()
    smoothed[:, :7] = _smooth_endpoint_preserving(actions[:, :7], state[:7])
    smoothed[:, 7:27] = _smooth_endpoint_preserving(actions[:, 7:27], state[7:27])
    return smoothed


def _validate_action_chunk(
    observation: Mapping[str, Any],
    action_chunk: list[dict[str, Any]],
    config: RobotPolicyConfig,
) -> None:
    if len(action_chunk) != config.deployment.wire_chunk_size:
        raise ProtocolError("COSMOS_INVALID_ACTION", "action chunk length does not match the deployment manifest")

    previous: dict[str, tuple[np.ndarray, np.ndarray]] = {
        side: _measured_side(observation, side) for side in ("left", "right")
    }
    min_norm = float(config.safety.quaternion_min_norm)
    for step in action_chunk:
        if not isinstance(step, Mapping):
            raise ProtocolError("COSMOS_INVALID_ACTION", "action step must be a mapping")
        for side in ("left", "right"):
            arm = step.get(f"arm_action_{side}")
            hand_value = step.get(f"hand_action_{side}")
            if not isinstance(arm, Mapping):
                raise ProtocolError("COSMOS_INVALID_ACTION", f"missing arm_action_{side}")
            position = np.asarray(arm.get("ee_pos"), dtype=np.float64)
            quaternion = np.asarray(arm.get("ee_quat"), dtype=np.float64)
            hand = np.asarray(hand_value, dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ProtocolError("COSMOS_INVALID_ACTION", f"arm_action_{side}.ee_pos must be finite 3D")
            if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
                raise ProtocolError("COSMOS_INVALID_ACTION", f"arm_action_{side}.ee_quat must be finite 4D")
            quaternion = _normalize_quaternion(quaternion, min_norm)
            if hand.shape != (20,) or not np.isfinite(hand).all():
                raise ProtocolError("COSMOS_INVALID_ACTION", f"hand_action_{side} must be finite 20D")

            previous_eef, previous_hand = previous[side]
            if float(np.linalg.norm(position - previous_eef[:3])) > config.safety.max_eef_step_m:
                raise ProtocolError("COSMOS_UNSAFE_ACTION", f"arm_action_{side} position step exceeds the safety limit")
            if _quaternion_angle_deg(previous_eef[3:], quaternion, min_norm) > config.safety.max_eef_step_deg:
                raise ProtocolError("COSMOS_UNSAFE_ACTION", f"arm_action_{side} rotation step exceeds the safety limit")
            if float(np.max(np.abs(hand - previous_hand))) > config.safety.max_hand_step_deg:
                raise ProtocolError("COSMOS_UNSAFE_ACTION", f"hand_action_{side} step exceeds the safety limit")
            previous[side] = (np.concatenate([position, quaternion]), hand)


def native_right_actions_to_wire(
    observation: Mapping[str, Any],
    native_actions: np.ndarray,
    config: RobotPolicyConfig,
) -> list[dict[str, Any]]:
    """Convert absolute right-chest 27D model actions into bilateral wire actions."""

    actions = np.asarray(native_actions, dtype=np.float32)
    expected_shape = (config.model.native_chunk_size, config.model.native_action_dim)
    if actions.shape != expected_shape or not np.isfinite(actions).all():
        raise ProtocolError("COSMOS_INVALID_ACTION", f"model action must have shape {expected_shape}")

    left_eef, left_hand_deg = _measured_side(observation, "left")
    right_current_eef, right_current_hand_deg = _measured_side(observation, "right")
    min_norm = float(config.safety.quaternion_min_norm)
    if config.model.trajectory_smoothing == "binomial5":
        actions = _smooth_native_right_actions(observation, actions, min_norm)
    result: list[dict[str, Any]] = []
    for native in actions:
        right_position = native[:3].astype(np.float32)
        right_quaternion = _normalize_quaternion(native[3:7], min_norm)
        right_hand_deg = np.rad2deg(native[7:27]).astype(np.float32)

        if config.model.service_mode == "small_motion":
            right_position = _limit_vector_delta(
                right_current_eef[:3], right_position, float(config.safety.small_motion_max_eef_m)
            ).astype(np.float32)
            right_quaternion = _limit_quaternion_delta(
                right_current_eef[3:],
                right_quaternion,
                float(config.safety.small_motion_max_eef_deg),
                min_norm,
            )
            hand_delta = np.clip(
                right_hand_deg - right_current_hand_deg,
                -float(config.safety.small_motion_max_hand_deg),
                float(config.safety.small_motion_max_hand_deg),
            )
            right_hand_deg = right_current_hand_deg + hand_delta

        result.append(
            {
                "arm_action_left": {"ee_pos": left_eef[:3].tolist(), "ee_quat": left_eef[3:].tolist()},
                "hand_action_left": left_hand_deg.tolist(),
                "arm_action_right": {
                    "ee_pos": right_position.tolist(),
                    "ee_quat": right_quaternion.tolist(),
                },
                "hand_action_right": right_hand_deg.tolist(),
            }
        )
    _validate_action_chunk(observation, result, config)
    return result


def _validate_joint_action_chunk(
    observation: Mapping[str, Any],
    action_chunk: list[dict[str, Any]],
    config: RobotPolicyConfig,
) -> None:
    if len(action_chunk) != config.deployment.wire_chunk_size:
        raise ProtocolError("COSMOS_INVALID_ACTION", "action chunk length does not match the deployment manifest")

    expected_keys = {
        "arm_joint_action_left",
        "hand_action_left",
        "arm_joint_action_right",
        "hand_action_right",
    }
    previous = {side: _measured_joint_side(observation, side) for side in ("left", "right")}
    for step in action_chunk:
        if not isinstance(step, Mapping) or set(step) != expected_keys:
            raise ProtocolError("COSMOS_INVALID_ACTION", "joint action step has an invalid envelope")
        for side in ("left", "right"):
            arm = np.asarray(step[f"arm_joint_action_{side}"], dtype=np.float64)
            hand = np.asarray(step[f"hand_action_{side}"], dtype=np.float64)
            if arm.shape != (7,) or not np.isfinite(arm).all():
                raise ProtocolError("COSMOS_INVALID_ACTION", f"arm_joint_action_{side} must be finite 7D")
            if hand.shape != (20,) or not np.isfinite(hand).all():
                raise ProtocolError("COSMOS_INVALID_ACTION", f"hand_action_{side} must be finite 20D")

            previous_arm, previous_hand = previous[side]
            arm_step_deg = np.max(np.abs(np.rad2deg(arm - previous_arm)))
            if float(arm_step_deg) > config.safety.max_arm_joint_step_deg:
                raise ProtocolError("COSMOS_UNSAFE_ACTION", f"arm_joint_action_{side} step exceeds the safety limit")
            if float(np.max(np.abs(hand - previous_hand))) > config.safety.max_hand_step_deg:
                raise ProtocolError("COSMOS_UNSAFE_ACTION", f"hand_action_{side} step exceeds the safety limit")
            previous[side] = (arm.astype(np.float32), hand.astype(np.float32))


def native_right_joint_actions_to_wire(
    observation: Mapping[str, Any],
    native_actions: np.ndarray,
    config: RobotPolicyConfig,
) -> list[dict[str, Any]]:
    """Convert absolute 27D arm/hand joint targets into bilateral wire actions."""

    actions = np.asarray(native_actions, dtype=np.float32)
    expected_shape = (config.model.native_chunk_size, config.model.native_action_dim)
    if actions.shape != expected_shape or not np.isfinite(actions).all():
        raise ProtocolError("COSMOS_INVALID_ACTION", f"model action must have shape {expected_shape}")

    left_arm_joint, left_hand_deg = _measured_joint_side(observation, "left")
    right_current_arm_joint, right_current_hand_deg = _measured_joint_side(observation, "right")
    if config.model.trajectory_smoothing == "binomial5":
        actions = _smooth_native_right_joint_actions(observation, actions)

    result: list[dict[str, Any]] = []
    for native in actions:
        right_arm_joint = native[:7].astype(np.float32)
        right_hand_deg = np.rad2deg(native[7:27]).astype(np.float32)

        if config.model.service_mode == "small_motion":
            arm_limit_rad = np.deg2rad(float(config.safety.small_motion_max_arm_joint_deg))
            right_arm_joint = right_current_arm_joint + np.clip(
                right_arm_joint - right_current_arm_joint,
                -arm_limit_rad,
                arm_limit_rad,
            )
            right_hand_deg = right_current_hand_deg + np.clip(
                right_hand_deg - right_current_hand_deg,
                -float(config.safety.small_motion_max_hand_deg),
                float(config.safety.small_motion_max_hand_deg),
            )

        result.append(
            {
                "arm_joint_action_left": left_arm_joint.tolist(),
                "hand_action_left": left_hand_deg.tolist(),
                "arm_joint_action_right": right_arm_joint.tolist(),
                "hand_action_right": right_hand_deg.tolist(),
            }
        )
    _validate_joint_action_chunk(observation, result, config)
    return result


class HoldAdapter(ModelAdapter):
    """Diagnostic adapter that returns measured bilateral hold targets."""

    @property
    def is_ready(self) -> bool:
        return True

    def infer(self, observation: Mapping[str, Any]) -> AdapterOutput:
        action = _hold_action(observation)
        chunk = [action for _ in range(self.config.deployment.wire_chunk_size)]
        _validate_action_chunk(observation, chunk, self.config)
        return AdapterOutput(action_chunk=chunk, inference_ms=0.0)


class JointHoldAdapter(ModelAdapter):
    """Diagnostic adapter that holds measured bilateral arm and hand joints."""

    @property
    def is_ready(self) -> bool:
        return True

    def infer(self, observation: Mapping[str, Any]) -> AdapterOutput:
        action = _joint_hold_action(observation)
        chunk = [action for _ in range(self.config.deployment.wire_chunk_size)]
        _validate_joint_action_chunk(observation, chunk, self.config)
        return AdapterOutput(action_chunk=chunk, inference_ms=0.0)


def _maybe_init_distributed() -> None:
    import torch
    from torch import distributed as dist

    if not dist.is_available() or dist.is_initialized():
        return
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if os.getenv("WORLD_SIZE") is not None and os.getenv("RANK") is not None:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return
    torch.cuda.set_device(0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    dist.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


def _verify_checkpoint_sha256(checkpoint_path: str, expected: str) -> None:
    """Verify a file hash or a deterministic relative-path/content directory hash."""

    from pathlib import Path

    root = Path(checkpoint_path).expanduser()
    if not root.exists():
        raise ValueError(f"checkpoint_path does not exist: {root}")
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint_path has no files: {root}")
    for path in files:
        if root.is_dir():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}")


def _decode_jpeg_rgb(payload: Mapping[str, Any], camera_name: str) -> np.ndarray:
    import cv2

    encoded = np.frombuffer(payload["data"], dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ProtocolError("COSMOS_INVALID_IMAGE", f"failed to decode JPEG for {camera_name}")
    declared_shape = tuple(int(value) for value in payload["shape"])
    if bgr.shape != declared_shape:
        raise ProtocolError("COSMOS_INVALID_IMAGE", f"decoded JPEG shape disagrees with {camera_name} metadata")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _compose_right_wrist_over_head(head_rgb: np.ndarray, wrist_rgb: np.ndarray) -> Any:
    import torch
    import torchvision.transforms.functional as transforms_f

    head = torch.from_numpy(head_rgb).permute(2, 0, 1).contiguous()
    wrist = torch.from_numpy(wrist_rgb).permute(2, 0, 1).contiguous()
    target_width = min(head.shape[-1], wrist.shape[-1])

    def resize_to_width(image: Any) -> Any:
        height, width = image.shape[-2:]
        target_height = max(1, round(height * target_width / width))
        if (height, width) == (target_height, target_width):
            return image
        return transforms_f.resize(
            image,
            [target_height, target_width],
            interpolation=transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )

    return torch.cat([resize_to_width(wrist), resize_to_width(head)], dim=-2)


class SingleRightHandCosmosAdapter(ModelAdapter):
    """Cosmos WAM adapter for 27D single-right-hand EEF policies."""

    def __init__(self, config: RobotPolicyConfig) -> None:
        super().__init__(config)
        self._ready = False
        self._lock = threading.Lock()
        self._load_model()
        if config.model.warmup:
            self._warmup()
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _load_model(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Cosmos single-right-hand adapter")

        from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
        from cosmos_framework.inference.common.args import ConfigFileType
        from cosmos_framework.inference.common.init import init_output_dir
        from cosmos_framework.inference.inference import OmniInference

        class _FrozenSetupArgs(OmniSetupArgs):
            def load_model_config_dict(self) -> dict:
                model_dict = super().load_model_config_dict()
                model_dict.setdefault("config", {}).setdefault("ema", {})["enabled"] = False
                return model_dict

        model_cfg = self.config.model
        if model_cfg.checkpoint_sha256 is not None:
            assert model_cfg.checkpoint_path is not None
            _verify_checkpoint_sha256(model_cfg.checkpoint_path, model_cfg.checkpoint_sha256)
        _maybe_init_distributed()
        setup_overrides = OmniSetupOverrides.model_validate(
            {
                "checkpoint_path": model_cfg.checkpoint_path,
                "config_file": model_cfg.config_file,
                "experiment": model_cfg.experiment,
                "experiment_overrides": model_cfg.experiment_overrides,
                "output_dir": model_cfg.output_dir,
                "credential_path": model_cfg.credential_path,
                "sampler": model_cfg.sampler,
                "use_ema_weights": model_cfg.use_ema_weights,
                # Robot-policy inference returns actions, not user-facing media.
                # Avoid loading optional text/video guardrail models and deps.
                "guardrails": False,
            }
        )
        setup_args = setup_overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        if setup_args.config_file_type != ConfigFileType.MODULE:
            setup_args = _FrozenSetupArgs.model_validate(setup_args.model_dump())
        self.pipe = OmniInference.create(setup_args)
        self.model = self.pipe.model
        self.model.eval()
        model_max_action_dim = getattr(self.model.config, "max_action_dim", None)
        if model_max_action_dim != model_cfg.max_action_dim:
            raise ValueError(
                f"deployment max_action_dim={model_cfg.max_action_dim} does not match model max_action_dim={model_max_action_dim}"
            )
        if not getattr(self.model.config, "action_gen", False):
            raise ValueError("loaded checkpoint does not have an action generation head")
        self.input_video_key = getattr(self.model, "input_video_key", None)
        if self.input_video_key is None:
            self.input_video_key = self.model.config.input_video_key

    def _build_batch(self, images_rgb: Mapping[str, np.ndarray], right_state: np.ndarray) -> dict[str, Any]:
        import torch

        from cosmos_framework.data.generator.action.action_processing import (
            ActionProcessingRecord,
            make_batched_action_processing_fields,
        )
        from cosmos_framework.data.generator.action.domain_utils import get_domain_id
        from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
        from cosmos_framework.data.generator.action.transforms import (
            build_sequence_plan_from_mode,
            find_closest_target_size,
            reflection_pad_to_target,
        )

        composed = _compose_right_wrist_over_head(images_rgb["head"], images_rgb["right_wrist"])
        target_frames = self.config.model.native_chunk_size + 1
        _, height, width = composed.shape
        video = torch.zeros((3, target_frames, height, width), dtype=torch.uint8)
        video[:, 0] = composed

        target_w, target_h = find_closest_target_size(height, width, self.config.model.resolution)
        padded: dict[str, Any] = {"video": video}
        reflection_pad_to_target(padded, ["video"], True, target_w, target_h)
        video = padded["video"]
        image_size = padded["image_size"]

        action = torch.zeros((target_frames, self.config.model.max_action_dim), dtype=torch.float32)
        action[0, : self.config.model.native_action_dim] = torch.from_numpy(right_state)
        sequence_plan = build_sequence_plan_from_mode(
            mode="wam",
            video_length=target_frames,
            action_length=target_frames,
            has_text=True,
        )
        prompt_data: dict[str, Any] = {
            "ai_caption": self.config.model.task,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.config.deployment.action_rate_hz),
            "image_size": image_size,
            "mode": "wam",
            "viewpoint": "concat_view",
            "additional_view_description": (
                "The upper view is from the right wrist-mounted camera. "
                "The lower view is from the head-mounted third-person camera."
            ),
        }
        formatted = ActionPromptJsonFormatter(caption_key="ai_caption")(prompt_data)["ai_caption"]
        prompt = json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)
        record = ActionProcessingRecord(raw_action_dim=self.config.model.native_action_dim, action_normalizer=None)
        return {
            self.input_video_key: [[video]],
            "action": [[action]],
            **make_batched_action_processing_fields(record, batch_size=1),
            "mode": ["wam"],
            "ai_caption": [prompt],
            "prompt": [prompt],
            "conditioning_fps": [torch.tensor(self.config.deployment.action_rate_hz, dtype=torch.long)],
            "image_size": image_size.unsqueeze(0).to(device="cuda"),
            "domain_id": [torch.tensor(get_domain_id(self.config.model.domain_name), dtype=torch.long)],
            "sequence_plan": [sequence_plan],
        }

    def _infer_native(self, images_rgb: Mapping[str, np.ndarray], right_state: np.ndarray) -> tuple[np.ndarray, float]:
        import torch

        batch = self._build_batch(images_rgb, right_state)
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                samples = self.model.generate_samples_from_batch(
                    batch,
                    guidance=float(self.config.model.guidance),
                    seed=[self.config.model.seed],
                    num_steps=int(self.config.model.num_steps),
                    shift=float(self.config.model.shift),
                )
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as exc:
                self._ready = False
                torch.cuda.empty_cache()
                raise ProtocolError(
                    "COSMOS_GPU_OOM", "Cosmos model worker ran out of GPU memory", fatal_session=True
                ) from exc
        inference_ms = (time.perf_counter() - started) * 1000.0
        action = samples["action"][0].float().squeeze(0).detach().cpu().numpy()
        expected_shape = (self.config.model.native_chunk_size + 1, self.config.model.native_action_dim)
        if action.shape != expected_shape:
            raise ProtocolError(
                "COSMOS_INVALID_ACTION", f"Cosmos returned action shape {action.shape}, expected {expected_shape}"
            )
        return np.asarray(action[1:], dtype=np.float32), inference_ms

    def _warmup(self) -> None:
        images = {
            name: np.zeros(tuple(self.config.image_preprocessing.warmup_shapes[name]), dtype=np.uint8)
            for name in self.config.deployment.camera_names
        }
        state = np.zeros((self.config.model.native_action_dim,), dtype=np.float32)
        state[6] = 1.0
        self._infer_native(images, state)

    def infer(self, observation: Mapping[str, Any]) -> AdapterOutput:
        if not self._ready:
            raise ProtocolError("COSMOS_NOT_READY", "Cosmos model is not ready", fatal_session=True)
        images = observation["images"]
        images_rgb = {name: _decode_jpeg_rgb(images[name], name) for name in self.config.deployment.camera_names}
        right_state = build_native_right_state(observation)
        native_actions, inference_ms = self._infer_native(images_rgb, right_state)
        return AdapterOutput(
            action_chunk=native_right_actions_to_wire(observation, native_actions, self.config),
            inference_ms=inference_ms,
        )


class SingleRightHandJointCosmosAdapter(SingleRightHandCosmosAdapter):
    """Cosmos WAM adapter for 27D absolute arm/hand-joint policies."""

    def _warmup(self) -> None:
        images = {
            name: np.zeros(tuple(self.config.image_preprocessing.warmup_shapes[name]), dtype=np.uint8)
            for name in self.config.deployment.camera_names
        }
        state = np.zeros((self.config.model.native_action_dim,), dtype=np.float32)
        self._infer_native(images, state)

    def infer(self, observation: Mapping[str, Any]) -> AdapterOutput:
        if not self._ready:
            raise ProtocolError("COSMOS_NOT_READY", "Cosmos model is not ready", fatal_session=True)
        images = observation["images"]
        images_rgb = {name: _decode_jpeg_rgb(images[name], name) for name in self.config.deployment.camera_names}
        right_state = build_native_right_joint_state(observation)
        native_actions, inference_ms = self._infer_native(images_rgb, right_state)
        return AdapterOutput(
            action_chunk=native_right_joint_actions_to_wire(observation, native_actions, self.config),
            inference_ms=inference_ms,
        )


def create_model_adapter(config: RobotPolicyConfig) -> ModelAdapter:
    if config.model.service_mode == "hold":
        if config.deployment.action_space == "joint":
            return JointHoldAdapter(config)
        return HoldAdapter(config)
    if config.deployment.action_space == "joint":
        return SingleRightHandJointCosmosAdapter(config)
    return SingleRightHandCosmosAdapter(config)


__all__ = [
    "AdapterOutput",
    "HoldAdapter",
    "JointHoldAdapter",
    "ModelAdapter",
    "SingleRightHandCosmosAdapter",
    "SingleRightHandJointCosmosAdapter",
    "build_native_right_joint_state",
    "build_native_right_state",
    "create_model_adapter",
    "native_right_joint_actions_to_wire",
    "native_right_actions_to_wire",
]
