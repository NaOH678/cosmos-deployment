# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validated deployment configuration for the protocol-v2 robot policy service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Self

import pydantic
from omegaconf import OmegaConf

CameraName = Literal["head", "left_wrist", "right_wrist"]
HandSide = Literal["left", "right"]
ServiceMode = Literal["full", "hold", "small_motion"]
ActionSpace = Literal["eef", "joint"]


class _ConfigModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")


class DeploymentConfig(_ConfigModel):
    protocol_version: Literal[2] = 2
    schema_version: Literal[2] = 2
    model_id: str
    mode: Literal["right", "left", "both"] = "right"
    arm_command_mode: ActionSpace = "eef"
    action_space: ActionSpace = "eef"
    camera_names: list[CameraName] = pydantic.Field(min_length=1)
    robot_layout: dict[str, Any] | None = None
    active_hand_sides: list[HandSide] = pydantic.Field(default_factory=lambda: ["right"])
    zero_filled_hand_sides: list[HandSide] = pydantic.Field(default_factory=lambda: ["left"])
    action_rate_hz: Literal[15.0] = 15.0
    wire_chunk_size: pydantic.PositiveInt = 32
    startup_handoff: bool = False

    @pydantic.field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_id must not be empty")
        return value

    @pydantic.field_validator("camera_names", "active_hand_sides", "zero_filled_hand_sides")
    @classmethod
    def _validate_unique_list(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("values must be unique")
        return value


class ModelConfig(_ConfigModel):
    service_mode: ServiceMode = "full"
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    config_file: str = "cosmos_framework/configs/base/config.py"
    experiment: str = "action_policy_singlerighthand_edge"
    experiment_overrides: list[str] = pydantic.Field(default_factory=list)
    output_dir: Path = Path("outputs/cosmos_protocol_v2")
    credential_path: str = "credentials/gcp_checkpoint.secret"
    sampler: Literal["unipc", "edm"] = "unipc"
    task: str = "make a sandwich"
    domain_name: Literal["singlerighthand"] = "singlerighthand"
    seed: int = 0
    guidance: pydantic.PositiveFloat = 3.0
    num_steps: pydantic.PositiveInt = 4
    shift: pydantic.PositiveFloat = 5.0
    trajectory_smoothing: Literal["none", "binomial5"] = "none"
    resolution: str = "480"
    native_action_dim: Literal[27] = 27
    native_chunk_size: Literal[32] = 32
    max_action_dim: pydantic.PositiveInt = 64
    use_ema_weights: bool = True
    warmup: bool = True

    @pydantic.model_validator(mode="after")
    def _validate_checkpoint(self) -> Self:
        if self.service_mode != "hold" and not self.checkpoint_path:
            raise ValueError("checkpoint_path is required for full and small_motion modes")
        if self.max_action_dim < self.native_action_dim:
            raise ValueError("max_action_dim must be >= native_action_dim")
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if self.checkpoint_sha256 is not None:
            digest = self.checkpoint_sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
            self.checkpoint_sha256 = digest
        return self


class ImagePreprocessingConfig(_ConfigModel):
    codec: Literal["jpeg"] = "jpeg"
    input_color_space: Literal["bgr8"] = "bgr8"
    model_color_space: Literal["rgb"] = "rgb"
    composition: Literal["right_wrist_over_head"] = "right_wrist_over_head"
    keep_aspect_ratio: Literal[True] = True
    padding: Literal["reflection"] = "reflection"
    warmup_shapes: dict[CameraName, tuple[pydantic.PositiveInt, pydantic.PositiveInt, Literal[3]]] = pydantic.Field(
        default_factory=lambda: {"head": (480, 640, 3), "right_wrist": (480, 640, 3)}
    )


class TimestampConfig(_ConfigModel):
    max_image_source_skew_s: pydantic.NonNegativeFloat | None = None


class CoordinateFramesConfig(_ConfigModel):
    native_action_type: Literal["absolute"] = "absolute"
    observation_right_eef: Literal["right_chest"] = "right_chest"
    model_right_eef: Literal["right_chest"] = "right_chest"
    wire_right_eef: Literal["right_chest"] = "right_chest"
    model_to_wire_transform: Literal["identity"] = "identity"
    quaternion_order: Literal["xyzw"] = "xyzw"


class SafetyConfig(_ConfigModel):
    max_eef_step_m: pydantic.PositiveFloat = 0.15
    max_eef_step_deg: pydantic.PositiveFloat = 75.0
    max_arm_joint_step_deg: pydantic.PositiveFloat = 75.0
    max_hand_step_deg: pydantic.PositiveFloat = 75.0
    quaternion_min_norm: pydantic.PositiveFloat = 1e-6
    small_motion_max_eef_m: pydantic.PositiveFloat = 0.01
    small_motion_max_eef_deg: pydantic.PositiveFloat = 5.0
    small_motion_max_arm_joint_deg: pydantic.PositiveFloat = 5.0
    small_motion_max_hand_deg: pydantic.PositiveFloat = 5.0


class AuthConfig(_ConfigModel):
    api_key_env: str = "COSMOS_POLICY_API_KEY"
    api_key_file: Path | None = None

    def load_api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if value is None and self.api_key_file is not None:
            value = self.api_key_file.expanduser().read_text(encoding="utf-8")
        if value is None or not value.strip():
            raise ValueError(f"API key is not configured; set {self.api_key_env!r} or auth.api_key_file")
        return value.strip()


class ServiceConfig(_ConfigModel):
    host: str = "0.0.0.0"
    port: pydantic.PositiveInt = 8000
    endpoint: Literal["/v1/robot-policy"] = "/v1/robot-policy"
    max_request_bytes: pydantic.PositiveInt = 16 * 1024 * 1024
    max_response_bytes: pydantic.PositiveInt = 4 * 1024 * 1024
    max_sessions: pydantic.PositiveInt = 8
    session_ttl_s: pydantic.PositiveFloat = 120.0
    max_inflight_inferences: pydantic.PositiveInt = 1
    inference_timeout_s: pydantic.PositiveFloat = 30.0

    @pydantic.field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if value > 65535:
            raise ValueError("port must be <= 65535")
        return value


class RobotPolicyConfig(_ConfigModel):
    deployment: DeploymentConfig
    model: ModelConfig
    image_preprocessing: ImagePreprocessingConfig = pydantic.Field(default_factory=ImagePreprocessingConfig)
    timestamps: TimestampConfig = pydantic.Field(default_factory=TimestampConfig)
    coordinate_frames: CoordinateFramesConfig = pydantic.Field(default_factory=CoordinateFramesConfig)
    safety: SafetyConfig = pydantic.Field(default_factory=SafetyConfig)
    auth: AuthConfig = pydantic.Field(default_factory=AuthConfig)
    service: ServiceConfig = pydantic.Field(default_factory=ServiceConfig)

    @pydantic.model_validator(mode="after")
    def _validate_contract(self) -> Self:
        if self.deployment.mode != "right":
            raise ValueError("the current singlerighthand adapter only supports deployment.mode='right'")
        if self.deployment.active_hand_sides != ["right"]:
            raise ValueError("singlerighthand deployment requires active_hand_sides=['right']")
        if self.deployment.zero_filled_hand_sides != ["left"]:
            raise ValueError("singlerighthand deployment requires zero_filled_hand_sides=['left']")
        if self.deployment.arm_command_mode != self.deployment.action_space:
            raise ValueError("deployment.arm_command_mode and deployment.action_space must match")
        if self.deployment.robot_layout is not None:
            action_key = "arm_joint" if self.deployment.action_space == "joint" else "eef"
            expected_action_layout = [
                {"side": "left", action_key: [0, 7], "hand": [7, 27]},
                {"side": "right", action_key: [27, 34], "hand": [34, 54]},
            ]
            if self.deployment.robot_layout.get("action_layout") != expected_action_layout:
                raise ValueError(
                    f"robot_layout.action_layout must describe bilateral {self.deployment.action_space} targets"
                )
        expected_cameras = ["head", "right_wrist"]
        if self.deployment.camera_names != expected_cameras:
            raise ValueError("singlerighthand deployment requires camera_names=['head', 'right_wrist'] in this order")
        if not set(expected_cameras).issubset(self.image_preprocessing.warmup_shapes):
            raise ValueError("warmup_shapes must include head and right_wrist")
        if self.model.native_chunk_size != self.deployment.wire_chunk_size:
            raise ValueError("native_chunk_size and wire_chunk_size must match; temporal resampling is unsupported")
        return self


def load_robot_policy_config(path: Path | str) -> RobotPolicyConfig:
    """Load and strictly validate one YAML deployment manifest."""

    config_path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Robot policy config must contain a mapping: {config_path}")
    return RobotPolicyConfig.model_validate(raw)


__all__ = [
    "ActionSpace",
    "CameraName",
    "RobotPolicyConfig",
    "ServiceMode",
    "load_robot_policy_config",
]
