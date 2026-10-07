"""Robot-side ROS 2 deployment client for replay and cloud policies."""

from __future__ import annotations

import argparse
import copy
from collections import deque
import json
import math
import os
import threading
import time
from typing import Any, Mapping, Optional
import uuid

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64MultiArray, Int8, String
from std_srvs.srv import SetBool, Trigger

from stereocamera.shared_frames import DEFAULT_DIRECTORY, SharedFrameReader

from .config import load_config, section
from .deployment_protocol import (
    IMAGE_CODECS,
    PROTOCOL_VERSION,
    LatestActionPlan,
    PendingActionChunk,
    PchipActionInterpolator,
    ScheduledAction,
    VelocityContinuousBoundaryInterpolator,
    blend_action_prefix,
    bridge_early_action_window,
    calculate_prefetch_lead,
    encode_color_image,
    extract_action_chunk,
    interpolate_action_pair,
    select_pending_action_window,
    smooth_action_chunk_butterworth,
    validate_splice_kinematics,
)
from .deployment_trace import DeploymentTraceWriter
from .paper_async_blend import build_overlap_plan
from .paper_observation_clock import observation_origin
from .fdm_async import (
    FDM_PROTOCOL_MODE,
    FeedbackAccumulator,
    FeedbackDraft,
    FeedbackQueueFull,
    FdmActionChunk,
    FdmFeedbackWorker,
    FdmProtocolConfig,
    FdmSessionLedger,
    build_action_request,
    build_bootstrap,
    build_hello,
    parse_action_chunk,
    robot_layout_for_action_mode,
    validate_hello_ack,
)
from .policy_transport import (
    PolicyAuthenticationError,
    PolicyTransport,
    create_policy_transport,
    policy_transport_kind,
)
from .recorder_node import (
    ACTIVE_HAND_CHOICES,
    CAMERA_TRANSPORT_CHOICES,
    active_hand_sides,
    decode_compressed_image,
    decode_raw_image,
    pose_message_to_array,
    stamp_to_seconds,
)
from .replay_core import quaternion_angle_rad
from .schema import (
    ARM_ACTION_SPACE_JOINT,
    RobotLayout,
    normalize_arm_action_space,
    normalize_quaternion_xyzw,
)


SIDES = ("left", "right")
COMMAND_LIFECYCLES = (2, 10)  # READY and TARGET_HOLD recovery
PI_PROTOCOL_MODE = "pi_v2"
PROTOCOL_V2_MODE = "protocol_v2"
SYNCHRONOUS_PROTOCOL_MODES = (PROTOCOL_V2_MODE, PI_PROTOCOL_MODE)

STARTUP_BYPASSED = "BYPASSED"
STARTUP_WAIT_READY = "WAIT_READY"
STARTUP_WAIT_BOOTSTRAP = "WAIT_BOOTSTRAP_CHUNK"
STARTUP_HANDOFF_ACTIVE = "HANDOFF_ACTIVE"
STARTUP_WAIT_FRESH = "WAIT_FRESH_CHUNK"
STARTUP_RUNNING = "RUNNING"
STARTUP_FAILED = "FAILED"
STARTUP_STATES = {
    STARTUP_BYPASSED,
    STARTUP_WAIT_READY,
    STARTUP_WAIT_BOOTSTRAP,
    STARTUP_HANDOFF_ACTIVE,
    STARTUP_WAIT_FRESH,
    STARTUP_RUNNING,
    STARTUP_FAILED,
}

CONTROLLER_HANDOFF_UNKNOWN = "UNKNOWN"
CONTROLLER_HANDOFF_IDLE = "IDLE"
CONTROLLER_HANDOFF_WAITING = "WAITING_TARGET"
CONTROLLER_HANDOFF_ACTIVE = "ACTIVE"
CONTROLLER_HANDOFF_COMPLETE = "COMPLETE"
CONTROLLER_HANDOFF_FAILED = "FAILED"


def _uses_synchronous_joint_actions(node: Any) -> bool:
    return bool(
        getattr(node, "protocol_mode", "") in SYNCHRONOUS_PROTOCOL_MODES
        and getattr(node, "policy_transport", "") == "http"
        and getattr(node, "policy_action_mode", "eef") == "joint"
        and getattr(
            node,
            "arm_command_mode",
            getattr(node, "policy_action_mode", "eef"),
        )
        == "joint"
    )


def _uses_cosmos_joint_wire(node: Any) -> bool:
    return bool(
        getattr(node, "protocol_mode", "") == PROTOCOL_V2_MODE
        and _uses_synchronous_joint_actions(node)
    )


def _hand_actions_are_radians(node: Any) -> bool:
    # LingBot FDM joint actions use radians for both arms and hands. Synchronous
    # protocol-v2 joint actions deliberately retain the existing degree-valued
    # WujiHand wire contract.
    return bool(
        getattr(node, "policy_action_mode", "eef") == "joint"
        and not _uses_synchronous_joint_actions(node)
    )


def _bounded_action_snapshot(actions, max_actions: int = 64) -> dict[str, Any]:
    """Freeze a bounded, image-free trajectory for background JSON encoding."""
    values = [] if actions is None else actions
    return {
        "actions": copy.deepcopy(list(values[:max_actions])),
        "action_count": len(values),
        "truncated": len(values) > max_actions,
        "max_actions": max_actions,
    }


def _trace_event(node: Any, event: str, **fields: Any) -> None:
    writer = getattr(node, "_trace_writer", None)
    if writer is not None:
        writer.record(event, **fields)


def _target_jump_metrics(
    node: Any,
    candidate_poses: Mapping[str, np.ndarray],
    candidate_hands: Mapping[str, np.ndarray],
    previous_poses: Optional[Mapping[str, Optional[np.ndarray]]] = None,
    previous_hands: Optional[Mapping[str, Optional[np.ndarray]]] = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if previous_poses is None:
        previous_poses = node._last_applied_pose
    if previous_hands is None:
        previous_hands = node._last_applied_hand
    position_jump_m = {}
    rotation_jump_deg = {}
    hand_jump_deg = {}
    joint_action_mode = (
        getattr(node, "policy_action_mode", "eef") == "joint"
    )
    for side in node.active_arm_sides:
        previous_pose = previous_poses[side]
        if previous_pose is not None:
            if joint_action_mode:
                # Keep the existing three-metric return contract. In joint
                # mode this field is the maximum arm-joint delta, not an EEF
                # rotation delta; trace emission labels it explicitly.
                rotation_jump_deg[side] = float(
                    np.degrees(
                        np.max(
                            np.abs(candidate_poses[side] - previous_pose),
                            initial=0.0,
                        )
                    )
                )
            else:
                position_jump_m[side] = float(
                    np.linalg.norm(
                        candidate_poses[side][:3] - previous_pose[:3]
                    )
                )
                rotation_jump_deg[side] = float(
                    np.degrees(
                        quaternion_angle_rad(
                            candidate_poses[side][3:7], previous_pose[3:7]
                        )
                    )
                )
    for side in node.active_hand_sides:
        previous_hand = previous_hands[side]
        if previous_hand is not None:
            hand_delta = float(
                np.max(
                    np.abs(candidate_hands[side] - previous_hand), initial=0.0
                )
            )
            hand_jump_deg[side] = (
                float(np.degrees(hand_delta))
                if _hand_actions_are_radians(node)
                else hand_delta
            )
    return position_jump_m, rotation_jump_deg, hand_jump_deg


class ServerIdentityMismatch(ValueError):
    """The decoded response does not identify the synchronous request."""


class PolicyServerError(RuntimeError):
    """A structured cloud error, including its session-fatal disposition."""

    def __init__(self, message: str, *, fatal_session: bool) -> None:
        super().__init__(message)
        self.fatal_session = bool(fatal_session)


class DeploymentNode(Node):
    def __init__(
        self,
        config: Mapping[str, Any],
        require_cameras: Optional[bool] = None,
        active_hand: str = "both",
        active_arm: str = "both",
        arm_command_mode: str = "eef",
        camera_transport: Optional[str] = None,
        server: Optional[str] = None,
    ):
        super().__init__("wuji_deployment")
        self.config = dict(config)
        self.topics = section(config, "topics")
        self.deployment_config = section(config, "deployment")
        self.protocol_mode = str(
            self.deployment_config.get("protocol_mode", PI_PROTOCOL_MODE)
        ).strip().lower()
        if self.protocol_mode not in (
            *SYNCHRONOUS_PROTOCOL_MODES,
            FDM_PROTOCOL_MODE,
        ):
            raise ValueError(
                "deployment.protocol_mode must be protocol_v2, legacy pi_v2, "
                "or fdm_async"
            )
        self.fdm_config: Optional[FdmProtocolConfig] = None
        if self.protocol_mode == FDM_PROTOCOL_MODE:
            self.fdm_config = FdmProtocolConfig.from_deployment_config(
                self.deployment_config
            )
        self.protocol_version = (
            PROTOCOL_VERSION
            if self.fdm_config is None
            else self.fdm_config.protocol_version
        )
        self.active_hand_sides = active_hand_sides(active_hand)
        self.zero_filled_hand_sides = tuple(
            side for side in SIDES if side not in self.active_hand_sides
        )
        self.active_arm_sides = active_hand_sides(active_arm)
        self.arm_command_mode = str(arm_command_mode).strip().lower()
        if self.arm_command_mode not in ("eef", "joint"):
            raise ValueError("arm_command_mode must be eef or joint")
        # Replay's established joint hardware path still consumes EEF actions
        # plus a legacy top-level arm_joint_action_* field. LingBot FDM and
        # Cosmos HTTP profiles declare their action semantics explicitly.
        self.policy_action_mode = (
            self.fdm_config.action_mode
            if self.fdm_config is not None
            else "eef"
        )
        if self.protocol_mode == PROTOCOL_V2_MODE:
            self.policy_action_mode = str(
                self.deployment_config.get("action_space", "eef")
            ).strip().lower()
            if self.policy_action_mode not in ("eef", "joint"):
                raise ValueError(
                    "deployment.action_space must be eef or joint"
                )
            configured_command_mode = str(
                self.deployment_config.get(
                    "arm_command_mode", self.policy_action_mode
                )
            ).strip().lower()
            if configured_command_mode != self.policy_action_mode:
                raise ValueError(
                    "Cosmos action_space must match arm_command_mode: "
                    f"action_space={self.policy_action_mode}, "
                    f"config={configured_command_mode}"
                )
            if self.arm_command_mode != configured_command_mode:
                raise ValueError(
                    "Cosmos arm_command_mode must match launch mode: "
                    f"config={configured_command_mode}, "
                    f"launch={self.arm_command_mode}"
                )
        if (
            self.fdm_config is not None
            and self.arm_command_mode != self.policy_action_mode
        ):
            raise ValueError(
                "LingBot-VA FDM action_mode must match arm_command_mode: "
                f"config={self.policy_action_mode}, "
                f"launch={self.arm_command_mode}"
            )
        if self.active_arm_sides != self.active_hand_sides:
            raise ValueError(
                "deployment requires matching arm/hand sides; supported modes "
                "are one arm + same-side hand, or both arms + both hands"
            )
        if (
            self.protocol_mode == FDM_PROTOCOL_MODE
            and self.active_hand_sides != ("right",)
        ):
            raise ValueError(
                "LingBot-VA right-arm/right-WujiHand profile requires "
                "--active-hand right"
            )
        configured_require = bool(
            self.deployment_config.get("require_cameras", True)
        )
        self.require_cameras = (
            configured_require
            if require_cameras is None
            else bool(require_cameras)
        )
        configured_camera_names = self.deployment_config.get(
            "camera_names",
            section(config, "recording").get("camera_names", []),
        )
        self.camera_names = (
            [str(name) for name in configured_camera_names]
            if self.require_cameras
            else []
        )
        self.camera_transport = str(
            camera_transport
            or self.deployment_config.get("camera_transport", "direct")
        ).strip().lower()
        if self.camera_transport not in CAMERA_TRANSPORT_CHOICES:
            raise ValueError(
                "camera_transport must be one of "
                f"{CAMERA_TRANSPORT_CHOICES}, got {self.camera_transport!r}"
            )
        self.camera_shared_memory_dir = str(
            self.deployment_config.get(
                "camera_shared_memory_dir", DEFAULT_DIRECTORY
            )
        )
        self._camera_readers: dict[str, SharedFrameReader] = {}
        self._camera_identity = {
            name: (0, 0) for name in self.camera_names
        }
        self.server = str(
            server
            or self.deployment_config.get("server", "tcp://127.0.0.1:5555")
        ).strip()
        self.policy_transport = policy_transport_kind(self.server)
        self.expected_arm_action_space = ""
        self._negotiated_arm_action_space = ""
        if (
            self.protocol_mode == PI_PROTOCOL_MODE
            and self.policy_transport == "http"
            and "expected_arm_action_space" in self.deployment_config
        ):
            self.expected_arm_action_space = normalize_arm_action_space(
                self.deployment_config["expected_arm_action_space"],
                "deployment.expected_arm_action_space",
            )
            self.policy_action_mode = (
                "joint"
                if self.expected_arm_action_space
                == ARM_ACTION_SPACE_JOINT
                else "eef"
            )
            if self.arm_command_mode != self.policy_action_mode:
                raise ValueError(
                    "Pi expected_arm_action_space must match "
                    "arm_command_mode: "
                    f"action_space={self.expected_arm_action_space}, "
                    f"launch={self.arm_command_mode}"
                )
            if self.active_arm_sides != ("right",):
                raise ValueError(
                    "Pi joint profile requires --active-hand right"
                )
        if (
            _uses_cosmos_joint_wire(self)
            and self.active_arm_sides != ("right",)
        ):
            raise ValueError(
                "Cosmos joint profile requires --active-hand right"
            )
        self.request_rate_hz = float(self.deployment_config.get("request_rate_hz", 30.0))
        self.request_timeout_ms = int(self.deployment_config.get("request_timeout_ms", 1000))
        self.reconnect_interval_s = float(
            self.deployment_config.get("reconnect_interval_s", 1.0)
        )
        self.publish_rate_hz = float(
            self.deployment_config.get("publish_rate_hz", 120.0)
        )
        self.action_rate_hz = float(
            self.deployment_config.get("action_rate_hz", 30.0)
        )
        self.action_interpolation_method = str(
            self.deployment_config.get(
                "action_interpolation_method", "linear_slerp"
            )
        ).strip().lower()
        configured_smoothing_method = str(
            self.deployment_config.get("action_smoothing_method", "none")
        ).strip().lower()
        # The hosted policy returns a complete future chunk. Local replay may
        # use one- or two-action responses, which cannot support a zero-phase
        # chunk filter and must retain its existing semantics.
        self.action_smoothing_method = (
            configured_smoothing_method
            if self.policy_transport == "http"
            else "none"
        )
        self.action_smoothing_cutoff_hz = float(
            self.deployment_config.get("action_smoothing_cutoff_hz", 3.0)
        )
        self.action_smoothing_order = int(
            self.deployment_config.get("action_smoothing_order", 3)
        )
        self.open_loop_horizon = int(
            self.deployment_config.get("open_loop_horizon", 25)
        )
        self.max_observation_age_s = float(
            self.deployment_config.get("max_observation_age_s", 0.5)
        )
        self.policy_start_timeout_s = float(
            self.deployment_config.get("policy_start_timeout_s", 2.0)
        )
        configured_startup_handoff_gate = bool(
            self.deployment_config.get(
                "startup_handoff_gate_enabled", True
            )
        )
        # Only the synchronous protocol-v2 HTTP path uses the optional
        # controller handoff barrier. Replay and FDM keep their established
        # startup contracts.
        self.startup_handoff_gate_enabled = bool(
            configured_startup_handoff_gate
            and self.protocol_mode in SYNCHRONOUS_PROTOCOL_MODES
            and self.policy_transport == "http"
        )
        self.startup_handoff_timeout_s = float(
            self.deployment_config.get(
                "startup_handoff_timeout_s", 3.0
            )
        )
        self.image_codec = str(
            self.deployment_config.get("image_codec", "jpeg")
        ).strip().lower()
        self.jpeg_quality = int(
            self.deployment_config.get("jpeg_quality", 90)
        )
        self.policy_http_path = str(
            self.deployment_config.get("policy_http_path", "/v1/robot-policy")
        )
        self.policy_http_api_key_env = str(
            self.deployment_config.get(
                "policy_http_api_key_env", "PI05_HTTP_API_KEY"
            )
        ).strip()
        self.policy_http_max_response_bytes = int(
            self.deployment_config.get(
                "policy_http_max_response_bytes", 32 * 1024 * 1024
            )
        )
        self.expected_model_id = str(
            self.deployment_config.get("policy_http_expected_model_id", "")
        ).strip()
        self.policy_http_expected_chunk_size = int(
            self.deployment_config.get("policy_http_expected_chunk_size", 50)
        )
        self._pi_joint_lower_rad: Optional[np.ndarray] = None
        self.first_step_anchor_on_measured_pose = bool(
            self.deployment_config.get("first_step_anchor_on_measured_pose", False)
        )
        self._pi_joint_upper_rad: Optional[np.ndarray] = None
        self._pi_joint_velocity_rad_s: Optional[np.ndarray] = None
        self._joint_acceleration_rad_s2: Optional[np.ndarray] = None
        self._pi_joint_command_velocity_rad_s: Optional[np.ndarray] = None
        self.joint_step_velocity_validation_enabled = bool(
            self.deployment_config.get(
                "joint_step_velocity_validation_enabled", True
            )
        )
        self.small_motion_max_arm_joint_step_rad = float(
            self.deployment_config.get(
                "small_motion_max_arm_joint_step_rad", math.inf
            )
        )
        if (
            not np.isfinite(self.small_motion_max_arm_joint_step_rad)
            and self.small_motion_max_arm_joint_step_rad != math.inf
        ) or self.small_motion_max_arm_joint_step_rad <= 0.0:
            raise ValueError(
                "small_motion_max_arm_joint_step_rad must be positive"
            )
        if _uses_synchronous_joint_actions(self):
            safety_key = "joint_safety"
            safety = self.deployment_config.get(safety_key)
            if not isinstance(safety, Mapping):
                # Preserve compatibility with the established Pi profile.
                safety_key = "pi_joint_safety"
                safety = self.deployment_config.get(safety_key)
            if not isinstance(safety, Mapping):
                raise ValueError(
                    "synchronous joint deployment requires "
                    "deployment.joint_safety (or legacy pi_joint_safety)"
                )

            def safety_vector(key: str) -> np.ndarray:
                values = np.asarray(safety.get(key), dtype=np.float64).reshape(-1)
                if values.shape != (7,) or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"deployment.{safety_key}.{key} must contain "
                        "7 finite values"
                    )
                return values

            lower_deg = safety_vector("position_lower_deg")
            upper_deg = safety_vector("position_upper_deg")
            velocity_deg_s = safety_vector("velocity_limit_deg_s")
            acceleration_deg_s2 = (
                safety_vector("acceleration_limit_deg_s2")
                if "acceleration_limit_deg_s2" in safety
                else None
            )
            command_velocity_deg_s = safety_vector(
                "command_velocity_limit_deg_s"
            )
            if np.any(lower_deg >= upper_deg):
                raise ValueError(
                    "joint lower limits must be below upper limits"
                )
            if np.any(velocity_deg_s <= 0.0):
                raise ValueError(
                    "joint velocity limits must be positive"
                )
            if (
                acceleration_deg_s2 is not None
                and np.any(acceleration_deg_s2 <= 0.0)
            ):
                raise ValueError(
                    "joint acceleration limits must be positive"
                )
            if np.any(command_velocity_deg_s <= 0.0):
                raise ValueError(
                    "joint command velocity limits must be positive"
                )
            self._pi_joint_lower_rad = np.radians(lower_deg)
            self._pi_joint_upper_rad = np.radians(upper_deg)
            self._pi_joint_velocity_rad_s = np.radians(velocity_deg_s)
            self._joint_acceleration_rad_s2 = (
                None
                if acceleration_deg_s2 is None
                else np.radians(acceleration_deg_s2)
            )
            self._pi_joint_command_velocity_rad_s = np.radians(
                command_velocity_deg_s
            )
        if self.fdm_config is not None:
            if not self.expected_model_id:
                self.expected_model_id = self.fdm_config.model_id
            if self.expected_model_id != self.fdm_config.model_id:
                raise ValueError(
                    "policy_http_expected_model_id must equal "
                    "deployment.fdm_async.model_id"
                )
        configured_prefetch = bool(
            self.deployment_config.get("prefetch_enabled", True)
        )
        # Replay uses short synthetic chunks and remains strictly sequential.
        # Adaptive prefetch is for the hosted, profile-sized HTTP policy
        # contract; the chunk length is model-specific (for example, pi0.5
        # uses 50 actions while LingBot-VA uses 48).
        self.prefetch_enabled = (
            configured_prefetch and self.policy_transport == "http"
        )
        self.prefetch_latency_window = int(
            self.deployment_config.get("prefetch_latency_window", 100)
        )
        self.prefetch_latency_min_samples = int(
            self.deployment_config.get("prefetch_latency_min_samples", 5)
        )
        self.prefetch_safety_actions = int(
            self.deployment_config.get("prefetch_safety_actions", 2)
        )
        self.prefetch_min_lead_actions = int(
            self.deployment_config.get("prefetch_min_lead_actions", 3)
        )
        self.prefetch_max_lead_actions = int(
            self.deployment_config.get("prefetch_max_lead_actions", 10)
        )
        self.prefetch_initial_lead_actions = int(
            self.deployment_config.get("prefetch_initial_lead_actions", 5)
        )
        self.prefetch_time_alignment_enabled = bool(
            self.deployment_config.get(
                "prefetch_time_alignment_enabled", True
            )
        )
        self.prefetch_fixed_skip_actions = self.deployment_config.get("prefetch_fixed_skip_actions")
        if self.prefetch_fixed_skip_actions is not None:
            value = self.prefetch_fixed_skip_actions
            if (isinstance(value, bool) or not isinstance(value, int) or value < 0
                    or value + self.open_loop_horizon > self.policy_http_expected_chunk_size
                    or not self.prefetch_enabled or self.prefetch_time_alignment_enabled
                    or self.policy_action_mode != "eef" or self.protocol_mode == FDM_PROTOCOL_MODE
                    or self.deployment_config.get("early_splice_enabled", False)):
                raise ValueError("fixed skip requires prefetched EEF, no time alignment, and sufficient chunk length")
        self.rtc_prefix_guidance_enabled = bool(self.deployment_config.get("rtc_prefix_guidance_enabled", False))
        if self.rtc_prefix_guidance_enabled:
            if (not self.prefetch_enabled or self.prefetch_time_alignment_enabled
                    or self.prefetch_fixed_skip_actions is not None
                    or self.policy_action_mode != "eef"
                    or self.deployment_config.get("early_splice_enabled", False)
                    or self.action_interpolation_method != "linear_slerp"
                    or self.open_loop_horizon + self.prefetch_max_lead_actions > self.policy_http_expected_chunk_size
                    or self.protocol_mode == FDM_PROTOCOL_MODE):
                raise ValueError("RTC requires prefetched EEF, prefix-count alignment, no early splice, and prefix budget")
        self.rtc_prefix_method = str(self.deployment_config.get("rtc_prefix_method", "identity_jacobian_soft_prefix_v1"))
        if self.rtc_prefix_method not in ("identity_jacobian_soft_prefix_v1", "hybrid_vjp_step2_soft_prefix_v1"):
            raise ValueError("Unknown RTC prefix method")
        self._rtc_soft_tail = ()
        self.early_splice_enabled = bool(self.deployment_config.get("early_splice_enabled", False))
        self.early_splice_bridge_max_steps = int(self.deployment_config.get("early_splice_bridge_max_steps", 0))
        if self.early_splice_bridge_max_steps < 0:
            raise ValueError("early_splice_bridge_max_steps must be nonnegative")
        self.early_splice_max_age_s = float(self.deployment_config.get("early_splice_max_age_s", 1.2))
        self.early_splice_request_interval_s = float(self.deployment_config.get("early_splice_request_interval_s", 0.2))
        self.early_splice_limits = {
            "max_speed_m_s": float(self.deployment_config.get("early_splice_max_speed_m_s", 0.25)),
            "max_acceleration_m_s2": float(self.deployment_config.get("early_splice_max_acceleration_m_s2", 1.0)),
            "max_rotation_deg_s": float(self.deployment_config.get("early_splice_max_rotation_deg_s", 60.0)),
            "max_hand_deg_s": float(self.deployment_config.get("early_splice_max_hand_deg_s", 90.0)),
        }
        if self.early_splice_enabled:
            if (self.policy_transport != "http" or self.policy_action_mode != "eef"
                    or self.action_interpolation_method != "linear_slerp"
                    or not self.prefetch_enabled or not self.prefetch_time_alignment_enabled
                    or self.protocol_mode == FDM_PROTOCOL_MODE):
                raise ValueError("early splice requires HTTP EEF linear_slerp with prefetch time alignment")
            for value in (self.early_splice_max_age_s, self.early_splice_request_interval_s,
                          *self.early_splice_limits.values()):
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("early splice limits must be finite and positive")
        self._policy_request_not_before = 0.0
        configured_blend_steps = int(
            self.deployment_config.get("boundary_blend_steps", 6)
        )
        self.boundary_blend_method = str(
            self.deployment_config.get(
                "boundary_blend_method", "smoothstep"
            )
        ).strip().lower()
        if self.early_splice_enabled and self.boundary_blend_method != "smoothstep":
            raise ValueError("early splice currently requires smoothstep boundary blending")
        self.sequential_boundary_blend_enabled = bool(
            self.deployment_config.get("sequential_boundary_blend_enabled", False)
        )
        self.sequential_request_settle_s = float(
            self.deployment_config.get("sequential_request_settle_s", 0.0)
        )
        if not np.isfinite(self.sequential_request_settle_s) or not 0 <= self.sequential_request_settle_s <= 0.25:
            raise ValueError("sequential_request_settle_s must be in [0, 0.25]")
        if self.sequential_boundary_blend_enabled and (
            self.prefetch_enabled or self.policy_transport != "http"
            or self.policy_action_mode != "eef" or self.protocol_mode == FDM_PROTOCOL_MODE
            or self.boundary_blend_method != "smoothstep"
        ):
            raise ValueError("sequential blending requires non-prefetch HTTP EEF smoothstep")
        self.boundary_blend_steps = (
            configured_blend_steps if (self.prefetch_enabled or self.sequential_boundary_blend_enabled) else 0
        )
        hand_blend_override = self.deployment_config.get("boundary_hand_blend_steps")
        self.boundary_hand_blend_steps = None if hand_blend_override is None else int(hand_blend_override)
        if self.boundary_hand_blend_steps is not None:
            if (self.policy_action_mode != "eef" or self.boundary_blend_method != "smoothstep"
                    or self.protocol_mode == FDM_PROTOCOL_MODE or self.early_splice_enabled):
                raise ValueError("boundary_hand_blend_steps requires EEF smoothstep with early splice disabled")
            if not 0 <= self.boundary_hand_blend_steps <= self.boundary_blend_steps:
                raise ValueError("boundary_hand_blend_steps must be between zero and boundary_blend_steps")
        configured_initial_blend_steps = int(
            self.deployment_config.get("initial_blend_steps", 0)
        )
        self.initial_blend_steps = (
            configured_initial_blend_steps if self.prefetch_enabled else 0
        )
        self.trace_enabled = bool(
            self.deployment_config.get("diagnostic_trace_enabled", True)
        )
        self.trace_directory = str(
            self.deployment_config.get(
                "diagnostic_trace_directory",
                "/home/wuji/datasets/tianji_wuji/diagnostics",
            )
        )
        self.trace_queue_size = int(
            self.deployment_config.get("diagnostic_trace_queue_size", 4096)
        )
        self.trace_policy_chunk_enabled = bool(
            self.deployment_config.get("diagnostic_policy_chunk_enabled", False)
        )
        self.trace_flush_interval_s = float(
            self.deployment_config.get(
                "diagnostic_trace_flush_interval_s", 0.5
            )
        )
        if self.policy_transport != "http" and self.fdm_config is None:
            # Local LMDB replay and LAN adapters have their own model IDs.
            self.expected_model_id = ""
        self._http_api_key = ""
        if self.policy_transport == "http":
            if not self.policy_http_api_key_env:
                raise ValueError("policy_http_api_key_env must not be empty")
            self._http_api_key = os.environ.get(
                self.policy_http_api_key_env, ""
            )
            if not self._http_api_key:
                raise ValueError(
                    f"HTTP policy endpoint requires non-empty environment "
                    f"variable {self.policy_http_api_key_env}"
                )
        if self.image_codec not in IMAGE_CODECS:
            raise ValueError(
                f"deployment image_codec must be one of {IMAGE_CODECS}"
            )
        if self.action_interpolation_method not in (
            "none",
            "linear_slerp",
            "pchip_slerp",
            "linear_joint",
            "pchip_joint",
        ):
            raise ValueError(
                "action_interpolation_method must be none, linear_slerp, "
                "pchip_slerp, linear_joint, or pchip_joint"
            )
        if self.policy_action_mode == "joint" and (
            self.action_interpolation_method
            not in ("none", "linear_joint", "pchip_joint")
        ):
            raise ValueError(
                "joint policy actions require a joint interpolation method"
            )
        if self.policy_action_mode == "eef" and (
            self.action_interpolation_method
            in ("linear_joint", "pchip_joint")
        ):
            raise ValueError(
                "EEF policy actions cannot use a joint interpolation method"
            )
        if self.action_smoothing_method not in ("none", "butterworth"):
            raise ValueError(
                "action_smoothing_method must be none or butterworth"
            )
        if (
            self.policy_action_mode == "joint"
            and self.action_smoothing_method != "none"
        ):
            raise ValueError(
                "joint policy actions currently require "
                "action_smoothing_method=none"
            )
        if (
            _uses_synchronous_joint_actions(self)
            and self.boundary_blend_method == "velocity_continuous"
            and (self.boundary_blend_steps > 0 or self.initial_blend_steps > 0)
        ):
            raise ValueError(
                "synchronous joint actions currently support smoothstep boundary "
                "blending only"
            )
        if (
            not 0.0
            < self.action_smoothing_cutoff_hz
            < 0.5 * self.action_rate_hz
        ):
            raise ValueError(
                "action_smoothing_cutoff_hz must be between zero and "
                "the configured action-rate Nyquist frequency"
            )
        if not 1 <= self.action_smoothing_order <= 8:
            raise ValueError("action_smoothing_order must be in [1, 8]")
        if (
            self.action_interpolation_method != "none"
            and self.publish_rate_hz < self.action_rate_hz
        ):
            raise ValueError(
                "publish_rate_hz must be at least action_rate_hz when "
                "action interpolation is enabled"
            )
        if min(
            self.request_rate_hz,
            self.request_timeout_ms,
            self.reconnect_interval_s,
            self.publish_rate_hz,
            self.action_rate_hz,
            self.open_loop_horizon,
            self.max_observation_age_s,
            self.policy_start_timeout_s,
            self.startup_handoff_timeout_s,
            self.policy_http_max_response_bytes,
            self.policy_http_expected_chunk_size,
        ) <= 0.0:
            raise ValueError("deployment rates and timeouts must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("deployment jpeg_quality must be in [1, 100]")
        if (
            self.prefetch_latency_window <= 0
            or self.prefetch_latency_min_samples <= 0
        ):
            raise ValueError("prefetch latency window sizes must be positive")
        if self.prefetch_latency_min_samples > self.prefetch_latency_window:
            raise ValueError(
                "prefetch_latency_min_samples must not exceed the latency window"
            )
        if self.prefetch_safety_actions < 0:
            raise ValueError("prefetch_safety_actions must be non-negative")
        if self.trace_queue_size <= 0 or self.trace_flush_interval_s <= 0.0:
            raise ValueError(
                "deployment diagnostic trace settings must be positive"
            )
        if not (
            0
            < self.prefetch_min_lead_actions
            <= self.prefetch_initial_lead_actions
            <= self.prefetch_max_lead_actions
            <= self.open_loop_horizon - (0 if self.rtc_prefix_guidance_enabled else 1)
        ):
            raise ValueError(
                "prefetch lead values must satisfy 0 < min <= initial <= max "
                "< open_loop_horizon (RTC also permits equality)"
            )
        if not 0 <= self.boundary_blend_steps < self.open_loop_horizon:
            raise ValueError(
                "boundary_blend_steps must be in [0, open_loop_horizon)"
            )
        if not 0 <= self.initial_blend_steps < self.open_loop_horizon:
            raise ValueError(
                "initial_blend_steps must be in [0, open_loop_horizon)"
            )
        if self.boundary_blend_method not in (
            "smoothstep",
            "velocity_continuous",
        ):
            raise ValueError(
                "boundary_blend_method must be smoothstep or "
                "velocity_continuous"
            )
        if (
            self.policy_transport == "http"
            and self.prefetch_enabled
            and self.prefetch_time_alignment_enabled
            and self.policy_http_expected_chunk_size
            < self.open_loop_horizon + self.prefetch_max_lead_actions + 1
        ):
            raise ValueError(
                "HTTP action chunk must leave enough actions for the configured "
                "horizon and maximum prefetch alignment"
            )
        if self.fdm_config is not None:
            if not self.require_cameras:
                raise ValueError("LingBot-VA FDM requires real camera feedback")
            if tuple(self.camera_names) != self.fdm_config.camera_names:
                raise ValueError(
                    "deployment.camera_names must match the negotiated FDM cameras"
                )
            if self.policy_http_expected_chunk_size != self.fdm_config.wire_chunk_size:
                raise ValueError("FDM expected HTTP chunk size must be 48")
            if self.open_loop_horizon != self.fdm_config.wire_chunk_size:
                raise ValueError("FDM open_loop_horizon must be 48")
            if not np.isclose(self.action_rate_hz, self.fdm_config.action_rate_hz):
                raise ValueError("deployment and FDM action rates must match")
            if self.prefetch_time_alignment_enabled:
                raise ValueError("FDM observation-time action alignment must be disabled")
            if self.boundary_blend_steps != 0 or self.initial_blend_steps != 0:
                raise ValueError("initial FDM profile requires all boundary blend disabled")
        self.paper_async_blend_enabled = bool(self.deployment_config.get("paper_async_blend_enabled", False))
        self.paper_async_stride_steps = int(self.deployment_config.get("paper_async_stride_steps", 16))
        self.paper_async_weight_curve = str(self.deployment_config.get("paper_async_weight_curve", "linear"))
        if self.paper_async_blend_enabled:
            if self.paper_async_weight_curve not in ("linear", "smoothstep"):
                raise ValueError("paper_async_weight_curve must be linear or smoothstep")
            if (self.policy_transport != "http" or self.policy_action_mode != "eef"
                    or self.protocol_mode == FDM_PROTOCOL_MODE
                    or self.action_interpolation_method != "linear_slerp"
                    or self.prefetch_enabled or self.prefetch_time_alignment_enabled
                    or self.prefetch_fixed_skip_actions is not None
                    or self.early_splice_enabled or self.rtc_prefix_guidance_enabled
                    or self.boundary_blend_steps or self.initial_blend_steps
                    or self.boundary_hand_blend_steps not in (None, 0)
                    or self.sequential_request_settle_s != 0.0
                    or self.sequential_boundary_blend_enabled
                    or self.action_smoothing_method != "none"):
                raise ValueError("paper async requires HTTP EEF linear_slerp, no legacy prefetch/alignment/blend/filter/RTC")
            if self.get_parameter("use_sim_time").value:
                raise ValueError("paper async camera clock requires use_sim_time=false")
            if not 1 <= self.paper_async_stride_steps < self.policy_http_expected_chunk_size:
                raise ValueError("paper async stride must be in [1, chunk_size)")
        self._paper_bootstrapped = False
        self._paper_next_request_at = 0.0
        self._paper_request_origins = {}
        self._paper_previous_clock_pair = None
        self._paper_last_issued_origin = None
        self._paper_underrun_reported = False
        self._latest: dict[str, tuple[float, Any]] = {}
        self._latest_lock = threading.Lock()
        self._action_plan = LatestActionPlan(
            max_actions=(self.policy_http_expected_chunk_size if self.paper_async_blend_enabled else self.open_loop_horizon),
        )
        self._applied_sequence = 0
        self._interpolation_current: Optional[ScheduledAction] = None
        self._interpolation_current_start_at = 0.0
        self._interpolation_chunk_id = 0
        self._interpolation_chunk_action_index = 0
        self._chunk_interpolator: Optional[PchipActionInterpolator] = None
        self._chunk_interpolator_chunk_id = 0
        self._interpolated_commands = 0
        self._last_applied_pose: dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
        }
        self._last_applied_hand: dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
        }
        self._last_applied_zsp: dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
        }
        self._previous_applied_pose: dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
        }
        self._previous_applied_hand: dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
        }
        self._previous_applied_at = 0.0
        self._last_applied_at = 0.0
        self._pi_joint_last_published_pose: dict[
            str, Optional[np.ndarray]
        ] = {side: None for side in SIDES}
        self._pi_joint_last_published_at = 0.0
        self._pi_joint_rate_limit_events = 0
        self._boundary_interpolator: Optional[
            VelocityContinuousBoundaryInterpolator
        ] = None
        self._boundary_interpolator_chunk_id = 0
        self._boundary_interpolator_start_at = 0.0
        self._lifecycle: Optional[int] = None
        self._session_lock = threading.Lock()
        self._session_id = uuid.uuid4().hex
        self._request_sequence = 0
        self._stop_event = threading.Event()
        self._network_wakeup = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._trace_writer: Optional[DeploymentTraceWriter] = None
        self._prefetch_lock = threading.Lock()
        self._stream_generation = 0
        self._request_inflight = False
        self._pending_chunk: Optional[PendingActionChunk] = None
        self._latency_samples_ms: deque[float] = deque(
            maxlen=self.prefetch_latency_window
        )
        self._prefetch_lead_actions = self.prefetch_initial_lead_actions
        self._prefetch_p99_rtt_ms = 0.0
        self._prefetch_requests = 0
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self._prefetch_activation_failures = 0
        self._stale_generation_responses = 0
        self._last_activation_skip_actions = 0
        self._total_activation_skip_actions = 0
        self._last_boundary_wait_ms = 0.0
        self._max_boundary_wait_ms = 0.0
        self._waiting_for_pending_since = 0.0
        self._last_action_due_at = 0.0
        self._active_chunk_id = 0
        self._active_chunk_action_index = 0
        self._active_chunk_skip_actions = 0
        self._active_chunk_prefetched = False
        self._active_chunk_blend_steps = 0
        self._requests = 0
        self._failures = 0
        self._last_latency_ms = 0.0
        self._last_observation_age_ms = 0.0
        self._last_request_bytes = 0
        self._last_response_bytes = 0
        self._last_model_id = ""
        self._last_server_inference_ms = 0.0
        self._last_http_status: Optional[int] = None
        self._transport_reconnects = 0
        self._server_identity_mismatches = 0
        self._server_ready = False
        self._last_server_error = "policy server has not completed handshake"
        self._ready_entered_at = 0.0
        self._valid_policy_stream_started = False
        self._standby_requested_for_policy = False
        self._controller_handoff_state = CONTROLLER_HANDOFF_UNKNOWN
        self._controller_handoff_progress = 0.0
        self._controller_handoff_payload: dict[str, Any] = {}
        self._startup_state = (
            STARTUP_WAIT_READY
            if self.startup_handoff_gate_enabled
            else STARTUP_BYPASSED
        )
        self._startup_state_entered_at = time.monotonic()
        self._startup_anchor_action: Optional[Mapping[str, Any]] = None
        self._startup_anchor_validated: Optional[
            Mapping[str, tuple[np.ndarray, np.ndarray, Any]]
        ] = None
        self._startup_anchor_poses: Optional[
            Mapping[str, np.ndarray]
        ] = None
        self._startup_anchor_hands: Optional[
            Mapping[str, np.ndarray]
        ] = None
        self._startup_anchor_request_id: Optional[int] = None
        self._startup_anchor_publish_count = 0
        self._startup_seen_controller_active = False
        self._startup_failure_reason = ""
        self._fdm_reset_lock = threading.Lock()
        self._fdm_delivery_validation_lock = threading.Lock()
        self._fdm_ledger: Optional[FdmSessionLedger] = None
        self._fdm_feedback_accumulator: Optional[FeedbackAccumulator] = None
        self._fdm_feedback_worker: Optional[FdmFeedbackWorker] = None
        self._fdm_bootstrapped = False
        self._fdm_hello_accepted = False
        self._fdm_active_wire_chunk_id: Optional[int] = None
        self._fdm_active_global_action_start: Optional[int] = None
        self._fdm_active_native_spans: tuple[Mapping[str, int], ...] = ()
        self._fdm_last_reset_reason = ""
        self._fdm_underrun_resets = 0
        self._fdm_hold_last_publishes = 0
        self._fdm_waiting_for_fresh_state_after_ready = False
        self._fdm_fresh_state_wait_reason = ""
        self._fdm_delivered_tail_pose: dict[
            str, Optional[np.ndarray]
        ] = {side: None for side in SIDES}
        self._fdm_delivered_tail_hand: dict[
            str, Optional[np.ndarray]
        ] = {side: None for side in SIDES}
        if self.fdm_config is not None:
            self._fdm_ledger = FdmSessionLedger(self.fdm_config)
            self._fdm_ledger.reset(self._session_id, self._stream_generation)
            self._fdm_feedback_accumulator = FeedbackAccumulator(
                self.fdm_config,
                self._next_request_identity,
            )
            self._fdm_feedback_accumulator.reset(
                self._session_id, self._stream_generation
            )
            self._fdm_feedback_worker = FdmFeedbackWorker(
                self.fdm_config,
                transport_factory=self._new_feedback_transport,
                snapshot_builder=self._build_fdm_feedback_snapshot,
                on_fatal=self._fdm_feedback_fatal,
                on_event=lambda event, fields: _trace_event(
                    self, event, **dict(fields)
                ),
                autostart=False,
            )
            self._fdm_feedback_worker.reset_session(
                self._session_id, self._stream_generation
            )

        self._create_subscriptions()
        if self.camera_names and self.camera_transport == "direct":
            self._camera_readers = {
                name: SharedFrameReader(
                    name, directory=self.camera_shared_memory_dir
                )
                for name in self.camera_names
            }
            self.create_timer(1.0 / 60.0, self._poll_direct_cameras)
        self._create_publishers()
        self._status_publisher = self.create_publisher(String, "~/status", 10)
        self.create_service(Trigger, "~/ready", self._ready_callback)
        self._tianji_enable_client = self.create_client(
            SetBool, "/tianji_arm_controller/set_enabled"
        )
        self.create_timer(1.0 / self.publish_rate_hz, self._apply_pending_action)
        self.create_timer(1.0, self._publish_status)
        self.create_timer(0.2, self._enforce_policy_start_lease)
        if self.trace_enabled:
            self._trace_writer = DeploymentTraceWriter(
                self.trace_directory,
                session_id=self._session_id,
                recording_run_id=str(self.deployment_config.get("recording_run_id", "")),
                queue_size=self.trace_queue_size,
                flush_interval_s=self.trace_flush_interval_s,
            )
            _trace_event(
                self,
                "session_start",
                session_id=self._session_id,
                recording_run_id=str(self.deployment_config.get("recording_run_id", "")),
                recording_source_pipeline_config=str(self.deployment_config.get("recording_source_pipeline_config", "")),
                early_splice_enabled=self.early_splice_enabled,
                early_splice_bridge_max_steps=self.early_splice_bridge_max_steps,
                early_splice_limits=dict(self.early_splice_limits),
                early_splice_max_age_s=self.early_splice_max_age_s,
                server=self.server,
                policy_transport=self.policy_transport,
                protocol_mode=self.protocol_mode,
                protocol_version=self.protocol_version,
                expected_arm_action_space=(
                    self.expected_arm_action_space
                ),
                active_arm_sides=list(self.active_arm_sides),
                active_hand_sides=list(self.active_hand_sides),
                action_rate_hz=self.action_rate_hz,
                publish_rate_hz=self.publish_rate_hz,
                action_interpolation_method=(
                    self.action_interpolation_method
                ),
                action_smoothing_method=self.action_smoothing_method,
                diagnostic_policy_chunk_enabled=self.trace_policy_chunk_enabled,
                action_smoothing_cutoff_hz=(
                    self.action_smoothing_cutoff_hz
                ),
                action_smoothing_order=self.action_smoothing_order,
                open_loop_horizon=self.open_loop_horizon,
                prefetch_enabled=self.prefetch_enabled,
                prefetch_initial_lead_actions=(
                    self.prefetch_initial_lead_actions
                ),
                prefetch_time_alignment_enabled=(
                    self.prefetch_time_alignment_enabled
                ),
                boundary_blend_steps=self.boundary_blend_steps,
                boundary_hand_blend_steps=(self.boundary_blend_steps if self.boundary_hand_blend_steps is None else self.boundary_hand_blend_steps),
                boundary_blend_method=self.boundary_blend_method,
                initial_blend_steps=self.initial_blend_steps,
                first_step_anchor_on_measured_pose=self.first_step_anchor_on_measured_pose,
                startup_handoff_gate_enabled=(
                    self.startup_handoff_gate_enabled
                ),
                startup_handoff_timeout_s=(
                    self.startup_handoff_timeout_s
                ),
                joint_step_velocity_validation_enabled=bool(
                    _uses_synchronous_joint_actions(self)
                    and self.joint_step_velocity_validation_enabled
                ),
                joint_acceleration_validation_enabled=bool(
                    _uses_synchronous_joint_actions(self)
                    and self._joint_acceleration_rad_s2 is not None
                ),
            )
        self._worker = threading.Thread(
            target=self._network_loop, name="wuji-policy-client", daemon=True
        )
        if self._fdm_feedback_worker is not None:
            self._fdm_feedback_worker.start()
        self._worker.start()
        self.get_logger().info(
            f"Deployment client ready: server={self.server}, "
            f"policy_transport={self.policy_transport}, "
            f"cameras={self.camera_names or '<disabled>'}, "
            f"camera_transport={self.camera_transport}, "
            f"image_codec={self.image_codec}, "
            f"active_arms={self.active_arm_sides}, "
            f"arm_command_mode={self.arm_command_mode}, "
            f"expected_arm_action_space="
            f"{self.expected_arm_action_space or '<legacy>'}, "
            f"active_hands={self.active_hand_sides}, "
            f"zero_filled_hands={self.zero_filled_hand_sides or '<none>'}; "
            f"protocol_mode={self.protocol_mode}; "
            f"diagnostic_trace={self._trace_path() or '<disabled>'}; "
            "waiting for Tianji READY"
        )

    def _trace_policy_chunk(
        self, actions, observation, *, request_id, generation, stage,
        action_rate_hz, observation_created_at,
    ) -> None:
        """Queue an immutable action/state snapshot outside the publish loop.

        server_output already includes any server-side smoothing. Images and
        credentials are excluded; this record supports local trajectory replay.
        """
        if not getattr(self, "trace_policy_chunk_enabled", False):
            return
        if getattr(self, "_trace_writer", None) is None:
            return
        _trace_event(
            self, "policy_action_chunk",
            request_id=request_id, generation=generation, stage=stage,
            action_rate_hz=action_rate_hz,
            observation_created_at=observation_created_at,
            source_timestamps=copy.deepcopy(observation.get("source_timestamps", {})),
            observation_state=copy.deepcopy({
                key: observation[key]
                for side in SIDES
                for key in (f"arm_state_{side}", f"hand_state_{side}")
                if key in observation
            }),
            actions=copy.deepcopy(actions),
        )

    def _trace_path(self) -> str:
        if self._trace_writer is None:
            return ""
        return str(self._trace_writer.path)

    def _trace_status(self) -> dict[str, Any]:
        if self._trace_writer is None:
            return {
                "enabled": False,
                "path": "",
                "queued_events": 0,
                "dropped_events": 0,
            }
        return {
            "enabled": True,
            "path": str(self._trace_writer.path),
            "queued_events": self._trace_writer.queued_events,
            "dropped_events": self._trace_writer.dropped_events,
            "writer_error": getattr(self._trace_writer, "writer_error", ""),
            "written_events": getattr(self._trace_writer, "written_events", 0),
            "closed_cleanly": getattr(self._trace_writer, "closed_cleanly", False),
        }

    def _store(self, key: str, message: Any, value: Any = None) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        timestamp = stamp_to_seconds(message, now)
        with self._latest_lock:
            self._latest[key] = (timestamp, message if value is None else value)

    def _create_subscriptions(self) -> None:
        arm_topics = self.topics["arm"]
        hand_topics = self.topics["hand"]
        for side in SIDES:
            arm = arm_topics[side]
            self.create_subscription(
                JointState,
                arm["state"],
                lambda msg, s=side: self._store(f"arm_state_{s}", msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                PoseStamped,
                arm["actual_eef"],
                lambda msg, s=side: self._store(
                    f"arm_eef_{s}", msg, pose_message_to_array(msg)
                ),
                qos_profile_sensor_data,
            )
            if side in self.active_hand_sides:
                hand = hand_topics[side]
                self.create_subscription(
                    JointState,
                    hand["state"],
                    lambda msg, s=side: self._store(f"hand_state_{s}", msg),
                    qos_profile_sensor_data,
                )

        lifecycle_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Int8,
            "/tianji_arm/lifecycle_state",
            self._lifecycle_callback,
            lifecycle_qos,
        )
        self.create_subscription(
            String,
            "/tianji_arm/handoff_state",
            self._handoff_state_callback,
            lifecycle_qos,
        )

        if self.camera_transport == "ros":
            camera_topics = self.topics.get("camera", {})
            for name in self.camera_names:
                camera = camera_topics[name]
                if camera.get("type", "raw") == "compressed":
                    self.create_subscription(
                        CompressedImage,
                        camera["topic"],
                        lambda msg, n=name: self._store(f"camera_{n}", msg),
                        qos_profile_sensor_data,
                    )
                else:
                    self.create_subscription(
                        Image,
                        camera["topic"],
                        lambda msg, n=name: self._store(f"camera_{n}", msg),
                        qos_profile_sensor_data,
                    )

    def _set_startup_state(self, state: str, *, reason: str = "") -> None:
        normalized = str(state).strip().upper()
        if normalized not in STARTUP_STATES:
            raise ValueError(f"invalid deployment startup state: {state!r}")
        previous = getattr(self, "_startup_state", STARTUP_BYPASSED)
        if normalized == previous and not reason:
            return
        self._startup_state = normalized
        self._startup_state_entered_at = time.monotonic()
        if normalized == STARTUP_FAILED:
            self._startup_failure_reason = str(reason)
        _trace_event(
            self,
            "startup_state_change",
            previous=previous,
            current=normalized,
            reason=str(reason),
            controller_handoff_state=getattr(
                self, "_controller_handoff_state", CONTROLLER_HANDOFF_UNKNOWN
            ),
        )

    def _reset_startup_gate(self, *, ready: bool) -> None:
        if not getattr(self, "startup_handoff_gate_enabled", False):
            DeploymentNode._set_startup_state(self, STARTUP_BYPASSED)
            return
        self._startup_anchor_action = None
        self._startup_anchor_validated = None
        self._startup_anchor_poses = None
        self._startup_anchor_hands = None
        self._startup_anchor_request_id = None
        self._startup_anchor_publish_count = 0
        self._startup_seen_controller_active = False
        self._startup_failure_reason = ""
        DeploymentNode._set_startup_state(
            self,
            STARTUP_WAIT_BOOTSTRAP if ready else STARTUP_WAIT_READY
        )

    def _handoff_state_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, Mapping):
                raise ValueError("handoff state is not a JSON object")
            state = str(payload.get("state", "")).strip().upper()
            if state not in {
                CONTROLLER_HANDOFF_IDLE,
                CONTROLLER_HANDOFF_WAITING,
                CONTROLLER_HANDOFF_ACTIVE,
                CONTROLLER_HANDOFF_COMPLETE,
                CONTROLLER_HANDOFF_FAILED,
            }:
                raise ValueError(f"unknown controller handoff state {state!r}")
            progress = float(payload.get("progress", 0.0))
        except Exception as exc:
            self.get_logger().warn(
                f"Ignored invalid Tianji handoff state: {exc}"
            )
            return
        previous = self._controller_handoff_state
        self._controller_handoff_state = state
        self._controller_handoff_progress = float(
            np.clip(progress, 0.0, 1.0)
        )
        self._controller_handoff_payload = dict(payload)
        if (
            self._startup_state == STARTUP_HANDOFF_ACTIVE
            and state == CONTROLLER_HANDOFF_ACTIVE
        ):
            self._startup_seen_controller_active = True
        if previous != state:
            _trace_event(
                self,
                "controller_handoff_state",
                previous=previous,
                current=state,
                progress=self._controller_handoff_progress,
            )
        self._network_wakeup.set()

    def _lifecycle_callback(self, message: Int8) -> None:
        previous = self._lifecycle
        self._lifecycle = int(message.data)
        if previous != self._lifecycle:
            _trace_event(
                self,
                "lifecycle_change",
                previous=previous,
                current=self._lifecycle,
            )
        if self._lifecycle == 2 and previous not in COMMAND_LIFECYCLES:
            if _uses_synchronous_joint_actions(self):
                DeploymentNode._reset_pi_joint_runtime_safety(self)
            self._ready_entered_at = time.monotonic()
            self._fdm_waiting_for_fresh_state_after_ready = bool(
                getattr(self, "fdm_config", None) is not None
                and self.fdm_config.state_history_enabled
            )
            self._fdm_fresh_state_wait_reason = ""
            self._valid_policy_stream_started = bool(
                getattr(self, "fdm_config", None) is not None
                and self._fdm_bootstrapped
            )
            self._standby_requested_for_policy = False
            DeploymentNode._reset_startup_gate(self, ready=True)
            self._network_wakeup.set()
        if self._lifecycle not in COMMAND_LIFECYCLES:
            self._fdm_waiting_for_fresh_state_after_ready = False
            self._fdm_fresh_state_wait_reason = ""
            if previous != self._lifecycle:
                if getattr(self, "fdm_config", None) is not None:
                    if previous in COMMAND_LIFECYCLES:
                        self._fdm_reset_session(
                            "hardware lifecycle left command state",
                            request_standby=False,
                        )
                    # Before first Enable, W0 is intentionally buffered while
                    # hardware remains in a non-command lifecycle state.
                else:
                    self._clear_action_stream(clear_active=True)
            self._ready_entered_at = 0.0
            self._valid_policy_stream_started = False
            self._standby_requested_for_policy = False
            DeploymentNode._reset_startup_gate(self, ready=False)
            if previous in COMMAND_LIFECYCLES:
                if _uses_synchronous_joint_actions(self):
                    DeploymentNode._reset_pi_joint_runtime_safety(self)
                # A new Recovery/Enable establishes a different Cartesian
                # neutral.  Do not compare its first valid policy target with
                # the final target from the previous hardware session.
                self._last_applied_pose = {"left": None, "right": None}
                self._last_applied_hand = {"left": None, "right": None}
                self._last_applied_zsp = {"left": None, "right": None}
                self._previous_applied_pose = {
                    "left": None,
                    "right": None,
                }
                self._previous_applied_hand = {
                    "left": None,
                    "right": None,
                }
                self._previous_applied_at = 0.0
                self._last_applied_at = 0.0

    def _enforce_policy_start_lease(self) -> None:
        if getattr(self, "startup_handoff_gate_enabled", False):
            if (
                self._lifecycle not in COMMAND_LIFECYCLES
                or self._standby_requested_for_policy
            ):
                return
            state = self._startup_state
            elapsed = time.monotonic() - self._startup_state_entered_at
            if state == STARTUP_FAILED:
                self._request_policy_standby(
                    self._startup_failure_reason
                    or "deployment startup handoff failed"
                )
                return
            if state == STARTUP_HANDOFF_ACTIVE:
                if elapsed <= self.startup_handoff_timeout_s:
                    return
                reason = (
                    "Tianji external handoff did not complete within "
                    f"{self.startup_handoff_timeout_s:.1f}s"
                )
                self._set_startup_state(STARTUP_FAILED, reason=reason)
                self._request_policy_standby(reason)
                return
            if state in (STARTUP_WAIT_BOOTSTRAP, STARTUP_WAIT_FRESH):
                if elapsed <= self.policy_start_timeout_s:
                    return
                phase = (
                    "bootstrap action"
                    if state == STARTUP_WAIT_BOOTSTRAP
                    else "fresh post-handoff action"
                )
                reason = (
                    f"no valid {phase} within "
                    f"{self.policy_start_timeout_s:.1f}s"
                )
                self._set_startup_state(STARTUP_FAILED, reason=reason)
                self._request_policy_standby(reason)
            return
        if (
            self._lifecycle not in COMMAND_LIFECYCLES
            or self._valid_policy_stream_started
            or self._ready_entered_at <= 0.0
            or self._standby_requested_for_policy
        ):
            return
        elapsed = time.monotonic() - self._ready_entered_at
        if elapsed <= self.policy_start_timeout_s:
            return
        if not self._tianji_enable_client.service_is_ready():
            return
        self._standby_requested_for_policy = True
        self._last_server_error = (
            f"no valid policy action within {self.policy_start_timeout_s:.1f}s "
            "after Tianji READY"
        )
        self.get_logger().error(
            self._last_server_error + "; requesting Tianji standby"
        )
        request = SetBool.Request()
        request.data = False
        self._tianji_enable_client.call_async(request)

    def _request_policy_standby(self, reason: str) -> None:
        self._last_server_error = str(reason)
        if self._standby_requested_for_policy:
            return
        if not self._tianji_enable_client.service_is_ready():
            return
        self._standby_requested_for_policy = True
        self.get_logger().error(
            self._last_server_error + "; requesting Tianji standby"
        )
        request = SetBool.Request()
        request.data = False
        self._tianji_enable_client.call_async(request)

    def _poll_direct_cameras(self) -> None:
        for name, reader in self._camera_readers.items():
            try:
                frame = reader.latest()
            except (OSError, ValueError):
                continue
            if frame is None:
                continue
            identity = (reader.producer_generation, frame.sequence)
            if identity == self._camera_identity[name]:
                continue
            self._camera_identity[name] = identity
            with self._latest_lock:
                self._latest[f"camera_{name}"] = (
                    frame.timestamp,
                    frame.image,
                )

    def _create_publishers(self) -> None:
        qos = qos_profile_sensor_data
        self._arm_publishers = {}
        self._zsp_publishers = {}
        self._hand_publishers = {}
        for side in SIDES:
            arm = self.topics["arm"][side]
            if side in self.active_arm_sides:
                if getattr(self, "arm_command_mode", "eef") == "joint":
                    self._arm_publishers[side] = self.create_publisher(
                        JointState, f"/{side}_arm/external_joint_target", qos
                    )
                else:
                    self._arm_publishers[side] = self.create_publisher(
                        PoseStamped, arm["external_target"], qos
                    )
                    self._zsp_publishers[side] = self.create_publisher(
                        Float64MultiArray, arm["external_zsp"], qos
                    )
            if side in self.active_hand_sides:
                hand = self.topics["hand"][side]
                self._hand_publishers[side] = self.create_publisher(
                    JointState, hand["command"], qos
                )

    def _snapshot(self) -> dict[str, tuple[float, Any]]:
        with self._latest_lock:
            return dict(self._latest)

    def _robot_layout_metadata(self) -> dict[str, Any]:
        layout = RobotLayout().metadata()
        if _uses_cosmos_joint_wire(self):
            layout["action_layout"] = [
                {
                    "side": entry["side"],
                    "arm_joint": list(entry["arm"]),
                    "hand": list(entry["hand"]),
                }
                for entry in layout["qpos_layout"]
            ]
            layout["action_dim"] = layout["state_dim"]
            layout["units"] = {
                "qpos.arm": "radian",
                "qpos.hand": "radian",
                "action.arm_joint": "radian",
                "action.hand": "degree",
            }
            return layout
        if getattr(self, "fdm_config", None) is None:
            return layout
        return robot_layout_for_action_mode(
            layout, self.fdm_config.action_mode
        )

    def _initial_command_anchors(
        self, first_action: Mapping[str, Any]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, Optional[np.ndarray]],
    ]:
        """Build a fresh first-chunk anchor from measured robot state."""

        snapshot = self._snapshot()
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        empty_previous = {side: None for side in SIDES}
        validated, _, model_hands = self._validate_response(
            first_action,
            previous_poses=empty_previous,
            previous_hands=empty_previous,
        )
        poses: dict[str, np.ndarray] = {}
        hands: dict[str, np.ndarray] = {}
        zsp: dict[str, Optional[np.ndarray]] = {}
        for side in SIDES:
            joint_mode = (
                getattr(self, "policy_action_mode", "eef") == "joint"
            )
            pose_key = (
                f"arm_state_{side}" if joint_mode else f"arm_eef_{side}"
            )
            if pose_key not in snapshot:
                raise ValueError(
                    f"initial bridge missing measured {side} "
                    + ("joint state" if joint_mode else "EEF state")
                )
            pose_stamp, pose_value = snapshot[pose_key]
            pose_age = now_ros - float(pose_stamp)
            if pose_age > self.max_observation_age_s:
                raise ValueError(
                    f"initial bridge {side} "
                    + ("joint state" if joint_mode else "EEF state")
                    + " is stale "
                    f"({pose_age * 1000.0:.1f} ms)"
                )
            pose = (
                np.radians(
                    np.asarray(pose_value.position, dtype=np.float32)
                ).reshape(-1)
                if joint_mode
                else np.asarray(pose_value, dtype=np.float32).reshape(-1)
            )
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise ValueError(
                    f"initial bridge has invalid measured {side} "
                    + ("joint state" if joint_mode else "EEF state")
                )
            pose = pose.copy()
            if not joint_mode:
                pose[3:7] = normalize_quaternion_xyzw(pose[3:7])
            poses[side] = pose

            if side in self.active_hand_sides:
                hand_key = f"hand_state_{side}"
                if hand_key not in snapshot:
                    raise ValueError(
                        f"initial bridge missing measured {side} hand state"
                    )
                hand_stamp, hand_message = snapshot[hand_key]
                hand_age = now_ros - float(hand_stamp)
                if hand_age > self.max_observation_age_s:
                    raise ValueError(
                        f"initial bridge {side} hand state is stale "
                        f"({hand_age * 1000.0:.1f} ms)"
                    )
                hand = np.asarray(
                    hand_message.position, dtype=np.float32
                ).reshape(-1)
                if not _hand_actions_are_radians(self):
                    hand = np.degrees(hand)
                if hand.shape != (20,) or not np.all(np.isfinite(hand)):
                    raise ValueError(
                        f"initial bridge has invalid measured {side} hand state"
                    )
                hands[side] = hand.astype(np.float32)
            else:
                # Inactive hands are never published.  Use the model value so
                # their placeholder side cannot affect the active-side bridge.
                hands[side] = model_hands[side].copy()
            target_zsp = validated[side][2]
            zsp[side] = (
                None if target_zsp is None else target_zsp.copy()
            )
        return poses, hands, zsp

    def _next_request_identity(self) -> tuple[str, int]:
        with self._session_lock:
            self._request_sequence += 1
            return self._session_id, self._request_sequence

    def _clear_action_stream(self, *, clear_active: bool = True) -> None:
        """Invalidate pending/in-flight work without allowing a stale install."""

        with self._prefetch_lock:
            self._stream_generation += 1
            generation = self._stream_generation
            self._request_inflight = False
            self._pending_chunk = None
            if getattr(self, "paper_async_blend_enabled", False):
                self._paper_request_origins.clear()
                self._paper_previous_clock_pair = None
                if clear_active:
                    self._paper_bootstrapped = False
                    self._paper_next_request_at = 0.0
                    self._paper_last_issued_origin = None
                    self._paper_underrun_reported = False
            self._waiting_for_pending_since = 0.0
            if clear_active:
                self._rtc_soft_tail = ()
                self._last_action_due_at = 0.0
                self._action_plan.clear()
                self._interpolation_current = None
                self._interpolation_previous = None
                self._interpolation_current_start_at = 0.0
                self._interpolation_chunk_id = 0
                self._interpolation_chunk_action_index = 0
                self._chunk_interpolator = None
                self._chunk_interpolator_chunk_id = 0
                self._boundary_interpolator = None
                self._boundary_interpolator_chunk_id = 0
                self._boundary_interpolator_start_at = 0.0
                self._active_chunk_action_index = 0
                self._active_chunk_skip_actions = 0
                self._active_chunk_prefetched = False
                self._active_chunk_blend_steps = 0
                self._fdm_active_wire_chunk_id = None
                self._fdm_active_global_action_start = None
                self._fdm_active_native_spans = ()
        _trace_event(
            self,
            "action_stream_clear",
            generation=generation,
            clear_active=bool(clear_active),
            lifecycle=self._lifecycle,
        )
        self._network_wakeup.set()

    def _claim_policy_request(self) -> Optional[tuple[int, bool]]:
        """Reserve the sole request slot and identify a prefetch request."""

        if (
            getattr(self, "startup_handoff_gate_enabled", False)
            and self._startup_state not in (
                STARTUP_WAIT_BOOTSTRAP,
                STARTUP_WAIT_FRESH,
                STARTUP_RUNNING,
            )
        ):
            # No HTTP inference may overlap the fixed-target controller ramp.
            return None
        if getattr(self, "paper_async_blend_enabled", False):
            return self._claim_paper_policy_request()
        with self._prefetch_lock:
            if (getattr(self, "early_splice_enabled", False)
                    and time.monotonic() < getattr(self, "_policy_request_not_before", 0.0)):
                return None
            if self._request_inflight or self._pending_chunk is not None:
                return None
            remaining = self._action_plan.remaining()
            if not self.prefetch_enabled and remaining == 0:
                last_due = getattr(self, "_last_action_due_at", 0.0)
                if last_due > 0 and time.monotonic() < last_due + getattr(self, "sequential_request_settle_s", 0.0):
                    return None
            is_prefetch = self.prefetch_enabled and remaining > 0
            request_immediately = (
                getattr(self, "protocol_mode", PI_PROTOCOL_MODE)
                == FDM_PROTOCOL_MODE
                and remaining > 0
            )
            if remaining > 0 and not request_immediately and (
                not self.prefetch_enabled
                or remaining > self._prefetch_lead_actions
            ):
                return None
            self._request_inflight = True
            if getattr(self, "early_splice_enabled", False):
                self._policy_request_not_before = time.monotonic() + self.early_splice_request_interval_s
            generation = self._stream_generation
            if is_prefetch:
                self._prefetch_requests += 1
            lead_actions = self._prefetch_lead_actions
        _trace_event(
            self,
            "policy_request_claim",
            generation=generation,
            is_prefetch=is_prefetch,
            remaining_actions=remaining,
            lead_actions=lead_actions,
        )
        return generation, is_prefetch

    def _register_paper_request_clock(self, request_id, generation, origin, metadata):
        with self._prefetch_lock:
            if generation != self._stream_generation:
                raise RuntimeError("paper async generation changed before send")
            previous = self._paper_last_issued_origin
            if previous is not None and origin < previous:
                raise ValueError("paper camera observation origin moved backwards")
            self._paper_request_origins[request_id] = (origin, metadata)
            self._paper_previous_clock_pair = metadata["clock_pair"]
            self._paper_last_issued_origin = origin
            self._paper_next_request_at = origin + self.paper_async_stride_steps / self.action_rate_hz

    def _claim_paper_policy_request(self):
        """One request per prediction-origin stride; no backlog or lead heuristic."""
        now = time.monotonic()
        with self._prefetch_lock:
            if self._request_inflight or self._pending_chunk is not None:
                return None
            if self._paper_bootstrapped and now < self._paper_next_request_at:
                return None
            self._request_inflight = True
            generation = self._stream_generation
            remaining = self._action_plan.remaining()
            deadline = self._paper_next_request_at
        _trace_event(self, "policy_request_claim", generation=generation,
                     is_prefetch=remaining > 0, remaining_actions=remaining,
                     paper_async=True, paper_stride_steps=self.paper_async_stride_steps,
                     request_not_before=deadline, trigger_lateness_ms=max(0., now-deadline)*1000 if deadline else 0.)
        return generation, remaining > 0

    def _release_policy_request(self, *, generation: Optional[int] = None) -> None:
        with self._prefetch_lock:
            if generation is None or generation == self._stream_generation:
                self._request_inflight = False

    def _record_complete_policy_latency(self, latency_ms: float) -> None:
        with self._prefetch_lock:
            self._latency_samples_ms.append(float(latency_ms))
            lead, p99_ms = calculate_prefetch_lead(
                self._latency_samples_ms,
                action_rate_hz=self.action_rate_hz,
                safety_actions=self.prefetch_safety_actions,
                minimum_actions=self.prefetch_min_lead_actions,
                maximum_actions=self.prefetch_max_lead_actions,
                initial_actions=self.prefetch_initial_lead_actions,
                minimum_samples=self.prefetch_latency_min_samples,
            )
            self._prefetch_lead_actions = lead
            self._prefetch_p99_rtt_ms = p99_ms
            sample_count = len(self._latency_samples_ms)
        _trace_event(
            self,
            "prefetch_latency_update",
            complete_rtt_ms=float(latency_ms),
            p99_complete_rtt_ms=p99_ms,
            lead_actions=lead,
            latency_samples=sample_count,
        )

    def _store_pending_chunk(
        self,
        pending: PendingActionChunk,
        *,
        generation: int,
    ) -> bool:
        with self._prefetch_lock:
            if generation == self._stream_generation:
                self._request_inflight = False
            if (
                generation != self._stream_generation
                or (
                    self._lifecycle not in COMMAND_LIFECYCLES
                    and getattr(self, "protocol_mode", PI_PROTOCOL_MODE)
                    != FDM_PROTOCOL_MODE
                )
            ):
                self._stale_generation_responses += 1
                stored = False
            else:
                if self._pending_chunk is not None:
                    raise RuntimeError("a pending policy chunk is already buffered")
                self._pending_chunk = pending
                self._valid_policy_stream_started = True
                stored = True
        _trace_event(
            self,
            "pending_chunk_store" if stored else "stale_policy_response",
            request_id=pending.request_id,
            response_generation=generation,
            current_generation=self._stream_generation,
            action_count=len(pending.actions),
            align_to_observation=pending.align_to_observation,
            wire_chunk_id=pending.wire_chunk_id,
            global_action_start=pending.global_action_start,
            native_spans=list(pending.native_spans),
        )
        return stored

    def _prefetch_status(self) -> dict[str, Any]:
        with self._prefetch_lock:
            return {
                "enabled": self.prefetch_enabled,
                "early_splice_enabled": getattr(self, "early_splice_enabled", False),
                "early_splice_last_rejection": getattr(self, "_early_splice_last_rejection", ""),
                "early_splice_exhausted_wait": bool(
                    getattr(self, "early_splice_enabled", False)
                    and self._action_plan.remaining() == 0
                    and self._pending_chunk is None
                ),
                "lead_actions": self._prefetch_lead_actions,
                "p99_complete_rtt_ms": self._prefetch_p99_rtt_ms,
                "latency_samples": len(self._latency_samples_ms),
                "inflight": self._request_inflight,
                "pending_chunk_ready": self._pending_chunk is not None,
                "requests": self._prefetch_requests,
                "hits": self._prefetch_hits,
                "misses": self._prefetch_misses,
                "activation_failures": self._prefetch_activation_failures,
                "stale_generation_responses": self._stale_generation_responses,
                "last_activation_skip_actions": (
                    self._last_activation_skip_actions
                ),
                "total_activation_skip_actions": (
                    self._total_activation_skip_actions
                ),
                "last_boundary_wait_ms": self._last_boundary_wait_ms,
                "max_boundary_wait_ms": self._max_boundary_wait_ms,
            }

    def _build_observation(
        self,
        request_id: Optional[int] = None,
        created_monotonic: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        snapshot = self._snapshot()
        required = []
        for side in SIDES:
            required.append(f"arm_state_{side}")
            required.append(f"arm_eef_{side}")
            if side in self.active_hand_sides:
                required.append(f"hand_state_{side}")
        required.extend(f"camera_{name}" for name in self.camera_names)
        if any(key not in snapshot for key in required):
            return None
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        if any(now_ros - snapshot[key][0] > 1.0 for key in required):
            return None

        observation: dict[str, Any] = {
            "protocol_version": getattr(
                self, "protocol_version", PROTOCOL_VERSION
            ),
            "schema_version": 2,
            "message_type": "observation",
            "session_id": str(
                session_id
                if session_id is not None
                else getattr(self, "_session_id", "test-session")
            ),
            "request_id": int(request_id or 0),
            "arms": list(SIDES),
            "robot_layout": DeploymentNode._robot_layout_metadata(self),
            "arm_command_mode": getattr(self, "arm_command_mode", "eef"),
            "action_space": getattr(self, "policy_action_mode", "eef"),
            "images": {},
            "timestamp": time.time(),
            "client_monotonic": float(
                time.monotonic()
                if created_monotonic is None
                else created_monotonic
            ),
            "source_timestamps": {},
        }
        for side in SIDES:
            arm_message: JointState = snapshot[f"arm_state_{side}"][1]
            arm_state = {
                "joint_pos": np.radians(np.asarray(arm_message.position, dtype=np.float32)),
                "joint_vel": np.radians(
                    np.asarray(arm_message.velocity, dtype=np.float32)
                    if arm_message.velocity
                    else np.zeros(7, dtype=np.float32)
                ),
                "joint_torque": (
                    np.asarray(arm_message.effort, dtype=np.float32)
                    if arm_message.effort
                    else np.zeros(7, dtype=np.float32)
                ),
            }
            eef = np.asarray(
                snapshot[f"arm_eef_{side}"][1], dtype=np.float32
            )
            arm_state.update(
                {
                    "ee_pos": eef[:3].copy(),
                    "ee_quat": eef[3:7].copy(),
                    "eef": eef.copy(),
                }
            )
            if side in self.active_hand_sides:
                hand_message: JointState = snapshot[f"hand_state_{side}"][1]
                hand_state = {
                    "joint_pos": np.asarray(hand_message.position, dtype=np.float32),
                    "joint_vel": (
                        np.asarray(hand_message.velocity, dtype=np.float32)
                        if hand_message.velocity
                        else np.zeros(20, dtype=np.float32)
                    ),
                    "joint_torque": (
                        np.asarray(hand_message.effort, dtype=np.float32)
                        if hand_message.effort
                        else np.zeros(20, dtype=np.float32)
                    ),
                }
            else:
                hand_state = {
                    "joint_pos": np.zeros(20, dtype=np.float32),
                    "joint_vel": np.zeros(20, dtype=np.float32),
                    "joint_torque": np.zeros(20, dtype=np.float32),
                }
            observation[f"arm_state_{side}"] = arm_state
            observation[f"hand_state_{side}"] = hand_state
            observation["source_timestamps"][f"arm_state_{side}"] = float(
                snapshot[f"arm_state_{side}"][0]
            )
            observation["source_timestamps"][f"arm_eef_{side}"] = float(
                snapshot[f"arm_eef_{side}"][0]
            )
            if side in self.active_hand_sides:
                observation["source_timestamps"][f"hand_state_{side}"] = float(
                    snapshot[f"hand_state_{side}"][0]
                )

        observation["active_hand_sides"] = list(self.active_hand_sides)
        observation["zero_filled_hand_sides"] = list(self.zero_filled_hand_sides)

        for name in self.camera_names:
            message = snapshot[f"camera_{name}"][1]
            if isinstance(message, np.ndarray):
                frame = np.ascontiguousarray(message, dtype=np.uint8)
            else:
                frame = (
                    decode_compressed_image(message)
                    if isinstance(message, CompressedImage)
                    else decode_raw_image(message)
                )
            camera_timestamp = float(snapshot[f"camera_{name}"][0])
            observation["images"][name] = encode_color_image(
                frame,
                codec=getattr(self, "image_codec", "raw"),
                jpeg_quality=getattr(self, "jpeg_quality", 90),
                timestamp=camera_timestamp,
            )
            observation["source_timestamps"][f"camera_{name}"] = camera_timestamp
        return observation

    def _new_transport(self) -> PolicyTransport:
        return create_policy_transport(
            self.server,
            timeout_ms=self.request_timeout_ms,
            http_policy_path=self.policy_http_path,
            http_api_key=self._http_api_key,
            http_max_response_bytes=self.policy_http_max_response_bytes,
        )

    def _new_feedback_transport(self) -> PolicyTransport:
        if self.fdm_config is None:
            raise RuntimeError("feedback transport is only valid in fdm_async mode")
        return create_policy_transport(
            self.server,
            timeout_ms=self.request_timeout_ms,
            http_policy_path=self.fdm_config.feedback_http_path,
            http_api_key=self._http_api_key,
            http_max_response_bytes=self.policy_http_max_response_bytes,
        )

    def _build_fdm_feedback_snapshot(
        self, draft: FeedbackDraft
    ) -> Optional[Mapping[str, Any]]:
        """Encode only a complete named-camera snapshot after action four.

        This method runs exclusively in the feedback worker.  The 120 Hz
        callback only records action metadata and never performs JPEG work.
        """

        snapshot = self._snapshot()
        camera_keys = [f"camera_{name}" for name in self.camera_names]
        if any(key not in snapshot for key in camera_keys):
            return None
        if any(
            float(snapshot[key][0]) < draft.keyframe_not_before
            for key in camera_keys
        ):
            return None
        return self._build_observation(
            request_id=draft.request_id,
            created_monotonic=time.monotonic(),
            session_id=draft.session_id,
        )

    def _exchange(
        self,
        transport: PolicyTransport,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        exchange = transport.exchange(request)
        self._last_request_bytes = exchange.request_bytes
        self._last_response_bytes = exchange.response_bytes
        self._last_http_status = transport.last_http_status
        self._transport_reconnects = transport.reconnects
        response = exchange.response
        if not isinstance(response, Mapping):
            raise ValueError("policy response is not a mapping")
        request_session_id = request.get("session_id")
        request_id = request.get("request_id")
        response_session_id = response.get("session_id")
        response_request_id = response.get("request_id")
        response_keys = sorted(str(key) for key in response.keys())
        self.get_logger().debug(
            "Policy round trip identity: "
            f"request_session_id={request_session_id!r} "
            f"request_id={request_id!r} "
            f"raw_response_keys={response_keys!r} "
            f"response_session_id={response_session_id!r} "
            f"response_request_id={response_request_id!r}"
        )
        response_protocol = response.get("protocol_version")
        expected_protocol = getattr(self, "protocol_version", PROTOCOL_VERSION)
        if response_protocol is None or int(response_protocol) != expected_protocol:
            raise ValueError(
                f"unsupported policy protocol_version={response_protocol}; "
                f"expected {expected_protocol}"
            )
        if response.get("error"):
            # Error envelopes may be created before the cloud handler has
            # extracted request identity.  Surface the actual server failure
            # instead of masking it as an identity mismatch.
            raw_error = response.get("error")
            raw_message = response.get("message")
            if isinstance(raw_error, str) and raw_error.strip():
                server_error = raw_error.strip()
            elif raw_message is not None and str(raw_message).strip():
                server_error = str(raw_message).strip()
            else:
                server_error = str(raw_error).strip()
            if len(server_error) > 1000:
                server_error = f"{server_error[:1000]}..."
            error_code = str(
                response.get("error_code", "POLICY_SERVER_ERROR")
            ).strip()
            if len(error_code) > 100:
                error_code = f"{error_code[:100]}..."
            fatal_session = bool(response.get("fatal_session", False))
            raise PolicyServerError(
                f"policy server error [{error_code}]: {server_error}; "
                f"fatal_session={fatal_session}; "
                f"raw_response_keys={response_keys!r} "
                f"response_session_id={response_session_id!r} "
                f"response_request_id={response_request_id!r}",
                fatal_session=fatal_session,
            )
        if (
            response_session_id != request_session_id
            or response_request_id != request_id
        ):
            self._server_identity_mismatches += 1
            raise ServerIdentityMismatch(
                "SERVER_IDENTITY_MISMATCH "
                f"request_session_id={request_session_id!r} "
                f"request_id={request_id!r} "
                f"raw_response_keys={response_keys!r} "
                f"response_session_id={response_session_id!r} "
                f"response_request_id={response_request_id!r}"
            )
        return response

    def _probe_server(self, transport: PolicyTransport) -> None:
        session_id, request_id = self._next_request_identity()
        expected_action_space = str(
            getattr(self, "expected_arm_action_space", "")
        )
        if expected_action_space:
            with self._session_lock:
                self._negotiated_arm_action_space = ""
            # A hello defines a new action-semantic negotiation boundary.
            # No target accepted under the preceding negotiation may survive.
            self._clear_action_stream(clear_active=True)
        hello_request = {
            "protocol_version": PROTOCOL_VERSION,
            "message_type": "hello",
            "session_id": session_id,
            "request_id": request_id,
            "robot_layout": DeploymentNode._robot_layout_metadata(self),
            "arm_command_mode": getattr(self, "arm_command_mode", "eef"),
            "action_space": getattr(self, "policy_action_mode", "eef"),
            "camera_names": list(self.camera_names),
        }
        if expected_action_space:
            hello_request["expected_arm_action_space"] = (
                expected_action_space
            )
        response = self._exchange(
            transport,
            hello_request,
        )
        if response.get("message_type") != "hello_ack":
            raise ValueError("policy server did not acknowledge deployment hello")
        if int(response.get("request_id", -1)) != request_id:
            raise ValueError("policy hello response request_id mismatch")
        model_id = str(response.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("policy hello response is missing model_id")
        if self.expected_model_id and model_id != self.expected_model_id:
            raise ValueError(
                f"policy model_id mismatch: expected {self.expected_model_id!r}, "
                f"received {model_id!r}"
            )
        try:
            action_rate_hz = float(response.get("action_rate_hz"))
        except (TypeError, ValueError) as exc:
            raise ValueError("policy hello response has invalid action_rate_hz") from exc
        if not np.isfinite(action_rate_hz) or action_rate_hz <= 0.0:
            raise ValueError("policy hello response has invalid action_rate_hz")
        if self.policy_transport == "http" and not np.isclose(
            action_rate_hz, self.action_rate_hz
        ):
            raise ValueError(
                f"policy action_rate_hz mismatch: expected {self.action_rate_hz}, "
                f"received {action_rate_hz}"
            )
        negotiated_action_space = ""
        if expected_action_space:
            negotiated_action_space = normalize_arm_action_space(
                response.get("arm_action_space"),
                "policy hello arm_action_space",
            )
            if negotiated_action_space != expected_action_space:
                raise ValueError(
                    "policy arm_action_space mismatch: expected "
                    f"{expected_action_space!r}, received "
                    f"{negotiated_action_space!r}"
                )
        with self._session_lock:
            self._last_model_id = model_id
            self._negotiated_arm_action_space = negotiated_action_space
            self._server_ready = True
            self._last_server_error = ""
        _trace_event(
            self,
            "policy_handshake_ready",
            request_id=request_id,
            model_id=model_id,
            action_rate_hz=action_rate_hz,
            arm_action_space=negotiated_action_space,
        )

    def _ready_callback(self, _request, response):
        with self._session_lock:
            response.success = bool(self._server_ready)
            model_id = self._last_model_id
            action_space = getattr(
                self, "_negotiated_arm_action_space", ""
            )
        if response.success:
            response.message = (
                f"policy server ready: {self.server}; "
                f"model={model_id or '<unknown>'}"
                + (
                    f"; arm_action_space={action_space}"
                    if action_space
                    else ""
                )
            )
        else:
            response.message = self._last_server_error
        return response

    def _validate_action_envelope(
        self, response: Mapping[str, Any], request_id: int
    ) -> None:
        if response.get("message_type") != "action_chunk":
            raise ValueError("policy response is not an action_chunk")
        echoed_request = response.get("request_id")
        if int(echoed_request or -1) != request_id:
            raise ValueError(
                f"policy response request_id={echoed_request} does not "
                f"match request_id={request_id}"
            )
        response_model_id = str(response.get("model_id", "")).strip()
        if not response_model_id:
            raise ValueError("policy action response is missing model_id")
        if response_model_id != self._last_model_id:
            raise ValueError("policy action response model_id mismatch")
        expected_action_space = str(
            getattr(self, "expected_arm_action_space", "")
        )
        negotiated_action_space = str(
            getattr(self, "_negotiated_arm_action_space", "")
        )
        if expected_action_space:
            if not negotiated_action_space:
                raise ValueError(
                    "policy action received before arm_action_space handshake"
                )
            response_action_space = normalize_arm_action_space(
                response.get("arm_action_space"),
                "policy action arm_action_space",
            )
            if (
                response_action_space
                != negotiated_action_space
                or response_action_space != expected_action_space
            ):
                raise ValueError(
                    "policy action arm_action_space changed within session: "
                    f"expected {expected_action_space!r}, received "
                    f"{response_action_space!r}"
                )

    def _reconnect_transport(
        self,
        transport: PolicyTransport,
        *,
        clear_active: bool = True,
    ) -> None:
        _trace_event(
            self,
            "policy_transport_reconnect",
            clear_active=bool(clear_active),
            error=self._last_server_error,
        )
        with self._session_lock:
            self._server_ready = False
            self._last_model_id = ""
            self._negotiated_arm_action_space = ""
        if _uses_synchronous_joint_actions(self):
            clear_active = True
        self._clear_action_stream(clear_active=clear_active)
        transport.reconnect()
        self._transport_reconnects = transport.reconnects

    def _fdm_feedback_fatal(self, reason: str) -> None:
        self._fdm_reset_session(str(reason), request_standby=True)

    def _fdm_reset_session(
        self,
        reason: str,
        *,
        request_standby: bool,
    ) -> None:
        """Invalidate every old-session queue/frontier and request bootstrap."""

        if self.fdm_config is None:
            return
        with self._fdm_reset_lock:
            with self._session_lock:
                old_session_id = self._session_id
                self._session_id = uuid.uuid4().hex
                self._request_sequence = 0
                new_session_id = self._session_id
                self._server_ready = False
                self._last_model_id = ""
            self._last_server_error = str(reason)
            self._fdm_last_reset_reason = str(reason)
            self._fdm_bootstrapped = False
            self._fdm_hello_accepted = False
            self._valid_policy_stream_started = False
            self._clear_action_stream(clear_active=True)
            generation = self._stream_generation
            assert self._fdm_ledger is not None
            assert self._fdm_feedback_accumulator is not None
            assert self._fdm_feedback_worker is not None
            with self._fdm_delivery_validation_lock:
                self._fdm_delivered_tail_pose = {
                    side: None for side in SIDES
                }
                self._fdm_delivered_tail_hand = {
                    side: None for side in SIDES
                }
                self._fdm_ledger.reset(new_session_id, generation)
            self._fdm_feedback_accumulator.reset(new_session_id, generation)
            self._fdm_feedback_worker.reset_session(new_session_id, generation)
            _trace_event(
                self,
                "fdm_session_reset",
                old_session_id=old_session_id,
                session_id=new_session_id,
                generation=generation,
                reason=str(reason),
                request_standby=bool(request_standby),
            )
        if (
            request_standby
            and self._lifecycle in COMMAND_LIFECYCLES
            and self._tianji_enable_client.service_is_ready()
        ):
            self._standby_requested_for_policy = True
            request = SetBool.Request()
            request.data = False
            self._tianji_enable_client.call_async(request)

    def _fdm_network_failure(
        self, transport: PolicyTransport, message: str
    ) -> None:
        self._server_ready = False
        self._last_server_error = str(message)
        self._failures += 1
        self._last_http_status = transport.last_http_status
        try:
            transport.reconnect()
        finally:
            self._transport_reconnects = transport.reconnects

    def _fdm_handle_exchange_failure(
        self, transport: PolicyTransport, exc: Exception
    ) -> bool:
        """Reconnect transient failures; reset immediately on fatal failures."""

        self._fdm_network_failure(transport, str(exc))
        fatal = bool(
            isinstance(exc, (PolicyAuthenticationError, ValueError))
            or (
                isinstance(exc, PolicyServerError)
                and exc.fatal_session
            )
        )
        if fatal:
            self._fdm_reset_session(
                f"FDM fatal action connection failure: {exc}",
                request_standby=True,
            )
        return fatal

    def _fdm_network_loop(self) -> None:
        """Action connection: hello/bootstrap, then idempotent wire prefetch."""

        assert self.fdm_config is not None
        assert self._fdm_ledger is not None
        try:
            transport = self._new_transport()
        except Exception as exc:
            self._last_server_error = str(exc)
            self.get_logger().error(str(exc))
            return
        hello_request: Optional[dict[str, Any]] = None
        bootstrap_request: Optional[dict[str, Any]] = None
        action_request: Optional[dict[str, Any]] = None
        request_generation = -1
        request_started_at = 0.0
        request_is_prefetch = False
        try:
            while not self._stop_event.is_set():
                with self._session_lock:
                    session_id = self._session_id
                generation = self._stream_generation
                if request_generation != generation:
                    hello_request = None
                    bootstrap_request = None
                    action_request = None
                    request_generation = generation

                if not self._fdm_hello_accepted:
                    if hello_request is None:
                        identity_session, request_id = self._next_request_identity()
                        if identity_session != session_id:
                            continue
                        hello_request = build_hello(
                            self.fdm_config,
                            session_id=session_id,
                            request_id=request_id,
                            robot_layout=self._robot_layout_metadata(),
                        )
                    try:
                        response = self._exchange(transport, hello_request)
                    except Exception as exc:
                        self._fdm_handle_exchange_failure(transport, exc)
                        self._stop_event.wait(
                            self.fdm_config.action_retry_backoff_s
                        )
                        continue
                    try:
                        validate_hello_ack(
                            response,
                            self.fdm_config,
                            session_id=session_id,
                            request_id=int(hello_request["request_id"]),
                        )
                    except Exception as exc:
                        self._fdm_reset_session(
                            f"FDM hello capability failure: {exc}",
                            request_standby=True,
                        )
                        continue
                    if generation != self._stream_generation:
                        continue
                    self._fdm_hello_accepted = True
                    self._last_model_id = self.fdm_config.model_id
                    hello_request = None
                    _trace_event(
                        self,
                        "fdm_hello_ack",
                        session_id=session_id,
                        generation=generation,
                    )

                if not self._fdm_bootstrapped:
                    if bootstrap_request is None:
                        identity_session, request_id = self._next_request_identity()
                        if identity_session != session_id:
                            continue
                        observation = self._build_observation(
                            request_id=request_id,
                            created_monotonic=time.monotonic(),
                            session_id=session_id,
                        )
                        if observation is None:
                            self._network_wakeup.wait(0.02)
                            self._network_wakeup.clear()
                            continue
                        bootstrap_request = build_bootstrap(
                            observation,
                            self.fdm_config,
                            session_id=session_id,
                            request_id=request_id,
                        )
                        request_started_at = time.monotonic()
                    try:
                        response = self._exchange(transport, bootstrap_request)
                    except Exception as exc:
                        self._fdm_handle_exchange_failure(transport, exc)
                        self._stop_event.wait(
                            self.fdm_config.action_retry_backoff_s
                        )
                        continue
                    if generation != self._stream_generation:
                        continue
                    try:
                        chunk = parse_action_chunk(
                            response,
                            self.fdm_config,
                            session_id=session_id,
                            request_id=int(bootstrap_request["request_id"]),
                            wire_chunk_id=0,
                            global_action_start=0,
                        )
                        self._fdm_validate_and_accept_action_chunk(
                            chunk, generation=generation
                        )
                        received_at = time.monotonic()
                        pending = PendingActionChunk(
                            actions=chunk.actions,
                            action_rate_hz=self.fdm_config.action_rate_hz,
                            observation_created_at=request_started_at,
                            response_received_at=received_at,
                            align_to_observation=False,
                            request_id=chunk.request_id,
                            generation=generation,
                            wire_chunk_id=chunk.wire_chunk_id,
                            global_action_start=chunk.global_action_start,
                            native_spans=tuple(
                                span.as_mapping() for span in chunk.native_spans
                            ),
                        )
                        if not self._store_pending_chunk(
                            pending, generation=generation
                        ):
                            continue
                    except Exception as exc:
                        self._fdm_reset_session(
                            f"FDM bootstrap response failure: {exc}",
                            request_standby=True,
                        )
                        continue
                    if generation != self._stream_generation:
                        continue
                    latency_ms = (time.monotonic() - request_started_at) * 1000.0
                    self._record_complete_policy_latency(latency_ms)
                    self._requests += 1
                    self._last_latency_ms = latency_ms
                    self._server_ready = True
                    self._last_server_error = ""
                    self._fdm_bootstrapped = True
                    bootstrap_request = None
                    _trace_event(
                        self,
                        "fdm_bootstrap_ready",
                        session_id=session_id,
                        generation=generation,
                        wire_chunk_id=0,
                        global_action_start=0,
                        latency_ms=latency_ms,
                    )
                    continue

                if self._lifecycle not in COMMAND_LIFECYCLES:
                    self._network_wakeup.wait(0.05)
                    self._network_wakeup.clear()
                    continue

                if action_request is None:
                    claim = self._claim_policy_request()
                    if claim is None:
                        self._network_wakeup.wait(0.02)
                        self._network_wakeup.clear()
                        continue
                    claimed_generation, request_is_prefetch = claim
                    if claimed_generation != generation:
                        self._release_policy_request()
                        continue
                    (
                        delivery_session,
                        delivery_generation,
                        wire_chunk_id,
                        global_action_start,
                    ) = self._fdm_ledger.next_delivery()
                    if (
                        delivery_session != session_id
                        or delivery_generation != generation
                    ):
                        self._release_policy_request()
                        continue
                    identity_session, request_id = self._next_request_identity()
                    if identity_session != session_id:
                        self._release_policy_request()
                        continue
                    action_request = build_action_request(
                        self.fdm_config,
                        session_id=session_id,
                        request_id=request_id,
                        wire_chunk_id=wire_chunk_id,
                        global_action_start=global_action_start,
                    )
                    request_started_at = time.monotonic()
                    request_generation = generation
                    _trace_event(
                        self,
                        "fdm_action_request_send",
                        request_id=request_id,
                        wire_chunk_id=wire_chunk_id,
                        global_action_start=global_action_start,
                        generation=generation,
                    )
                try:
                    response = self._exchange(transport, action_request)
                except Exception as exc:
                    # Keep action_request unchanged across reconnects. The
                    # control callback owns the transient pending-miss reset
                    # deadline. Fatal auth/session/protocol failures reset now.
                    self._fdm_handle_exchange_failure(transport, exc)
                    self._stop_event.wait(
                        self.fdm_config.action_retry_backoff_s
                    )
                    continue
                if generation != self._stream_generation:
                    action_request = None
                    continue
                try:
                    chunk = parse_action_chunk(
                        response,
                        self.fdm_config,
                        session_id=session_id,
                        request_id=int(action_request["request_id"]),
                        wire_chunk_id=int(action_request["wire_chunk_id"]),
                        global_action_start=int(
                            action_request["global_action_start"]
                        ),
                    )
                    self._fdm_validate_and_accept_action_chunk(
                        chunk, generation=generation
                    )
                    received_at = time.monotonic()
                    pending = PendingActionChunk(
                        actions=chunk.actions,
                        action_rate_hz=self.fdm_config.action_rate_hz,
                        observation_created_at=request_started_at,
                        response_received_at=received_at,
                        align_to_observation=False,
                        request_id=chunk.request_id,
                        generation=generation,
                        wire_chunk_id=chunk.wire_chunk_id,
                        global_action_start=chunk.global_action_start,
                        native_spans=tuple(
                            span.as_mapping() for span in chunk.native_spans
                        ),
                    )
                    if not self._store_pending_chunk(
                        pending, generation=generation
                    ):
                        action_request = None
                        continue
                except Exception as exc:
                    self._release_policy_request()
                    self._fdm_reset_session(
                        f"FDM action response failure: {exc}",
                        request_standby=True,
                    )
                    action_request = None
                    continue
                if generation != self._stream_generation:
                    action_request = None
                    continue
                latency_ms = (time.monotonic() - request_started_at) * 1000.0
                self._record_complete_policy_latency(latency_ms)
                self._requests += 1
                self._last_latency_ms = latency_ms
                self._server_ready = True
                self._last_server_error = ""
                _trace_event(
                    self,
                    "fdm_action_chunk_ready",
                    request_id=chunk.request_id,
                    wire_chunk_id=chunk.wire_chunk_id,
                    global_action_start=chunk.global_action_start,
                    native_spans=[
                        span.as_mapping() for span in chunk.native_spans
                    ],
                    complete_rtt_ms=latency_ms,
                    prefetched=bool(request_is_prefetch),
                )
                action_request = None
        finally:
            transport.close()

    def _network_loop(self) -> None:
        if self.fdm_config is not None:
            self._fdm_network_loop()
            return
        try:
            transport = self._new_transport()
        except Exception as exc:
            self._server_ready = False
            self._last_server_error = str(exc)
            self.get_logger().error(str(exc))
            return
        interval = 1.0 / self.request_rate_hz
        next_probe_at = 0.0
        try:
            while not self._stop_event.is_set():
                started = time.monotonic()
                # Gate replay/policy progression on the same lifecycle that
                # authorizes Tianji motion.  This also keeps Wuji hands still
                # before the operator explicitly enables the session.
                if self._lifecycle not in COMMAND_LIFECYCLES:
                    if started >= next_probe_at:
                        try:
                            self._probe_server(transport)
                        except Exception as exc:
                            self._server_ready = False
                            self._last_server_error = str(exc)
                            self._failures += 1
                            self.get_logger().warn(
                                f"Policy handshake failed: {exc}"
                            )
                            self._last_http_status = transport.last_http_status
                            self._reconnect_transport(
                                transport, clear_active=True
                            )
                        next_probe_at = (
                            time.monotonic() + self.reconnect_interval_s
                        )
                    self._stop_event.wait(min(interval, 0.1))
                    continue
                if not self._server_ready:
                    if started >= next_probe_at:
                        try:
                            self._probe_server(transport)
                        except Exception as exc:
                            self._server_ready = False
                            self._last_server_error = str(exc)
                            self._failures += 1
                            self.get_logger().warn(
                                f"Policy handshake failed: {exc}"
                            )
                            self._last_http_status = transport.last_http_status
                            self._reconnect_transport(
                                transport,
                                clear_active=self._action_plan.remaining() == 0,
                            )
                        next_probe_at = (
                            time.monotonic() + self.reconnect_interval_s
                        )
                    self._stop_event.wait(min(interval, 0.1))
                    continue

                claim = self._claim_policy_request()
                if claim is None:
                    self._network_wakeup.wait(min(interval, 0.1))
                    self._network_wakeup.clear()
                    continue
                generation, is_prefetch = claim
                session_id, request_id = self._next_request_identity()
                observation_created_at = time.monotonic()
                observation = self._build_observation(
                    request_id=request_id,
                    created_monotonic=observation_created_at,
                    session_id=session_id,
                )
                if observation is None:
                    self._release_policy_request(generation=generation)
                    self._network_wakeup.wait(min(interval, 0.1))
                    self._network_wakeup.clear()
                    continue
                source_clock_now = self.get_clock().now().nanoseconds * 1e-9
                paper_clock_monotonic = time.monotonic()
                source_timestamps = observation.get("source_timestamps", {})
                _trace_event(
                    self,
                    "policy_request_send",
                    request_id=request_id,
                    generation=generation,
                    is_prefetch=is_prefetch,
                    observation_created_at=observation_created_at,
                    active_remaining_at_send=self._action_plan.remaining(),
                    source_timestamps=dict(source_timestamps),
                    source_age_ms={key: (source_clock_now - float(stamp)) * 1000.0
                                   for key, stamp in source_timestamps.items()},
                    source_age_clock="ros",
                    source_clock_at_send=source_clock_now,
                    observation_build_ms=(time.monotonic() - observation_created_at) * 1000.0,
                )
                try:
                    if getattr(self, "paper_async_blend_enabled", False):
                        origin, clock_metadata = observation_origin(
                            observation, source_clock_ros=source_clock_now,
                            captured_monotonic=paper_clock_monotonic,
                            previous_clock_pair=self._paper_previous_clock_pair,
                            max_age_s=self.max_observation_age_s)
                        self._register_paper_request_clock(request_id, generation, origin, clock_metadata)
                        _trace_event(self, "paper_async_request_clock", request_id=request_id,
                                     generation=generation, observation_origin=origin,
                                     request_not_before=self._paper_next_request_at,
                                     stride_steps=self.paper_async_stride_steps, **clock_metadata)
                    rtc_prefix_steps = None
                    if getattr(self, "rtc_prefix_guidance_enabled", False):
                        from .rtc_prefix import build_prefix
                        with self._prefetch_lock:
                            if generation != self._stream_generation:
                                raise RuntimeError("RTC stream changed before snapshot")
                            queue = self._action_plan.snapshot()
                            tail = self._rtc_soft_tail if queue else ()
                        prefix = build_prefix(queue, tail, horizon=self.open_loop_horizon, method=self.rtc_prefix_method)
                        rtc_prefix_steps = prefix["committed_steps"]
                        observation["rtc_prefix"] = prefix
                        _trace_event(self, "rtc_prefix_request", request_id=request_id,
                                     generation=generation, prefix=prefix)
                    response = self._exchange(transport, observation)
                    if rtc_prefix_steps is not None:
                        ack = response.get("rtc_prefix_guidance", {})
                        if (ack.get("method") != self.rtc_prefix_method
                                or ack.get("committed_steps") != rtc_prefix_steps):
                            raise ValueError("RTC server did not acknowledge prefix guidance")
                        _trace_event(self, "rtc_prefix_response", request_id=request_id, guidance=ack)

                    self._validate_action_envelope(
                        response, request_id
                    )
                    server_timing = response.get("server_timing")
                    if isinstance(server_timing, Mapping):
                        self._last_server_inference_ms = float(
                            server_timing.get("inference_ms", 0.0)
                        )
                    received_at = time.monotonic()
                    observation_age = received_at - observation_created_at
                    self._last_observation_age_ms = observation_age * 1000.0
                    if observation_age > self.max_observation_age_s:
                        raise TimeoutError(
                            f"policy response used a {observation_age:.3f}s-old "
                            "observation"
                        )
                    actions, action_rate_hz = extract_action_chunk(
                        response,
                        default_rate_hz=self.action_rate_hz,
                    )
                    actions = [
                        DeploymentNode._canonicalize_action(self, action)
                        for action in actions
                    ]
                    self._trace_policy_chunk(
                        actions, observation, request_id=request_id,
                        generation=generation, stage="server_output",
                        action_rate_hz=action_rate_hz,
                        observation_created_at=observation_created_at,
                    )
                    observation_eef_positions = None
                    if (
                        self.policy_action_mode == "eef"
                        and getattr(self, "first_step_anchor_on_measured_pose", False)
                    ):
                        observation_eef_positions = {
                            side: np.asarray(
                                observation[f"arm_state_{side}"]["eef"],
                                dtype=np.float32,
                            ).copy()
                            for side in SIDES
                        }
                    observation_joint_positions = {
                        side: np.asarray(
                            observation[f"arm_state_{side}"]["joint_pos"],
                            dtype=np.float32,
                        ).reshape(7)
                        for side in SIDES
                    }
                    observation_hand_positions = {
                        side: np.degrees(
                            np.asarray(
                                observation[f"hand_state_{side}"][
                                    "joint_pos"
                                ],
                                dtype=np.float32,
                            ).reshape(20)
                        ).astype(np.float32)
                        for side in SIDES
                    }
                    try:
                        self._validate_action_chunk(
                            actions,
                            action_rate_hz,
                            observation_eef_positions=observation_eef_positions,
                            observation_joint_positions=(
                                observation_joint_positions
                            ),
                            observation_hand_positions=(
                                observation_hand_positions
                            ),
                        )
                    except ValueError as exc:
                        raise ValueError(f"raw policy chunk: {exc}") from exc
                    if self.action_smoothing_method == "butterworth":
                        actions = smooth_action_chunk_butterworth(
                            actions,
                            rate_hz=action_rate_hz,
                            cutoff_hz=self.action_smoothing_cutoff_hz,
                            order=self.action_smoothing_order,
                            sides=SIDES,
                        )
                        # Filtering runs in the network worker, never in the
                        # 120 Hz publisher callback. Revalidate the trajectory
                        # that will actually be installed and executed.
                        self._trace_policy_chunk(
                            actions, observation, request_id=request_id,
                            generation=generation, stage="post_smoothing",
                            action_rate_hz=action_rate_hz,
                            observation_created_at=observation_created_at,
                        )
                        try:
                            self._validate_action_chunk(
                                actions,
                                action_rate_hz,
                                observation_eef_positions=observation_eef_positions,
                                observation_joint_positions=(
                                    observation_joint_positions
                                ),
                                observation_hand_positions=(
                                    observation_hand_positions
                                ),
                            )
                        except ValueError as exc:
                            raise ValueError(
                                f"post-smoothing policy chunk: {exc}"
                            ) from exc
                    pending = PendingActionChunk(
                        actions=tuple(actions),
                        action_rate_hz=action_rate_hz,
                        observation_created_at=observation_created_at,
                        response_received_at=received_at,
                        align_to_observation=is_prefetch,
                        request_id=request_id,
                        generation=generation,
                        rtc_prefix_steps=rtc_prefix_steps,
                    )
                    if not self._store_pending_chunk(
                        pending, generation=generation
                    ):
                        if getattr(self, "paper_async_blend_enabled", False):
                            with self._prefetch_lock:
                                self._paper_request_origins.pop(request_id, None)
                        self.get_logger().debug(
                            "Discarded policy response from an obsolete "
                            "lifecycle generation"
                        )
                        continue
                    completed_at = time.monotonic()
                    complete_latency_ms = (
                        completed_at - observation_created_at
                    ) * 1000.0
                    self._record_complete_policy_latency(complete_latency_ms)
                    self._requests += 1
                    self._server_ready = True
                    self._last_server_error = ""
                    self._last_latency_ms = complete_latency_ms
                    _trace_event(
                        self,
                        "policy_response_ready",
                        request_id=request_id,
                        generation=generation,
                        is_prefetch=is_prefetch,
                        action_count=len(actions),
                        complete_rtt_ms=complete_latency_ms,
                        observation_age_ms=observation_age * 1000.0,
                        server_inference_ms=self._last_server_inference_ms,
                        active_remaining_at_response=(
                            self._action_plan.remaining()
                        ),
                    )
                except Exception as exc:
                    if getattr(self, "paper_async_blend_enabled", False):
                        with self._prefetch_lock:
                            self._paper_request_origins.pop(request_id, None)
                    self._release_policy_request(generation=generation)
                    self._server_ready = False
                    self._last_server_error = str(exc)
                    self._failures += 1
                    _trace_event(
                        self,
                        "policy_request_failed",
                        request_id=request_id,
                        generation=generation,
                        is_prefetch=is_prefetch,
                        error=str(exc),
                        active_remaining=self._action_plan.remaining(),
                    )
                    self.get_logger().warn(f"Policy request failed: {exc}")
                    if (
                        _uses_synchronous_joint_actions(self)
                        and self._lifecycle in COMMAND_LIFECYCLES
                    ):
                        self._request_policy_standby(
                            f"Pi joint policy response rejected: {exc}"
                        )
                    self._last_http_status = transport.last_http_status
                    self._reconnect_transport(
                        transport,
                        clear_active=not is_prefetch,
                    )
                    self._stop_event.wait(self.reconnect_interval_s)
                    next_probe_at = time.monotonic()
        finally:
            transport.close()

    @staticmethod
    def _action_for_side(response: Mapping[str, Any], side: str):
        arm = response.get(f"arm_action_{side}")
        hand = response.get(f"hand_action_{side}")
        return arm, hand

    def _canonicalize_action(
        self, response: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Map the Cosmos joint wire shape onto the internal joint shape."""

        if not _uses_cosmos_joint_wire(self):
            return response
        action = dict(response)
        for side in SIDES:
            key = f"arm_joint_action_{side}"
            if key not in response:
                raise ValueError(
                    f"Cosmos joint response is missing {key}"
                )
            # Ignore an optional EEF compatibility field. Joint control,
            # validation, blending and interpolation consume only this value.
            action[f"arm_action_{side}"] = {
                "joint_pos": response[key]
            }
        return action

    def _validate_response(
        self,
        response: Mapping[str, Any],
        previous_poses: Optional[Mapping[str, Optional[np.ndarray]]] = None,
        previous_hands: Optional[Mapping[str, Optional[np.ndarray]]] = None,
    ):
        if previous_poses is None:
            previous_poses = self._last_applied_pose
        if previous_hands is None:
            previous_hands = self._last_applied_hand
        validated = {}
        candidate_poses = {}
        candidate_hands = {}
        joint_action_mode = (
            getattr(self, "policy_action_mode", "eef") == "joint"
        )
        for side in SIDES:
            arm, hand = self._action_for_side(response, side)
            if not isinstance(arm, Mapping) or hand is None:
                raise ValueError(f"response missing paired {side} arm/hand action")
            if joint_action_mode:
                joint_rad = np.asarray(
                    arm.get("joint_pos"), dtype=np.float32
                ).reshape(-1)
                hand_rad = np.asarray(hand, dtype=np.float32).reshape(-1)
                if joint_rad.shape != (7,) or not np.all(
                    np.isfinite(joint_rad)
                ):
                    raise ValueError(f"invalid {side} 7-DoF joint action")
                if hand_rad.shape != (20,) or not np.all(
                    np.isfinite(hand_rad)
                ):
                    raise ValueError(f"invalid {side} 20-DoF hand action")
                if _uses_synchronous_joint_actions(self):
                    lower = getattr(self, "_pi_joint_lower_rad", None)
                    upper = getattr(self, "_pi_joint_upper_rad", None)
                    velocity = getattr(
                        self, "_pi_joint_velocity_rad_s", None
                    )
                    if lower is None or upper is None or velocity is None:
                        raise ValueError(
                            "Pi joint safety limits are not configured"
                        )
                    lower = np.asarray(lower, dtype=np.float32).reshape(7)
                    upper = np.asarray(upper, dtype=np.float32).reshape(7)
                    violations = np.flatnonzero(
                        np.logical_or(joint_rad < lower, joint_rad > upper)
                    )
                    if violations.size:
                        index = int(violations[0])
                        raise ValueError(
                            f"unsafe {side} joint {index + 1} position "
                            f"{np.degrees(joint_rad[index]):.2f} deg outside "
                            f"[{np.degrees(lower[index]):.2f}, "
                            f"{np.degrees(upper[index]):.2f}] deg"
                        )
                previous_joint = previous_poses[side]
                if previous_joint is not None:
                    delta = np.abs(joint_rad - previous_joint)
                    if (
                        _uses_synchronous_joint_actions(self)
                        and getattr(
                            self,
                            "joint_step_velocity_validation_enabled",
                            True,
                        )
                    ):
                        max_step = (
                            np.asarray(
                                self._pi_joint_velocity_rad_s,
                                dtype=np.float32,
                            ).reshape(7)
                            / float(self.action_rate_hz)
                        )
                        configured_step = float(
                            getattr(
                                self,
                                "small_motion_max_arm_joint_step_rad",
                                math.inf,
                            )
                        )
                        if np.isfinite(configured_step):
                            max_step = np.minimum(max_step, configured_step)
                        violations = np.flatnonzero(
                            delta > max_step + 1e-6
                        )
                        if violations.size:
                            index = int(violations[0])
                            raise ValueError(
                                f"unsafe {side} joint {index + 1} "
                                "step/velocity: "
                                f"delta={np.degrees(delta[index]):.2f} deg, "
                                f"limit={np.degrees(max_step[index]):.2f} "
                                f"deg at {self.action_rate_hz:.1f} Hz"
                            )
                    elif (
                        not _uses_synchronous_joint_actions(self)
                        and float(np.max(delta, initial=0.0))
                        > np.radians(75.0)
                    ):
                        raise ValueError(f"unsafe {side} policy joint jump")
                previous_hand = previous_hands[side]
                if (
                    previous_hand is not None
                    and float(
                        np.max(
                            np.abs(hand_rad - previous_hand), initial=0.0
                        )
                    )
                    > (
                        np.radians(75.0)
                        if _hand_actions_are_radians(self)
                        else 75.0
                    )
                ):
                    raise ValueError(f"unsafe {side} policy hand jump")
                validated[side] = (joint_rad, hand_rad, None)
                candidate_poses[side] = joint_rad
                candidate_hands[side] = hand_rad
                continue

            position = np.asarray(arm.get("ee_pos"), dtype=np.float32).reshape(-1)
            quaternion = normalize_quaternion_xyzw(arm.get("ee_quat"))
            hand_deg = np.asarray(hand, dtype=np.float32).reshape(-1)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"invalid {side} EE position")
            if hand_deg.shape != (20,) or not np.all(np.isfinite(hand_deg)):
                raise ValueError(f"invalid {side} 20-DoF hand action")
            previous = previous_poses[side]
            pose = np.concatenate([position, quaternion]).astype(np.float32)
            if previous is not None:
                position_step_m = float(np.linalg.norm(position - previous[:3]))
                if position_step_m > 0.15:
                    raise ValueError(
                        f"unsafe {side} policy position jump: "
                        f"step={position_step_m:.6f} m exceeds 0.15 m; "
                        f"from={previous[:3].tolist()} to={position.tolist()}"
                    )
                rotation_step_rad = quaternion_angle_rad(quaternion, previous[3:7])
                if rotation_step_rad > np.radians(75.0):
                    raise ValueError(
                        f"unsafe {side} policy rotation jump: "
                        f"step={np.degrees(rotation_step_rad):.6f} deg exceeds 75 deg"
                    )
            previous_hand = previous_hands[side]
            if (
                previous_hand is not None
                and float(np.max(np.abs(hand_deg - previous_hand), initial=0.0))
                > 75.0
            ):
                raise ValueError(f"unsafe {side} policy hand jump")
            zsp = arm.get("zsp")
            if zsp is not None:
                zsp = np.asarray(zsp, dtype=np.float32).reshape(-1)
                if zsp.shape != (3,) or not np.all(np.isfinite(zsp)):
                    raise ValueError(f"invalid {side} ZSP")
            if getattr(self, "arm_command_mode", "eef") == "joint":
                joint_rad = np.asarray(
                    response.get(f"arm_joint_action_{side}"),
                    dtype=np.float32,
                ).reshape(-1)
                if joint_rad.shape != (7,) or not np.all(
                    np.isfinite(joint_rad)
                ):
                    raise ValueError(f"invalid {side} 7-DoF joint action")
            validated[side] = (pose, hand_deg, zsp)
            candidate_poses[side] = pose
            candidate_hands[side] = hand_deg
        return validated, candidate_poses, candidate_hands

    def _pi_joint_measured_positions(
        self, sides: tuple[str, ...] = SIDES
    ) -> dict[str, np.ndarray]:
        """Return fresh measured Tianji qpos in protocol radians."""

        if not _uses_synchronous_joint_actions(self):
            raise RuntimeError("measured joint guard is only valid for Pi joint")
        snapshot = self._snapshot()
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        measured: dict[str, np.ndarray] = {}
        for side in sides:
            key = f"arm_state_{side}"
            if key not in snapshot:
                raise ValueError(
                    f"Pi joint safety missing measured {side} arm state"
                )
            stamp, message = snapshot[key]
            age = now_ros - float(stamp)
            if age > self.max_observation_age_s:
                raise ValueError(
                    f"Pi joint safety measured {side} arm state is stale "
                    f"({age * 1000.0:.1f} ms)"
                )
            joint_rad = np.radians(
                np.asarray(message.position, dtype=np.float32)
            ).reshape(-1)
            if joint_rad.shape != (7,) or not np.all(np.isfinite(joint_rad)):
                raise ValueError(
                    f"Pi joint safety measured {side} qpos must be finite 7D"
                )
            measured[side] = joint_rad.astype(np.float32, copy=False)
        return measured

    def _reset_pi_joint_runtime_safety(self) -> None:
        """Clear only the Pi joint publish-limiter runtime state."""

        self._pi_joint_last_published_pose = {
            side: None for side in SIDES
        }
        self._pi_joint_last_published_at = 0.0

    def _limit_pi_joint_publish_target(
        self,
        validated: Mapping[str, tuple[np.ndarray, np.ndarray, Any]],
        *,
        measured: Mapping[str, np.ndarray],
        now: float,
    ) -> tuple[
        dict[str, tuple[np.ndarray, np.ndarray, Any]],
        dict[str, np.ndarray],
        bool,
    ]:
        """Slew-limit final Pi arm targets on the 120 Hz publish timeline.

        Protocol validation still checks the model's native 30 Hz trajectory.
        This final limiter is deliberately downstream of interpolation so a
        slow physical response cannot turn into a false chunk rejection.
        """

        if not _uses_synchronous_joint_actions(self):
            poses = {
                side: np.asarray(values[0], dtype=np.float32).copy()
                for side, values in validated.items()
            }
            return dict(validated), poses, False
        limits = np.asarray(
            self._pi_joint_command_velocity_rad_s, dtype=np.float32
        ).reshape(7)
        last_at = float(
            getattr(self, "_pi_joint_last_published_at", 0.0)
        )
        nominal_dt = 1.0 / float(self.publish_rate_hz)
        if last_at <= 0.0:
            dt = nominal_dt
        else:
            # Do not turn an executor stall into a large catch-up command.
            dt = float(np.clip(now - last_at, 0.0, 2.0 * nominal_dt))
        max_step = limits * max(dt, np.finfo(np.float32).eps)
        limited_validated = dict(validated)
        published_poses = {
            side: np.asarray(values[0], dtype=np.float32).copy()
            for side, values in validated.items()
        }
        was_limited = False
        last_published = getattr(
            self,
            "_pi_joint_last_published_pose",
            {side: None for side in SIDES},
        )
        for side in self.active_arm_sides:
            requested, hand, zsp = validated[side]
            requested_joint = np.asarray(
                requested, dtype=np.float32
            ).reshape(7)
            previous = last_published.get(side)
            if previous is None:
                previous = np.asarray(
                    measured[side], dtype=np.float32
                ).reshape(7)
            else:
                previous = np.asarray(previous, dtype=np.float32).reshape(7)
            delta = requested_joint - previous
            limited_delta = np.clip(delta, -max_step, max_step)
            limited_joint = (previous + limited_delta).astype(np.float32)
            if not np.allclose(
                limited_joint, requested_joint, rtol=0.0, atol=1e-7
            ):
                was_limited = True
            limited_validated[side] = (limited_joint, hand, zsp)
            published_poses[side] = limited_joint
        if was_limited:
            self._pi_joint_rate_limit_events = (
                getattr(self, "_pi_joint_rate_limit_events", 0) + 1
            )
        return limited_validated, published_poses, was_limited

    def _record_pi_joint_published_target(
        self, published_poses: Mapping[str, np.ndarray], *, now: float
    ) -> None:
        if not _uses_synchronous_joint_actions(self):
            return
        for side in self.active_arm_sides:
            self._pi_joint_last_published_pose[side] = np.asarray(
                published_poses[side], dtype=np.float32
            ).reshape(7).copy()
        self._pi_joint_last_published_at = float(now)

    def _validate_pi_joint_measured_boundary(
        self, action: Mapping[str, Any]
    ) -> None:
        if not _uses_synchronous_joint_actions(self):
            return
        empty_hands = {side: None for side in SIDES}
        self._validate_response(
            action,
            previous_poses=DeploymentNode._pi_joint_measured_positions(self),
            previous_hands=empty_hands,
        )

    def _validate_action_chunk(
        self,
        actions: list[Mapping[str, Any]],
        action_rate_hz: float,
        *,
        observation_eef_positions: Optional[
            Mapping[str, np.ndarray]
        ] = None,
        observation_joint_positions: Optional[
            Mapping[str, np.ndarray]
        ] = None,
        observation_hand_positions: Optional[
            Mapping[str, np.ndarray]
        ] = None,
    ) -> None:
        if self.policy_transport == "http":
            if len(actions) != self.policy_http_expected_chunk_size:
                raise ValueError(
                    "HTTP policy returned unexpected action chunk size: "
                    f"expected {self.policy_http_expected_chunk_size}, "
                    f"received {len(actions)}"
                )
            if not np.isclose(action_rate_hz, self.action_rate_hz):
                raise ValueError(
                    f"HTTP policy action_rate_hz mismatch: expected "
                    f"{self.action_rate_hz}, received {action_rate_hz}"
                )

        if _uses_cosmos_joint_wire(self):
            if (
                observation_joint_positions is None
                or observation_hand_positions is None
            ):
                raise ValueError(
                    "Cosmos joint validation requires observation anchors"
                )
            anchors = {
                side: np.asarray(
                    observation_joint_positions[side], dtype=np.float32
                ).reshape(7)
                for side in SIDES
            }
            hand_anchors = {
                side: np.asarray(
                    observation_hand_positions[side], dtype=np.float32
                ).reshape(20)
                for side in SIDES
            }
            self._validate_action_sequence(
                actions,
                previous_poses=anchors,
                previous_hands=hand_anchors,
            )
            inactive_sides = tuple(
                side for side in SIDES if side not in self.active_arm_sides
            )
            for action_index, action in enumerate(actions):
                for side in inactive_sides:
                    joint = np.asarray(
                        action[f"arm_action_{side}"]["joint_pos"],
                        dtype=np.float32,
                    ).reshape(7)
                    hand = np.asarray(
                        action[f"hand_action_{side}"], dtype=np.float32
                    ).reshape(20)
                    if not np.allclose(
                        joint, anchors[side], rtol=0.0, atol=1e-6
                    ):
                        raise ValueError(
                            f"inactive {side} arm does not hold observation "
                            f"at action {action_index}"
                        )
                    if not np.allclose(
                        hand, hand_anchors[side], rtol=0.0, atol=1e-4
                    ):
                        raise ValueError(
                            f"inactive {side} hand does not hold observation "
                            f"at action {action_index}"
                        )
            return

        if _uses_synchronous_joint_actions(self):
            empty_previous = {side: None for side in SIDES}
            self._validate_action_sequence(
                actions,
                previous_poses=empty_previous,
                previous_hands=empty_previous,
            )
            # The first chunk has no accepted command boundary yet.  Anchor
            # it to fresh measured qpos before it can enter the pending slot.
            if (
                getattr(self, "_active_chunk_id", 0) == 0
                and all(
                    value is None
                    for value in self._last_applied_pose.values()
                )
            ):
                self._validate_pi_joint_measured_boundary(actions[0])
            return
        if (
            getattr(self, "policy_action_mode", "eef") == "eef"
            and getattr(self, "first_step_anchor_on_measured_pose", False)
        ):
            if observation_eef_positions is None or observation_hand_positions is None:
                raise ValueError("EEF chunk validation requires observation anchors")
            poses, hands = {}, {}
            for side in SIDES:
                pose = np.asarray(observation_eef_positions[side], dtype=np.float32).copy()
                hand = np.asarray(observation_hand_positions[side], dtype=np.float32).copy()
                if (
                    pose.shape != (7,) or hand.shape != (20,)
                    or not np.all(np.isfinite(pose))
                    or not np.all(np.isfinite(hand))
                ):
                    raise ValueError(f"invalid {side} EEF observation anchor")
                pose[3:] = normalize_quaternion_xyzw(pose[3:])
                poses[side], hands[side] = pose, hand
            # Raw step 0 belongs to the request's observation time. The live
            # command may have moved while inference ran. Activation separately
            # validates the aligned/blended window against the live command.
            self._validate_action_sequence(
                actions, previous_poses=poses, previous_hands=hands
            )
            return
        self._validate_action_sequence(actions)

    def _fdm_validate_and_accept_action_chunk(
        self,
        chunk: FdmActionChunk,
        *,
        generation: int,
    ) -> bool:
        """Validate a delivered wire chunk against the preceding wire tail.

        A prefetched Wn may arrive while the robot is near the beginning of
        Wn-1. Comparing Wn[0] with the currently applied target therefore
        spans almost 48 model actions and creates a false safety jump. The
        delivery-time boundary is Wn-1[-1] -> Wn[0]; activation later checks
        that same boundary again against the target actually executed.
        """

        assert self.fdm_config is not None
        assert self._fdm_ledger is not None
        actions = list(chunk.actions)
        with self._fdm_delivery_validation_lock:
            if int(generation) != self._stream_generation:
                raise RuntimeError("stale FDM action chunk generation")
            (
                delivery_session,
                delivery_generation,
                next_wire_chunk_id,
                _next_global_action,
            ) = self._fdm_ledger.next_delivery()
            if (
                chunk.session_id != delivery_session
                or int(generation) != delivery_generation
            ):
                # Preserve the ledger's canonical session error text.
                return self._fdm_ledger.accept_action_chunk(chunk)
            if chunk.wire_chunk_id < next_wire_chunk_id:
                # An exact retry was already validated; the ledger still
                # verifies that its immutable payload fingerprint matches.
                return self._fdm_ledger.accept_action_chunk(chunk)
            if chunk.wire_chunk_id != next_wire_chunk_id:
                return self._fdm_ledger.accept_action_chunk(chunk)

            self._validate_action_sequence(
                actions,
                previous_poses=self._fdm_delivered_tail_pose,
                previous_hands=self._fdm_delivered_tail_hand,
            )
            accepted = self._fdm_ledger.accept_action_chunk(chunk)
            if accepted:
                empty_previous = {side: None for side in SIDES}
                _, tail_poses, tail_hands = self._validate_response(
                    actions[-1],
                    previous_poses=empty_previous,
                    previous_hands=empty_previous,
                )
                self._fdm_delivered_tail_pose = {
                    side: tail_poses[side].copy() for side in SIDES
                }
                self._fdm_delivered_tail_hand = {
                    side: tail_hands[side].copy() for side in SIDES
                }
            return accepted

    def _validate_action_sequence(
        self,
        actions: list[Mapping[str, Any]],
        previous_poses: Optional[
            Mapping[str, Optional[np.ndarray]]
        ] = None,
        previous_hands: Optional[
            Mapping[str, Optional[np.ndarray]]
        ] = None,
    ) -> None:
        """Validate one sequence from the command that is currently applied."""

        if previous_poses is None:
            previous_poses = self._last_applied_pose
        if previous_hands is None:
            previous_hands = self._last_applied_hand
        mutable_previous_poses = {
            side: (
                None
                if previous_poses[side] is None
                else previous_poses[side].copy()
            )
            for side in SIDES
        }
        mutable_previous_hands = {
            side: (
                None
                if previous_hands[side] is None
                else previous_hands[side].copy()
            )
            for side in SIDES
        }
        previous_joint_velocity = {side: None for side in SIDES}
        for action_index, action in enumerate(actions):
            prior_poses = {
                side: (
                    None
                    if mutable_previous_poses[side] is None
                    else mutable_previous_poses[side].copy()
                )
                for side in SIDES
            }
            try:
                _, candidate_poses, candidate_hands = self._validate_response(
                    action,
                    previous_poses=mutable_previous_poses,
                    previous_hands=mutable_previous_hands,
                )
            except ValueError as exc:
                raise ValueError(f"waypoint {action_index}: {exc}") from exc
            if _uses_synchronous_joint_actions(self):
                acceleration_limits = getattr(
                    self, "_joint_acceleration_rad_s2", None
                )
                for side in SIDES:
                    if prior_poses[side] is None:
                        continue
                    velocity = (
                        candidate_poses[side] - prior_poses[side]
                    ) * float(self.action_rate_hz)
                    previous_velocity = previous_joint_velocity[side]
                    if (
                        acceleration_limits is not None
                        and previous_velocity is not None
                    ):
                        acceleration = np.abs(
                            velocity - previous_velocity
                        ) * float(self.action_rate_hz)
                        limits = np.asarray(
                            acceleration_limits, dtype=np.float32
                        ).reshape(7)
                        violations = np.flatnonzero(
                            acceleration > limits + 1e-6
                        )
                        if violations.size:
                            index = int(violations[0])
                            raise ValueError(
                                f"unsafe {side} joint {index + 1} "
                                "acceleration: "
                                f"value={np.degrees(acceleration[index]):.2f} "
                                "deg/s^2, "
                                f"limit={np.degrees(limits[index]):.2f} "
                                "deg/s^2"
                            )
                    previous_joint_velocity[side] = velocity
            mutable_previous_poses.update(candidate_poses)
            mutable_previous_hands.update(candidate_hands)

    def _activate_pending_chunk(
        self,
        now: float,
        *,
        schedule_start_at: Optional[float] = None,
    ) -> bool:
        """Install at exhaustion, or replace validated future at a model boundary.

        Early splice is opt-in and retains the committed endpoint. Rejection
        consumes only the pending response, never the old plan. Exhaustion
        keeps the existing controller watchdog behavior; this is not an active
        deceleration/hold controller and makes no hardware-safety guarantee.
        """

        if getattr(self, "paper_async_blend_enabled", False):
            return self._activate_paper_pending(now, schedule_start_at=schedule_start_at)
        activation_started = time.perf_counter()
        error = None
        activation_trace = None
        aligned_before_splice = None
        bridge_input_actions = None
        actions = None
        skipped = None
        blend_steps = 0
        measured_poses = None
        measured_hands = None
        activation_stage = "alignment"
        chunk_interpolator = None
        boundary_interpolator = None
        boundary_start_at = 0.0
        boundary_velocity_metrics: dict[str, dict[str, float]] = {}
        blend_method = "none"
        splice_kinematics: dict[str, float] = {}
        early_bridge_steps = 0
        initial_bridge = False
        bridge_anchor_poses: Optional[dict[str, np.ndarray]] = None
        bridge_anchor_hands: Optional[dict[str, np.ndarray]] = None
        bridge_anchor_zsp: Optional[
            dict[str, Optional[np.ndarray]]
        ] = None
        with self._prefetch_lock:
            splice_enabled = getattr(self, "early_splice_enabled", False)
            committed = self._action_plan.peek_next()
            replaced_remaining = self._action_plan.remaining()
            if self._pending_chunk is None:
                return False
            if committed is not None:
                # Do not change the endpoint of a segment already interpolating.
                # Wait until that endpoint is due and preserve it as new head.
                if not splice_enabled or float(now) < committed.due_at:
                    return False
            pending = self._pending_chunk
            interval = 1.0 / pending.action_rate_hz
            if schedule_start_at is None:
                nominal_start = (
                    self._last_action_due_at + interval
                    if self._last_action_due_at > 0.0
                    else float(now)
                )
                schedule_start = max(float(now), nominal_start)
            else:
                schedule_start = float(schedule_start_at)
            if committed is not None:
                schedule_start = committed.due_at
            try:
                if splice_enabled and pending.generation != self._stream_generation:
                    raise ValueError("pending chunk belongs to a stale generation")
                if splice_enabled:
                    if abs(pending.action_rate_hz - self.action_rate_hz) > 1e-6:
                        raise ValueError("splice action rate changed")
                    if (max(float(now), schedule_start) - pending.observation_created_at > self.early_splice_max_age_s
                            or pending.observation_created_at > float(now)):
                        raise TimeoutError("splice observation expired or is in the future")
                    if committed is not None and float(now) - committed.due_at >= interval:
                        raise TimeoutError("splice boundary missed by a full model period")
                aligned_actions, skipped = select_pending_action_window(
                    pending,
                    activation_at=schedule_start,
                    horizon=self.open_loop_horizon,
                    fixed_skip_actions=getattr(self, "prefetch_fixed_skip_actions", None),
                    align_to_observation=(
                        pending.align_to_observation
                        and getattr(
                            self,
                            "prefetch_time_alignment_enabled",
                            True,
                        )
                    ),
                )
                fdm_config = getattr(self, "fdm_config", None)
                if fdm_config is not None:
                    if skipped != 0 or len(aligned_actions) != fdm_config.wire_chunk_size:
                        raise RuntimeError(
                            "FDM activation must install all 48 actions with no skip"
                        )
                    if (
                        pending.wire_chunk_id is None
                        or pending.global_action_start is None
                    ):
                        raise RuntimeError("FDM pending chunk lacks wire/global identity")
                aligned_before_splice = aligned_actions
                if committed is not None:
                    aligned_actions = [committed.action, *aligned_actions[1:]]
                    bridge_input_actions = aligned_actions
                    bridge_max = getattr(self, "early_splice_bridge_max_steps", 0)
                    if bridge_max > 0:
                        activation_stage = "early_bridge"
                        current = getattr(self, "_interpolation_current", None)
                        if current is None:
                            raise ValueError("early bridge requires previous model waypoint")
                        aligned_actions, early_bridge_steps, splice_kinematics = bridge_early_action_window(
                            aligned_actions, previous_action=current.action,
                            rate_hz=pending.action_rate_hz, max_bridge_steps=bridge_max,
                            arm_sides=self.active_arm_sides, hand_sides=self.active_hand_sides,
                            limits=self.early_splice_limits,
                        )
                actions = aligned_actions
                activation_stage = "action_validation"
                empty_previous = {side: None for side in SIDES}
                _, raw_first_poses, raw_first_hands = self._validate_response(
                    aligned_actions[0],
                    previous_poses=empty_previous,
                    previous_hands=empty_previous,
                )
                blend_steps = 0
                anchors_ready = all(
                    self._last_applied_pose[side] is not None
                    and self._last_applied_hand[side] is not None
                    for side in SIDES
                )
                initial_bridge = bool(
                    self._active_chunk_id == 0
                    and getattr(self, "initial_blend_steps", 0) > 0
                )
                previous_poses = getattr(
                    self,
                    "_previous_applied_pose",
                    {side: None for side in SIDES},
                )
                previous_hands = getattr(
                    self,
                    "_previous_applied_hand",
                    {side: None for side in SIDES},
                )
                previous_dt_s = max(
                    0.0,
                    float(getattr(self, "_last_applied_at", 0.0))
                    - float(getattr(self, "_previous_applied_at", 0.0)),
                )
                if initial_bridge:
                    (
                        bridge_anchor_poses,
                        bridge_anchor_hands,
                        bridge_anchor_zsp,
                    ) = self._initial_command_anchors(aligned_actions[0])
                    blend_steps = min(
                        int(self.initial_blend_steps), len(aligned_actions)
                    )
                    blend_method = getattr(
                        self, "boundary_blend_method", "smoothstep"
                    )
                    previous_poses = {side: None for side in SIDES}
                    previous_hands = {side: None for side in SIDES}
                    previous_dt_s = 0.0
                elif (
                    (pending.align_to_observation or getattr(self, "sequential_boundary_blend_enabled", False))
                    and committed is None
                    and self.boundary_blend_steps > 0
                    and anchors_ready
                ):
                    bridge_anchor_poses = self._last_applied_pose
                    if _uses_synchronous_joint_actions(self):
                        # The model timeline may be ahead of the slew-limited
                        # hardware target. Join a new chunk from the target
                        # actually published, not from the raw policy sample.
                        published_anchor = getattr(
                            self,
                            "_pi_joint_last_published_pose",
                            {side: None for side in SIDES},
                        )
                        bridge_anchor_poses = {
                            side: (
                                published_anchor[side]
                                if side in self.active_arm_sides
                                and published_anchor[side] is not None
                                else self._last_applied_pose[side]
                            )
                            for side in SIDES
                        }
                    bridge_anchor_hands = self._last_applied_hand
                    bridge_anchor_zsp = self._last_applied_zsp
                    blend_steps = min(
                        self.boundary_blend_steps, len(aligned_actions)
                    )
                    blend_method = getattr(
                        self, "boundary_blend_method", "smoothstep"
                    )
                if blend_steps > 0 and bridge_anchor_poses is not None:
                    if blend_method == "velocity_continuous":
                        if initial_bridge:
                            boundary_start_at = schedule_start
                            # Keep the first model deadline one period after
                            # the measured zero-velocity anchor. Action k then
                            # remains due exactly (k + 1) model periods later.
                            schedule_start += interval
                        else:
                            boundary_start_at = schedule_start - interval
                        boundary_interpolator = (
                            VelocityContinuousBoundaryInterpolator(
                                aligned_actions,
                                anchor_poses=bridge_anchor_poses,
                                anchor_hands=bridge_anchor_hands,
                                anchor_zsp=bridge_anchor_zsp,
                                previous_poses=previous_poses,
                                previous_hands=previous_hands,
                                previous_dt_s=previous_dt_s,
                                blend_steps=blend_steps,
                                rate_hz=pending.action_rate_hz,
                                sides=SIDES,
                            )
                        )
                        boundary_velocity_metrics = (
                            boundary_interpolator.velocity_metrics()
                        )
                    else:
                        actions = blend_action_prefix(
                            aligned_actions,
                            anchor_poses=bridge_anchor_poses,
                            anchor_hands=bridge_anchor_hands,
                            anchor_zsp=bridge_anchor_zsp,
                            blend_steps=blend_steps,
                            hand_blend_steps=(None if initial_bridge else getattr(self, "boundary_hand_blend_steps", None)),
                            sides=SIDES,
                            action_mode=getattr(
                                self, "policy_action_mode", "eef"
                            ),
                        )
                metric_previous_poses = (
                    bridge_anchor_poses
                    if bridge_anchor_poses is not None
                    else self._last_applied_pose
                )
                metric_previous_hands = (
                    bridge_anchor_hands
                    if bridge_anchor_hands is not None
                    else self._last_applied_hand
                )
                raw_position_jump_m, raw_rotation_jump_deg, raw_hand_jump_deg = (
                    _target_jump_metrics(
                        self,
                        raw_first_poses,
                        raw_first_hands,
                        previous_poses=metric_previous_poses,
                        previous_hands=metric_previous_hands,
                    )
                )
                first_action = actions[0]
                if boundary_interpolator is not None:
                    first_action = boundary_interpolator.sample(interval)
                _, first_poses, first_hands = self._validate_response(
                    first_action,
                    previous_poses=empty_previous,
                    previous_hands=empty_previous,
                )
                position_jump_m, rotation_jump_deg, hand_jump_deg = (
                    _target_jump_metrics(
                        self,
                        first_poses,
                        first_hands,
                        previous_poses=metric_previous_poses,
                        previous_hands=metric_previous_hands,
                    )
                )
                # Full response validation happened in the network worker.
                # Validate the selected splice again against the command that
                # actually reached the boundary while inference was running.
                if boundary_interpolator is not None:
                    validation_interval = 1.0 / float(self.publish_rate_hz)
                    validation_count = max(
                        1,
                        math.ceil(
                            boundary_interpolator.duration_s
                            / validation_interval
                        ),
                    )
                    bridge_actions = [
                        boundary_interpolator.sample(
                            min(
                                boundary_interpolator.duration_s,
                                (index + 1) * validation_interval,
                            )
                        )
                        for index in range(validation_count)
                    ]
                    self._validate_action_sequence(
                        bridge_actions + actions[blend_steps:],
                        previous_poses=metric_previous_poses,
                        previous_hands=metric_previous_hands,
                    )
                else:
                    self._validate_action_sequence(
                        actions,
                        previous_poses=metric_previous_poses,
                        previous_hands=metric_previous_hands,
                    )
                if (
                    getattr(self, "action_interpolation_method", "none")
                    in ("pchip_slerp", "pchip_joint")
                ):
                    chunk_interpolator = PchipActionInterpolator(
                        actions,
                        sides=SIDES,
                        action_mode=getattr(
                            self, "policy_action_mode", "eef"
                        ),
                    )
                if splice_enabled:
                    activation_stage = "kinematic_guard"
                    # Use model waypoints, not the previous 120 Hz published
                    # sample, to measure velocities on the model clock.
                    current = getattr(self, "_interpolation_current", None)
                    # Check entry and exit of the modified neighborhood,
                    # not every finite difference inside an otherwise valid
                    # model prediction. Full sequence legality is checked above.
                    # Early replacement preserves the committed head: include two
                    # following endpoints. Blending modifies the whole prefix:
                    # include all of it plus two untouched endpoints to expose
                    # a jump or velocity change at the end of the blend.
                    guard_count = max(3, early_bridge_steps + 3) if committed is not None else max(2, blend_steps + 2)
                    samples = list(actions[:guard_count])
                    if current is not None:
                        samples.insert(0, current.action)
                        if schedule_start - current.due_at > interval * 1.5:
                            # A held command has zero incoming velocity.
                            samples.insert(0, current.action)
                        elif committed is None:
                            previous = getattr(self, "_interpolation_previous", None)
                            if previous is not None and abs(current.due_at - previous.due_at - interval) < interval * 0.1:
                                samples.insert(0, previous.action)
                            else:
                                samples.insert(0, current.action)
                    pose_samples = {side: [] for side in self.active_arm_sides}
                    hand_samples = {side: [] for side in self.active_hand_sides}
                    if current is None:
                        # Startup/reset must also pass the limits against fresh
                        # measured state, including zero incoming velocity.
                        measured_poses, measured_hands, _ = self._initial_command_anchors(actions[0])
                        for side in pose_samples:
                            pose_samples[side].extend([measured_poses[side]] * 2)
                        for side in hand_samples:
                            hand_samples[side].extend([measured_hands[side]] * 2)
                    for sample in samples:
                        _, poses, hands = self._validate_response(
                            sample, previous_poses=empty_previous, previous_hands=empty_previous)
                        for side in pose_samples:
                            pose_samples[side].append(poses[side])
                        for side in hand_samples:
                            hand_samples[side].append(hands[side])
                    splice_kinematics = validate_splice_kinematics(
                        pose_samples, hand_samples, rate_hz=pending.action_rate_hz,
                        **self.early_splice_limits)
                activation_stage = "install"
                self._action_plan.install(
                    actions,
                    observation_created_at=pending.observation_created_at,
                    received_at=pending.response_received_at,
                    rate_hz=pending.action_rate_hz,
                    schedule_start_at=schedule_start,
                    replace_from_sequence=(committed.sequence if committed is not None else None),
                )
            except Exception as exc:
                error = exc
                if splice_enabled:
                    self._policy_request_not_before = float(now) + self.early_splice_request_interval_s
                    self._early_splice_last_rejection = str(exc)
                self._pending_chunk = None
                self._prefetch_activation_failures += 1
            else:
                self._pending_chunk = None
                self._last_activation_skip_actions = skipped
                self._total_activation_skip_actions += skipped
                waited_for_pending = self._waiting_for_pending_since > 0.0
                if pending.align_to_observation:
                    if waited_for_pending:
                        wait_ms = max(
                            0.0,
                            (float(now) - self._waiting_for_pending_since)
                            * 1000.0,
                        )
                        self._last_boundary_wait_ms = wait_ms
                        self._max_boundary_wait_ms = max(
                            self._max_boundary_wait_ms, wait_ms
                        )
                    else:
                        self._prefetch_hits += 1
                        self._last_boundary_wait_ms = 0.0
                self._waiting_for_pending_since = 0.0
                if initial_bridge and bridge_anchor_poses is not None:
                    self._last_applied_pose = {
                        side: bridge_anchor_poses[side].copy()
                        for side in SIDES
                    }
                    self._last_applied_hand = {
                        side: bridge_anchor_hands[side].copy()
                        for side in SIDES
                    }
                    self._last_applied_zsp = {
                        side: (
                            None
                            if bridge_anchor_zsp[side] is None
                            else bridge_anchor_zsp[side].copy()
                        )
                        for side in SIDES
                    }
                    self._previous_applied_pose = {
                        side: None for side in SIDES
                    }
                    self._previous_applied_hand = {
                        side: None for side in SIDES
                    }
                    self._previous_applied_at = 0.0
                    self._last_applied_at = (
                        boundary_start_at
                        if boundary_start_at > 0.0
                        else schedule_start - interval
                    )
                self._rtc_soft_tail = tuple(pending.actions[skipped + len(actions):skipped + len(actions) + 2])
                self._active_chunk_id += 1
                self._chunk_interpolator = chunk_interpolator
                self._chunk_interpolator_chunk_id = self._active_chunk_id
                self._boundary_interpolator = boundary_interpolator
                self._boundary_interpolator_chunk_id = (
                    self._active_chunk_id
                    if boundary_interpolator is not None
                    else 0
                )
                self._boundary_interpolator_start_at = boundary_start_at
                self._active_chunk_action_index = 0
                self._active_chunk_skip_actions = skipped
                self._active_chunk_prefetched = pending.align_to_observation
                self._active_chunk_blend_steps = blend_steps
                if fdm_config is not None:
                    self._fdm_active_wire_chunk_id = pending.wire_chunk_id
                    self._fdm_active_global_action_start = (
                        pending.global_action_start
                    )
                    self._fdm_active_native_spans = pending.native_spans
                activation_trace = {
                    "request_id": pending.request_id,
                    "early_splice": committed is not None,
                    "early_bridge_steps": early_bridge_steps,
                    "early_bridge_modified_indices": [1, early_bridge_steps - 1] if early_bridge_steps else [],
                    "replaced_remaining_actions": replaced_remaining,
                    "alignment_clock": "request_observation_created_monotonic",
                    "splice_kinematics": splice_kinematics,
                    "committed_sequence": committed.sequence if committed is not None else None,
                    "generation": pending.generation,
                    "chunk_id": self._active_chunk_id,
                    "prefetched": pending.align_to_observation,
                    "time_alignment_enabled": bool(
                        pending.align_to_observation
                        and getattr(
                            self,
                            "prefetch_time_alignment_enabled",
                            True,
                        )
                    ),
                    "waited_for_pending": waited_for_pending,
                    "skipped_actions": skipped,
                    "blend_steps": blend_steps,
                    "hand_blend_steps": (blend_steps if initial_bridge or getattr(self, "boundary_hand_blend_steps", None) is None else min(blend_steps, self.boundary_hand_blend_steps)),
                    "wire_chunk_id": pending.wire_chunk_id,
                    "global_action_start": pending.global_action_start,
                    "native_spans": list(pending.native_spans),
                    "blend_method": blend_method,
                    "initial_bridge": initial_bridge,
                    "boundary_velocity": boundary_velocity_metrics,
                    "installed_actions": len(actions),
                    "observation_to_activation_ms": (
                        schedule_start - pending.observation_created_at
                    )
                    * 1000.0,
                    "response_to_activation_ms": (
                        schedule_start - pending.response_received_at
                    )
                    * 1000.0,
                    "boundary_wait_ms": self._last_boundary_wait_ms,
                    "schedule_start_at": schedule_start,
                    "activation_callback_at": float(now),
                    "position_jump_m": position_jump_m,
                    (
                        "arm_joint_jump_deg"
                        if getattr(self, "policy_action_mode", "eef")
                        == "joint"
                        else "rotation_jump_deg"
                    ): rotation_jump_deg,
                    "hand_jump_deg": hand_jump_deg,
                    "raw_position_jump_m": raw_position_jump_m,
                    (
                        "raw_arm_joint_jump_deg"
                        if getattr(self, "policy_action_mode", "eef")
                        == "joint"
                        else "raw_rotation_jump_deg"
                    ): raw_rotation_jump_deg,
                    "raw_hand_jump_deg": raw_hand_jump_deg,
                    "action_smoothing_method": getattr(
                        self, "action_smoothing_method", "none"
                    ),
                    "action_smoothing_cutoff_hz": getattr(
                        self, "action_smoothing_cutoff_hz", 0.0
                    ),
                }
        computation_ms = (time.perf_counter() - activation_started) * 1000.0
        if activation_trace is not None:
            activation_trace["activation_compute_ms"] = computation_ms
        if (getattr(self, "trace_policy_chunk_enabled", False)
                and getattr(self, "_trace_writer", None) is not None):
            def waypoint_snapshot(waypoint):
                if waypoint is None:
                    return None
                return {"sequence": waypoint.sequence, "due_at": waypoint.due_at,
                        "action": copy.deepcopy(waypoint.action)}
            snapshot_started = time.perf_counter()
            _trace_event(
                self, "policy_splice_snapshot",
                request_id=pending.request_id, generation=pending.generation,
                chunk_id=self._active_chunk_id,
                status="rejected" if error is not None else "activated",
                rejection_reason=str(error) if error is not None else "",
                rejection_stage=activation_stage if error is not None else "",
                error_class=type(error).__name__ if error is not None else "",
                observation_created_at=pending.observation_created_at,
                response_received_at=pending.response_received_at,
                action_rate_hz=pending.action_rate_hz,
                activation_callback_at=float(now), schedule_start_at=schedule_start,
                alignment_clock="request_observation_created_monotonic",
                skipped_actions=skipped,
                replaced_remaining_actions=replaced_remaining,
                early_bridge_steps=early_bridge_steps,
                early_bridge_max_steps=getattr(self, "early_splice_bridge_max_steps", 0),
                early_bridge_modified_indices=[1, early_bridge_steps - 1] if early_bridge_steps else [],
                blend_steps=blend_steps if actions is not None else 0,
                hand_blend_steps=(blend_steps if initial_bridge or getattr(self, "boundary_hand_blend_steps", None) is None else min(blend_steps, self.boundary_hand_blend_steps)),
                blend_method=blend_method,
                computation_ms=computation_ms,
                pending_actions=_bounded_action_snapshot(pending.actions),
                aligned_actions=_bounded_action_snapshot(aligned_before_splice),
                bridge_input_actions=_bounded_action_snapshot(bridge_input_actions),
                final_actions=_bounded_action_snapshot(actions),
                current=waypoint_snapshot(getattr(self, "_interpolation_current", None)),
                previous=waypoint_snapshot(getattr(self, "_interpolation_previous", None)),
                committed=waypoint_snapshot(committed),
                bridge_anchor_poses=copy.deepcopy(bridge_anchor_poses),
                bridge_anchor_hands=copy.deepcopy(bridge_anchor_hands),
                bridge_anchor_zsp=copy.deepcopy(bridge_anchor_zsp),
                measured_anchor_poses=copy.deepcopy(measured_poses),
                measured_anchor_hands=copy.deepcopy(measured_hands),
                kinematic_limits=dict(getattr(self, "early_splice_limits", {})),
                splice_kinematics=splice_kinematics,
                diagnostic_snapshot_ms=(time.perf_counter() - snapshot_started) * 1000.0,
            )
        if error is not None:
            self._failures += 1
            self._last_server_error = f"pending chunk activation failed: {error}"
            _trace_event(
                self,
                "pending_chunk_activation_failed",
                request_id=pending.request_id,
                generation=pending.generation,
                error=str(error),
                activation_compute_ms=computation_ms,
                rejection_stage=activation_stage,
                error_class=type(error).__name__,
                activation_callback_at=float(now),
            )
            self.get_logger().warn(self._last_server_error)
            if _uses_synchronous_joint_actions(self):
                self._request_policy_standby(self._last_server_error)
            self._network_wakeup.set()
            return False
        if activation_trace is not None:
            _trace_event(self, "pending_chunk_activate", **activation_trace)
        return True

    def _activate_paper_pending(self, now, *, schedule_start_at=None):
        """Post-hoc overlap replacement; no anchor blend, RTC or new limits."""
        started = time.perf_counter()
        error = None
        event = None
        with self._prefetch_lock:
            pending = self._pending_chunk
            if pending is None:
                return False
            old = self._action_plan.snapshot()
            if old and float(now) < old[0].due_at:
                return False  # Keep the segment already being interpolated.
            try:
                if pending.generation != self._stream_generation:
                    raise ValueError("paper pending belongs to stale generation")
                if abs(pending.action_rate_hz - self.action_rate_hz) > 1e-6:
                    raise ValueError("paper action rate changed")
                origin, clock_metadata = self._paper_request_origins[pending.request_id]
                bootstrap = not self._paper_bootstrapped
                if bootstrap:
                    if old:
                        raise ValueError("bootstrap must not replace an active plan")
                    actions = list(pending.actions)
                    start = float(now)
                    execution_origin = start - 1.0 / pending.action_rate_hz
                    weights = [1.0] * len(actions)
                    indices = list(range(len(actions)))
                    overlap_end = None
                    protected = None
                else:
                    activation_at = float(now) if schedule_start_at is None else max(float(now), float(schedule_start_at))
                    plan = build_overlap_plan(old, pending.actions, new_origin=origin,
                                              rate_hz=pending.action_rate_hz,
                                              activation_at=activation_at,
                                              weight_curve=getattr(self, "paper_async_weight_curve", "linear"))
                    actions = list(plan.actions)
                    start = plan.start_at
                    execution_origin = origin
                    weights = list(plan.new_weights)
                    indices = list(plan.new_sample_indices)
                    overlap_end = plan.overlap_end_at
                    protected = plan.protected_sequence
                # Preserve all existing geometric/command safety validation.
                self._validate_action_sequence(actions)
                if len(actions) > self._action_plan._max_actions:
                    raise ValueError("paper plan would exceed broker capacity")
                self._action_plan.install(
                    actions, observation_created_at=pending.observation_created_at,
                    received_at=pending.response_received_at, rate_hz=pending.action_rate_hz,
                    schedule_start_at=start, replace_from_sequence=protected)
                self._pending_chunk = None
                self._paper_request_origins.pop(pending.request_id, None)
                if bootstrap:
                    self._paper_next_request_at = execution_origin + self.paper_async_stride_steps / self.action_rate_hz
                self._paper_bootstrapped = True
                self._paper_underrun_reported = False
                self._active_chunk_id += 1
                self._active_chunk_action_index = 0
                self._active_chunk_skip_actions = max(0, int(math.floor(indices[0] + 1e-7)))
                self._active_chunk_prefetched = bool(old)
                self._active_chunk_blend_steps = 0  # This mode has no legacy anchor blend.
                self._last_activation_skip_actions = self._active_chunk_skip_actions
                self._total_activation_skip_actions += self._active_chunk_skip_actions
                self._chunk_interpolator = None
                self._boundary_interpolator = None
                self._boundary_interpolator_start_at = 0.0
                self._rtc_soft_tail = ()
                self._waiting_for_pending_since = 0.0
                event = dict(request_id=pending.request_id, generation=pending.generation,
                             chunk_id=self._active_chunk_id, bootstrap=bootstrap,
                             prediction_origin=origin, execution_origin=execution_origin,
                             scheduled_handoff_at=start,
                             camera_to_handoff_ms=(start-origin)*1000,
                             response_to_handoff_ms=(start-pending.response_received_at)*1000,
                             request_not_before=self._paper_next_request_at,
                             protected_sequence=protected,
                             protected_reinstalled_sequence=(self._action_plan.peek_next().sequence if protected is not None else None),
                             overlap_end_at=overlap_end,
                             installed_actions=len(actions), raw_prediction_actions=len(pending.actions),
                             weight_curve=getattr(self, "paper_async_weight_curve", "linear"),
                             new_weights=weights, new_sample_indices=indices,
                             old_queue_deadlines=[w.due_at for w in old],
                             observation_clock=clock_metadata)
                if getattr(self, "trace_policy_chunk_enabled", False):
                    event.update(old_plan=_bounded_action_snapshot([w.action for w in old]),
                                 raw_prediction=_bounded_action_snapshot(pending.actions),
                                 installed_plan=_bounded_action_snapshot(actions))
            except Exception as exc:
                error = str(exc)
                self._pending_chunk = None
                self._paper_request_origins.pop(pending.request_id, None)
                self._prefetch_activation_failures += 1
        if error is not None:
            _trace_event(self, "paper_async_activation_rejected", request_id=pending.request_id,
                         generation=pending.generation, error=error,
                         old_plan_preserved=True)
            self.get_logger().warn(f"Paper async activation rejected: {error}")
            self._network_wakeup.set()
            return False
        event["activation_compute_ms"] = (time.perf_counter()-started)*1000
        _trace_event(self, "paper_async_activate", **event)
        _trace_event(self, "pending_chunk_activate", request_id=pending.request_id,
                     generation=pending.generation, chunk_id=self._active_chunk_id,
                     paper_async=True, installed_actions=len(actions), prefetched=bool(old),
                     skipped_actions=self._active_chunk_skip_actions,
                     blend_steps=0, hand_blend_steps=0,
                     blend_method="moving_overlap_" + getattr(self, "paper_async_weight_curve", "linear"),
                     schedule_start_at=start, activation_callback_at=float(now))
        self._network_wakeup.set()
        return True

    def _mark_prefetch_miss(self, now: float) -> bool:
        if getattr(self, "paper_async_blend_enabled", False):
            if (self._paper_bootstrapped and self._action_plan.remaining() == 0
                    and not self._paper_underrun_reported):
                self._paper_underrun_reported = True
                _trace_event(self, "paper_async_underrun", at=float(now),
                             last_action_due_at=self._last_action_due_at,
                             request_inflight=self._request_inflight,
                             pending_ready=self._pending_chunk is not None,
                             next_request_not_before=self._paper_next_request_at)
            return False  # Keep the existing wait/watchdog behavior unchanged.
        if not self.prefetch_enabled or not self._valid_policy_stream_started:
            return False
        newly_missed = False
        reset_for_underrun = False
        with self._prefetch_lock:
            if (
                self._pending_chunk is None
                and self._waiting_for_pending_since <= 0.0
            ):
                self._waiting_for_pending_since = float(now)
                self._prefetch_misses += 1
                newly_missed = True
                miss_count = self._prefetch_misses
                inflight = self._request_inflight
            if (
                getattr(self, "fdm_config", None) is not None
                and self._waiting_for_pending_since > 0.0
                and float(now) - self._waiting_for_pending_since
                >= self.fdm_config.pending_miss_timeout_s
            ):
                reset_for_underrun = True
        if newly_missed:
            _trace_event(
                self,
                "prefetch_boundary_miss",
                miss_count=miss_count,
                boundary_at=float(now),
                request_inflight=inflight,
                lead_actions=self._prefetch_lead_actions,
            )
        if reset_for_underrun:
            self._fdm_underrun_resets += 1
            self._fdm_reset_session(
                "FDM pending action chunk underrun timeout",
                request_standby=True,
            )
            return True
        self._network_wakeup.set()
        return False

    def _publish_validated_command(
        self,
        validated: Mapping[str, tuple[np.ndarray, np.ndarray, Any]],
        command: Optional[Mapping[str, Any]] = None,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        for side, (pose, hand_deg, zsp) in validated.items():
            if side in self.active_arm_sides:
                if getattr(self, "arm_command_mode", "eef") == "joint":
                    if getattr(self, "policy_action_mode", "eef") == "joint":
                        joint_rad = np.asarray(
                            pose, dtype=np.float32
                        ).reshape(7)
                    else:
                        if command is None:
                            raise ValueError(
                                "joint command payload is unavailable"
                            )
                        joint_rad = np.asarray(
                            command[f"arm_joint_action_{side}"],
                            dtype=np.float32,
                        ).reshape(7)
                    joint_message = JointState()
                    joint_message.header.stamp = stamp
                    joint_message.position = joint_rad.tolist()
                    self._arm_publishers[side].publish(joint_message)
                else:
                    pose_message = PoseStamped()
                    pose_message.header.stamp = stamp
                    pose_message.header.frame_id = f"{side}_chest"
                    pose_message.pose.position.x = float(pose[0])
                    pose_message.pose.position.y = float(pose[1])
                    pose_message.pose.position.z = float(pose[2])
                    pose_message.pose.orientation.x = float(pose[3])
                    pose_message.pose.orientation.y = float(pose[4])
                    pose_message.pose.orientation.z = float(pose[5])
                    pose_message.pose.orientation.w = float(pose[6])
                    self._arm_publishers[side].publish(pose_message)
                    if zsp is not None:
                        zsp_message = Float64MultiArray()
                        zsp_message.data = zsp.tolist()
                        self._zsp_publishers[side].publish(zsp_message)
            if side in self.active_hand_sides:
                hand_message = JointState()
                hand_message.header.stamp = stamp
                hand_message.position = (
                    np.asarray(hand_deg, dtype=np.float32).tolist()
                    if _hand_actions_are_radians(self)
                    else np.radians(hand_deg).tolist()
                )
                self._hand_publishers[side].publish(hand_message)

    def _fdm_actual_wire_action(
        self, action: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return the final locally selected 54D waypoint sent for control."""

        empty_previous = {side: None for side in SIDES}
        validated, _, _ = self._validate_response(
            action,
            previous_poses=empty_previous,
            previous_hands=empty_previous,
        )
        actual: dict[str, Any] = {}
        for side in SIDES:
            pose, hand_deg, _zsp = validated[side]
            if getattr(self, "policy_action_mode", "eef") == "joint":
                actual[f"arm_action_{side}"] = {
                    "joint_pos": np.asarray(
                        pose, dtype=np.float32
                    ).reshape(7).tolist(),
                }
                actual[f"hand_action_{side}"] = np.asarray(
                    hand_deg, dtype=np.float32
                ).reshape(20).tolist()
            else:
                actual[f"arm_action_{side}"] = {
                    "ee_pos": pose[:3].copy(),
                    "ee_quat": pose[3:7].copy(),
                }
                actual[f"hand_action_{side}"] = hand_deg.copy()
        return actual

    def _fdm_current_qpos_sample(self) -> tuple[np.ndarray, float]:
        """Return the latest measured 54D qpos at one 30 Hz action tick."""

        snapshot = self._snapshot()
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        qpos_parts: list[np.ndarray] = []
        source_timestamps: list[float] = []
        for side in SIDES:
            arm_key = f"arm_state_{side}"
            if arm_key not in snapshot:
                raise ValueError(
                    f"state-history missing measured {side} arm state"
                )
            arm_stamp, arm_message = snapshot[arm_key]
            if now_ros - float(arm_stamp) > self.max_observation_age_s:
                raise ValueError(f"state-history measured {side} arm state is stale")
            arm_qpos = np.radians(
                np.asarray(arm_message.position, dtype=np.float32)
            ).reshape(-1)
            if arm_qpos.shape != (7,) or not np.all(np.isfinite(arm_qpos)):
                raise ValueError(
                    f"state-history measured {side} arm qpos must be finite 7D"
                )
            qpos_parts.append(arm_qpos.astype(np.float32, copy=False))
            source_timestamps.append(float(arm_stamp))

            if side in self.active_hand_sides:
                hand_key = f"hand_state_{side}"
                if hand_key not in snapshot:
                    raise ValueError(
                        f"state-history missing measured {side} hand state"
                    )
                hand_stamp, hand_message = snapshot[hand_key]
                if now_ros - float(hand_stamp) > self.max_observation_age_s:
                    raise ValueError(
                        f"state-history measured {side} hand state is stale"
                    )
                hand_qpos = np.asarray(
                    hand_message.position, dtype=np.float32
                ).reshape(-1)
                if hand_qpos.shape != (20,) or not np.all(
                    np.isfinite(hand_qpos)
                ):
                    raise ValueError(
                        f"state-history measured {side} hand qpos must be finite 20D"
                    )
                qpos_parts.append(hand_qpos.copy())
                source_timestamps.append(float(hand_stamp))
            else:
                qpos_parts.append(np.zeros(20, dtype=np.float32))

        qpos = np.concatenate(qpos_parts).astype(np.float32, copy=False)
        if qpos.shape != (54,):
            raise AssertionError(
                f"state-history qpos has invalid shape {qpos.shape}"
            )
        return qpos, max(source_timestamps)

    def _fdm_record_executed_action(
        self,
        action: Mapping[str, Any],
        *,
        chunk_action_index: int,
    ) -> None:
        if getattr(self, "fdm_config", None) is None:
            return
        assert self._fdm_ledger is not None
        assert self._fdm_feedback_accumulator is not None
        assert self._fdm_feedback_worker is not None
        if (
            self._fdm_active_global_action_start is None
            or self._fdm_active_wire_chunk_id is None
        ):
            raise RuntimeError("executed FDM action has no active wire identity")
        global_action_index = (
            self._fdm_active_global_action_start + int(chunk_action_index)
        )
        with self._session_lock:
            session_id = self._session_id
        generation = self._stream_generation
        actual_action = self._fdm_actual_wire_action(action)
        executed_at = self.get_clock().now().nanoseconds * 1e-9
        qpos = None
        qpos_timestamp = None
        if self.fdm_config.state_history_enabled:
            qpos, qpos_timestamp = self._fdm_current_qpos_sample()
        executed_frontier = self._fdm_ledger.record_execution(
            session_id, global_action_index
        )
        draft = self._fdm_feedback_accumulator.record(
            session_id=session_id,
            generation=generation,
            global_action_index=global_action_index,
            action=actual_action,
            executed_at=executed_at,
            keyframe_not_before=executed_at,
            qpos=qpos,
            qpos_timestamp=qpos_timestamp,
        )
        _trace_event(
            self,
            "fdm_action_executed",
            wire_chunk_id=self._fdm_active_wire_chunk_id,
            native_spans=list(self._fdm_active_native_spans),
            global_action_index=global_action_index,
            executed_frontier=executed_frontier,
        )
        if draft is not None:
            self._fdm_feedback_worker.submit(draft)
            _trace_event(
                self,
                "fdm_feedback_queued",
                request_id=draft.request_id,
                feedback_id=draft.feedback_id,
                feedback_seq=draft.feedback_seq,
                global_action_start=draft.global_action_start,
                action_count=len(draft.executed_actions),
            )

    def _publish_startup_anchor(self, now: float) -> bool:
        """Republish the fixed bootstrap target without advancing model time."""

        validated = self._startup_anchor_validated
        candidate_poses = self._startup_anchor_poses
        candidate_hands = self._startup_anchor_hands
        if (
            validated is None
            or candidate_poses is None
            or candidate_hands is None
        ):
            return False
        DeploymentNode._publish_validated_command(self, validated)
        last_applied_at = float(getattr(self, "_last_applied_at", 0.0))
        if last_applied_at > 0.0:
            self._previous_applied_pose = {
                side: (
                    None
                    if self._last_applied_pose[side] is None
                    else self._last_applied_pose[side].copy()
                )
                for side in SIDES
            }
            self._previous_applied_hand = {
                side: (
                    None
                    if self._last_applied_hand[side] is None
                    else self._last_applied_hand[side].copy()
                )
                for side in SIDES
            }
            self._previous_applied_at = last_applied_at
        self._last_applied_pose = {
            side: candidate_poses[side].copy() for side in SIDES
        }
        self._last_applied_hand = {
            side: candidate_hands[side].copy() for side in SIDES
        }
        self._last_applied_zsp = {
            side: (
                None
                if validated[side][2] is None
                else validated[side][2].copy()
            )
            for side in SIDES
        }
        self._last_applied_at = float(now)
        self._interpolated_commands += 1
        self._startup_anchor_publish_count += 1
        return True

    def _startup_handoff_tick(self, now: float) -> bool:
        """Drive the Pi HTTP bootstrap/handoff/replan barrier.

        Returns True while normal action-plan scheduling must remain paused.
        """

        if not getattr(self, "startup_handoff_gate_enabled", False):
            return False
        state = self._startup_state
        if state == STARTUP_RUNNING:
            return False
        if state in (STARTUP_WAIT_READY, STARTUP_FAILED):
            return True

        if state == STARTUP_WAIT_BOOTSTRAP:
            if self._controller_handoff_state == CONTROLLER_HANDOFF_FAILED:
                reason = "Tianji controller rejected the external handoff"
                self._set_startup_state(STARTUP_FAILED, reason=reason)
                self._request_policy_standby(reason)
                return True
            with self._prefetch_lock:
                pending = self._pending_chunk
                if pending is not None:
                    self._pending_chunk = None
                    self._waiting_for_pending_since = 0.0
            if pending is None:
                return True
            try:
                anchor_action = pending.actions[0]
                DeploymentNode._validate_pi_joint_measured_boundary(
                    self, anchor_action
                )
                validated, poses, hands = self._validate_response(
                    anchor_action
                )
            except Exception as exc:
                self._failures += 1
                self._last_server_error = (
                    f"invalid startup handoff target: {exc}"
                )
                _trace_event(
                    self,
                    "startup_anchor_rejected",
                    request_id=pending.request_id,
                    error=str(exc),
                )
                self._network_wakeup.set()
                return True
            self._startup_anchor_action = anchor_action
            self._startup_anchor_validated = validated
            self._startup_anchor_poses = poses
            self._startup_anchor_hands = hands
            self._startup_anchor_request_id = pending.request_id
            self._startup_seen_controller_active = (
                self._controller_handoff_state
                == CONTROLLER_HANDOFF_ACTIVE
            )
            self._set_startup_state(STARTUP_HANDOFF_ACTIVE)
            self._publish_startup_anchor(now)
            _trace_event(
                self,
                "startup_anchor_latched",
                request_id=pending.request_id,
                discarded_chunk_actions=len(pending.actions),
                **(
                    {
                        "arm_joint_rad": {
                            side: poses[side].tolist()
                            for side in self.active_arm_sides
                        }
                    }
                    if getattr(self, "policy_action_mode", "eef")
                    == "joint"
                    else {
                        "arm_eef": {
                            side: poses[side].tolist()
                            for side in self.active_arm_sides
                        }
                    }
                ),
                hand_deg={
                    side: hands[side].tolist()
                    for side in self.active_hand_sides
                },
            )
            return True

        if state == STARTUP_HANDOFF_ACTIVE:
            self._publish_startup_anchor(now)
            if self._controller_handoff_state == CONTROLLER_HANDOFF_FAILED:
                reason = "Tianji controller reported external handoff failure"
                self._set_startup_state(STARTUP_FAILED, reason=reason)
                self._request_policy_standby(reason)
                return True
            completed_at = float(
                self._controller_handoff_payload.get(
                    "completed_at_monotonic", 0.0
                )
            )
            current_handoff_completed = bool(
                self._controller_handoff_state
                == CONTROLLER_HANDOFF_COMPLETE
                and (
                    self._startup_seen_controller_active
                    or completed_at >= self._startup_state_entered_at
                )
            )
            if not current_handoff_completed:
                return True

            bootstrap_request_id = self._startup_anchor_request_id
            bootstrap_publish_count = self._startup_anchor_publish_count
            # Invalidate every bootstrap-generation response before allowing
            # the post-handoff observation to leave the robot.
            self._clear_action_stream(clear_active=True)
            self._valid_policy_stream_started = False
            self._set_startup_state(STARTUP_WAIT_FRESH)
            _trace_event(
                self,
                "startup_bootstrap_discarded",
                request_id=bootstrap_request_id,
                anchor_publish_count=bootstrap_publish_count,
                controller_completed_at=completed_at,
            )
            self._network_wakeup.set()
            return True

        if state == STARTUP_WAIT_FRESH:
            self._publish_startup_anchor(now)
            with self._prefetch_lock:
                pending_ready = self._pending_chunk is not None
            if not pending_ready:
                return True
            if not self._activate_pending_chunk(now):
                return True
            self._set_startup_state(STARTUP_RUNNING)
            self._valid_policy_stream_started = True
            _trace_event(
                self,
                "startup_policy_clock_started",
                bootstrap_request_id=self._startup_anchor_request_id,
                active_chunk_id=self._active_chunk_id,
                anchor_publish_count=self._startup_anchor_publish_count,
            )
            return False

        return True

    def _fdm_wait_for_fresh_state_after_ready(self, now: float) -> bool:
        """Gate W0 until feedback can sample a fresh post-Enable qpos.

        Tianji pauses state publication during its Enable stability check. The
        READY lifecycle sample can therefore arrive a few milliseconds before
        the first resumed joint-state sample.  Do not consume or publish W0 in
        that window because the required state-history feedback would contain
        the pre-Enable sample.
        """

        if not getattr(
            self, "_fdm_waiting_for_fresh_state_after_ready", False
        ):
            return False
        try:
            self._fdm_current_qpos_sample()
        except Exception as exc:
            reason = str(exc)
            if reason != getattr(self, "_fdm_fresh_state_wait_reason", ""):
                self._fdm_fresh_state_wait_reason = reason
                _trace_event(
                    self,
                    "fdm_wait_fresh_state_after_ready",
                    reason=reason,
                )
            entered_at = float(getattr(self, "_ready_entered_at", 0.0))
            if (
                entered_at > 0.0
                and float(now) - entered_at
                >= float(self.policy_start_timeout_s)
            ):
                self._fdm_waiting_for_fresh_state_after_ready = False
                self._fdm_reset_session(
                    "FDM READY did not receive fresh measured state: "
                    f"{reason}",
                    request_standby=True,
                )
            return True
        self._fdm_waiting_for_fresh_state_after_ready = False
        self._fdm_fresh_state_wait_reason = ""
        _trace_event(self, "fdm_fresh_state_after_ready")
        return False

    def _apply_pending_action(self) -> None:
        """Publish normally; aggregate callback timing at most once a second."""
        if getattr(self, "_trace_writer", None) is None:
            return DeploymentNode._apply_pending_action_impl(self)
        started = time.perf_counter()
        stats = getattr(self, "_callback_timing", None)
        if stats is None:
            stats = dict(window_start=started, previous_start=started, count=0,
                         duration_sum=0.0, duration_max=0.0, interval_sum=0.0,
                         interval_max=0.0, interval_count=0, over_budget=0)
            self._callback_timing = stats
        previous_start = getattr(self, "_last_callback_timing_start", None)
        self._last_callback_timing_start = started
        interval = 0.0 if previous_start is None else started - previous_start
        if previous_start is not None:
            stats["interval_sum"] += interval
            stats["interval_max"] = max(stats["interval_max"], interval)
            stats["interval_count"] += 1
        stats["previous_start"] = started
        try:
            return DeploymentNode._apply_pending_action_impl(self)
        finally:
            elapsed = time.perf_counter() - started
            stats["count"] += 1
            stats["duration_sum"] += elapsed
            stats["duration_max"] = max(stats["duration_max"], elapsed)
            stats["over_budget"] += int(elapsed > 1.0 / self.publish_rate_hz)
            if started - stats["window_start"] >= 1.0:
                _trace_event(
                    self, "callback_timing", callback_count=stats["count"],
                    window_duration_s=started - stats["window_start"],
                    duration_mean_ms=1000 * stats["duration_sum"] / stats["count"],
                    duration_max_ms=1000 * stats["duration_max"],
                    interval_mean_ms=1000 * stats["interval_sum"] / max(1, stats["interval_count"]),
                    interval_max_ms=1000 * stats["interval_max"],
                    callbacks_over_period=stats["over_budget"],
                    nominal_period_ms=1000 / self.publish_rate_hz,
                )
                self._callback_timing = None

    def _apply_pending_action_impl(self) -> None:
        if self._lifecycle not in COMMAND_LIFECYCLES:
            return
        now = time.monotonic()
        if DeploymentNode._startup_handoff_tick(self, now):
            return
        if DeploymentNode._fdm_wait_for_fresh_state_after_ready(self, now):
            return
        if (self._action_plan.remaining() == 0 or getattr(self, "early_splice_enabled", False)
                or getattr(self, "paper_async_blend_enabled", False)):
            self._activate_pending_chunk(now)
        scheduled = self._action_plan.pop_due(now)
        new_model_waypoint = scheduled is not None
        if scheduled is not None:
            sequence = scheduled.sequence
            if sequence <= self._applied_sequence:
                return
            try:
                _, model_poses, model_hands = self._validate_response(
                    scheduled.action
                )
            except Exception as exc:
                self._applied_sequence = sequence
                _trace_event(
                    self,
                    "policy_action_rejected",
                    sequence=sequence,
                    error=str(exc),
                )
                self.get_logger().error(f"Rejected policy action: {exc}")
                return

            interval = 1.0 / self.action_rate_hz
            dispatch_lag = max(0.0, float(now) - scheduled.due_at)
            self._interpolation_previous = getattr(self, "_interpolation_current", None)
            self._interpolation_current = scheduled
            # A genuine executor stall causes LatestActionPlan to move the
            # next deadline to now + interval. Start the interpolation clock
            # at now as well, rather than catching up through the old segment.
            self._interpolation_current_start_at = (
                float(now) if dispatch_lag >= interval else scheduled.due_at
            )
            self._applied_sequence = sequence
            self._last_action_due_at = scheduled.due_at

            remaining = self._action_plan.remaining()
            chunk_action_index = self._active_chunk_action_index
            self._active_chunk_action_index += 1
            self._interpolation_chunk_id = self._active_chunk_id
            self._interpolation_chunk_action_index = chunk_action_index
            joint_action_mode = (
                getattr(self, "policy_action_mode", "eef") == "joint"
            )
            _trace_event(
                self,
                "action_dispatch",
                sequence=sequence,
                chunk_id=self._active_chunk_id,
                chunk_action_index=chunk_action_index,
                chunk_skip_actions=self._active_chunk_skip_actions,
                chunk_prefetched=self._active_chunk_prefetched,
                chunk_blend_steps=self._active_chunk_blend_steps,
                scheduled_due_at=scheduled.due_at,
                dispatched_at=float(now),
                dispatch_lag_ms=dispatch_lag * 1000.0,
                remaining_actions=remaining,
                **({"arm_joint_rad": {
                    side: model_poses[side].tolist()
                    for side in self.active_arm_sides
                }} if joint_action_mode else {"arm_eef": {
                    side: model_poses[side].tolist()
                    for side in self.active_arm_sides
                }}),
                **({"hand_rad": {
                    side: model_hands[side].tolist()
                    for side in self.active_hand_sides
                }} if _hand_actions_are_radians(self) else {"hand_deg": {
                    side: model_hands[side].tolist()
                    for side in self.active_hand_sides
                }}),
            )

        current = getattr(self, "_interpolation_current", None)
        if current is None:
            if self._action_plan.remaining() == 0:
                self._mark_prefetch_miss(now)
            return

        next_waypoint = self._action_plan.peek_next()
        interpolation_method = getattr(
            self, "action_interpolation_method", "none"
        )
        hold_last_for_pending = bool(
            not new_model_waypoint
            and next_waypoint is None
            and self._action_plan.remaining() == 0
            and getattr(self, "fdm_config", None) is not None
            and self.fdm_config.pending_miss_policy == "hold_last"
        )
        if not new_model_waypoint and (
            interpolation_method == "none" or next_waypoint is None
        ):
            if self._action_plan.remaining() == 0:
                reset_for_underrun = self._mark_prefetch_miss(now)
                if reset_for_underrun:
                    return
            if not hold_last_for_pending:
                return

        command = current.action
        fraction = 0.0
        to_sequence = current.sequence
        sampling_method = (
            "fdm_hold_last" if hold_last_for_pending else "waypoint"
        )
        if (
            interpolation_method
            in (
                "linear_slerp",
                "pchip_slerp",
                "linear_joint",
                "pchip_joint",
            )
            and next_waypoint is not None
        ):
            segment_start = float(
                getattr(
                    self,
                    "_interpolation_current_start_at",
                    current.due_at,
                )
            )
            duration = next_waypoint.due_at - segment_start
            if duration <= 0.0:
                fraction = 1.0
            else:
                fraction = float(
                    np.clip((float(now) - segment_start) / duration, 0.0, 1.0)
                )
            chunk_interpolator = getattr(self, "_chunk_interpolator", None)
            chunk_id = getattr(self, "_interpolation_chunk_id", 0)
            chunk_action_index = getattr(
                self, "_interpolation_chunk_action_index", 0
            )
            if (
                interpolation_method in ("pchip_slerp", "pchip_joint")
                and chunk_interpolator is not None
                and getattr(self, "_chunk_interpolator_chunk_id", 0)
                == chunk_id
                and chunk_action_index
                < chunk_interpolator.action_count - 1
            ):
                command = chunk_interpolator.sample(
                    chunk_action_index + fraction
                )
                if getattr(self, "policy_action_mode", "eef") == "joint":
                    sampling_method = (
                        "pchip_joint"
                        if chunk_interpolator.pchip_enabled
                        else "linear_joint_fallback"
                    )
                else:
                    sampling_method = (
                        "pchip_slerp"
                        if chunk_interpolator.pchip_enabled
                        else "linear_slerp_fallback"
                    )
            else:
                command = interpolate_action_pair(
                    current.action,
                    next_waypoint.action,
                    fraction=fraction,
                    sides=SIDES,
                    action_mode=getattr(
                        self, "policy_action_mode", "eef"
                    ),
                )
                sampling_method = (
                    (
                        "linear_joint_boundary"
                        if interpolation_method == "pchip_joint"
                        else "linear_joint"
                    )
                    if getattr(self, "policy_action_mode", "eef") == "joint"
                    else (
                        "linear_slerp_boundary"
                        if interpolation_method == "pchip_slerp"
                        else "linear_slerp"
                    )
                )
            to_sequence = next_waypoint.sequence

        boundary_interpolator = getattr(
            self, "_boundary_interpolator", None
        )
        if (
            boundary_interpolator is not None
            and getattr(self, "_boundary_interpolator_chunk_id", 0)
            == getattr(self, "_active_chunk_id", 0)
        ):
            boundary_elapsed = max(
                0.0,
                float(now)
                - float(
                    getattr(self, "_boundary_interpolator_start_at", now)
                ),
            )
            if boundary_elapsed <= boundary_interpolator.duration_s:
                command = boundary_interpolator.sample(boundary_elapsed)
                fraction = float(
                    np.clip(
                        boundary_elapsed / boundary_interpolator.duration_s,
                        0.0,
                        1.0,
                    )
                )
                sampling_method = "velocity_continuous_boundary"
            else:
                self._boundary_interpolator = None
                self._boundary_interpolator_chunk_id = 0
                self._boundary_interpolator_start_at = 0.0

        try:
            validated, candidate_poses, candidate_hands = self._validate_response(
                command
            )
        except Exception as exc:
            _trace_event(
                self,
                "interpolated_command_rejected",
                from_sequence=current.sequence,
                to_sequence=to_sequence,
                fraction=fraction,
                error=str(exc),
            )
            self.get_logger().error(f"Rejected interpolated command: {exc}")
            return

        publish_validated = validated
        published_poses = candidate_poses
        pi_joint_rate_limited = False
        if _uses_synchronous_joint_actions(self):
            try:
                measured_poses = DeploymentNode._pi_joint_measured_positions(
                    self, tuple(self.active_arm_sides)
                )
                (
                    publish_validated,
                    published_poses,
                    pi_joint_rate_limited,
                ) = DeploymentNode._limit_pi_joint_publish_target(
                    self,
                    validated,
                    measured=measured_poses,
                    now=now,
                )
            except Exception as exc:
                reason = f"Pi joint runtime safety failed: {exc}"
                _trace_event(
                    self,
                    "pi_joint_runtime_safety_failed",
                    error=str(exc),
                )
                self._request_policy_standby(reason)
                return

        DeploymentNode._publish_validated_command(
            self, publish_validated, command=command
        )
        ros_publish_complete_at = time.monotonic()
        previous_publish_at = getattr(self, "_trace_last_ros_publish_at", None)
        self._trace_last_ros_publish_at = ros_publish_complete_at
        DeploymentNode._record_pi_joint_published_target(
            self, published_poses, now=now
        )
        if hold_last_for_pending:
            self._fdm_hold_last_publishes = (
                getattr(self, "_fdm_hold_last_publishes", 0) + 1
            )
        last_applied_at = float(getattr(self, "_last_applied_at", 0.0))
        if last_applied_at > 0.0:
            self._previous_applied_pose = {
                side: (
                    None
                    if self._last_applied_pose[side] is None
                    else self._last_applied_pose[side].copy()
                )
                for side in SIDES
            }
            self._previous_applied_hand = {
                side: (
                    None
                    if self._last_applied_hand[side] is None
                    else self._last_applied_hand[side].copy()
                )
                for side in SIDES
            }
            self._previous_applied_at = last_applied_at
        self._last_applied_pose.update(candidate_poses)
        self._last_applied_hand.update(candidate_hands)
        self._last_applied_zsp.update(
            {
                side: None if values[2] is None else values[2].copy()
                for side, values in validated.items()
            }
        )
        self._last_applied_at = float(now)
        self._interpolated_commands = (
            getattr(self, "_interpolated_commands", 0) + 1
        )
        arm_trace = (
            {
                "arm_joint_rad": {
                    side: published_poses[side].tolist()
                    for side in self.active_arm_sides
                }
            }
            if getattr(self, "policy_action_mode", "eef") == "joint"
            else {
                "arm_eef": {
                    side: candidate_poses[side].tolist()
                    for side in self.active_arm_sides
                }
            }
        )
        hand_trace = (
            {
                "hand_rad": {
                    side: candidate_hands[side].tolist()
                    for side in self.active_hand_sides
                }
            }
            if _hand_actions_are_radians(self)
            else {
                "hand_deg": {
                    side: candidate_hands[side].tolist()
                    for side in self.active_hand_sides
                }
            }
        )
        pi_joint_trace = (
            {
                "arm_joint_requested_rad": {
                    side: candidate_poses[side].tolist()
                    for side in self.active_arm_sides
                },
                "pi_joint_rate_limited": pi_joint_rate_limited,
            }
            if _uses_synchronous_joint_actions(self)
            else {}
        )
        _trace_event(
            self,
            "command_publish",
            interpolation_method=interpolation_method,
            sampling_method=sampling_method,
            from_sequence=current.sequence,
            to_sequence=to_sequence,
            fraction=fraction,
            chunk_id=getattr(self, "_interpolation_chunk_id", 0),
            chunk_action_index=getattr(
                self, "_interpolation_chunk_action_index", 0
            ),
            hold_last_for_pending=hold_last_for_pending,
            published_at=float(now),
            ros_publish_complete_at=ros_publish_complete_at,
            ros_publish_interval_ms=(None if previous_publish_at is None else
                                     (ros_publish_complete_at - previous_publish_at) * 1000.0),
            callback_to_ros_publish_ms=(ros_publish_complete_at - float(now)) * 1000.0,
            **arm_trace,
            **hand_trace,
            **pi_joint_trace,
        )

        if new_model_waypoint:
            try:
                DeploymentNode._fdm_record_executed_action(
                    self,
                    command,
                    chunk_action_index=chunk_action_index,
                )
            except Exception as exc:
                self.get_logger().error(f"FDM execution feedback failed: {exc}")
                if (
                    getattr(self, "fdm_config", None) is not None
                    and not isinstance(exc, FeedbackQueueFull)
                ):
                    self._fdm_reset_session(
                        f"FDM execution feedback failed: {exc}",
                        request_standby=True,
                    )
                return

        if not new_model_waypoint:
            return
        remaining = self._action_plan.remaining()
        if remaining == 0:
            next_due = scheduled.due_at + 1.0 / self.action_rate_hz
            if not self._activate_pending_chunk(
                now, schedule_start_at=next_due
            ):
                self._mark_prefetch_miss(now)
        elif (
            (getattr(self, "paper_async_blend_enabled", False)
             and now >= self._paper_next_request_at)
            or (self.prefetch_enabled and remaining <= self._prefetch_lead_actions)
        ):
            self._network_wakeup.set()

    def status_dict(self) -> dict[str, Any]:
        with self._session_lock:
            session_id = self._session_id
        fdm_status: dict[str, Any] = {"enabled": False}
        if self.fdm_config is not None:
            assert self._fdm_ledger is not None
            assert self._fdm_feedback_accumulator is not None
            assert self._fdm_feedback_worker is not None
            fdm_status = {
                "enabled": True,
                "hello_accepted": self._fdm_hello_accepted,
                "bootstrapped": self._fdm_bootstrapped,
                "active_wire_chunk_id": self._fdm_active_wire_chunk_id,
                "active_global_action_start": (
                    self._fdm_active_global_action_start
                ),
                "active_native_spans": list(self._fdm_active_native_spans),
                "underrun_resets": self._fdm_underrun_resets,
                "hold_last_publishes": self._fdm_hold_last_publishes,
                "pending_miss_policy": self.fdm_config.pending_miss_policy,
                "pending_miss_timeout_s": (
                    self.fdm_config.pending_miss_timeout_s
                ),
                "last_reset_reason": self._fdm_last_reset_reason,
                "frontiers": self._fdm_ledger.status(),
                "feedback_accumulator": (
                    self._fdm_feedback_accumulator.status()
                ),
                "feedback_worker": self._fdm_feedback_worker.status(),
            }
        return {
            "server": self.server,
            "policy_transport": self.policy_transport,
            "last_http_status": self._last_http_status,
            "transport_reconnects": self._transport_reconnects,
            "server_identity_mismatches": self._server_identity_mismatches,
            "lifecycle": self._lifecycle,
            "requests": self._requests,
            "failures": self._failures,
            "last_latency_ms": self._last_latency_ms,
            "last_observation_age_ms": self._last_observation_age_ms,
            "last_request_bytes": self._last_request_bytes,
            "last_response_bytes": self._last_response_bytes,
            "last_model_id": self._last_model_id,
            "last_server_inference_ms": self._last_server_inference_ms,
            "server_ready": self._server_ready,
            "last_server_error": self._last_server_error,
            "valid_policy_stream_started": self._valid_policy_stream_started,
            "policy_start_timeout_s": self.policy_start_timeout_s,
            "startup_handoff": {
                "enabled": self.startup_handoff_gate_enabled,
                "state": self._startup_state,
                "state_elapsed_s": max(
                    0.0, time.monotonic() - self._startup_state_entered_at
                ),
                "timeout_s": self.startup_handoff_timeout_s,
                "controller_state": self._controller_handoff_state,
                "controller_progress": self._controller_handoff_progress,
                "controller_payload": dict(
                    self._controller_handoff_payload
                ),
                "bootstrap_request_id": self._startup_anchor_request_id,
                "anchor_publish_count": (
                    self._startup_anchor_publish_count
                ),
                "seen_controller_active": (
                    self._startup_seen_controller_active
                ),
                "failure_reason": self._startup_failure_reason,
            },
            "last_action_sequence": self._applied_sequence,
            "cameras": self.camera_names,
            "camera_transport": self.camera_transport,
            "active_arm_sides": list(self.active_arm_sides),
            "active_hand_sides": list(self.active_hand_sides),
            "zero_filled_hand_sides": list(self.zero_filled_hand_sides),
            "protocol_mode": self.protocol_mode,
            "protocol_version": self.protocol_version,
            "action_mode": self.policy_action_mode,
            "expected_arm_action_space": self.expected_arm_action_space,
            "negotiated_arm_action_space": (
                self._negotiated_arm_action_space
            ),
            "session_id": session_id,
            "image_codec": self.image_codec,
            "paper_async_blend_enabled": getattr(self, "paper_async_blend_enabled", False),
            "paper_async_stride_steps": getattr(self, "paper_async_stride_steps", 16),
            "paper_async_weight_curve": getattr(self, "paper_async_weight_curve", "linear"),
            "initial_blend_steps": self.initial_blend_steps,
            "boundary_blend_steps": self.boundary_blend_steps,
            "boundary_blend_method": self.boundary_blend_method,
            "action_interpolation_method": self.action_interpolation_method,
            "action_smoothing_method": self.action_smoothing_method,
            "action_smoothing_cutoff_hz": (
                self.action_smoothing_cutoff_hz
            ),
            "action_smoothing_order": self.action_smoothing_order,
            "published_commands": self._interpolated_commands,
            "pi_joint_runtime_safety": {
                "enabled": _uses_synchronous_joint_actions(self),
                "command_velocity_limit_deg_s": (
                    []
                    if self._pi_joint_command_velocity_rad_s is None
                    else np.degrees(
                        self._pi_joint_command_velocity_rad_s
                    ).tolist()
                ),
                "rate_limit_events": self._pi_joint_rate_limit_events,
            },
            "action_plan": self._action_plan.status(),
            "prefetch": self._prefetch_status(),
            "fdm": fdm_status,
            "diagnostic_trace": self._trace_status(),
        }

    def _publish_status(self) -> None:
        message = String()
        message.data = json.dumps(self.status_dict())
        self._status_publisher.publish(message)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._network_wakeup.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
        if self._fdm_feedback_worker is not None:
            self._fdm_feedback_worker.close(timeout_s=3.0)
        if self._trace_writer is not None:
            _trace_event(
                self,
                "session_stop",
                requests=self._requests,
                failures=self._failures,
                dropped_trace_events=self._trace_writer.dropped_events,
                writer_error=getattr(self._trace_writer, "writer_error", ""),
                recording_run_id=str(self.deployment_config.get("recording_run_id", "")),
                action_interpolation_method=(
                    self.action_interpolation_method
                ),
                action_smoothing_method=self.action_smoothing_method,
                action_smoothing_cutoff_hz=(
                    self.action_smoothing_cutoff_hz
                ),
                action_smoothing_order=self.action_smoothing_order,
                boundary_blend_method=self.boundary_blend_method,
                initial_blend_steps=self.initial_blend_steps,
                published_commands=self._interpolated_commands,
                prefetch=self._prefetch_status(),
                action_plan=self._action_plan.status(),
            )
            self._trace_writer.close(timeout_s=3.0)
        for reader in self._camera_readers.values():
            reader.close()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ROS 2 policy deployment client")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--server",
        default=None,
        help=(
            "override deployment.server, for example tcp://10.0.0.5:5555 "
            "or https://policy.example"
        ),
    )
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--camera-transport",
        choices=CAMERA_TRANSPORT_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--active-hand",
        choices=ACTIVE_HAND_CHOICES,
        default="both",
        help="connected physical Wuji Hand side(s)",
    )
    parser.add_argument(
        "--arm-command-mode",
        choices=("eef", "joint"),
        default="eef",
    )
    return parser.parse_args(argv)


def main(argv=None):
    import sys

    raw_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    config = load_config(args.config)
    rclpy.init(args=raw_argv)
    node = DeploymentNode(
        config,
        require_cameras=not args.no_camera,
        active_hand=args.active_hand,
        active_arm=args.active_hand,
        arm_command_mode=args.arm_command_mode,
        camera_transport=args.camera_transport,
        server=args.server,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
