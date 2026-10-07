import threading
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from builtin_interfaces.msg import Time as TimeMessage

from wuji_data_pipeline.config import load_config
from wuji_data_pipeline.deployment_node import DeploymentNode
from wuji_data_pipeline.deployment_protocol import LatestActionPlan
from wuji_data_pipeline.fdm_async import (
    FDM_PROTOCOL_MODE,
    FdmProtocolConfig,
)


def fdm_config(**overrides):
    config_path = Path(__file__).parents[1] / "config" / "lingbot_va_fdm.yaml"
    deployment = load_config(str(config_path))["deployment"]
    raw = dict(deployment["fdm_async"])
    raw.update(overrides)
    deployment = dict(deployment)
    deployment["fdm_async"] = raw
    return FdmProtocolConfig.from_deployment_config(deployment)


def lightweight_prefetch_node():
    node = object.__new__(DeploymentNode)
    node._trace_writer = None
    node._prefetch_lock = threading.Lock()
    node._request_inflight = False
    node._pending_chunk = None
    node._stream_generation = 5
    node._action_plan = LatestActionPlan(max_actions=48)
    node._action_plan.install(
        [{"index": index} for index in range(48)],
        observation_created_at=0.0,
        received_at=0.0,
        rate_hz=30.0,
        schedule_start_at=0.0,
    )
    node.prefetch_enabled = True
    node.protocol_mode = FDM_PROTOCOL_MODE
    node._prefetch_lead_actions = 3
    node._prefetch_requests = 0
    return node


def test_fdm_prefetch_claim_does_not_consume_or_replace_active_48_buffer():
    node = lightweight_prefetch_node()
    claim = DeploymentNode._claim_policy_request(node)

    assert claim == (5, True)
    assert node._request_inflight
    assert node._action_plan.remaining() == 48
    first = node._action_plan.pop_due(0.0)
    assert first.action["index"] == 0
    assert node._action_plan.remaining() == 47


def test_fdm_pending_miss_holds_then_resets_without_advancing_plan():
    node = object.__new__(DeploymentNode)
    node._trace_writer = None
    node.prefetch_enabled = True
    node._valid_policy_stream_started = True
    node._prefetch_lock = threading.Lock()
    node._pending_chunk = None
    node._waiting_for_pending_since = 0.0
    node._prefetch_misses = 0
    node._request_inflight = True
    node._prefetch_lead_actions = 10
    node.fdm_config = fdm_config(pending_miss_timeout_s=0.2)
    node._fdm_underrun_resets = 0
    node._network_wakeup = threading.Event()
    resets = []
    node._fdm_reset_session = lambda reason, request_standby: resets.append(
        (reason, request_standby)
    )

    assert node.fdm_config.pending_miss_policy == "hold_last"
    assert DeploymentNode._mark_prefetch_miss(node, 10.0) is False
    assert node._prefetch_misses == 1
    assert resets == []
    assert DeploymentNode._mark_prefetch_miss(node, 10.19) is False
    assert resets == []
    assert DeploymentNode._mark_prefetch_miss(node, 10.21) is True
    assert len(resets) == 1
    assert resets[0][1] is True
    assert node._fdm_underrun_resets == 1


def test_pre_enable_lifecycle_updates_preserve_bootstrapped_w0():
    node = object.__new__(DeploymentNode)
    node._trace_writer = None
    node._lifecycle = None
    node.fdm_config = fdm_config()
    node._fdm_bootstrapped = True
    node._ready_entered_at = 0.0
    node._valid_policy_stream_started = True
    node._standby_requested_for_policy = False
    node._network_wakeup = threading.Event()
    clears = []
    node._clear_action_stream = lambda clear_active: clears.append(clear_active)

    DeploymentNode._lifecycle_callback(node, SimpleNamespace(data=0))
    assert clears == []
    DeploymentNode._lifecycle_callback(node, SimpleNamespace(data=2))
    assert node._valid_policy_stream_started is True


def test_dedicated_pi_config_remains_protocol_v2_profile():
    config_path = Path(__file__).parents[1] / "config" / "pi05_protocol_v2.yaml"
    config = load_config(str(config_path))
    deployment = config["deployment"]

    assert deployment["protocol_mode"] == "pi_v2"
    assert deployment["policy_http_expected_chunk_size"] == 50
    assert deployment["expected_arm_action_space"] == "joint_position"
    assert deployment["action_interpolation_method"] == "linear_joint"
    assert deployment["open_loop_horizon"] == 30
    assert deployment["policy_http_expected_model_id"] == (
        "checkpoints/pi05_singlerighthand_dropper_100_joint/"
        "dropper_joint_4gpu_v3/30000"
    )
    assert deployment["pi_joint_safety"] == {
        "position_lower_deg": [
            -170.0, -120.0, -170.0, -140.0, -170.0, -60.0, -90.0
        ],
        "position_upper_deg": [
            170.0, 120.0, 170.0, 78.0, 170.0, 60.0, 90.0
        ],
        "velocity_limit_deg_s": [180.0] * 7,
        "command_velocity_limit_deg_s": [180.0] * 7,
    }
    assert deployment["startup_handoff_gate_enabled"] is True
    assert "fdm_async" not in deployment


def test_joint_fdm_profile_enables_measured_state_history():
    config = fdm_config()

    assert config.pending_miss_policy == "hold_last"
    assert config.pending_miss_timeout_s == 5.0
    assert config.action_mode == "joint"
    assert config.state_history_enabled is True
    assert config.model_id


def test_pi_and_fdm_profiles_use_the_same_deployment_model_id_field():
    package_root = Path(__file__).parents[1]
    pi_deployment = load_config(
        str(package_root / "config" / "pi05_protocol_v2.yaml")
    )["deployment"]
    fdm_deployment = load_config(
        str(package_root / "config" / "lingbot_va_fdm.yaml")
    )["deployment"]

    assert pi_deployment["policy_http_expected_model_id"]
    assert fdm_deployment["policy_http_expected_model_id"]
    assert (
        fdm_deployment["fdm_async"]["model_id"]
        == fdm_deployment["policy_http_expected_model_id"]
    )
    assert (
        FdmProtocolConfig.from_deployment_config(fdm_deployment).model_id
        == fdm_deployment["policy_http_expected_model_id"]
    )


def test_fdm_legacy_nested_model_id_is_compatible_but_cannot_conflict():
    config_path = Path(__file__).parents[1] / "config" / "lingbot_va_fdm.yaml"
    deployment = dict(load_config(str(config_path))["deployment"])
    expected_model_id = deployment.pop("policy_http_expected_model_id")
    deployment["fdm_async"] = dict(deployment["fdm_async"])
    deployment["fdm_async"]["model_id"] = expected_model_id

    assert (
        FdmProtocolConfig.from_deployment_config(deployment).model_id
        == expected_model_id
    )

    deployment["policy_http_expected_model_id"] = "different-checkpoint"
    with pytest.raises(ValueError, match="must equal"):
        FdmProtocolConfig.from_deployment_config(deployment)


def test_fdm_profile_keeps_state_trace_arm_topics_complete():
    config_path = Path(__file__).parents[1] / "config" / "lingbot_va_fdm.yaml"
    topics = load_config(str(config_path))["topics"]

    for side in ("left", "right"):
        assert set(topics["arm"][side]) >= {
            "state",
            "command",
            "actual_eef",
            "target_eef",
            "zsp",
            "external_target",
            "external_zsp",
        }


def test_fdm_feedback_action_is_the_locally_normalized_54d_waypoint():
    node = object.__new__(DeploymentNode)
    node.policy_action_mode = "eef"
    node.arm_command_mode = "eef"
    node._last_applied_pose = {"left": None, "right": None}
    node._last_applied_hand = {"left": None, "right": None}
    node._action_for_side = DeploymentNode._action_for_side
    node._validate_response = lambda response, **kwargs: DeploymentNode._validate_response(
        node, response, **kwargs
    )
    action = {}
    for side in ("left", "right"):
        action[f"arm_action_{side}"] = {
            "ee_pos": np.array([0.4, 0.0, 0.2], dtype=np.float32),
            # Deliberately non-unit model output; local validation normalizes it.
            "ee_quat": np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32),
            "zsp": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }
        action[f"hand_action_{side}"] = np.arange(20, dtype=np.float32)

    actual = DeploymentNode._fdm_actual_wire_action(node, action)

    assert set(actual) == {
        "arm_action_left",
        "hand_action_left",
        "arm_action_right",
        "hand_action_right",
    }
    assert np.allclose(actual["arm_action_right"]["ee_quat"], [0, 0, 0, 1])
    assert "zsp" not in actual["arm_action_right"]
    assert actual["hand_action_right"].shape == (20,)


def test_joint_fdm_feedback_action_uses_nested_joint_pos_and_radians():
    node = object.__new__(DeploymentNode)
    node.policy_action_mode = "joint"
    node.arm_command_mode = "joint"
    node._last_applied_pose = {"left": None, "right": None}
    node._last_applied_hand = {"left": None, "right": None}
    node._action_for_side = DeploymentNode._action_for_side
    node._validate_response = lambda response, **kwargs: DeploymentNode._validate_response(
        node, response, **kwargs
    )
    action = {}
    for index, side in enumerate(("left", "right")):
        action[f"arm_action_{side}"] = {
            "joint_pos": (np.arange(7, dtype=np.float32) * 0.01 + index).tolist()
        }
        action[f"hand_action_{side}"] = (
            np.arange(20, dtype=np.float32) * 0.02 + index
        ).tolist()

    actual = DeploymentNode._fdm_actual_wire_action(node, action)

    assert set(actual["arm_action_right"]) == {"joint_pos"}
    assert len(actual["arm_action_right"]["joint_pos"]) == 7
    assert len(actual["hand_action_right"]) == 20
    assert np.allclose(
        actual["arm_action_right"]["joint_pos"],
        action["arm_action_right"]["joint_pos"],
    )
    assert np.allclose(
        actual["hand_action_right"], action["hand_action_right"]
    )
    assert "ee_pos" not in actual["arm_action_right"]
    assert "ee_quat" not in actual["arm_action_right"]


def test_joint_fdm_publishes_ros_arm_and_hand_targets_in_radians():
    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    node = object.__new__(DeploymentNode)
    node.policy_action_mode = "joint"
    node.arm_command_mode = "joint"
    node.active_arm_sides = ("right",)
    node.active_hand_sides = ("right",)
    node._last_applied_pose = {"left": None, "right": None}
    node._last_applied_hand = {"left": None, "right": None}
    node._arm_publishers = {"right": Publisher()}
    node._hand_publishers = {"right": Publisher()}
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: TimeMessage())
    )
    action = {}
    for side in ("left", "right"):
        action[f"arm_action_{side}"] = {"joint_pos": [0.2] * 7}
        action[f"hand_action_{side}"] = [0.3] * 20
    validated, _, _ = DeploymentNode._validate_response(node, action)

    DeploymentNode._publish_validated_command(
        node, validated, command=action
    )

    assert np.allclose(
        node._arm_publishers["right"].messages[-1].position, [0.2] * 7
    )
    assert np.allclose(
        node._hand_publishers["right"].messages[-1].position, [0.3] * 20
    )


def test_fdm_qpos_sample_uses_dataset_54d_order_and_units():
    node = object.__new__(DeploymentNode)
    node.active_hand_sides = ("right",)
    node.max_observation_age_s = 1.0
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)
    )
    left_arm_deg = np.arange(7, dtype=np.float32)
    right_arm_deg = np.arange(10, 17, dtype=np.float32)
    right_hand_rad = np.arange(20, dtype=np.float32) * 0.01
    node._snapshot = lambda: {
        "arm_state_left": (
            9.99,
            SimpleNamespace(position=left_arm_deg.tolist()),
        ),
        "arm_state_right": (
            9.98,
            SimpleNamespace(position=right_arm_deg.tolist()),
        ),
        "hand_state_right": (
            9.97,
            SimpleNamespace(position=right_hand_rad.tolist()),
        ),
    }

    qpos, timestamp = DeploymentNode._fdm_current_qpos_sample(node)

    assert qpos.shape == (54,)
    assert np.allclose(qpos[0:7], np.radians(left_arm_deg))
    assert np.allclose(qpos[7:27], 0.0)
    assert np.allclose(qpos[27:34], np.radians(right_arm_deg))
    assert np.allclose(qpos[34:54], right_hand_rad)
    assert timestamp == 9.99
