from pathlib import Path
from collections import deque
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time as TimeMessage
from sensor_msgs.msg import JointState

import wuji_data_pipeline.deployment_node as deployment_module
import wuji_data_pipeline.deployment_session as deployment_session_module
from wuji_data_pipeline.deployment_session import (
    _configured_arm_command_mode,
    _fdm_state_history_hand_first_enabled,
    _request_ordered_enable,
    _synchronous_http_startup_handoff_enabled,
    _wait_for_deployment_nodes,
)
from wuji_data_pipeline.deployment_node import (
    CONTROLLER_HANDOFF_ACTIVE,
    CONTROLLER_HANDOFF_COMPLETE,
    CONTROLLER_HANDOFF_WAITING,
    DeploymentNode,
    PolicyServerError,
    ServerIdentityMismatch,
    STARTUP_HANDOFF_ACTIVE,
    STARTUP_RUNNING,
    STARTUP_WAIT_BOOTSTRAP,
    STARTUP_WAIT_FRESH,
)
from wuji_data_pipeline.deployment_protocol import (
    LatestActionPlan,
    PendingActionChunk,
)
from wuji_data_pipeline.fdm_async import FdmActionChunk, FdmSessionLedger
from wuji_data_pipeline.policy_transport import (
    PolicyAuthenticationError,
    PolicyExchange,
)


def _node():
    node = SimpleNamespace(
        _action_for_side=DeploymentNode._action_for_side,
        _last_applied_pose={"left": None, "right": None},
        _last_applied_hand={"left": None, "right": None},
        _last_applied_zsp={"left": None, "right": None},
        policy_action_mode="eef",
        _active_chunk_id=0,
        _active_chunk_action_index=0,
        _active_chunk_skip_actions=0,
        _active_chunk_prefetched=False,
        _active_chunk_blend_steps=0,
    )
    node._validate_response = (
        lambda response, previous_poses=None, previous_hands=None:
        DeploymentNode._validate_response(
            node,
            response,
            previous_poses=previous_poses,
            previous_hands=previous_hands,
        )
    )
    node._validate_action_sequence = (
        lambda actions, previous_poses=None, previous_hands=None:
        DeploymentNode._validate_action_sequence(
            node,
            actions,
            previous_poses=previous_poses,
            previous_hands=previous_hands,
        )
    )
    return node


def test_ordered_enable_completes_hand_before_tianji_request():
    calls = []

    class FakeSessionNode:
        def set_hands_enabled(self, active_hand, enabled, abort_event=None):
            calls.append(("hand", active_hand, enabled))
            return True, f"hand enabled={enabled}"

        def call(self, name, request, **kwargs):
            calls.append(("arm", name, request.data))
            return True, "arm enable accepted"

    accepted, hands_enabled, arm_requested, message = _request_ordered_enable(
        FakeSessionNode(), "right"
    )

    assert accepted
    assert hands_enabled
    assert arm_requested
    assert calls == [
        ("hand", "right", True),
        ("arm", "enable", True),
    ]
    assert "arm enable accepted" in message


def test_ordered_enable_rolls_hand_back_when_tianji_request_fails():
    calls = []

    class FakeSessionNode:
        def set_hands_enabled(self, active_hand, enabled, abort_event=None):
            calls.append(("hand", active_hand, enabled))
            return True, f"hand enabled={enabled}"

        def call(self, name, request, **kwargs):
            calls.append(("arm", name, request.data))
            return False, "arm enable failed"

    accepted, hands_enabled, arm_requested, message = _request_ordered_enable(
        FakeSessionNode(), "right"
    )

    assert not accepted
    assert not hands_enabled
    assert arm_requested
    assert calls == [
        ("hand", "right", True),
        ("arm", "enable", True),
        ("hand", "right", False),
    ]
    assert "arm enable failed" in message


def _response(hand_value=0.0):
    response = {}
    for index, side in enumerate(("left", "right")):
        response[f"arm_action_{side}"] = {
            "ee_pos": [0.4, 0.1 * index, 0.5],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "zsp": [0.0, 1.0, 0.0],
        }
        response[f"hand_action_{side}"] = [hand_value] * 20
    return response


def _joint_response(value=0.0):
    response = {}
    for side in ("left", "right"):
        response[f"arm_action_{side}"] = {"joint_pos": [value] * 7}
        response[f"hand_action_{side}"] = [value] * 20
    return response


def _pi_joint_node():
    node = _node()
    node.protocol_mode = "pi_v2"
    node.policy_transport = "http"
    node.expected_arm_action_space = "joint_position"
    node._negotiated_arm_action_space = "joint_position"
    node.policy_action_mode = "joint"
    node.arm_command_mode = "joint"
    node.action_rate_hz = 30.0
    node.max_observation_age_s = 0.5
    node._pi_joint_lower_rad = np.radians(
        [-170.0, -120.0, -170.0, -140.0, -170.0, -60.0, -90.0]
    )
    node._pi_joint_upper_rad = np.radians(
        [170.0, 120.0, 170.0, 78.0, 170.0, 60.0, 90.0]
    )
    node._pi_joint_velocity_rad_s = np.radians([180.0] * 7)
    node._pi_joint_command_velocity_rad_s = np.radians([180.0] * 7)
    node.publish_rate_hz = 120.0
    node.active_arm_sides = ("right",)
    node.active_hand_sides = ("right",)
    node._pi_joint_last_published_pose = {
        "left": None,
        "right": None,
    }
    node._pi_joint_last_published_at = 0.0
    node._pi_joint_rate_limit_events = 0
    return node


def _cosmos_joint_node():
    node = _pi_joint_node()
    node.protocol_mode = "protocol_v2"
    node.expected_arm_action_space = ""
    node._negotiated_arm_action_space = ""
    node.action_rate_hz = 15.0
    node.small_motion_max_arm_joint_step_rad = 0.0872665
    node._joint_acceleration_rad_s2 = np.radians(
        [450.0, 450.0, 900.0, 900.0, 900.0, 900.0, 900.0]
    )
    node.policy_http_expected_chunk_size = 32
    return node


def _cosmos_joint_wire_action(value=0.0):
    response = {}
    for side in ("left", "right"):
        response[f"arm_joint_action_{side}"] = [value] * 7
        response[f"hand_action_{side}"] = [value] * 20
    return response


def _prefetch_node():
    node = _node()
    node._action_plan = LatestActionPlan(max_actions=25)
    node._prefetch_lock = threading.Lock()
    node._network_wakeup = threading.Event()
    node._stream_generation = 3
    node._request_inflight = False
    node._pending_chunk = None
    node._latency_samples_ms = deque(maxlen=100)
    node.prefetch_enabled = True
    node._prefetch_lead_actions = 5
    node._prefetch_p99_rtt_ms = 0.0
    node._prefetch_requests = 0
    node._prefetch_hits = 0
    node._prefetch_misses = 0
    node._prefetch_activation_failures = 0
    node._stale_generation_responses = 0
    node._last_activation_skip_actions = 0
    node._total_activation_skip_actions = 0
    node._last_boundary_wait_ms = 0.0
    node._max_boundary_wait_ms = 0.0
    node._waiting_for_pending_since = 0.0
    node._last_action_due_at = 0.0
    node._active_chunk_id = 0
    node._active_chunk_action_index = 0
    node._active_chunk_skip_actions = 0
    node._active_chunk_prefetched = False
    node._active_chunk_blend_steps = 0
    node._valid_policy_stream_started = True
    node._lifecycle = 2
    node.active_arm_sides = ("right",)
    node.active_hand_sides = ("right",)
    node._failures = 0
    node._last_server_error = ""
    node.action_rate_hz = 30.0
    node.publish_rate_hz = 120.0
    node.open_loop_horizon = 25
    node.prefetch_safety_actions = 2
    node.prefetch_min_lead_actions = 3
    node.prefetch_max_lead_actions = 10
    node.prefetch_initial_lead_actions = 5
    node.prefetch_latency_min_samples = 5
    node.boundary_blend_steps = 6
    node.boundary_blend_method = "smoothstep"
    node.get_logger = lambda: SimpleNamespace(
        warn=lambda _message: None,
        error=lambda _message: None,
        debug=lambda _message: None,
    )
    return node


def test_dual_54d_response_validation_accepts_paired_20d_hands():
    validated, poses, hands = DeploymentNode._validate_response(_node(), _response())

    assert set(validated) == {"left", "right"}
    assert poses["left"].shape == (7,)
    assert hands["right"].shape == (20,)


def test_joint_replay_requires_finite_7d_motor_targets():
    node = _node()
    node.arm_command_mode = "joint"
    response = _response()
    for side in ("left", "right"):
        response[f"arm_joint_action_{side}"] = [0.1] * 7

    DeploymentNode._validate_response(node, response)

    response["arm_joint_action_right"] = [0.1] * 6
    with pytest.raises(ValueError, match="right 7-DoF joint"):
        DeploymentNode._validate_response(node, response)


def test_joint_fdm_parses_nested_arm_and_radian_hand_without_eef():
    node = _node()
    node.policy_action_mode = "joint"
    node.arm_command_mode = "joint"

    validated, joints, hands = DeploymentNode._validate_response(
        node, _joint_response(0.2)
    )

    assert set(validated) == {"left", "right"}
    assert np.allclose(joints["right"], [0.2] * 7)
    assert np.allclose(hands["right"], [0.2] * 20)
    assert validated["right"][2] is None

    invalid = _joint_response(0.2)
    invalid["arm_action_right"]["joint_pos"] = [0.2] * 6
    with pytest.raises(ValueError, match="right 7-DoF joint"):
        DeploymentNode._validate_response(node, invalid)


def test_pi_joint_parses_absolute_arm_radians_and_degree_hands():
    node = _pi_joint_node()
    action = _joint_response(0.2)
    action["hand_action_right"] = [12.0] * 20

    validated, joints, hands = DeploymentNode._validate_response(
        node, action
    )

    assert np.allclose(joints["right"], [0.2] * 7)
    assert np.allclose(hands["right"], [12.0] * 20)
    assert validated["right"][2] is None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda action: action["arm_action_right"].update(
                {"joint_pos": [0.0] * 6}
            ),
            "right 7-DoF joint",
        ),
        (
            lambda action: action["arm_action_right"].update(
                {"joint_pos": [0.0] * 6 + [float("nan")]}
            ),
            "right 7-DoF joint",
        ),
        (
            lambda action: action.update(
                {"arm_action_right": {"ee_pos": [0.4, 0.0, 0.5],
                                      "ee_quat": [0.0, 0.0, 0.0, 1.0]}}
            ),
            "right 7-DoF joint",
        ),
        (
            lambda action: action.update(
                {"hand_action_right": [0.0] * 19}
            ),
            "right 20-DoF hand",
        ),
    ],
)
def test_pi_joint_rejects_wrong_shape_nonfinite_and_eef_only(mutate, message):
    node = _pi_joint_node()
    action = _joint_response(0.0)
    mutate(action)

    with pytest.raises(ValueError, match=message):
        DeploymentNode._validate_response(node, action)


def test_eef_mode_rejects_joint_only_action():
    with pytest.raises(ValueError):
        DeploymentNode._validate_response(_node(), _joint_response(0.0))


def test_pi_joint_rejects_position_limit_and_30hz_velocity_violation():
    node = _pi_joint_node()
    outside = _joint_response(0.0)
    outside["arm_action_right"]["joint_pos"][5] = np.radians(61.0)
    with pytest.raises(ValueError, match="joint 6 position"):
        DeploymentNode._validate_response(node, outside)

    previous = {
        side: np.zeros(7, dtype=np.float32) for side in ("left", "right")
    }
    hands = {side: None for side in ("left", "right")}
    too_fast = _joint_response(0.0)
    # PNVA velocity is 180 deg/s, hence the 30 Hz per-step ceiling is 6 deg.
    too_fast["arm_action_right"]["joint_pos"][0] = np.radians(6.1)
    with pytest.raises(ValueError, match="step/velocity"):
        DeploymentNode._validate_response(
            node,
            too_fast,
            previous_poses=previous,
            previous_hands=hands,
        )


def test_missing_side_and_large_hand_jump_are_rejected():
    response = _response()
    del response["hand_action_right"]
    with pytest.raises(ValueError, match="paired right"):
        DeploymentNode._validate_response(_node(), response)

    node = _node()
    node._last_applied_hand = {
        "left": np.zeros(20, dtype=np.float32),
        "right": np.zeros(20, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="hand jump"):
        DeploymentNode._validate_response(node, _response(hand_value=90.0))


def test_http_validates_all_50_actions_before_installing_chunk():
    node = _node()
    node.policy_transport = "http"
    node.policy_http_expected_chunk_size = 50
    node.action_rate_hz = 30.0
    node._validate_response = (
        lambda response, previous_poses=None, previous_hands=None:
        DeploymentNode._validate_response(
            node,
            response,
            previous_poses=previous_poses,
            previous_hands=previous_hands,
        )
    )
    actions = [_response() for _ in range(50)]

    DeploymentNode._validate_action_chunk(node, actions, 30.0)

    with pytest.raises(ValueError, match="chunk size"):
        DeploymentNode._validate_action_chunk(node, actions[:49], 30.0)
    with pytest.raises(ValueError, match="action_rate_hz mismatch"):
        DeploymentNode._validate_action_chunk(node, actions, 20.0)

    actions[1]["arm_action_right"]["ee_pos"][0] += 0.2
    with pytest.raises(
        ValueError,
        match=r"waypoint 1: unsafe right policy position jump: step=0\.200000 m exceeds 0\.15 m",
    ):
        DeploymentNode._validate_action_chunk(node, actions, 30.0)


def test_eef_chunk_uses_request_anchor_without_relaxing_live_boundary():
    node = _node()
    node.policy_transport = "http"
    node.policy_http_expected_chunk_size = 2
    node.action_rate_hz = 15.0
    node.first_step_anchor_on_measured_pose = True
    _, poses, hands = DeploymentNode._validate_response(node, _response())
    node._last_applied_pose = {side: pose.copy() for side, pose in poses.items()}
    node._last_applied_pose["right"][0] += 0.3
    actions = [_response(), _response()]
    DeploymentNode._validate_action_chunk(
        node, actions, 15.0,
        observation_eef_positions=poses,
        observation_hand_positions=hands,
    )
    # Acceptance of the old-time raw prefix must not replace the live anchor
    # or bypass the check used when installing an execution window.
    with pytest.raises(ValueError, match="waypoint 0: unsafe right policy position jump"):
        node._validate_action_sequence(actions)
    with pytest.raises(ValueError, match="requires observation anchors"):
        DeploymentNode._validate_action_chunk(node, actions, 15.0)


def test_policy_chunk_trace_preserves_response_and_observation_for_offline_replay(tmp_path):
    import json
    from wuji_data_pipeline.deployment_trace import DeploymentTraceWriter

    writer = DeploymentTraceWriter(tmp_path, session_id="chunk_capture")
    node = _node()
    node.trace_policy_chunk_enabled = True
    node._trace_writer = writer
    actions = [_response(), _response()]
    observation = {
        "arm_state_right": {"eef": np.array([0.1, 0.2, 0.3, 0, 0, 0, 1], dtype=np.float32)},
        "hand_state_right": {"joint_pos": np.zeros(20, dtype=np.float32)},
        "images": {"head": {"data": b"excluded"}},
        "api_key": "excluded",
    }
    original_position = list(actions[0]["arm_action_right"]["ee_pos"])
    try:
        DeploymentNode._trace_policy_chunk(
            node, actions, observation, request_id=12, generation=4,
            stage="server_output", action_rate_hz=15.0,
            observation_created_at=100.0,
        )
        actions[0]["arm_action_right"]["ee_pos"][0] += 1.0
        observation["arm_state_right"]["eef"][0] = 99.0
    finally:
        writer.close()
    records = [json.loads(line) for line in writer.path.read_text().splitlines()]
    record = next(item for item in records if item["event"] == "policy_action_chunk")
    assert record["stage"] == "server_output"
    assert record["request_id"] == 12
    assert record["actions"][0]["arm_action_right"]["ee_pos"] == original_position
    assert record["observation_state"]["arm_state_right"]["eef"][0] == pytest.approx(0.1)
    assert set(record["observation_state"]) == {"arm_state_right", "hand_state_right"}
    assert "excluded" not in writer.path.read_text()


def test_policy_chunk_trace_is_opt_in():
    node = _node()
    node._trace_writer = SimpleNamespace(record=lambda *a, **kw: pytest.fail("unexpected trace"))
    DeploymentNode._trace_policy_chunk(
        node, None, None, request_id=1, generation=1, stage="server_output",
        action_rate_hz=15.0, observation_created_at=0.0,
    )


@pytest.mark.parametrize("bad_index", [0, 1])
def test_eef_request_anchor_still_rejects_first_and_internal_jumps(bad_index):
    node = _node()
    node.policy_transport = "http"
    node.policy_http_expected_chunk_size = 2
    node.action_rate_hz = 15.0
    node.first_step_anchor_on_measured_pose = True
    _, poses, hands = DeploymentNode._validate_response(node, _response())
    actions = [_response(), _response()]
    actions[bad_index]["arm_action_right"]["ee_pos"][0] += 0.2
    with pytest.raises(ValueError, match=f"waypoint {bad_index}: unsafe right policy position jump"):
        DeploymentNode._validate_action_chunk(
            node, actions, 15.0,
            observation_eef_positions=poses,
            observation_hand_positions=hands,
        )


def test_pi_joint_validates_measured_first_target_and_all_50_steps():
    node = _pi_joint_node()
    node.policy_http_expected_chunk_size = 50
    node._active_chunk_id = 0
    node._action_plan = LatestActionPlan(max_actions=30)
    now_s = 10.0
    measured = SimpleNamespace(position=[0.0] * 7)
    node._snapshot = lambda: {
        "arm_state_left": (now_s, measured),
        "arm_state_right": (now_s, measured),
    }
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=int(now_s * 1e9))
    )
    node._pi_joint_measured_positions = lambda: (
        DeploymentNode._pi_joint_measured_positions(node)
    )
    node._validate_pi_joint_measured_boundary = lambda action: (
        DeploymentNode._validate_pi_joint_measured_boundary(node, action)
    )
    actions = [_joint_response(0.001 * index) for index in range(50)]

    DeploymentNode._validate_action_chunk(node, actions, 30.0)

    actions[20]["arm_action_right"]["joint_pos"][0] += np.radians(6.1)
    with pytest.raises(ValueError, match="step/velocity"):
        DeploymentNode._validate_action_chunk(node, actions, 30.0)


def test_pi_joint_first_target_must_be_close_to_measured_qpos():
    node = _pi_joint_node()
    now_s = 10.0
    measured = SimpleNamespace(position=[0.0] * 7)
    node._snapshot = lambda: {
        "arm_state_left": (now_s, measured),
        "arm_state_right": (now_s, measured),
    }
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=int(now_s * 1e9))
    )
    target = _joint_response(0.0)
    target["arm_action_right"]["joint_pos"][0] = np.radians(6.1)

    with pytest.raises(ValueError, match="step/velocity"):
        DeploymentNode._validate_pi_joint_measured_boundary(node, target)


def test_pi_joint_final_publish_target_is_slew_limited_at_120hz():
    node = _pi_joint_node()
    target = _joint_response(0.0)
    target["arm_action_right"]["joint_pos"] = [1.0] * 7
    validated, _, _ = DeploymentNode._validate_response(
        node,
        target,
        previous_poses={side: None for side in ("left", "right")},
        previous_hands={side: None for side in ("left", "right")},
    )
    measured = {"right": np.zeros(7, dtype=np.float32)}

    limited, published, was_limited = (
        DeploymentNode._limit_pi_joint_publish_target(
            node, validated, measured=measured, now=10.0
        )
    )

    assert was_limited
    assert np.allclose(
        published["right"], np.radians([1.5] * 7), atol=1e-7
    )
    assert np.allclose(limited["right"][0], published["right"])
    DeploymentNode._record_pi_joint_published_target(
        node, published, now=10.0
    )

    _, published_2, _ = DeploymentNode._limit_pi_joint_publish_target(
        node,
        validated,
        measured=measured,
        now=10.0 + 1.0 / 120.0,
    )
    assert np.allclose(
        published_2["right"], np.radians([3.0] * 7), atol=1e-7
    )
    DeploymentNode._record_pi_joint_published_target(
        node, published_2, now=10.0 + 1.0 / 120.0
    )
    published_4 = None
    for tick in (2, 3):
        _, published_4, _ = DeploymentNode._limit_pi_joint_publish_target(
            node,
            validated,
            measured=measured,
            now=10.0 + tick / 120.0,
        )
        DeploymentNode._record_pi_joint_published_target(
            node, published_4, now=10.0 + tick / 120.0
        )
    # Four 120 Hz ticks cover one 30 Hz model period, so a protocol-valid
    # 6-degree model step is not allowed to accumulate any local lag.
    assert np.allclose(
        published_4["right"], np.radians([6.0] * 7), atol=1e-7
    )


def test_non_pi_joint_path_is_not_slew_limited():
    node = _node()
    node.policy_action_mode = "joint"
    node.arm_command_mode = "joint"
    node.active_arm_sides = ("right",)
    validated, _, _ = DeploymentNode._validate_response(
        node, _joint_response(0.5)
    )

    limited, published, was_limited = (
        DeploymentNode._limit_pi_joint_publish_target(
            node,
            validated,
            measured={"right": np.zeros(7, dtype=np.float32)},
            now=10.0,
        )
    )

    assert not was_limited
    assert np.allclose(limited["right"][0], [0.5] * 7)
    assert np.allclose(published["right"], [0.5] * 7)


def test_fdm_prefetch_validates_future_wire_against_previous_wire_tail():
    node = _node()
    node.policy_transport = "http"
    node.policy_http_expected_chunk_size = 48
    node.action_rate_hz = 30.0
    node.fdm_config = SimpleNamespace(
        action_rate_hz=30.0,
        wire_chunk_size=48,
    )
    node._stream_generation = 0
    node._fdm_delivery_validation_lock = threading.Lock()
    node._fdm_ledger = FdmSessionLedger(node.fdm_config)
    node._fdm_ledger.reset("session-a", 0)
    node._fdm_delivered_tail_pose = {
        "left": None,
        "right": None,
    }
    node._fdm_delivered_tail_hand = {
        "left": None,
        "right": None,
    }

    def moving_chunk(global_start):
        actions = []
        for offset in range(48):
            action = _response()
            position_x = 0.4 + 0.004 * (global_start + offset)
            for side in ("left", "right"):
                action[f"arm_action_{side}"]["ee_pos"][0] = position_x
            actions.append(action)
        return actions

    def chunk(wire_chunk_id, actions):
        return FdmActionChunk(
            session_id="session-a",
            request_id=wire_chunk_id + 1,
            wire_chunk_id=wire_chunk_id,
            global_action_start=wire_chunk_id * 48,
            actions=tuple(actions),
            native_spans=(),
            server_timing={},
        )

    w0 = moving_chunk(0)
    w1 = moving_chunk(48)
    assert DeploymentNode._fdm_validate_and_accept_action_chunk(
        node, chunk(0, w0), generation=0
    )

    # W1 arrives while the robot is still near W0[0]. A current-target check
    # crosses 48 future steps and falsely exceeds the 0.15 m safety threshold.
    node._last_applied_pose = {
        side: np.array(
            [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        for side in ("left", "right")
    }
    node._last_applied_hand = {
        side: np.zeros(20, dtype=np.float32)
        for side in ("left", "right")
    }
    with pytest.raises(ValueError, match="position jump"):
        DeploymentNode._validate_action_chunk(node, w1, 30.0)

    # Delivery validation uses the true W0[-1] -> W1[0] boundary instead.
    assert DeploymentNode._fdm_validate_and_accept_action_chunk(
        node, chunk(1, w1), generation=0
    )
    assert node._fdm_ledger.status()["delivered_frontier"] == 96
    assert np.isclose(node._fdm_delivered_tail_pose["right"][0], 0.78)

    # Internal per-step safety validation remains active.
    w2 = moving_chunk(96)
    w2[1]["arm_action_right"]["ee_pos"][0] += 0.2
    with pytest.raises(ValueError, match="position jump"):
        DeploymentNode._fdm_validate_and_accept_action_chunk(
            node, chunk(2, w2), generation=0
        )
    assert node._fdm_ledger.status()["delivered_frontier"] == 96


def test_single_right_hand_observation_zero_fills_absent_left_hand():
    now_s = 10.0
    arm_message = SimpleNamespace(
        position=[0.0] * 7,
        velocity=[],
        effort=[],
    )
    right_hand_message = SimpleNamespace(
        position=[0.25] * 20,
        velocity=[],
        effort=[],
    )
    snapshot = {
        "arm_state_left": (now_s, arm_message),
        "arm_eef_left": (now_s, np.array([0.4, 0.2, 0.5, 0, 0, 0, 1])),
        "arm_state_right": (now_s, arm_message),
        "arm_eef_right": (now_s, np.array([0.4, -0.2, 0.5, 0, 0, 0, 1])),
        "hand_state_right": (now_s, right_hand_message),
    }
    node = SimpleNamespace(
        active_hand_sides=("right",),
        zero_filled_hand_sides=("left",),
        camera_names=[],
        _snapshot=lambda: snapshot,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(now_s * 1e9))
        ),
    )

    observation = DeploymentNode._build_observation(node)

    assert observation is not None
    assert observation["active_hand_sides"] == ["right"]
    assert observation["zero_filled_hand_sides"] == ["left"]
    assert np.array_equal(
        observation["hand_state_left"]["joint_pos"], np.zeros(20)
    )
    assert np.allclose(observation["hand_state_right"]["joint_pos"], 0.25)
    assert "franka_state_left" not in observation
    assert "xhand_state_left" not in observation


def test_joint_fdm_observation_keeps_measured_eef_state():
    now_s = 10.0
    arm_message = SimpleNamespace(
        position=[10.0] * 7,
        velocity=[],
        effort=[],
    )
    hand_message = SimpleNamespace(
        position=[0.25] * 20,
        velocity=[],
        effort=[],
    )
    node = SimpleNamespace(
        policy_action_mode="joint",
        fdm_config=SimpleNamespace(action_mode="joint"),
        active_hand_sides=("right",),
        zero_filled_hand_sides=("left",),
        camera_names=[],
        _snapshot=lambda: {
            "arm_state_left": (now_s, arm_message),
            "arm_state_right": (now_s, arm_message),
            "arm_eef_left": (now_s, np.asarray([0.1, 0.2, 0.3, 0, 0, 0, 1])),
            "arm_eef_right": (now_s, np.asarray([0.4, 0.5, 0.6, 0, 0, 0, 1])),
            "hand_state_right": (now_s, hand_message),
        },
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(now_s * 1e9))
        ),
    )

    observation = DeploymentNode._build_observation(node)

    assert observation is not None
    assert np.allclose(
        observation["arm_state_right"]["joint_pos"],
        np.radians([10.0] * 7),
    )
    assert np.allclose(observation["arm_state_right"]["ee_pos"], [0.4, 0.5, 0.6])
    assert np.allclose(observation["arm_state_right"]["ee_quat"], [0, 0, 0, 1])
    assert observation["robot_layout"]["units"]["action.arm"] == "radian"
    assert observation["robot_layout"]["action_layout"][1]["arm"] == [27, 34]


def test_prefetch_claims_one_request_only_when_active_reaches_lead():
    node = _prefetch_node()
    now = time.monotonic()
    node._action_plan.install(
        [_response() for _ in range(6)],
        observation_created_at=now,
        received_at=now,
        rate_hz=30.0,
    )

    assert DeploymentNode._claim_policy_request(node) is None
    node._action_plan.pop_due(now)
    claim = DeploymentNode._claim_policy_request(node)

    assert claim == (3, True)
    assert node._prefetch_requests == 1
    assert DeploymentNode._claim_policy_request(node) is None


def test_tcp_replay_does_not_prefetch_an_unconsumed_chunk():
    node = _prefetch_node()
    node.prefetch_enabled = False
    now = time.monotonic()
    node._action_plan.install(
        [_response() for _ in range(5)],
        observation_created_at=now,
        received_at=now,
        rate_hz=30.0,
    )

    assert DeploymentNode._claim_policy_request(node) is None
    node._action_plan.clear()
    assert DeploymentNode._claim_policy_request(node) == (3, False)
    assert node._prefetch_requests == 0


@pytest.mark.parametrize(
    ("protocol_mode", "server", "expected"),
    [
        ("protocol_v2", "https://policy.example", True),
        ("pi_v2", "https://policy.example", True),
        ("pi_v2", "tcp://127.0.0.1:5555", False),
        ("fdm_async", "https://policy.example", False),
    ],
)
def test_session_enables_fixed_handoff_only_for_synchronous_http(
    tmp_path, protocol_mode, server, expected
):
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(
        "deployment:\n"
        f"  protocol_mode: {protocol_mode}\n"
        "  startup_handoff_gate_enabled: true\n"
        f"  server: {server}\n"
    )
    args = SimpleNamespace(config=str(config_path), server=None)

    assert _synchronous_http_startup_handoff_enabled(args) is expected


@pytest.mark.parametrize(
    ("protocol_mode", "state_history_enabled", "expected"),
    [
        ("fdm_async", True, True),
        ("fdm_async", False, False),
        ("protocol_v2", True, False),
        ("pi_v2", False, False),
    ],
)
def test_hand_first_enable_is_scoped_to_fdm_state_history(
    tmp_path, protocol_mode, state_history_enabled, expected
):
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(
        "deployment:\n"
        f"  protocol_mode: {protocol_mode}\n"
        "  fdm_async:\n"
        "    state_history:\n"
        f"      enabled: {'true' if state_history_enabled else 'false'}\n"
    )
    args = SimpleNamespace(config=str(config_path), server=None)

    assert _fdm_state_history_hand_first_enabled(args) is expected


@pytest.mark.parametrize(
    ("protocol_mode", "server", "action_space", "expected"),
    [
        ("fdm_async", "https://policy.example", "joint", "joint"),
        ("fdm_async", "https://policy.example", "eef", "eef"),
        ("pi_v2", "https://policy.example", "joint_position", "joint"),
        ("pi_v2", "https://policy.example", "eef_pose", "eef"),
        ("pi_v2", "tcp://127.0.0.1:5555", "joint_position", "eef"),
    ],
)
def test_session_derives_arm_command_mode_from_isolated_profile(
    tmp_path, protocol_mode, server, action_space, expected
):
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(
        "deployment:\n"
        f"  protocol_mode: {protocol_mode}\n"
        f"  server: {server}\n"
        f"  expected_arm_action_space: {action_space}\n"
        "  fdm_async:\n"
        f"    action_mode: {action_space}\n"
    )
    args = SimpleNamespace(config=str(config_path), server=None)

    assert _configured_arm_command_mode(args) == expected


def test_session_derives_cosmos_joint_mode_from_profile(tmp_path):
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(
        "deployment:\n"
        "  protocol_mode: protocol_v2\n"
        "  server: https://policy.example\n"
        "  arm_command_mode: joint\n"
        "  action_space: joint\n"
    )
    args = SimpleNamespace(config=str(config_path), server=None)

    assert _configured_arm_command_mode(args) == "joint"


def test_session_waits_for_required_deployment_nodes(monkeypatch):
    required = {
        "/tianji_arm_controller",
        "/wuji_deployment",
        "/right_hand/wujihand_driver",
    }

    class Node:
        def __init__(self):
            self.spin_count = 0

        def visible_nodes(self):
            return required if self.spin_count else set()

    node = Node()

    def spin_once(_node, timeout_sec):
        assert timeout_sec <= 0.05
        node.spin_count += 1

    monkeypatch.setattr(
        deployment_session_module.rclpy,
        "spin_once",
        spin_once,
    )

    assert _wait_for_deployment_nodes(node, "right", timeout_s=1.0) == []
    assert node.spin_count == 1


def test_session_reports_nodes_missing_after_startup_timeout():
    node = SimpleNamespace(visible_nodes=lambda: {"/wuji_deployment"})

    assert _wait_for_deployment_nodes(node, "right", timeout_s=0.0) == [
        "/right_hand/wujihand_driver",
        "/tianji_arm_controller",
    ]


def test_pi_startup_handoff_discards_bootstrap_and_replans_before_clock():
    node = _prefetch_node()
    node.startup_handoff_gate_enabled = True
    node._startup_state = STARTUP_WAIT_BOOTSTRAP
    node._startup_state_entered_at = 10.0
    node._startup_anchor_action = None
    node._startup_anchor_validated = None
    node._startup_anchor_poses = None
    node._startup_anchor_hands = None
    node._startup_anchor_request_id = None
    node._startup_anchor_publish_count = 0
    node._startup_seen_controller_active = False
    node._startup_failure_reason = ""
    node._controller_handoff_state = CONTROLLER_HANDOFF_WAITING
    node._controller_handoff_progress = 0.0
    node._controller_handoff_payload = {}
    node._interpolated_commands = 0
    node._applied_sequence = 0
    node._previous_applied_pose = {"left": None, "right": None}
    node._previous_applied_hand = {"left": None, "right": None}
    node._previous_applied_at = 0.0
    node._last_applied_at = 0.0
    node._arm_publishers = {"right": _Publisher()}
    node._zsp_publishers = {"right": _Publisher()}
    node._hand_publishers = {"right": _Publisher()}
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: TimeMessage())
    )
    node._set_startup_state = lambda state, reason="": (
        DeploymentNode._set_startup_state(node, state, reason=reason))
    node._publish_startup_anchor = lambda now: (
        DeploymentNode._publish_startup_anchor(node, now))
    node._request_policy_standby = lambda _reason: pytest.fail(
        "valid startup must not request standby")
    node._clear_action_stream = lambda clear_active=True: (
        DeploymentNode._clear_action_stream(
            node, clear_active=clear_active
        ))

    bootstrap = PendingActionChunk(
        actions=tuple(_response(hand_value=5.0) for _ in range(50)),
        action_rate_hz=30.0,
        observation_created_at=9.8,
        response_received_at=9.9,
        align_to_observation=False,
        request_id=11,
        generation=node._stream_generation,
    )
    node._pending_chunk = bootstrap

    assert DeploymentNode._startup_handoff_tick(node, 10.0) is True
    assert node._startup_state == STARTUP_HANDOFF_ACTIVE
    assert node._action_plan.remaining() == 0
    assert node._applied_sequence == 0
    assert len(node._arm_publishers["right"].messages) == 1
    assert DeploymentNode._claim_policy_request(node) is None

    node._controller_handoff_state = CONTROLLER_HANDOFF_ACTIVE
    node._startup_seen_controller_active = True
    assert DeploymentNode._startup_handoff_tick(node, 10.6) is True
    assert node._action_plan.remaining() == 0

    old_generation = node._stream_generation
    node._controller_handoff_state = CONTROLLER_HANDOFF_COMPLETE
    node._controller_handoff_payload = {
        "completed_at_monotonic": 11.2
    }
    assert DeploymentNode._startup_handoff_tick(node, 11.2) is True
    assert node._startup_state == STARTUP_WAIT_FRESH
    assert node._stream_generation == old_generation + 1
    assert node._action_plan.remaining() == 0
    assert DeploymentNode._claim_policy_request(node) == (
        node._stream_generation,
        False,
    )
    DeploymentNode._release_policy_request(node)

    fresh = PendingActionChunk(
        actions=tuple(_response(hand_value=6.0) for _ in range(50)),
        action_rate_hz=30.0,
        observation_created_at=11.2,
        response_received_at=11.3,
        align_to_observation=False,
        request_id=12,
        generation=node._stream_generation,
    )
    node._pending_chunk = fresh

    def activate_fresh(now, schedule_start_at=None):
        del schedule_start_at
        pending = node._pending_chunk
        node._pending_chunk = None
        node._action_plan.install(
            list(pending.actions[:25]),
            observation_created_at=pending.observation_created_at,
            received_at=pending.response_received_at,
            rate_hz=pending.action_rate_hz,
            schedule_start_at=now,
        )
        node._active_chunk_id += 1
        return True

    node._activate_pending_chunk = activate_fresh
    assert DeploymentNode._startup_handoff_tick(node, 11.3) is False
    assert node._startup_state == STARTUP_RUNNING
    assert node._action_plan.remaining() == 25
    assert node._applied_sequence == 0


def test_pending_prefetch_activates_aligned_25_step_window():
    node = _prefetch_node()
    actions = []
    for index in range(50):
        action = _response()
        action["step"] = index
        actions.append(action)
    node._pending_chunk = PendingActionChunk(
        actions=tuple(actions),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.05,
        align_to_observation=True,
    )
    node._last_action_due_at = 10.2

    activated = DeploymentNode._activate_pending_chunk(
        node,
        10.2,
        schedule_start_at=10.0 + 7.0 / 30.0,
    )

    assert activated is True
    assert node._pending_chunk is None
    assert node._action_plan.remaining() == 25
    assert node._last_activation_skip_actions == 7
    assert node._prefetch_hits == 1
    first = node._action_plan.pop_due(10.0 + 7.0 / 30.0 + 1e-6)
    assert first.action["step"] == 7


def test_pi_joint_future_chunk_does_not_reject_transient_measured_lag():
    node = _prefetch_node()
    pi_defaults = _pi_joint_node()
    for name in (
        "protocol_mode",
        "policy_transport",
        "expected_arm_action_space",
        "_negotiated_arm_action_space",
        "policy_action_mode",
        "arm_command_mode",
        "_pi_joint_lower_rad",
        "_pi_joint_upper_rad",
        "_pi_joint_velocity_rad_s",
    ):
        setattr(node, name, getattr(pi_defaults, name))
    node._active_chunk_id = 1
    node._last_applied_pose = {
        side: np.zeros(7, dtype=np.float32) for side in ("left", "right")
    }
    node._last_applied_hand = {
        side: np.zeros(20, dtype=np.float32) for side in ("left", "right")
    }
    node._last_applied_zsp = {"left": None, "right": None}
    node._pi_joint_last_published_pose = {
        "left": None,
        "right": np.full(7, 0.02, dtype=np.float32),
    }
    # The arm can transiently trail the last published target by more than the
    # 30 Hz 6-degree model step. Future chunks must not call this bootstrap-only
    # measured-boundary check.
    node._validate_pi_joint_measured_boundary = lambda _action: pytest.fail(
        "future chunk must not use instantaneous measured boundary rejection"
    )
    actions = []
    for index in range(50):
        action = _joint_response(0.0)
        for side in ("left", "right"):
            action[f"arm_action_{side}"]["joint_pos"] = [
                0.001 * index
            ] * 7
        actions.append(action)
    node._pending_chunk = PendingActionChunk(
        actions=tuple(actions),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.05,
        align_to_observation=True,
    )

    assert DeploymentNode._activate_pending_chunk(
        node, 10.2, schedule_start_at=10.2
    )
    assert node._action_plan.remaining() == 25
    first = node._action_plan.pop_due(10.2 + 1e-6)
    assert first is not None
    assert first.action["arm_action_right"]["joint_pos"][0] > 0.015


def test_pending_activation_builds_pchip_sampler_once_for_selected_window():
    node = _prefetch_node()
    node.action_interpolation_method = "pchip_slerp"
    actions = []
    for index in range(50):
        action = _response(hand_value=float(index))
        for side in ("left", "right"):
            action[f"arm_action_{side}"]["ee_pos"][0] = 0.4 + 0.001 * index
        actions.append(action)
    node._pending_chunk = PendingActionChunk(
        actions=tuple(actions),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.05,
        align_to_observation=False,
    )

    activated = DeploymentNode._activate_pending_chunk(
        node, 10.1, schedule_start_at=10.1
    )

    assert activated is True
    assert node._chunk_interpolator.action_count == 25
    assert node._chunk_interpolator.pchip_enabled is True
    assert node._chunk_interpolator_chunk_id == node._active_chunk_id
    midpoint = node._chunk_interpolator.sample(0.5)
    assert 0.4 < midpoint["arm_action_right"]["ee_pos"][0] < 0.401


def test_pending_activation_blends_six_aligned_steps_from_last_target():
    node = _prefetch_node()
    for side in ("left", "right"):
        node._last_applied_pose[side] = np.array(
            [0.4, 0.0 if side == "left" else 0.1, 0.5, 0, 0, 0, 1],
            dtype=np.float32,
        )
        node._last_applied_hand[side] = np.zeros(20, dtype=np.float32)
        node._last_applied_zsp[side] = np.array(
            [0.0, 1.0, 0.0], dtype=np.float32
        )
    actions = []
    for index in range(50):
        action = _response(hand_value=12.0)
        action["arm_action_left"]["ee_pos"][0] = 0.46
        action["arm_action_right"]["ee_pos"][0] = 0.46
        action["step"] = index
        actions.append(action)
    node._pending_chunk = PendingActionChunk(
        actions=tuple(actions),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.05,
        align_to_observation=True,
        request_id=8,
        generation=3,
    )
    schedule_start = 10.0 + 6.0 / 30.0

    activated = DeploymentNode._activate_pending_chunk(
        node, schedule_start, schedule_start_at=schedule_start
    )
    first = node._action_plan.pop_due(schedule_start + 1e-6)

    alpha = (1.0 / 6.0) ** 2 * (3.0 - 2.0 / 6.0)
    assert activated is True
    assert node._active_chunk_blend_steps == 6
    assert first.action["step"] == 6
    assert np.isclose(
        first.action["arm_action_right"]["ee_pos"][0],
        0.4 + alpha * 0.06,
    )
    assert np.isclose(
        first.action["hand_action_right"][0], alpha * 12.0
    )


def test_pending_activation_can_select_velocity_continuous_boundary():
    node = _prefetch_node()
    node.boundary_blend_method = "velocity_continuous"
    node.action_interpolation_method = "pchip_slerp"
    publish_dt = 1.0 / node.publish_rate_hz
    for side in ("left", "right"):
        y = 0.0 if side == "left" else 0.1
        node._last_applied_pose[side] = np.array(
            [0.4, y, 0.5, 0, 0, 0, 1], dtype=np.float32
        )
        node._last_applied_hand[side] = np.zeros(20, dtype=np.float32)
        node._last_applied_zsp[side] = np.array(
            [0.0, 1.0, 0.0], dtype=np.float32
        )
    node._previous_applied_pose = {
        side: pose.copy() for side, pose in node._last_applied_pose.items()
    }
    node._previous_applied_hand = {
        side: hand.copy() for side, hand in node._last_applied_hand.items()
    }
    node._previous_applied_at = 9.9 - publish_dt
    node._last_applied_at = 9.9
    actions = []
    for index in range(50):
        action = _response(hand_value=0.2 * index)
        for side in ("left", "right"):
            action[f"arm_action_{side}"]["ee_pos"][0] = 0.42 + 0.001 * index
        action["step"] = index
        actions.append(action)
    node._pending_chunk = PendingActionChunk(
        actions=tuple(actions),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.05,
        align_to_observation=True,
        request_id=9,
        generation=3,
    )
    schedule_start = 10.0 + 6.0 / 30.0

    activated = DeploymentNode._activate_pending_chunk(
        node, schedule_start, schedule_start_at=schedule_start
    )
    first = node._action_plan.pop_due(schedule_start + 1e-6)
    bridge_end = node._boundary_interpolator.sample(
        node._boundary_interpolator.duration_s
    )

    assert activated is True
    assert node._active_chunk_blend_steps == 6
    assert node._boundary_interpolator_chunk_id == node._active_chunk_id
    assert np.isclose(
        node._boundary_interpolator_start_at,
        schedule_start - 1.0 / 30.0,
    )
    # The scheduled plan retains the raw time-aligned model actions. The
    # continuous bridge is sampled only by the 120 Hz publication path.
    assert first.action["step"] == 6
    assert np.isclose(first.action["arm_action_right"]["ee_pos"][0], 0.426)
    assert np.allclose(
        bridge_end["arm_action_right"]["ee_pos"],
        actions[11]["arm_action_right"]["ee_pos"],
    )


def test_first_chunk_bridges_from_fresh_measured_robot_state():
    node = _prefetch_node()
    node.boundary_blend_method = "velocity_continuous"
    node.initial_blend_steps = 15
    node.action_interpolation_method = "pchip_slerp"
    node.max_observation_age_s = 0.5
    node._latest_lock = threading.Lock()
    measured_poses = {
        "left": np.array(
            [0.0, 0.69, 0.18, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        "right": np.array(
            [
                0.57,
                -0.22,
                0.275,
                0.0,
                np.sqrt(0.5),
                0.0,
                np.sqrt(0.5),
            ],
            dtype=np.float32,
        ),
    }
    right_hand = JointState()
    right_hand.position = np.radians(np.linspace(0.0, 4.0, 20)).tolist()
    node._latest = {
        "arm_eef_left": (100.0, measured_poses["left"]),
        "arm_eef_right": (100.0, measured_poses["right"]),
        "hand_state_right": (100.0, right_hand),
    }
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=int(100.1e9))
    )
    node._snapshot = lambda: DeploymentNode._snapshot(node)
    node._initial_command_anchors = (
        lambda first_action: DeploymentNode._initial_command_anchors(
            node, first_action
        )
    )
    actions = []
    for index in range(50):
        action = _response(hand_value=20.0 + 0.2 * index)
        action["arm_action_left"]["ee_pos"] = [0.0, 0.69, 0.18]
        action["arm_action_right"]["ee_pos"] = [
            0.55 - 0.001 * index,
            -0.25,
            0.27,
        ]
        action["step"] = index
        actions.append(action)
    node._pending_chunk = PendingActionChunk(
        actions=tuple(actions),
        action_rate_hz=30.0,
        observation_created_at=99.95,
        response_received_at=100.0,
        align_to_observation=False,
        request_id=1,
        generation=3,
    )

    activated = DeploymentNode._activate_pending_chunk(
        node, 100.1, schedule_start_at=100.1
    )
    first_due = node._action_plan.pop_due(100.1 + 1.0 / 30.0 + 1e-6)
    bridge_start = node._boundary_interpolator.sample(0.0)
    bridge_end = node._boundary_interpolator.sample(
        node._boundary_interpolator.duration_s
    )

    assert activated is True
    assert node._active_chunk_blend_steps == 15
    assert np.isclose(node._boundary_interpolator_start_at, 100.1)
    assert first_due.action["step"] == 0
    assert np.allclose(
        bridge_start["arm_action_right"]["ee_pos"],
        measured_poses["right"][:3],
    )
    assert np.allclose(
        bridge_start["hand_action_right"],
        np.linspace(0.0, 4.0, 20),
        atol=1e-5,
    )
    assert np.allclose(
        bridge_end["arm_action_right"]["ee_pos"],
        actions[14]["arm_action_right"]["ee_pos"],
    )
    assert np.allclose(
        node._last_applied_pose["right"], measured_poses["right"]
    )


def test_first_chunk_refuses_a_stale_measured_anchor():
    node = _prefetch_node()
    node.boundary_blend_method = "velocity_continuous"
    node.initial_blend_steps = 15
    node.action_interpolation_method = "pchip_slerp"
    node.max_observation_age_s = 0.5
    node._latest_lock = threading.Lock()
    right_hand = JointState()
    right_hand.position = [0.0] * 20
    node._latest = {
        "arm_eef_left": (99.0, np.array([0, 0.69, 0.18, 0, 0, 0, 1])),
        "arm_eef_right": (99.0, np.array([0.57, -0.22, 0.27, 0, 0, 0, 1])),
        "hand_state_right": (99.0, right_hand),
    }
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=int(100.0e9))
    )
    node._snapshot = lambda: DeploymentNode._snapshot(node)
    node._initial_command_anchors = (
        lambda first_action: DeploymentNode._initial_command_anchors(
            node, first_action
        )
    )
    node._pending_chunk = PendingActionChunk(
        actions=tuple(_response() for _ in range(50)),
        action_rate_hz=30.0,
        observation_created_at=99.9,
        response_received_at=99.95,
        align_to_observation=False,
    )

    activated = DeploymentNode._activate_pending_chunk(
        node, 100.0, schedule_start_at=100.0
    )

    assert activated is False
    assert node._action_plan.remaining() == 0
    assert "EEF state is stale" in node._last_server_error


def test_stale_generation_response_never_enters_pending_slot():
    node = _prefetch_node()
    node._request_inflight = True
    pending = PendingActionChunk(
        actions=tuple(_response() for _ in range(50)),
        action_rate_hz=30.0,
        observation_created_at=1.0,
        response_received_at=1.1,
        align_to_observation=True,
    )

    stored = DeploymentNode._store_pending_chunk(
        node, pending, generation=node._stream_generation - 1
    )

    assert stored is False
    assert node._pending_chunk is None
    # A stale response must not release the current generation request slot.
    assert node._request_inflight is True
    assert node._stale_generation_responses == 1


def test_prefetch_miss_is_counted_once_until_pending_arrives():
    node = _prefetch_node()

    DeploymentNode._mark_prefetch_miss(node, 10.0)
    DeploymentNode._mark_prefetch_miss(node, 10.1)

    assert node._prefetch_misses == 1
    assert node._waiting_for_pending_since == 10.0


def test_last_active_action_arms_pending_chunk_on_next_30hz_deadline():
    node = _prefetch_node()
    node._activate_pending_chunk = (
        lambda now, schedule_start_at=None: DeploymentNode._activate_pending_chunk(
            node, now, schedule_start_at=schedule_start_at
        )
    )
    node._mark_prefetch_miss = (
        lambda now: DeploymentNode._mark_prefetch_miss(node, now)
    )
    node.active_arm_sides = ("right",)
    node.active_hand_sides = ("right",)
    node._arm_publishers = {"right": _Publisher()}
    node._zsp_publishers = {"right": _Publisher()}
    node._hand_publishers = {"right": _Publisher()}
    node._applied_sequence = 0
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: TimeMessage())
    )
    now = time.monotonic()
    node._action_plan.install(
        [_response()],
        observation_created_at=now,
        received_at=now,
        rate_hz=30.0,
    )
    pending_actions = []
    for index in range(50):
        action = _response()
        action["step"] = index
        pending_actions.append(action)
    node._pending_chunk = PendingActionChunk(
        actions=tuple(pending_actions),
        action_rate_hz=30.0,
        observation_created_at=now - 0.1,
        response_received_at=now - 0.02,
        align_to_observation=True,
    )

    DeploymentNode._apply_pending_action(node)

    assert len(node._arm_publishers["right"].messages) == 1
    assert node._pending_chunk is None
    assert node._action_plan.remaining() == 25
    next_action = node._action_plan.pop_due(now + 1.0 / 30.0 + 0.01)
    assert next_action is not None
    assert next_action.action["step"] >= 4


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_pi_joint_publish_uses_only_right_joint_target_and_degree_hand_path():
    node = _pi_joint_node()
    node.active_arm_sides = ("right",)
    node.active_hand_sides = ("right",)
    arm_publishers = {side: _Publisher() for side in ("left", "right")}
    hand_publishers = {side: _Publisher() for side in ("left", "right")}
    node._arm_publishers = arm_publishers
    node._hand_publishers = hand_publishers
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: TimeMessage())
    )
    action = _joint_response(0.0)
    right_joint = np.linspace(-0.2, 0.2, 7)
    right_hand_deg = np.linspace(0.0, 19.0, 20)
    action["arm_action_right"]["joint_pos"] = right_joint.tolist()
    action["hand_action_right"] = right_hand_deg.tolist()
    validated, _, _ = DeploymentNode._validate_response(node, action)

    DeploymentNode._publish_validated_command(node, validated)

    assert len(arm_publishers["left"].messages) == 0
    assert len(hand_publishers["left"].messages) == 0
    assert len(arm_publishers["right"].messages) == 1
    assert len(hand_publishers["right"].messages) == 1
    assert np.allclose(
        arm_publishers["right"].messages[0].position, right_joint
    )
    assert np.allclose(
        hand_publishers["right"].messages[0].position,
        np.radians(right_hand_deg),
    )


def test_pi_joint_apply_path_rate_limits_final_hardware_publish(monkeypatch):
    node = _pi_joint_node()
    node._lifecycle = 2
    node.startup_handoff_gate_enabled = False
    node._action_plan = LatestActionPlan(max_actions=30)
    target = _joint_response(0.0)
    target["arm_action_right"]["joint_pos"] = np.radians(
        [20.0] * 7
    ).tolist()
    node._action_plan.install(
        [target],
        observation_created_at=10.0,
        received_at=10.0,
        rate_hz=30.0,
        schedule_start_at=10.0,
    )
    node._applied_sequence = 0
    node.action_interpolation_method = "none"
    node.prefetch_enabled = False
    node._activate_pending_chunk = lambda *_args, **_kwargs: False
    node._mark_prefetch_miss = lambda *_args, **_kwargs: False
    node._previous_applied_pose = {"left": None, "right": None}
    node._previous_applied_hand = {"left": None, "right": None}
    node._previous_applied_at = 0.0
    node._last_applied_at = 0.0
    node._interpolation_current = None
    node._interpolated_commands = 0
    node._arm_publishers = {"right": _Publisher()}
    node._hand_publishers = {"right": _Publisher()}
    measured_message = JointState()
    measured_message.position = [0.0] * 7
    node._snapshot = lambda: {
        "arm_state_right": (10.0, measured_message)
    }
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(
            nanoseconds=int(10.0 * 1e9),
            to_msg=lambda: TimeMessage(),
        )
    )
    node.get_logger = lambda: SimpleNamespace(
        warn=lambda _message: None,
        error=lambda _message: None,
    )
    node._request_policy_standby = lambda reason: pytest.fail(reason)
    monkeypatch.setattr(deployment_module.time, "monotonic", lambda: 10.0)

    DeploymentNode._apply_pending_action(node)

    assert len(node._arm_publishers["right"].messages) == 1
    assert np.allclose(
        node._arm_publishers["right"].messages[0].position,
        np.radians([1.5] * 7),
        atol=1e-7,
    )
    assert np.allclose(
        node._last_applied_pose["right"],
        np.radians([20.0] * 7),
    )
    assert np.allclose(
        node._pi_joint_last_published_pose["right"],
        np.radians([1.5] * 7),
        atol=1e-7,
    )


def test_single_right_arm_and_hand_publish_skip_inactive_left_side():
    arm_publishers = {side: _Publisher() for side in ("left", "right")}
    zsp_publishers = {side: _Publisher() for side in ("left", "right")}
    hand_publishers = {"right": _Publisher()}
    node = _node()
    node.active_arm_sides = ("right",)
    node.active_hand_sides = ("right",)
    node._lifecycle = 2
    node._action_plan = LatestActionPlan(
        max_actions=8, max_schedule_lag_s=1.0
    )
    now = time.monotonic()
    node._action_plan.install(
        [_response(hand_value=15.0)],
        observation_created_at=now,
        received_at=now,
        rate_hz=30.0,
    )
    node._applied_sequence = 0
    node.action_rate_hz = 30.0
    node.prefetch_enabled = False
    node._activate_pending_chunk = lambda *_args, **_kwargs: False
    node._mark_prefetch_miss = lambda *_args, **_kwargs: None
    node._validate_response = lambda response: DeploymentNode._validate_response(
        node, response
    )
    node._arm_publishers = arm_publishers
    node._zsp_publishers = zsp_publishers
    node._hand_publishers = hand_publishers
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: TimeMessage())
    )
    node.get_logger = lambda: SimpleNamespace(
        warn=lambda _message: None,
        error=lambda _message: None,
    )

    DeploymentNode._apply_pending_action(node)

    assert len(arm_publishers["left"].messages) == 0
    assert len(arm_publishers["right"].messages) == 1
    assert len(hand_publishers["right"].messages) == 1
    assert node._applied_sequence == 1

    # The legacy Pi path has no FDM hold policy and keeps its existing
    # exhausted-plan behavior.
    DeploymentNode._apply_pending_action(node)
    assert len(arm_publishers["right"].messages) == 1
    assert len(hand_publishers["right"].messages) == 1
    assert node._applied_sequence == 1


def test_fdm_pending_chunk_republishes_last_target_without_execution_advance(
    monkeypatch,
):
    arm_publishers = {"right": _Publisher()}
    zsp_publishers = {"right": _Publisher()}
    hand_publishers = {"right": _Publisher()}
    node = _node()
    node.active_arm_sides = ("right",)
    node.active_hand_sides = ("right",)
    node._lifecycle = 2
    node._action_plan = LatestActionPlan(max_actions=48)
    node._action_plan.install(
        [_response(hand_value=8.0)],
        observation_created_at=100.0,
        received_at=100.0,
        rate_hz=30.0,
        schedule_start_at=100.0,
    )
    node._applied_sequence = 0
    node._interpolation_current = None
    node._interpolation_current_start_at = 0.0
    node._interpolated_commands = 0
    node._fdm_hold_last_publishes = 0
    node._active_chunk_action_index = 47
    node.action_interpolation_method = "linear_slerp"
    node.action_rate_hz = 30.0
    node.prefetch_enabled = True
    node.fdm_config = SimpleNamespace(pending_miss_policy="hold_last")
    node._activate_pending_chunk = lambda *_args, **_kwargs: False
    miss_ticks = []
    node._mark_prefetch_miss = lambda now: miss_ticks.append(now) or False
    node._arm_publishers = arm_publishers
    node._zsp_publishers = zsp_publishers
    node._hand_publishers = hand_publishers
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: TimeMessage())
    )
    node.get_logger = lambda: SimpleNamespace(
        warn=lambda _message: None,
        error=lambda _message: None,
    )
    executed = []

    def record_execution(_node, action, *, chunk_action_index):
        executed.append((action, chunk_action_index))

    monkeypatch.setattr(
        DeploymentNode,
        "_fdm_record_executed_action",
        record_execution,
    )
    current_time = [100.0]
    monkeypatch.setattr(
        deployment_module.time, "monotonic", lambda: current_time[0]
    )

    # Dispatch the genuine final 30 Hz waypoint in Wn.
    DeploymentNode._apply_pending_action(node)
    assert len(executed) == 1
    assert executed[0][1] == 47
    assert node._active_chunk_action_index == 48

    # Four 120 Hz control ticks while Wn+1 is pending repeat the exact target.
    # They are watchdog keepalives, not model actions or feedback entries.
    for tick in range(1, 5):
        current_time[0] = 100.0 + tick / 120.0
        DeploymentNode._apply_pending_action(node)

    assert len(arm_publishers["right"].messages) == 5
    assert len(hand_publishers["right"].messages) == 5
    assert all(
        np.isclose(message.pose.position.x, 0.4)
        for message in arm_publishers["right"].messages
    )
    assert all(
        np.isclose(np.degrees(message.position[0]), 8.0)
        for message in hand_publishers["right"].messages
    )
    assert node._fdm_hold_last_publishes == 4
    assert node._active_chunk_action_index == 48
    assert node._applied_sequence == 1
    assert len(executed) == 1
    assert len(miss_ticks) == 5

    # Once the next real wire action arrives, its index zero executes exactly
    # once and normal execution/feedback accounting resumes.
    next_action = _response(hand_value=9.0)
    node._action_plan.install(
        [next_action],
        observation_created_at=current_time[0],
        received_at=current_time[0],
        rate_hz=30.0,
        schedule_start_at=current_time[0],
    )
    node._active_chunk_action_index = 0
    DeploymentNode._apply_pending_action(node)

    assert len(executed) == 2
    assert executed[1][0] is next_action
    assert executed[1][1] == 0
    assert node._active_chunk_action_index == 1
    assert node._applied_sequence == 2


def test_deployment_resamples_30hz_waypoints_at_120hz(monkeypatch):
    arm_publishers = {"right": _Publisher()}
    zsp_publishers = {"right": _Publisher()}
    hand_publishers = {"right": _Publisher()}
    node = _node()
    node.active_arm_sides = ("right",)
    node.active_hand_sides = ("right",)
    node._lifecycle = 2
    node._action_plan = LatestActionPlan(max_actions=8)
    start = _response(hand_value=0.0)
    end = _response(hand_value=12.0)
    for side in ("left", "right"):
        start[f"arm_action_{side}"]["ee_pos"][0] = 0.4
        end[f"arm_action_{side}"]["ee_pos"][0] = 0.44
    node._action_plan.install(
        [start, end],
        observation_created_at=100.0,
        received_at=100.0,
        rate_hz=30.0,
    )
    node._applied_sequence = 0
    node._interpolation_current = None
    node._interpolation_current_start_at = 0.0
    node._interpolated_commands = 0
    node.action_interpolation_method = "linear_slerp"
    node.action_rate_hz = 30.0
    node.prefetch_enabled = False
    node._activate_pending_chunk = lambda *_args, **_kwargs: False
    node._mark_prefetch_miss = lambda *_args, **_kwargs: None
    node._validate_response = lambda response: DeploymentNode._validate_response(
        node, response
    )
    node._arm_publishers = arm_publishers
    node._zsp_publishers = zsp_publishers
    node._hand_publishers = hand_publishers
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: TimeMessage())
    )
    node.get_logger = lambda: SimpleNamespace(
        warn=lambda _message: None,
        error=lambda _message: None,
    )
    current_time = [100.0]
    monkeypatch.setattr(
        deployment_module.time, "monotonic", lambda: current_time[0]
    )

    for tick in range(5):
        current_time[0] = 100.0 + tick / 120.0
        DeploymentNode._apply_pending_action(node)

    positions = [
        message.pose.position.x for message in arm_publishers["right"].messages
    ]
    hands_deg = [
        np.degrees(message.position[0])
        for message in hand_publishers["right"].messages
    ]
    assert np.allclose(positions, [0.4, 0.41, 0.42, 0.43, 0.44])
    assert np.allclose(hands_deg, [0.0, 3.0, 6.0, 9.0, 12.0])
    assert node._applied_sequence == 2
    assert node._interpolated_commands == 5


def test_leaving_command_lifecycle_resets_previous_session_guards():
    node = SimpleNamespace(
        _lifecycle=2,
        _action_plan=LatestActionPlan(
            max_actions=8, max_schedule_lag_s=1.0
        ),
        _last_applied_pose={"left": np.ones(7), "right": np.ones(7)},
        _last_applied_hand={"left": np.ones(20), "right": np.ones(20)},
    )
    node._clear_action_stream = lambda clear_active=True: (
        node._action_plan.clear() if clear_active else None
    )

    DeploymentNode._lifecycle_callback(node, SimpleNamespace(data=3))

    assert node._lifecycle == 3
    assert node._last_applied_pose == {"left": None, "right": None}
    assert node._last_applied_hand == {"left": None, "right": None}
    assert node._last_applied_zsp == {"left": None, "right": None}


def test_ready_service_refuses_enable_until_policy_handshake():
    node = SimpleNamespace(
        _session_lock=threading.Lock(),
        _server_ready=False,
        _last_server_error="timed out",
        server="tcp://policy:5555",
        _last_model_id="",
    )
    response = SimpleNamespace(success=None, message="")

    DeploymentNode._ready_callback(node, None, response)
    assert response.success is False
    assert response.message == "timed out"

    node._server_ready = True
    node._last_model_id = "checkpoint-42"
    DeploymentNode._ready_callback(node, None, response)
    assert response.success is True
    assert "checkpoint-42" in response.message


class _HelloTransport:
    last_http_status = 200
    reconnects = 0

    def __init__(self, response_updates=None):
        self.response_updates = dict(response_updates or {})
        self.requests = []

    def exchange(self, request):
        self.requests.append(request)
        response = {
            "protocol_version": 2,
            "message_type": "hello_ack",
            "session_id": request["session_id"],
            "request_id": request["request_id"],
            "model_id": "checkpoint-99999",
            "action_rate_hz": 30.0,
        }
        response.update(self.response_updates)
        return PolicyExchange(response, 100, 120)


def _hello_node():
    node = SimpleNamespace(
        _session_lock=threading.Lock(),
        _request_sequence=0,
        _session_id="session-abc",
        _server_identity_mismatches=0,
        camera_names=["head", "right_wrist"],
        expected_model_id="checkpoint-99999",
        policy_transport="http",
        action_rate_hz=30.0,
        _last_request_bytes=0,
        _last_response_bytes=0,
        _last_http_status=None,
        _transport_reconnects=0,
        _last_model_id="",
        expected_arm_action_space="",
        _negotiated_arm_action_space="",
        _server_ready=False,
        _last_server_error="not ready",
        _action_plan=LatestActionPlan(
            max_actions=8, max_schedule_lag_s=1.0
        ),
        get_logger=lambda: SimpleNamespace(debug=lambda _message: None),
    )
    node._next_request_identity = lambda: DeploymentNode._next_request_identity(
        node
    )
    node._exchange = lambda transport, request: DeploymentNode._exchange(
        node, transport, request
    )
    node._clear_action_stream = lambda clear_active=True: (
        node._action_plan.clear() if clear_active else None
    )
    return node


def test_hello_validates_identity_model_and_rate():
    node = _hello_node()
    transport = _HelloTransport()

    DeploymentNode._probe_server(node, transport)

    assert node._server_ready is True
    assert node._last_model_id == "checkpoint-99999"
    assert transport.requests[0]["camera_names"] == ["head", "right_wrist"]

    for updates, message in (
        ({"session_id": "wrong"}, "SERVER_IDENTITY_MISMATCH"),
        ({"request_id": 999}, "SERVER_IDENTITY_MISMATCH"),
        ({"model_id": "wrong"}, "model_id mismatch"),
        ({"action_rate_hz": 20.0}, "action_rate_hz mismatch"),
    ):
        node = _hello_node()
        with pytest.raises(ValueError, match=message):
            DeploymentNode._probe_server(node, _HelloTransport(updates))


def test_pi_joint_hello_locks_explicit_action_space():
    node = _hello_node()
    node.protocol_mode = "pi_v2"
    node.expected_arm_action_space = "joint_position"
    transport = _HelloTransport(
        {"arm_action_space": "joint_position"}
    )

    DeploymentNode._probe_server(node, transport)

    assert node._server_ready is True
    assert node._negotiated_arm_action_space == "joint_position"
    assert (
        transport.requests[0]["expected_arm_action_space"]
        == "joint_position"
    )


def test_cosmos_joint_hello_advertises_top_level_joint_profile():
    node = _hello_node()
    cosmos = _cosmos_joint_node()
    node.protocol_mode = cosmos.protocol_mode
    node.policy_action_mode = cosmos.policy_action_mode
    node.arm_command_mode = cosmos.arm_command_mode
    transport = _HelloTransport()

    DeploymentNode._probe_server(node, transport)

    request = transport.requests[0]
    assert request["arm_command_mode"] == "joint"
    assert request["action_space"] == "joint"
    assert request["robot_layout"]["action_layout"] == [
        {"side": "left", "arm_joint": [0, 7], "hand": [7, 27]},
        {"side": "right", "arm_joint": [27, 34], "hand": [34, 54]},
    ]
    assert request["robot_layout"]["units"] == {
        "qpos.arm": "radian",
        "qpos.hand": "radian",
        "action.arm_joint": "radian",
        "action.hand": "degree",
    }


def test_cosmos_joint_wire_canonicalization_does_not_require_eef():
    node = _cosmos_joint_node()
    wire_action = _cosmos_joint_wire_action(0.01)

    action = DeploymentNode._canonicalize_action(node, wire_action)
    validated, joints, hands = DeploymentNode._validate_response(
        node,
        action,
        previous_poses={side: None for side in ("left", "right")},
        previous_hands={side: None for side in ("left", "right")},
    )

    assert np.allclose(joints["right"], [0.01] * 7)
    assert np.allclose(hands["right"], [0.01] * 20)
    assert validated["right"][2] is None

    wire_action.pop("arm_joint_action_right")
    with pytest.raises(ValueError, match="arm_joint_action_right"):
        DeploymentNode._canonicalize_action(node, wire_action)


def test_cosmos_joint_chunk_checks_observation_hold_step_and_acceleration():
    node = _cosmos_joint_node()
    joint_anchors = {
        side: np.zeros(7, dtype=np.float32)
        for side in ("left", "right")
    }
    hand_anchors = {
        side: np.zeros(20, dtype=np.float32)
        for side in ("left", "right")
    }

    def canonical_chunk():
        return [
            DeploymentNode._canonicalize_action(
                node, _cosmos_joint_wire_action(0.0)
            )
            for _ in range(32)
        ]

    actions = canonical_chunk()
    DeploymentNode._validate_action_chunk(
        node,
        actions,
        15.0,
        observation_joint_positions=joint_anchors,
        observation_hand_positions=hand_anchors,
    )

    inactive_moves = canonical_chunk()
    inactive_moves[4]["arm_action_left"]["joint_pos"][0] = 0.001
    with pytest.raises(ValueError, match="inactive left arm"):
        DeploymentNode._validate_action_chunk(
            node,
            inactive_moves,
            15.0,
            observation_joint_positions=joint_anchors,
            observation_hand_positions=hand_anchors,
        )

    large_step = canonical_chunk()
    large_step[1]["arm_action_right"]["joint_pos"][0] = np.radians(5.1)
    with pytest.raises(ValueError, match="step/velocity"):
        DeploymentNode._validate_action_chunk(
            node,
            large_step,
            15.0,
            observation_joint_positions=joint_anchors,
            observation_hand_positions=hand_anchors,
        )

    high_acceleration = canonical_chunk()
    high_acceleration[1]["arm_action_right"]["joint_pos"][0] = np.radians(1.0)
    high_acceleration[2]["arm_action_right"]["joint_pos"][0] = np.radians(-1.0)
    with pytest.raises(ValueError, match="acceleration"):
        DeploymentNode._validate_action_chunk(
            node,
            high_acceleration,
            15.0,
            observation_joint_positions=joint_anchors,
            observation_hand_positions=hand_anchors,
        )


def test_cosmos_joint_chunk_can_disable_dynamic_prevalidation():
    node = _cosmos_joint_node()
    node.joint_step_velocity_validation_enabled = False
    node._joint_acceleration_rad_s2 = None
    joint_anchors = {
        side: np.zeros(7, dtype=np.float32)
        for side in ("left", "right")
    }
    hand_anchors = {
        side: np.zeros(20, dtype=np.float32)
        for side in ("left", "right")
    }
    actions = [
        DeploymentNode._canonicalize_action(
            node, _cosmos_joint_wire_action(0.0)
        )
        for _ in range(32)
    ]
    actions[1]["arm_action_right"]["joint_pos"][0] = np.radians(100.0)

    DeploymentNode._validate_action_chunk(
        node,
        actions,
        15.0,
        observation_joint_positions=joint_anchors,
        observation_hand_positions=hand_anchors,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"arm_action_space": "eef_pose"},
        {"arm_action_space": 7},
    ],
)
def test_pi_joint_hello_rejects_missing_wrong_or_nonstring_action_space(
    updates,
):
    node = _hello_node()
    node.protocol_mode = "pi_v2"
    node.expected_arm_action_space = "joint_position"

    with pytest.raises(ValueError, match="arm_action_space"):
        DeploymentNode._probe_server(node, _HelloTransport(updates))

    assert node._server_ready is False
    assert node._negotiated_arm_action_space == ""


def test_response_identity_mismatch_is_distinct_server_protocol_error():
    node = _hello_node()
    session_id, request_id = DeploymentNode._next_request_identity(node)
    transport = _HelloTransport({"session_id": "server-stale-session"})
    request = {
        "session_id": session_id,
        "request_id": request_id,
    }

    with pytest.raises(ServerIdentityMismatch) as captured:
        DeploymentNode._exchange(node, transport, request)

    message = str(captured.value)
    assert "SERVER_IDENTITY_MISMATCH" in message
    assert "request_session_id='session-abc'" in message
    assert "raw_response_keys=" in message
    assert "response_session_id='server-stale-session'" in message
    assert node._server_identity_mismatches == 1


def test_server_error_is_not_masked_by_empty_error_envelope_identity():
    node = _hello_node()
    session_id, request_id = DeploymentNode._next_request_identity(node)
    transport = _HelloTransport(
        {
            "error": "invalid observation images",
            "session_id": None,
            "request_id": None,
        }
    )

    with pytest.raises(RuntimeError, match="invalid observation images"):
        DeploymentNode._exchange(
            node,
            transport,
            {"session_id": session_id, "request_id": request_id},
        )

    assert node._server_identity_mismatches == 0


def test_boolean_server_error_surfaces_cloud_code_and_message():
    node = _hello_node()
    session_id, request_id = DeploymentNode._next_request_identity(node)
    transport = _HelloTransport(
        {
            "error": True,
            "error_code": "BOOTSTRAP_SCHEMA_ERROR",
            "message": "images.right_wrist has invalid shape",
            "fatal_session": True,
            "session_id": "",
            "request_id": -1,
        }
    )

    with pytest.raises(PolicyServerError) as captured:
        DeploymentNode._exchange(
            node,
            transport,
            {"session_id": session_id, "request_id": request_id},
        )

    message = str(captured.value)
    assert "BOOTSTRAP_SCHEMA_ERROR" in message
    assert "images.right_wrist has invalid shape" in message
    assert "fatal_session=True" in message
    assert captured.value.fatal_session is True
    assert node._server_identity_mismatches == 0


@pytest.mark.parametrize(
    "failure",
    [
        PolicyAuthenticationError("status 401"),
        ServerIdentityMismatch("wrong session"),
        ValueError("invalid protocol response"),
        PolicyServerError("fatal cloud session", fatal_session=True),
    ],
)
def test_fdm_fatal_exchange_failures_reset_and_request_standby(failure):
    node = SimpleNamespace()
    network_failures = []
    resets = []
    node._fdm_network_failure = lambda transport, message: (
        network_failures.append((transport, message))
    )
    node._fdm_reset_session = lambda reason, request_standby: resets.append(
        (reason, request_standby)
    )
    transport = object()

    fatal = DeploymentNode._fdm_handle_exchange_failure(
        node, transport, failure
    )

    assert fatal is True
    assert len(network_failures) == 1
    assert len(resets) == 1
    assert resets[0][1] is True


def test_fdm_transient_exchange_failure_waits_for_pending_safety_ceiling():
    node = SimpleNamespace()
    network_failures = []
    resets = []
    node._fdm_network_failure = lambda transport, message: (
        network_failures.append((transport, message))
    )
    node._fdm_reset_session = lambda reason, request_standby: resets.append(
        (reason, request_standby)
    )
    transport = object()

    fatal = DeploymentNode._fdm_handle_exchange_failure(
        node, transport, TimeoutError("cloud request still pending")
    )

    assert fatal is False
    assert len(network_failures) == 1
    assert resets == []


def test_transport_reconnect_preserves_protocol_session_and_clears_actions():
    node = _hello_node()
    old_session = node._session_id
    DeploymentNode._next_request_identity(node)
    now = time.monotonic()
    node._action_plan.install(
        [_response()],
        observation_created_at=now,
        received_at=now,
        rate_hz=30.0,
    )
    node._server_ready = True
    node._last_model_id = "checkpoint-99999"
    node._negotiated_arm_action_space = "eef_pose"
    transport = SimpleNamespace(
        reconnects=0,
        reconnect=lambda: setattr(transport, "reconnects", transport.reconnects + 1),
    )

    DeploymentNode._reconnect_transport(node, transport)

    assert node._session_id == old_session
    assert node._request_sequence == 1
    assert node._action_plan.remaining() == 0
    assert node._server_ready is False
    assert node._last_model_id == ""
    assert node._negotiated_arm_action_space == ""
    assert node._transport_reconnects == 1


def test_pi_joint_reconnect_always_clears_old_active_buffer_and_space():
    node = _hello_node()
    node.protocol_mode = "pi_v2"
    node.policy_transport = "http"
    node.policy_action_mode = "joint"
    node.expected_arm_action_space = "joint_position"
    node._negotiated_arm_action_space = "joint_position"
    now = time.monotonic()
    node._action_plan.install(
        [_joint_response(0.0)],
        observation_created_at=now,
        received_at=now,
        rate_hz=30.0,
    )
    transport = SimpleNamespace(
        reconnects=0,
        reconnect=lambda: setattr(
            transport, "reconnects", transport.reconnects + 1
        ),
    )

    DeploymentNode._reconnect_transport(
        node, transport, clear_active=False
    )

    assert node._action_plan.remaining() == 0
    assert node._negotiated_arm_action_space == ""


def test_action_envelope_rejects_stale_request_and_model():
    node = SimpleNamespace(
        _last_model_id="checkpoint-99999",
        expected_arm_action_space="",
        _negotiated_arm_action_space="",
    )
    response = {
        "message_type": "action_chunk",
        "request_id": 10,
        "model_id": "checkpoint-99999",
    }
    DeploymentNode._validate_action_envelope(node, response, 10)

    with pytest.raises(ValueError, match="request_id"):
        DeploymentNode._validate_action_envelope(node, response, 11)
    response["request_id"] = 11
    response["model_id"] = "old-checkpoint"
    with pytest.raises(ValueError, match="model_id mismatch"):
        DeploymentNode._validate_action_envelope(node, response, 11)


def test_pi_joint_action_envelope_cannot_change_negotiated_space():
    node = SimpleNamespace(
        _last_model_id="checkpoint-99999",
        expected_arm_action_space="joint_position",
        _negotiated_arm_action_space="joint_position",
    )
    response = {
        "message_type": "action_chunk",
        "request_id": 10,
        "model_id": "checkpoint-99999",
        "arm_action_space": "joint_position",
    }
    DeploymentNode._validate_action_envelope(node, response, 10)

    response["arm_action_space"] = "eef_pose"
    with pytest.raises(ValueError, match="changed within session"):
        DeploymentNode._validate_action_envelope(node, response, 10)

    del response["arm_action_space"]
    with pytest.raises(ValueError, match="arm_action_space"):
        DeploymentNode._validate_action_envelope(node, response, 10)


def test_missing_first_policy_action_requests_tianji_standby():
    requests = []
    client = SimpleNamespace(
        service_is_ready=lambda: True,
        call_async=lambda request: requests.append(request),
    )
    node = SimpleNamespace(
        _lifecycle=2,
        _valid_policy_stream_started=False,
        _ready_entered_at=time.monotonic() - 3.0,
        _standby_requested_for_policy=False,
        policy_start_timeout_s=2.0,
        _last_server_error="",
        _tianji_enable_client=client,
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    DeploymentNode._enforce_policy_start_lease(node)

    assert node._standby_requested_for_policy is True
    assert len(requests) == 1
    assert requests[0].data is False


def test_fdm_waits_for_fresh_state_after_ready_without_consuming_w0():
    resets = []
    node = SimpleNamespace(
        _fdm_waiting_for_fresh_state_after_ready=True,
        _fdm_fresh_state_wait_reason="",
        _ready_entered_at=10.0,
        policy_start_timeout_s=2.0,
        _trace_writer=None,
        _fdm_current_qpos_sample=lambda: (_ for _ in ()).throw(
            ValueError("state-history measured left arm state is stale")
        ),
        _fdm_reset_session=lambda reason, request_standby: resets.append(
            (reason, request_standby)
        ),
    )

    assert DeploymentNode._fdm_wait_for_fresh_state_after_ready(node, 10.1)
    assert node._fdm_waiting_for_fresh_state_after_ready is True
    assert resets == []

    node._fdm_current_qpos_sample = lambda: (np.zeros(54), 10.2)
    assert not DeploymentNode._fdm_wait_for_fresh_state_after_ready(node, 10.2)
    assert node._fdm_waiting_for_fresh_state_after_ready is False


def test_fdm_fresh_state_gate_times_out_to_standby():
    resets = []
    node = SimpleNamespace(
        _fdm_waiting_for_fresh_state_after_ready=True,
        _fdm_fresh_state_wait_reason="",
        _ready_entered_at=10.0,
        policy_start_timeout_s=2.0,
        _trace_writer=None,
        _fdm_current_qpos_sample=lambda: (_ for _ in ()).throw(
            ValueError("state-history measured left arm state is stale")
        ),
        _fdm_reset_session=lambda reason, request_standby: resets.append(
            (reason, request_standby)
        ),
    )

    assert DeploymentNode._fdm_wait_for_fresh_state_after_ready(node, 12.1)
    assert node._fdm_waiting_for_fresh_state_after_ready is False
    assert resets == [(
        "FDM READY did not receive fresh measured state: "
        "state-history measured left arm state is stale",
        True,
    )]


def test_deployment_launch_routes_active_hand_to_drivers_and_client():
    source = (Path(__file__).parents[1] / "launch" / "deployment.launch.py").read_text()

    assert 'DeclareLaunchArgument(\n            "active_hand"' in source
    assert 'DeclareLaunchArgument(\n            "active_arm"' not in source
    assert "condition=IfCondition(enable_left_hand)" in source
    assert "condition=IfCondition(enable_right_hand)" in source
    assert '"--active-hand"' in source
    assert '"--active-arm"' in source
    assert '"active_arm": active_hand' in source
    assert '"control_source": "external"' in source
    assert '"arm_hardware_mode": arm_hardware_mode' in source
    assert '"control_rate": 120.0' in source
    assert '"state_publish_rate": 500.0' in source
    assert "max_initial_ik_offset_deg" not in source
    assert "max_ik_frame_jump_deg" not in source
    assert "max_joint_command_step_deg" not in source
    assert "teleop_max_target_offset_m" not in source
    assert '"teleop_position_scale": 1.25' in source
    assert '"impedance_velocity_ratio": 30' in source
    assert '"impedance_acceleration_ratio": 30' in source
    assert '"auto_enable": False' in source
    assert '"command_ramp_duration": 1.0' in source
    assert '"handoff_hold_sec": 0.2' in source
    assert '"handoff_ramp_sec": 1.0' in source
    assert '"recovery_max_speed_deg_s": 5.0' in source
    assert '"recovery_max_accel_deg_s2": 10.0' in source
    assert '"impedance_max_drift_deg": 3.0' in source


def test_sequential_activation_blends_without_skipping():
    node = _prefetch_node()
    node.prefetch_enabled = False
    node.sequential_boundary_blend_enabled = True
    node.boundary_hand_blend_steps = 2
    for side in ("left", "right"):
        node._last_applied_pose[side] = np.array(
            [0.4, 0.0 if side == "left" else 0.1, 0.5, 0, 0, 0, 1],
            dtype=np.float32,
        )
        node._last_applied_hand[side] = np.zeros(20, dtype=np.float32)
        node._last_applied_zsp[side] = np.array(
            [0.0, 1.0, 0.0], dtype=np.float32
        )
    actions = []
    for index in range(50):
        action = _response(hand_value=12.0)
        action["arm_action_left"]["ee_pos"][0] = 0.46
        action["arm_action_right"]["ee_pos"][0] = 0.46
        action["step"] = index
        actions.append(action)
    node._pending_chunk = PendingActionChunk(
        actions=tuple(actions),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.05,
        align_to_observation=False,
        request_id=8,
        generation=3,
    )
    schedule_start = 10.0 + 6.0 / 30.0

    activated = DeploymentNode._activate_pending_chunk(
        node, schedule_start, schedule_start_at=schedule_start
    )
    first = node._action_plan.pop_due(schedule_start + 1e-6)

    alpha = (1.0 / 6.0) ** 2 * (3.0 - 2.0 / 6.0)
    assert activated is True
    assert node._active_chunk_blend_steps == 6
    assert first.action["step"] == 0
    assert np.isclose(
        first.action["arm_action_right"]["ee_pos"][0],
        0.4 + alpha * 0.06,
    )
    assert np.isclose(
        first.action["hand_action_right"][0], 0.5 * 12.0
    )



def test_sequential_request_waits_after_last_waypoint(monkeypatch):
    node = _prefetch_node()
    node.prefetch_enabled = False
    node.sequential_request_settle_s = 0.1
    node._last_action_due_at = 10.0
    monkeypatch.setattr(deployment_module.time, "monotonic", lambda: 10.05)
    assert DeploymentNode._claim_policy_request(node) is None
    assert not node._request_inflight
    monkeypatch.setattr(deployment_module.time, "monotonic", lambda: 10.11)
    assert DeploymentNode._claim_policy_request(node) == (3, False)
