"""Offline-only model-clock and node scheduling tests; no ROS node constructed."""

from types import SimpleNamespace, MethodType
import threading

import numpy as np
import pytest

from wuji_data_pipeline.paper_async_blend import build_overlap_plan, sample_prediction
from wuji_data_pipeline.deployment_protocol import (
    LatestActionPlan,
    PendingActionChunk,
    ScheduledAction,
)
from wuji_data_pipeline.deployment_node import DeploymentNode


def action(x):
    return {
        **{
            f"arm_action_{side}": {"ee_pos": [x, 0, 0], "ee_quat": [0, 0, 0, 1]}
            for side in ("left", "right")
        },
        **{f"hand_action_{side}": [x] * 20 for side in ("left", "right")},
    }


def old_queue(values, start=1.0, rate=10.0):
    return [
        ScheduledAction(
            sequence=i + 1,
            due_at=start + i / rate,
            response_received_at=0.0,
            action=action(x),
        )
        for i, x in enumerate(values)
    ]


def test_first_future_row_and_no_extrapolation():
    actions = [action(1), action(2)]
    assert sample_prediction(actions, origin=0, at=0.1, rate_hz=10) == actions[0]
    assert sample_prediction(actions, origin=0, at=0.15, rate_hz=10)[
        "hand_action_right"
    ][0] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        sample_prediction(actions, origin=0, at=0, rate_hz=10)


def test_moving_overlap_uses_old_tail_and_identical_arm_hand_weight():
    old = old_queue([0, 1, 2, 3])
    plan = build_overlap_plan(
        old, [action(10)] * 10, new_origin=0.9, rate_hz=10, activation_at=1.0
    )
    assert plan.actions[0] is old[0].action
    assert plan.new_weights[:4] == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert plan.actions[1]["arm_action_right"]["ee_pos"][0] == pytest.approx(4.0)
    assert plan.actions[1]["hand_action_right"][0] == pytest.approx(4.0)
    assert plan.actions[3]["hand_action_right"][0] == 10
    assert len(plan.actions) == 10  # complete remaining new support, not old horizon


def test_absolute_time_fractional_observation_alignment():
    plan = build_overlap_plan(
        old_queue([0, 0, 0]),
        [action(i) for i in range(32)],
        new_origin=0.93,
        rate_hz=10,
        activation_at=1.0,
    )
    assert plan.new_sample_indices[1] == pytest.approx(0.7)
    assert plan.actions[2]["hand_action_right"][0] == pytest.approx(1.7)


def test_no_tail_does_not_invent_old_motion():
    plan = build_overlap_plan(
        [], [action(i) for i in range(32)], new_origin=0, rate_hz=10, activation_at=0.85
    )
    assert plan.actions[0]["hand_action_right"][0] == pytest.approx(7.5)
    assert set(plan.new_weights) == {1.0}
    with pytest.raises(ValueError):
        build_overlap_plan(
            [], [action(1)] * 3, new_origin=0, rate_hz=10, activation_at=1.0
        )


def test_preserves_quaternion_normalization_and_shortest_path():
    new = [action(1)] * 5
    for a in new:
        a["arm_action_right"]["ee_quat"] = [0, 0, 0, -1]
    plan = build_overlap_plan(
        old_queue([0, 0, 0]), new, new_origin=0.9, rate_hz=10, activation_at=1.0
    )
    assert np.linalg.norm(
        plan.actions[1]["arm_action_right"]["ee_quat"]
    ) == pytest.approx(1)


def fake_node():
    n = SimpleNamespace(
        _prefetch_lock=threading.Lock(),
        _action_plan=LatestActionPlan(max_actions=32),
        _pending_chunk=None,
        _request_inflight=False,
        _stream_generation=1,
        _paper_bootstrapped=False,
        _paper_next_request_at=0.0,
        _paper_request_origins={},
        paper_async_stride_steps=16,
        action_rate_hz=15.0,
        _active_chunk_id=0,
        _prefetch_activation_failures=0,
        _total_activation_skip_actions=0,
        _network_wakeup=threading.Event(),
        trace_policy_chunk_enabled=False,
        get_logger=lambda: SimpleNamespace(warn=lambda s: None),
        _validate_action_sequence=lambda a: None,
    )
    n._claim_paper_policy_request = MethodType(
        DeploymentNode._claim_paper_policy_request, n
    )
    return n


def put_pending(n, origin=9.0, request=1):
    n._pending_chunk = PendingActionChunk(
        actions=tuple(action(i * 0.001) for i in range(32)),
        action_rate_hz=15,
        observation_created_at=origin,
        response_received_at=origin + 0.4,
        align_to_observation=True,
        request_id=request,
        generation=1,
    )
    n._paper_request_origins[request] = (origin, {})


def test_bootstrap_keeps32_without_delay_skip_and_stride_execution_origin():
    n = fake_node()
    put_pending(n)
    assert DeploymentNode._activate_paper_pending(n, 10.0)
    q = n._action_plan.snapshot()
    assert len(q) == 32 and q[0].due_at == 10.0
    assert q[0].action == action(0)
    assert n._paper_next_request_at == pytest.approx(11.0)  # (10-dt)+16dt


def test_claim_single_inflight_no_catchup_burst(monkeypatch):
    n = fake_node()
    n._paper_bootstrapped = True
    n._paper_next_request_at = 10.0
    monkeypatch.setattr(
        "wuji_data_pipeline.deployment_node.time.monotonic", lambda: 9.0
    )
    assert n._claim_paper_policy_request() is None
    monkeypatch.setattr(
        "wuji_data_pipeline.deployment_node.time.monotonic", lambda: 20.0
    )
    assert n._claim_paper_policy_request() == (1, False)
    assert n._claim_paper_policy_request() is None


def test_midsegment_wait_and_then_protected_endpoint_exact():
    n = fake_node()
    put_pending(n)
    DeploymentNode._activate_paper_pending(n, 10.0)
    n._action_plan.pop_due(10.0)
    before = n._action_plan.snapshot()[0]
    put_pending(n, origin=9.9, request=2)
    assert not DeploymentNode._activate_paper_pending(n, 10.03)
    assert DeploymentNode._activate_paper_pending(n, before.due_at)
    after = n._action_plan.snapshot()[0]
    assert after.action == before.action and after.due_at == before.due_at
    assert n._paper_next_request_at == pytest.approx(
        11.0
    )  # activation doesn't drift scheduler


def test_stale_generation_does_not_replace_old_plan():
    n = fake_node()
    put_pending(n)
    DeploymentNode._activate_paper_pending(n, 10.0)
    before = n._action_plan.snapshot()
    put_pending(n, request=2)
    n._stream_generation = 2
    assert not DeploymentNode._activate_paper_pending(n, 10.0)
    assert n._action_plan.snapshot() == before


@pytest.mark.parametrize("curve", ["linear", "smoothstep"])
def test_real_publish_callback_overlap_sequence_and_interpolation(monkeypatch, curve):
    n = fake_node()
    n.paper_async_weight_curve = curve
    n.paper_async_blend_enabled = True
    n._lifecycle = 2
    n._applied_sequence = 0
    n._last_action_due_at = 0.0
    n.active_arm_sides = ("right",)
    n.active_hand_sides = ("right",)
    n.policy_action_mode = "eef"
    n.action_interpolation_method = "linear_slerp"
    n.prefetch_enabled = False
    n._interpolation_current = None
    n._last_applied_pose = {s: None for s in ("left", "right")}
    n._last_applied_hand = {s: None for s in ("left", "right")}
    n._last_applied_zsp = {s: None for s in ("left", "right")}
    n._activate_pending_chunk = MethodType(DeploymentNode._activate_pending_chunk, n)
    n._activate_paper_pending = MethodType(DeploymentNode._activate_paper_pending, n)
    n._mark_prefetch_miss = lambda now: False

    def validate(a):
        poses = {
            s: np.asarray(
                a[f"arm_action_{s}"]["ee_pos"] + a[f"arm_action_{s}"]["ee_quat"]
            )
            for s in ("left", "right")
        }
        hands = {s: np.asarray(a[f"hand_action_{s}"]) for s in ("left", "right")}
        return {s: (poses[s], hands[s], None) for s in poses}, poses, hands

    n._validate_response = validate
    published = []
    monkeypatch.setattr(
        DeploymentNode, "_startup_handoff_tick", lambda self, now: False
    )
    monkeypatch.setattr(
        DeploymentNode, "_fdm_wait_for_fresh_state_after_ready", lambda self, now: False
    )
    monkeypatch.setattr(
        DeploymentNode,
        "_publish_validated_command",
        lambda self, validated, command: published.append(command),
    )
    monkeypatch.setattr(
        DeploymentNode, "_fdm_record_executed_action", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        DeploymentNode, "_record_pi_joint_published_target", lambda *a, **kw: None
    )
    clock = [10.0]
    monkeypatch.setattr(
        "wuji_data_pipeline.deployment_node.time.monotonic", lambda: clock[0]
    )
    put_pending(n)
    DeploymentNode._apply_pending_action_impl(n)
    protected = n._action_plan.peek_next()
    first_sequence = n._applied_sequence
    put_pending(n, origin=9.9, request=2)
    clock[0] = 10.03
    DeploymentNode._apply_pending_action_impl(n)
    assert n._pending_chunk is not None
    assert n._applied_sequence == first_sequence
    clock[0] = protected.due_at
    DeploymentNode._apply_pending_action_impl(n)
    assert n._pending_chunk is None
    assert n._applied_sequence > protected.sequence
    assert published[-1]["hand_action_right"] == pytest.approx(
        protected.action["hand_action_right"]
    )
    new_current = n._interpolation_current
    new_next = n._action_plan.peek_next()
    clock[0] = protected.due_at + 1 / 30
    DeploymentNode._apply_pending_action_impl(n)
    expected = 0.5 * (
        new_current.action["hand_action_right"][0]
        + new_next.action["hand_action_right"][0]
    )
    assert published[-1]["hand_action_right"][0] == pytest.approx(expected)
    assert n._applied_sequence == new_current.sequence


def test_reconnect_clear_preserves_active_paper_timeline():
    n = fake_node()
    n.paper_async_blend_enabled = True
    n._lifecycle = 2
    put_pending(n)
    assert DeploymentNode._activate_paper_pending(n, 10.0)
    before = n._action_plan.snapshot()
    deadline = n._paper_next_request_at
    n._paper_request_origins[99] = (10.0, {})
    DeploymentNode._clear_action_stream(n, clear_active=False)
    assert n._paper_bootstrapped
    assert n._paper_next_request_at == deadline
    assert n._action_plan.snapshot() == before
    assert n._paper_request_origins == {}
    assert n._stream_generation == 2
    assert not n._request_inflight


def test_full_clear_restarts_bootstrap_and_drops_origins():
    n = fake_node()
    n.paper_async_blend_enabled = True
    n._lifecycle = 2
    put_pending(n)
    assert DeploymentNode._activate_paper_pending(n, 10.0)
    n._paper_request_origins[99] = (10.0, {})
    DeploymentNode._clear_action_stream(n, clear_active=True)
    assert not n._paper_bootstrapped
    assert n._paper_next_request_at == 0.0
    assert n._paper_request_origins == {}
    assert n._action_plan.remaining() == 0


def test_protected_head_plus32_capacity_rejection_is_atomic():
    n = fake_node()
    put_pending(n)
    assert DeploymentNode._activate_paper_pending(n, 10.0)
    before = n._action_plan.snapshot()
    put_pending(n, origin=10.0, request=2)
    # New row0 at old protected deadline+dt gives 33 slots, including head.
    assert not DeploymentNode._activate_paper_pending(n, 10.0)
    assert n._action_plan.snapshot() == before
    assert n._prefetch_activation_failures == 1
    assert 2 not in n._paper_request_origins


def test_camera_origin_regression_rejected_but_repeat_permitted():
    n = fake_node()
    n._paper_last_issued_origin = None
    metadata = {"clock_pair": {"source_clock_ros": 100.0, "captured_monotonic": 10.0}}
    DeploymentNode._register_paper_request_clock(n, 1, 1, 10.0, metadata)
    deadline = n._paper_next_request_at
    with pytest.raises(ValueError, match="backwards"):
        DeploymentNode._register_paper_request_clock(n, 2, 1, 9.9, metadata)
    assert n._paper_next_request_at == deadline
    assert 2 not in n._paper_request_origins
    DeploymentNode._register_paper_request_clock(n, 3, 1, 10.0, metadata)
    assert n._paper_next_request_at == deadline


def test_paper_underrun_trace_once_without_command_changes():
    n = fake_node()
    n.paper_async_blend_enabled = True
    n._paper_bootstrapped = True
    n._paper_underrun_reported = False
    n._last_action_due_at = 10.0
    events = []
    n._trace_writer = SimpleNamespace(record=lambda *a, **kw: events.append((a, kw)))
    assert DeploymentNode._mark_prefetch_miss(n, 10.1) is False
    assert DeploymentNode._mark_prefetch_miss(n, 10.2) is False
    assert len(events) == 1
    assert events[0][0] == ("paper_async_underrun",)
    assert n._action_plan.remaining() == 0


def test_smoothstep_preserves_timeline_endpoints_and_shared_weights():
    old = old_queue([0, 1, 2, 3, 4])
    kwargs = dict(new_origin=0.9, rate_hz=10, activation_at=1.0)
    linear = build_overlap_plan(old, [action(10)] * 10, **kwargs)
    smooth = build_overlap_plan(old, [action(10)] * 10, weight_curve="smoothstep", **kwargs)
    assert smooth.start_at == linear.start_at
    assert smooth.overlap_end_at == linear.overlap_end_at
    assert smooth.new_sample_indices == linear.new_sample_indices
    assert smooth.protected_sequence == linear.protected_sequence
    assert smooth.actions[0] is old[0].action
    assert smooth.actions[4:] == linear.actions[4:]
    assert smooth.new_weights[:5] == pytest.approx([0, 0.15625, 0.5, 0.84375, 1])
    for i, w in enumerate(smooth.new_weights[:5]):
        expected = (1-w)*i + w*10
        assert smooth.actions[i]["arm_action_right"]["ee_pos"][0] == pytest.approx(expected)
        assert smooth.actions[i]["hand_action_right"] == pytest.approx([expected]*20)
    with pytest.raises(ValueError, match="weight curve"):
        build_overlap_plan(old, [action(10)] * 10, weight_curve="invalid", **kwargs)
