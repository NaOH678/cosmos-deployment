"""Independent node-level counterexamples; no ROS graph or robot is created."""
import threading
from types import SimpleNamespace, MethodType

import numpy as np
import pytest

from wuji_data_pipeline.deployment_node import DeploymentNode
from wuji_data_pipeline.deployment_protocol import LatestActionPlan, PendingActionChunk


def action(x=0.4):
    return {**{f'arm_action_{s}': {'ee_pos': [x, 0.1, 0.5],
                                  'ee_quat': [0., 0., 0., 1.]}
               for s in ('left', 'right')},
            **{f'hand_action_{s}': [0.] * 20 for s in ('left', 'right')}}


def node():
    n = SimpleNamespace(
        _action_plan=LatestActionPlan(max_actions=16), _prefetch_lock=threading.Lock(),
        _network_wakeup=threading.Event(), _stream_generation=3,
        _pending_chunk=None, _request_inflight=False, _last_action_due_at=0.,
        _active_chunk_id=1, _active_chunk_action_index=0,
        _last_applied_pose={s: np.array([.4, .1, .5, 0, 0, 0, 1]) for s in ('left', 'right')},
        _last_applied_hand={s: np.zeros(20) for s in ('left', 'right')},
        _last_applied_zsp={s: None for s in ('left', 'right')},
        _prefetch_hits=0, _prefetch_requests=0, _prefetch_activation_failures=0,
        _total_activation_skip_actions=0, _waiting_for_pending_since=0.,
        _max_boundary_wait_ms=0., _failures=0, _lifecycle=2,
        _valid_policy_stream_started=True, _prefetch_lead_actions=12,
        _interpolation_current=None, action_rate_hz=15., publish_rate_hz=120.,
        policy_action_mode='eef', active_arm_sides=('right',), active_hand_sides=('right',),
        open_loop_horizon=16, boundary_blend_steps=0, boundary_blend_method='smoothstep',
        action_interpolation_method='linear_slerp', prefetch_enabled=True,
        early_splice_enabled=True, early_splice_max_age_s=1.2,
        early_splice_request_interval_s=.2,
        early_splice_limits=dict(max_speed_m_s=.25, max_acceleration_m_s2=1.,
                                 max_rotation_deg_s=60., max_hand_deg_s=90.),
        get_logger=lambda: SimpleNamespace(warn=lambda _: None))
    n._action_for_side = DeploymentNode._action_for_side
    for name in ('_validate_response', '_validate_action_sequence'):
        setattr(n, name, MethodType(getattr(DeploymentNode, name), n))
    n._initial_command_anchors = lambda _: (n._last_applied_pose, n._last_applied_hand, n._last_applied_zsp)
    return n


def install_old(n, count=16):
    n._action_plan.install([action() for _ in range(count)], observation_created_at=10.,
                          received_at=10., rate_hz=15., schedule_start_at=10.)
    n._interpolation_current = n._action_plan.pop_due(10.)
    n._last_action_due_at = 10.


def pending(n, x=.4, observation=10., generation=3):
    n._pending_chunk = PendingActionChunk(actions=tuple(action(x) for _ in range(32)),
        action_rate_hz=15., observation_created_at=observation, response_received_at=10.02,
        align_to_observation=True, generation=generation)


def test_120hz_callbacks_preserve_committed_endpoint_until_15hz_boundary():
    n = node(); install_old(n); pending(n, .401)
    committed = n._action_plan.peek_next()
    for tick in range(1, 8):
        assert not DeploymentNode._activate_pending_chunk(n, 10. + tick / 120.)
        assert n._action_plan.peek_next() is committed
        assert n._pending_chunk is not None
    assert DeploymentNode._activate_pending_chunk(n, committed.due_at)
    assert n._action_plan.peek_next().action == committed.action
    assert n._action_plan.peek_next().due_at == committed.due_at
    assert n._pending_chunk is None


@pytest.mark.parametrize('kind', ['discontinuous', 'expired', 'generation', 'future'])
def test_rejected_candidate_is_consumed_once_and_old_plan_preserved(kind):
    n = node(); install_old(n)
    pending(n, x=.8 if kind == 'discontinuous' else .4,
            observation=8. if kind == 'expired' else 11. if kind == 'future' else 10.,
            generation=2 if kind == 'generation' else 3)
    committed = n._action_plan.peek_next()
    assert not DeploymentNode._activate_pending_chunk(n, committed.due_at)
    assert n._pending_chunk is None
    assert n._action_plan.peek_next() is committed
    assert n._prefetch_activation_failures == 1
    assert not DeploymentNode._activate_pending_chunk(n, committed.due_at + 1/120)
    assert n._prefetch_activation_failures == 1


def test_rejection_cannot_bypass_limits_after_queue_exhaustion_and_can_recover():
    n = node(); install_old(n, count=3); pending(n, .8)
    assert not DeploymentNode._activate_pending_chunk(n, 10. + 1/15)
    while n._action_plan.remaining():
        due = n._action_plan.peek_next().due_at
        n._interpolation_current = n._action_plan.pop_due(due)
        n._last_action_due_at = due
    pending(n, .8)
    assert not DeploymentNode._activate_pending_chunk(n, 10.2)
    assert n._action_plan.remaining() == 0
    pending(n, .4)
    assert DeploymentNode._activate_pending_chunk(n, 10.4)
    assert n._action_plan.remaining() > 0


def test_request_cooldown_and_single_inflight(monkeypatch):
    n = node(); n._policy_request_not_before = 10.2
    monkeypatch.setattr('wuji_data_pipeline.deployment_node.time.monotonic', lambda: 10.1)
    assert DeploymentNode._claim_policy_request(n) is None
    monkeypatch.setattr('wuji_data_pipeline.deployment_node.time.monotonic', lambda: 10.21)
    assert DeploymentNode._claim_policy_request(n) is not None
    assert DeploymentNode._claim_policy_request(n) is None


def test_exhausted_boundary_checks_previous_velocity_not_only_new_internal_steps():
    from wuji_data_pipeline.deployment_protocol import ScheduledAction
    n = node()
    n._interpolation_previous = ScheduledAction(1, 10. - 1/15, 9., action(.39))
    n._interpolation_current = ScheduledAction(2, 10., 9., action(.4))
    n._last_action_due_at = 10.
    # Incoming +0.15 m/s; new segment -0.15 m/s: both individually below
    # speed limit, but reversal acceleration is 4.5 m/s² (>1.0).
    pending(n, .39)
    assert not DeploymentNode._activate_pending_chunk(n, 10. + 1/15)
    assert 'acceleration' in n._early_splice_last_rejection


def test_bootstrap_uses_measured_pose_and_zero_incoming_velocity():
    n = node(); n._active_chunk_id = 0; pending(n, .41)
    # Only 1cm displacement: below 0.25m/s speed limit, but from rest
    # 2.25m/s² exceeds acceleration limit. No running plan may bypass it.
    assert not DeploymentNode._activate_pending_chunk(n, 10.02)
    assert 'acceleration' in n._early_splice_last_rejection


def test_full_publish_loop_reject_exhaust_and_resume_without_big_target(monkeypatch):
    n = node(); install_old(n, count=3)
    n._applied_sequence = n._interpolation_current.sequence
    n._active_chunk_skip_actions = n._active_chunk_blend_steps = 0
    n._active_chunk_prefetched = False
    n._prefetch_misses = 0
    n._activate_pending_chunk = MethodType(DeploymentNode._activate_pending_chunk, n)
    n._mark_prefetch_miss = MethodType(DeploymentNode._mark_prefetch_miss, n)
    published = []
    monkeypatch.setattr(DeploymentNode, '_publish_validated_command',
                        lambda self, validated, command=None: published.append(command))
    monkeypatch.setattr(DeploymentNode, '_record_pi_joint_published_target', lambda *a, **kw: None)
    pending(n, .8)
    for tick in range(1, 49):
        now = 10. + tick / 120.
        monkeypatch.setattr('wuji_data_pipeline.deployment_node.time.monotonic', lambda now=now: now)
        if tick == 30:
            pending(n, .4, observation=now-.1)
        DeploymentNode._apply_pending_action(n)
    assert n._prefetch_activation_failures == 1
    assert n._prefetch_misses == 1
    assert n._active_chunk_id == 2
    assert len(published) > 20
    assert all(abs(c['arm_action_right']['ee_pos'][0] - .4) < 1e-6 for c in published)


def test_boundary_gate_does_not_reclassify_unmodified_distant_model_motion():
    n = node(); install_old(n); pending(n)
    from dataclasses import replace
    actions = list(n._pending_chunk.actions)
    # Smooth splice, then a small distant model change that exceeds the
    # experimental acceleration limit. Ordinary full-response validation
    # still runs; the splice-only check must not reject distant model content.
    for i in range(10, len(actions)):
        actions[i] = action(.41)
    n._pending_chunk = replace(n._pending_chunk, actions=tuple(actions))
    assert DeploymentNode._activate_pending_chunk(n, 10. + 1/15)


def test_blend_prefix_later_step_cannot_escape_boundary_gate():
    from wuji_data_pipeline.deployment_protocol import ScheduledAction
    n = node(); n.boundary_blend_steps = 4
    n.early_splice_limits['max_acceleration_m_s2'] = 100.
    n._interpolation_current = ScheduledAction(1, 10., 9., action(.4))
    n._last_action_due_at = 10.
    pending(n, .46)
    # Four-step smoothstep: first move 9.375mm is 0.1406m/s (<.25),
    # second move 20.625mm is 0.3094m/s (>.25). Checking only new0 fails.
    assert not DeploymentNode._activate_pending_chunk(n, 10.4)
    assert 'speed_m_s' in n._early_splice_last_rejection


def test_minimal_bridge_preserves_committed_endpoint_tail_and_original_inputs():
    import copy
    from wuji_data_pipeline.deployment_protocol import bridge_early_action_window
    original = [action(.4), *[action(.5) for _ in range(15)]]
    saved = copy.deepcopy(original)
    limits = dict(max_speed_m_s=.4, max_acceleration_m_s2=5., max_rotation_deg_s=120., max_hand_deg_s=200.)
    bridged, count, _ = bridge_early_action_window(
        original, previous_action=action(.4), rate_hz=15., max_bridge_steps=8,
        arm_sides=('right',), hand_sides=('right',), limits=limits)
    assert 2 <= count <= 8
    assert bridged[0] is original[0]
    assert all(a is b for a, b in zip(bridged[count:], original[count:]))
    assert original == saved
    with pytest.raises(ValueError):
        bridge_early_action_window(original, previous_action=action(.4), rate_hz=15.,
            max_bridge_steps=count-1, arm_sides=('right',), hand_sides=('right',), limits=limits)


def test_failed_bridge_keeps_old_queue_and_does_not_rewrite_committed():
    n = node(); install_old(n); pending(n, 1.4)
    n.early_splice_bridge_max_steps = 8
    committed = n._action_plan.peek_next()
    assert not DeploymentNode._activate_pending_chunk(n, committed.due_at)
    assert n._action_plan.peek_next() is committed
    assert n._action_plan.remaining() == 15
    assert n._pending_chunk is None


def test_full_publish_loop_with_bridge_limits_targets_and_reaches_model(monkeypatch):
    n = node(); install_old(n)
    n.early_splice_bridge_max_steps = 8
    n.early_splice_limits.update(max_speed_m_s=.4, max_acceleration_m_s2=5.)
    n._applied_sequence = n._interpolation_current.sequence
    n._active_chunk_skip_actions = n._active_chunk_blend_steps = 0
    n._active_chunk_prefetched = False
    n._prefetch_misses = 0
    n._activate_pending_chunk = MethodType(DeploymentNode._activate_pending_chunk, n)
    n._mark_prefetch_miss = MethodType(DeploymentNode._mark_prefetch_miss, n)
    published = []
    monkeypatch.setattr(DeploymentNode, '_publish_validated_command',
                        lambda self, validated, command=None: published.append(command))
    monkeypatch.setattr(DeploymentNode, '_record_pi_joint_published_target', lambda *a, **kw: None)
    pending(n, .5)
    for tick in range(1, 97):
        now = 10. + tick / 120.
        monkeypatch.setattr('wuji_data_pipeline.deployment_node.time.monotonic', lambda now=now: now)
        DeploymentNode._apply_pending_action(n)
    assert n._prefetch_activation_failures == 0
    assert n._active_chunk_id == 2
    xs = np.array([c['arm_action_right']['ee_pos'][0] for c in published])
    assert xs[0] == pytest.approx(.4)
    assert xs[-1] == pytest.approx(.5)
    assert (np.abs(np.diff(xs)) * 120).max() <= .4001
