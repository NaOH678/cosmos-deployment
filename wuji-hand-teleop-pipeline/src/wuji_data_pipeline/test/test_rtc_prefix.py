import numpy as np
import pytest
from wuji_data_pipeline.rtc_prefix import build_prefix
from wuji_data_pipeline.deployment_protocol import LatestActionPlan, PendingActionChunk, select_pending_action_window


def action(index):
    return dict(arm_action_right=dict(ee_pos=[index/100., 0., 0.], ee_quat=[0.,0.,0.,1.]),
                hand_action_right=[90.] * 20)


def test_prefix_snapshots_committed_queue_preserves_units_and_soft_tail():
    plan = LatestActionPlan(max_actions=20)
    plan.install([action(i) for i in range(20)], observation_created_at=0., received_at=1., rate_hz=15.)
    for i in range(9):
        plan.pop_due(1. + i/15.)
    snapshot = plan.snapshot()
    prefix = build_prefix(snapshot, [action(20), action(21)], horizon=20)
    assert prefix['committed_steps'] == 11
    assert len(prefix['actions']) == 13
    np.testing.assert_allclose(prefix['actions'][0][7:], np.pi/2)
    assert prefix['actions'][0][0] == .09
    plan.pop_due(2.)
    assert len(snapshot) == 11  # Subsequent dispatch cannot mutate the request.


def test_prefix_alignment_consumes_exact_committed_count_not_elapsed_rtt():
    chunk = PendingActionChunk(tuple(action(i) for i in range(32)), 15., 0., .8, True, rtc_prefix_steps=11)
    result, skip = select_pending_action_window(chunk, activation_at=.84, horizon=20, align_to_observation=False)
    assert skip == 11
    assert len(result) == 20
    assert result[0] == action(11)
    with pytest.raises(ValueError):
        select_pending_action_window(chunk, activation_at=.84, horizon=20, align_to_observation=True)
    with pytest.raises(ValueError):
        select_pending_action_window(chunk, activation_at=.84, horizon=20, align_to_observation=False, fixed_skip_actions=6)
    with pytest.raises(TimeoutError):
        select_pending_action_window(chunk, activation_at=.84, horizon=24, align_to_observation=False)


def test_bootstrap_empty_prefix_and_budget_failure():
    assert build_prefix((), (), horizon=20)['actions'] == []
    plan = LatestActionPlan(max_actions=20)
    plan.install([action(i) for i in range(13)], observation_created_at=0., received_at=1., rate_hz=15.)
    with pytest.raises(ValueError):
        build_prefix(plan.snapshot(), (), horizon=20)
