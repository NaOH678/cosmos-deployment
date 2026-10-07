"""Independent numerical checks for the offline target-limit screen."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'async_trace_audit', Path(__file__).resolve().parents[2] / 'scripts/audit_cosmos_async_trace.py')
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def dispatch(x, t, chunk=1):
    return dict(event='action_dispatch', chunk_id=chunk, scheduled_due_at=t,
                arm_eef={'right': [x, 0, 0, 0, 0, 0, 1]}, hand_deg={'right': [0]*20})


def test_analytic_reversal_counts_acceleration_and_retains_chunk_boundary():
    rows = [dict(event='session_start', action_rate_hz=15),
            dispatch(0, 0), dispatch(.01, 1/15), dispatch(0, 2/15, chunk=2)]
    result = audit.dispatched_kinematics(rows)
    second = result['chunks']['2']
    assert second['maximum']['speed_m_s'] == pytest.approx(.15)
    assert second['maximum']['acceleration_m_s2'] == pytest.approx(4.5)
    assert second['exceedances'] == {'acceleration_m_s2': 1}


def test_stream_reset_does_not_create_fake_jump():
    rows = [dict(event='session_start', action_rate_hz=15), dispatch(0, 0),
            dict(event='action_stream_clear'), dispatch(1, 1/15)]
    assert audit.dispatched_kinematics(rows)['chunks']['1']['maximum'] == {}


def test_gap_is_not_misrepresented_as_model_period_acceleration():
    rows = [dict(event='session_start', action_rate_hz=15), dispatch(0, 0),
            dispatch(.01, 1/15), dispatch(1, 1), dispatch(1.01, 1+1/15)]
    metrics = audit.dispatched_kinematics(rows)['chunks']['1']['maximum']
    assert metrics['speed_m_s'] == pytest.approx(.15)
    assert 'acceleration_m_s2' not in metrics


def test_exploration_limits_are_explicit_and_do_not_change_default_screen():
    rows = [dict(event='session_start', action_rate_hz=15),
            dispatch(0, 0), dispatch(.01, 1/15), dispatch(0, 2/15)]
    limits = dict(speed_m_s=.4, acceleration_m_s2=5., rotation_deg_s=120., hand_deg_s=200.)
    assert audit.dispatched_kinematics(rows, limits)['chunks_exceeding_any_limit'] == 0
    assert audit.dispatched_kinematics(rows)['chunks_exceeding_any_limit'] == 1
    with pytest.raises(ValueError, match='finite and positive'):
        audit.dispatched_kinematics(rows, {**limits, 'speed_m_s': float('nan')})
