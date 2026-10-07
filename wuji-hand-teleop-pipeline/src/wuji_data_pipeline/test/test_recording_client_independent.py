"""Client recording tests independent of production ROS/network startup."""
import json
from pathlib import Path
from types import SimpleNamespace

from test_async_splice_independent import node, install_old, pending
from wuji_data_pipeline.deployment_node import DeploymentNode, _bounded_action_snapshot
from wuji_data_pipeline.deployment_trace import DeploymentTraceWriter


def capture_node():
    n = node(); install_old(n)
    n.trace_policy_chunk_enabled = True
    events = []
    n._trace_writer = SimpleNamespace(record=lambda event, **fields: events.append(dict(event=event, **fields)))
    return n, events


def test_activated_snapshot_contains_all_stages_with_original_identity():
    n, events = capture_node(); pending(n, .5)
    n.early_splice_bridge_max_steps = 8
    n.early_splice_limits.update(max_speed_m_s=.4, max_acceleration_m_s2=5.)
    original = n._pending_chunk
    assert DeploymentNode._activate_pending_chunk(n, 10. + 1/15)
    snapshot = next(r for r in events if r['event'] == 'policy_splice_snapshot')
    assert snapshot['status'] == 'activated'
    assert snapshot['generation'] == original.generation
    assert snapshot['request_id'] == original.request_id
    assert snapshot['pending_actions']['action_count'] == 32
    assert snapshot['final_actions']['action_count'] == 16
    assert snapshot['committed']['action'] == snapshot['final_actions']['actions'][0]
    assert snapshot['bridge_input_actions']['actions'][0] == snapshot['committed']['action']
    assert snapshot['early_bridge_steps'] > 0
    original.actions[0]['arm_action_right']['ee_pos'][0] = -100
    assert snapshot['pending_actions']['actions'][0]['arm_action_right']['ee_pos'][0] == .5


def test_expired_snapshot_is_complete_even_before_window_selection():
    n, events = capture_node(); pending(n, observation=8.)
    assert not DeploymentNode._activate_pending_chunk(n, 10.+1/15)
    snapshot = next(r for r in events if r['event'] == 'policy_splice_snapshot')
    assert snapshot['status'] == 'rejected'
    assert 'expired' in snapshot['rejection_reason']
    assert snapshot['pending_actions']['action_count'] == 32
    assert snapshot['final_actions']['action_count'] == 0
    assert len([r for r in events if r['event'] == 'policy_splice_snapshot']) == 1


def test_snapshot_explicitly_marks_truncation_without_mutating_source():
    source = [{'value': [i]} for i in range(70)]
    saved = _bounded_action_snapshot(source)
    assert saved['action_count'] == 70 and saved['truncated']
    assert len(saved['actions']) == saved['max_actions'] == 64
    source[0]['value'][0] = -1
    assert saved['actions'][0]['value'][0] == 0


def test_client_writer_disk_failure_is_observable_and_does_not_raise(tmp_path, monkeypatch):
    original_open = Path.open
    def fail_trace(self, *args, **kwargs):
        if self.name.startswith('deployment_trace_'):
            raise OSError('test disk unavailable')
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_trace)
    writer = DeploymentTraceWriter(tmp_path, session_id='diagnostic-test', queue_size=2)
    writer.record('event', request_id=1)
    writer.close()
    assert 'test disk unavailable' in writer.writer_error
