import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('recording_bundle_audit',
    Path(__file__).resolve().parents[2] / 'scripts/audit_cosmos_recording_bundle.py')
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def write_json(path, value):
    path.write_text(json.dumps(value))


def bundle(path, failed=False):
    server = path / 'server/run_1'; server.mkdir(parents=True)
    client = path / 'client'; client.mkdir()
    write_json(server / 'manifest.json', {'recording_run_id': 'r1'})
    write_json(server / 'stats.json', {'written': 1, 'enqueued': 1, 'drained': True,
                                      'dropped': 0, 'errors': 0, 'stats_errors': 0})
    arrays = {'head': np.zeros((2, 3, 3), np.uint8), 'right_wrist': np.zeros((2, 3, 3), np.uint8),
              'state': np.zeros(27), 'raw_actions': np.zeros((32, 27))}
    if failed:
        del arrays['raw_actions']
    np.savez(server / '00000001.npz', **arrays)
    write_json(server / '00000001.json', {'session_id': 's1', 'request_id': 7,
        'status': 'failed' if failed else 'complete'})
    rows = [{'event': 'session_start', 'session_id': 's1', 'recording_run_id': 'r1'},
            {'event': 'policy_request_send', 'request_id': 7}]
    if not failed:
        rows.append({'event': 'policy_response_ready', 'request_id': 7})
    rows.append({'event': 'trace_summary', 'recording_run_id': 'r1', 'closed_cleanly': True,
                 'dropped_events': 0})
    (client / 'deployment_trace_example.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    (client / 'deployment_state_example.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'event': 'trace_start', 'recording_run_id': 'r1'},
        {'event': 'trace_summary', 'recording_run_id': 'r1', 'writer': {'dropped_events': 0}}])+'\n')
    return server, client


def test_complete_bundle_joins_request_identity(tmp_path):
    bundle(tmp_path)
    report = audit.audit_bundle(tmp_path, finalized=True)
    assert report['status'] == 'complete'
    assert report['correlation']['matched_successful_responses'] == 1


def test_inference_failure_with_partial_arrays_is_valid_recording(tmp_path):
    bundle(tmp_path, failed=True)
    report = audit.audit_bundle(tmp_path, finalized=True)
    assert report['status'] == 'complete'
    assert report['counts']['failed_inference_captures'] == 1
    assert report['counts']['failed_captures_with_partial_arrays'] == 1


def test_success_missing_committed_npz_is_explicit_failure(tmp_path):
    server, _ = bundle(tmp_path)
    (server / '00000001.npz').unlink()
    assert audit.audit_bundle(tmp_path, finalized=True)['status'] == 'failure'


def test_missing_closing_summary_is_running_then_incomplete(tmp_path):
    _, client = bundle(tmp_path)
    path = client / 'deployment_trace_example.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
    assert audit.audit_bundle(tmp_path)['status'] == 'running'
    assert audit.audit_bundle(tmp_path, finalized=True)['status'] == 'incomplete'


def test_writer_error_detected_even_when_manifest_creation_failed(tmp_path):
    server, _ = bundle(tmp_path)
    (server / 'manifest.json').unlink()
    write_json(server / 'stats.json', {'errors': 1, 'written': 1, 'drained': True})
    assert audit.audit_bundle(tmp_path, finalized=True)['status'] == 'failure'


def test_wrong_request_or_run_identity_is_explicit_failure(tmp_path):
    server, _ = bundle(tmp_path)
    write_json(server / '00000001.json', {'session_id': 'wrong', 'request_id': 7, 'status': 'complete'})
    assert audit.audit_bundle(tmp_path, finalized=True)['status'] == 'failure'
    write_json(server / 'manifest.json', {'recording_run_id': 'wrong-run'})
    assert any('Inconsistent' in x['detail'] for x in audit.audit_bundle(tmp_path)['issues'])


def test_no_files_is_unavailable_not_false_complete(tmp_path):
    assert audit.audit_bundle(tmp_path, finalized=True)['status'] == 'unavailable'
