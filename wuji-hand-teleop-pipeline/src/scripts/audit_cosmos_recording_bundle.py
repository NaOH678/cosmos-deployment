#!/usr/bin/env python3
"""Audit a Cosmos recording bundle offline, without ROS/model/GPU access.

Exit 1 only for explicit recording corruption/loss/mismatches. Missing evidence
is incomplete/unavailable; writers without a closing marker may still be
running unless --finalized is supplied. Failed inference captures are valid
recordings and are counted separately from recording failures.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np


def audit_bundle(directory, finalized=False):
    root = Path(directory).resolve()
    issues = []
    counts = collections.Counter()
    captured, responded, sent = set(), set(), set()
    run_ids = set()

    def issue(kind, path, detail):
        issues.append(dict(kind=kind, path=str(path), detail=detail))

    def read_json(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            issue('failure', path, f'Unreadable JSON: {type(exc).__name__}')
            return None

    def run_identity(value, path):
        if value:
            run_ids.add(str(value))
        else:
            issue('unavailable', path, 'No recording_run_id')

    for folder in sorted((root / 'server').glob('run_*')):
        if not folder.is_dir():
            continue
        counts['server_recorders'] += 1
        manifest_path = folder / 'manifest.json'
        if manifest_path.exists():
            manifest = read_json(manifest_path) or {}
        else:
            issue('incomplete' if finalized else 'running', manifest_path, 'No server recorder manifest')
            manifest = {}
        run_identity(manifest.get('recording_run_id'), manifest_path)
        deployment = manifest.get('config', {}).get('deployment', {})
        model = manifest.get('config', {}).get('model', {})
        cameras = deployment.get('camera_names', ['head', 'right_wrist'])
        width = int(model.get('native_action_dim', 27))
        stats_path = folder / 'stats.json'
        stats = read_json(stats_path) if stats_path.exists() else None
        if stats is None:
            issue('incomplete' if finalized else 'running', stats_path, 'No recorder statistics')
        else:
            if any(stats.get(k, 0) for k in ('dropped', 'errors', 'stats_errors')):
                issue('failure', stats_path, 'Recorder reports dropped/error/stat-write counts')
            drained = bool(stats.get('drained', False))
            if not drained:
                issue('incomplete' if finalized else 'running', stats_path, 'Recorder has not drained')
        committed = 0
        for info_path in sorted(folder.glob('*.json')):
            if not info_path.stem.isdigit():
                continue
            info = read_json(info_path)
            if info is None:
                continue
            committed += 1
            counts['server_requests'] += 1
            failed = info.get('status') == 'failed'
            counts['failed_inference_captures' if failed else 'successful_inference_captures'] += 1
            identity = (info.get('session_id'), info.get('request_id'))
            if identity[0] is None or identity[1] is None:
                issue('failure', info_path, 'Missing session_id/request_id')
            elif identity in captured:
                issue('failure', info_path, 'Duplicate session_id/request_id')
            else:
                captured.add(identity)
            npz_path = info_path.with_suffix('.npz')
            if not npz_path.exists():
                issue('failure', npz_path, 'Committed request JSON lacks NPZ')
                continue
            try:
                with np.load(npz_path, allow_pickle=False) as arrays:
                    required = [*cameras, 'state', 'raw_actions']
                    missing = [key for key in required if key not in arrays.files]
                    if missing and not failed:
                        issue('failure', npz_path, 'Successful capture lacks: '+','.join(missing))
                    elif missing:
                        counts['failed_captures_with_partial_arrays'] += 1
                    for key in arrays.files:
                        value = arrays[key]
                        if key in info.get('shapes', {}) and list(value.shape) != info['shapes'][key]:
                            issue('failure', npz_path, f'{key} shape differs from metadata')
                        finite = bool(np.isfinite(value).all())
                        if key in info.get('finite', {}) and finite != info['finite'][key]:
                            issue('failure', npz_path, f'{key} finite flag differs from array')
                        if not finite and not failed:
                            issue('failure', npz_path, f'Successful output contains nonfinite {key}')
                        if key in cameras and (value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3):
                            issue('failure', npz_path, f'{key} is not HWC RGB uint8')
                        if key == 'state' and value.shape != (width,):
                            issue('failure', npz_path, f'state shape must be ({width},)')
                        if key == 'raw_actions' and (value.ndim != 2 or value.shape[1] != width or value.shape[0] != 32):
                            issue('failure', npz_path, f'raw_actions expected 32x{width} for this Cosmos profile')
            except (OSError, ValueError, TypeError) as exc:
                issue('failure', npz_path, f'Cannot inspect NPZ: {type(exc).__name__}')
        if stats and stats.get('drained') and committed != stats.get('written'):
            issue('failure', folder, 'Committed JSON count differs from drained written count')
        for npz_path in folder.glob('*.npz'):
            if not npz_path.with_suffix('.json').exists():
                issue('incomplete' if finalized else 'running', npz_path, 'NPZ has no JSON commit marker')

    for trace_path in sorted((root / 'client').glob('*.jsonl')):
        counts['client_trace_files'] += 1
        session = None
        summary = None
        state_file = trace_path.name.startswith('deployment_state_')
        counts['state_trace_files' if state_file else 'deployment_trace_files'] += 1
        decisions, snapshots = set(), set()
        try:
            with trace_path.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        issue('failure' if finalized else 'incomplete', trace_path,
                              f'Invalid JSONL at line {line_number}')
                        continue
                    event = row.get('event')
                    if event in ('session_start', 'trace_start'):
                        run_identity(row.get('recording_run_id'), trace_path)
                        session = row.get('session_id')
                    if event == 'policy_request_send':
                        sent.add((session, row.get('request_id')))
                    elif event == 'policy_response_ready':
                        responded.add((session, row.get('request_id')))
                    elif event in ('pending_chunk_activate', 'pending_chunk_activation_failed'):
                        decisions.add((row.get('generation'), row.get('request_id'),
                                       'activated' if event == 'pending_chunk_activate' else 'rejected'))
                    elif event == 'policy_splice_snapshot':
                        snapshots.add((row.get('generation'), row.get('request_id'), row.get('status')))
                        counts['splice_snapshots'] += 1
                        if row.get('status') == 'rejected':
                            counts['rejected_splice_snapshots'] += 1
                        for stage in ('pending_actions', 'aligned_actions', 'bridge_input_actions', 'final_actions'):
                            snapshot = row.get(stage)
                            if snapshot is None:
                                issue('failure', trace_path, f'Splice snapshot lacks {stage}')
                            elif snapshot.get('truncated'):
                                issue('incomplete', trace_path, f'Splice snapshot truncated {stage}')
                    elif event == 'trace_summary':
                        summary = row
        except OSError as exc:
            issue('failure', trace_path, f'Cannot read trace: {type(exc).__name__}')
            continue
        if decisions - snapshots:
            issue('incomplete', trace_path, f'{len(decisions - snapshots)} splice decisions lack full stage snapshots')
        if summary is None:
            issue('incomplete' if finalized else 'running', trace_path, 'No terminal trace_summary')
        else:
            run_identity(summary.get('recording_run_id'), trace_path)
            stats = summary.get('writer', {}) if state_file else summary
            if stats.get('dropped_events', 0) or stats.get('writer_error'):
                issue('failure', trace_path, 'Trace writer reports loss/error')
            if not state_file and not summary.get('closed_cleanly', False):
                issue('incomplete', trace_path, 'Client trace did not close cleanly')
    if len(run_ids) > 1:
        issue('failure', root, 'Inconsistent recording_run_id across files')
    unmatched = responded - captured
    if unmatched:
        issue('failure' if finalized else 'running', root,
              f'{len(unmatched)} successful client responses have no committed server capture')
    if not counts['server_recorders']:
        issue('unavailable', root / 'server', 'No server recording manifest')
    if not counts['client_trace_files']:
        issue('unavailable', root / 'client', 'No client traces')
    elif not counts['state_trace_files']:
        issue('unavailable', root / 'client', 'No measured state/action trace')
    kinds = {i['kind'] for i in issues}
    status = next((k for k in ('failure', 'incomplete', 'running', 'unavailable') if k in kinds), 'complete')
    return dict(directory=str(root), finalized=finalized, status=status,
                counts=dict(counts), recording_run_ids=sorted(run_ids),
                correlation={'client_requests': len(sent), 'client_successful_responses': len(responded),
                             'server_captures': len(captured), 'matched_successful_responses': len(responded & captured),
                             'requests_without_server_capture': len(sent - captured)},
                issues=issues,
                note='Offline completeness/correlation audit, not model determinism or robot safety validation. Failed inference captures remain valid evidence; their arrays may be partial.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--finalized', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_bundle(args.directory, args.finalized)
    rendered = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end='')
    raise SystemExit(1 if report['status'] == 'failure' else 0)


if __name__ == '__main__':
    main()
