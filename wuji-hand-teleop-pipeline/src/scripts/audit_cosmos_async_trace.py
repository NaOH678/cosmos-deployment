#!/usr/bin/env python3
"""Read-only offline timing and whole-session tracking audit; no ROS imports.

Usage: audit_cosmos_async_trace.py TRACE [--state-trace STATE] [--output JSON]
Tracking errors compare latest received actual/target poses, not synchronized
physical measurements. Only complete READY snapshots within publish intervals
are used. Missing evidence is reported as unavailable, never zero.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(count=len(a), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()))



def dispatched_kinematics(rows, limits=None):
    """Screen recorded targets at their model rate; not a new-policy rollout."""
    limits = dict(limits or dict(speed_m_s=.25, acceleration_m_s2=1., rotation_deg_s=60., hand_deg_s=90.))
    if any(not np.isfinite(value) or value <= 0 for value in limits.values()):
        raise ValueError("Screen limits must be finite and positive")
    chunks = {}
    previous = previous_velocity = None
    rate = None
    for row in rows:
        event = row['event']
        if event == 'session_start':
            rate = row['action_rate_hz']
        if event == 'action_stream_clear':
            previous = previous_velocity = None
        if event != 'action_dispatch' or 'right' not in row.get('arm_eef', {}):
            continue
        chunk = chunks.setdefault(str(row['chunk_id']), {'waypoints': 0, 'exceedances': {}, 'maximum': {}, 'violations': []})
        chunk['waypoints'] += 1
        if previous is not None:
            # Large gaps invalidate incoming velocity continuity. They must not
            # masquerade as one 15 Hz period or imply measured acceleration.
            dt = row['scheduled_due_at'] - previous['scheduled_due_at']
            if dt <= 0 or abs(dt - 1/rate) > .1/rate:
                previous = row
                previous_velocity = None
                continue
            a, b = np.array(previous['arm_eef']['right']), np.array(row['arm_eef']['right'])
            velocity = (b[:3] - a[:3]) * rate
            qa, qb = a[3:7] / np.linalg.norm(a[3:7]), b[3:7] / np.linalg.norm(b[3:7])
            metrics = {'speed_m_s': float(np.linalg.norm(velocity)),
                       'rotation_deg_s': float(np.degrees(2*np.arccos(np.clip(abs(qa @ qb), 0., 1.))) * rate)}
            if previous_velocity is not None:
                metrics['acceleration_m_s2'] = float(np.linalg.norm(velocity - previous_velocity) * rate)
            if 'right' in previous.get('hand_deg', {}) and 'right' in row.get('hand_deg', {}):
                metrics['hand_deg_s'] = float(np.abs(np.array(row['hand_deg']['right']) - previous['hand_deg']['right']).max() * rate)
            for key, value in metrics.items():
                chunk['maximum'][key] = max(chunk['maximum'].get(key, 0.), value)
                if value > limits[key]:
                    chunk['exceedances'][key] = chunk['exceedances'].get(key, 0) + 1
                    chunk['violations'].append({'metric': key, 'value': value,
                                                'chunk_action_index': row.get('chunk_action_index')})
            previous_velocity = velocity
        previous = row
    return {'limits': limits, 'chunks': chunks,
            'chunks_exceeding_any_limit': sum(bool(c['exceedances']) for c in chunks.values()),
            'chunk_count': len(chunks),
            'note': 'Recorded dispatched EEF/hand targets only, including cross-chunk differences. Uniform model-period adjacent points only; not measured motion, not new-policy acceptance or closed-loop prediction.'}


def analyze(trace, state_trace=None, limits=None):
    rows = [json.loads(line) for line in Path(trace).read_text().splitlines() if line]
    counts = collections.Counter(r['event'] for r in rows)
    response = [r for r in rows if r['event'] == 'policy_response_ready']
    activate = [r for r in rows if r['event'] == 'pending_chunk_activate']
    intervals, segments, previous, start, end = [], [], None, None, None
    for r in rows:
        if r['event'] == 'action_stream_clear':
            if start is not None:
                segments.append((start, end))
            previous = start = end = None
        if r['event'] == 'command_publish':
            now = r['published_at']
            if previous is not None:
                intervals.append(1000 * (now - previous))
            if start is None:
                start = now
            end = previous = now
    if start is not None:
        segments.append((start, end))
    report = {'trace': str(Path(trace).resolve()), 'event_counts': dict(counts),
              'complete_rtt_ms': distribution([r['complete_rtt_ms'] for r in response]),
              'server_inference_ms': distribution([r['server_inference_ms'] for r in response]),
              'publish_interval_ms': distribution(intervals),
              'publish_gaps_over_25ms': sum(v > 25 for v in intervals),
              'activation_count': len(activate),
              'prefetch_activation_count': sum(bool(r.get('prefetched')) for r in activate),
              'policy_request_image_payload_available': False,
              'note': 'Image payloads are not stored by these deployment traces. No closed-loop counterfactual or model inference is performed.'}
    for key in ('observation_to_activation_ms', 'response_to_activation_ms', 'boundary_wait_ms'):
        report[key] = distribution([r[key] for r in activate if r.get('prefetched') and key in r])
    if state_trace:
        errors, ages, skipped = [], [], collections.Counter()
        for line in Path(state_trace).read_text().splitlines():
            r = json.loads(line)
            if r.get('event') != 'snapshot':
                continue
            t = r['monotonic_ns'] / 1e9
            if not any(a <= t <= b for a, b in segments):
                continue
            if not r.get('ready') or not r.get('complete'):
                skipped['not_complete_ready'] += 1
                continue
            streams = r['streams']
            keys = ('right.arm_actual_eef', 'right.arm_external_target')
            if any(k not in streams for k in keys):
                skipped['missing_pose'] += 1
                continue
            actual, target = (streams[k] for k in keys)
            if actual['data'].get('frame_id') != target['data'].get('frame_id'):
                skipped['frame_mismatch'] += 1
                continue
            error = np.linalg.norm(np.array(actual['data']['position_m']) - target['data']['position_m'])
            errors.append(error * 100)
            ages.append(max(actual['age_ms'], target['age_ms']))
        report['tracking'] = {'state_trace': str(Path(state_trace).resolve()),
                              'external_target_vs_actual_cm': distribution(errors),
                              'paired_stream_max_receive_age_ms': distribution(ages),
                              'excluded_snapshots': dict(skipped),
                              'note': 'Latest received poses in matching frames, complete READY snapshots during command publication. Not timestamp-interpolated ground truth.'}
    else:
        report['tracking'] = None
    report['recorded_target_limit_screen'] = dispatched_kinematics(rows, limits)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--state-trace', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--max-speed-m-s', type=float, default=.25)
    p.add_argument('--max-acceleration-m-s2', type=float, default=1.)
    p.add_argument('--max-rotation-deg-s', type=float, default=60.)
    p.add_argument('--max-hand-deg-s', type=float, default=90.)
    a = p.parse_args()
    limits = dict(speed_m_s=a.max_speed_m_s, acceleration_m_s2=a.max_acceleration_m_s2,
                  rotation_deg_s=a.max_rotation_deg_s, hand_deg_s=a.max_hand_deg_s)
    rendered = json.dumps(analyze(a.trace, a.state_trace, limits), indent=2, allow_nan=False) + '\n'
    if a.output:
        a.output.write_text(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
