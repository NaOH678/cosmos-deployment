#!/usr/bin/env python3
"""Offline authenticated HTTP policy benchmark. Never imports ROS or publishes commands.

Replay decoded saved RGB through ONE JPEG encoding shared across endpoints.
JPEG recompression changes pixels relative to original NPZ; comparisons are
between these identical HTTP inputs, not bitwise against the original capture.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np
import yaml


def prepare_observation(npz_path, metadata_path, config, quality=95):
    dep = config['deployment']
    if dep.get('action_space', 'eef') != 'eef':
        raise ValueError('This benchmark supports recorded EEF profiles only')
    metadata = json.loads(Path(metadata_path).read_text())
    recorded = metadata['measured_state']
    request = dict(protocol_version=dep.get('protocol_version', 2), schema_version=dep.get('schema_version', 2),
                   message_type='observation', robot_layout=dep['robot_layout'], arms=['left', 'right'],
                   action_space='eef', arm_command_mode='eef',
                   active_hand_sides=dep.get('active_hand_sides', ['right']),
                   zero_filled_hand_sides=dep.get('zero_filled_hand_sides', ['left']), images={}, source_timestamps={})
    for side in ('left', 'right'):
        arm = copy.deepcopy(recorded[f'arm_state_{side}'])
        eef = arm['eef']; arm['ee_pos'] = eef[:3]; arm['ee_quat'] = eef[3:]
        arm.setdefault('joint_vel', [0.]*7); arm.setdefault('joint_torque', [0.]*7)
        hand = copy.deepcopy(recorded[f'hand_state_{side}'])
        hand.setdefault('joint_vel', [0.]*20); hand.setdefault('joint_torque', [0.]*20)
        arm = {k: np.asarray(v, dtype=np.float32) for k, v in arm.items()}
        hand = {k: np.asarray(v, dtype=np.float32) for k, v in hand.items()}
        request[f'arm_state_{side}'] = arm; request[f'hand_state_{side}'] = hand
    hashes = {}
    with np.load(npz_path, allow_pickle=False) as arrays:
        native_state = np.r_[request['arm_state_right']['eef'], request['hand_state_right']['joint_pos']].astype(np.float32)
        if not np.array_equal(native_state, arrays['state']):
            raise ValueError('Recorded measured_state does not exactly reproduce saved native state')
        for name in dep['camera_names']:
            image = arrays[name]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError('Saved image must be RGB uint8 HWC')
            ok, encoded = cv2.imencode('.jpg', np.ascontiguousarray(image[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError('JPEG encoding failed')
            data = encoded.tobytes()
            stamp = float(metadata.get('image_timestamps', {}).get(name, metadata.get('request_timestamp', 0.)))
            request['images'][name] = dict(codec='jpeg', color_space='bgr8', shape=list(image.shape), timestamp=stamp, data=data)
            request['source_timestamps']['camera_'+name] = stamp
            hashes[name] = hashlib.sha256(data).hexdigest()
    return request, hashes


def wire_vector(response):
    rows = []
    for action in response['action_chunk']:
        arm = action['arm_action_right']
        rows.append(arm['ee_pos'] + arm['ee_quat'] + action['hand_action_right'])
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (32, 27) or not np.isfinite(values).all():
        raise ValueError('Expected finite 32x27 wire EEF actions')
    if np.any(np.linalg.norm(values[:,3:7],axis=1)<1e-8):
        raise ValueError('Wire actions contain a zero quaternion')
    return values



def compare_wire(values, reference):
    distances=np.linalg.norm(values[:,:3]-reference[:,:3],axis=1)
    q=values[:,3:7];r=reference[:,3:7]
    dots=np.sum(q*r,axis=1)/(np.linalg.norm(q,axis=1)*np.linalg.norm(r,axis=1))
    return dict(reference_xyz_max_m=float(distances.max()), reference_xyz_mean_m=float(distances.mean()),
                reference_rotation_max_deg=float(np.degrees(2*np.arccos(np.clip(abs(dots),0.,1.))).max()),
                reference_hand_max_deg=float(np.max(abs(values[:,7:]-reference[:,7:]))),
                reference_wire_exact=bool(np.array_equal(values,reference)))


def statistics(values):
    return {'count': len(values), 'p50_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'p99_ms': float(np.percentile(values, 99)),
            'max_ms': float(max(values))} if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', action='append', required=True, help='label=http://host:port; may repeat')
    p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--api-key-env', default='COSMOS_POLICY_API_KEY')
    p.add_argument('--api-key-file', type=Path)
    p.add_argument('--policy-path')
    p.add_argument('--transport-module', type=Path, default=Path(__file__).resolve().parents[1]/'wuji_data_pipeline/wuji_data_pipeline/policy_transport.py')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--timeout-ms', type=int, default=120000)
    args = p.parse_args()
    if args.warmup < 0 or args.repeats < 1 or not 1 <= args.jpeg_quality <= 100:
        p.error('Invalid warmup/repeats/JPEG quality')
    key = args.api_key_file.read_text().strip() if args.api_key_file else os.environ.get(args.api_key_env, '')
    if not key:
        p.error('API key unavailable; supply environment variable or key file, never a command-line key')
    spec = importlib.util.spec_from_file_location('cosmos_http_existing_transport', args.transport_module)
    transport_module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = transport_module; spec.loader.exec_module(transport_module)
    config = yaml.safe_load(args.config.read_text()); dep = config['deployment']
    preparation_started=time.perf_counter()
    request, hashes = prepare_observation(args.observation, args.metadata, config, args.jpeg_quality)
    preparation_ms=(time.perf_counter()-preparation_started)*1000
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(note=__doc__.strip(), timing_scope='RTT covers existing transport pickle encode, HTTP exchange and response unpickle; excludes observation/JPEG preparation, loop payload construction and robot scheduling', observation_preparation_ms=preparation_ms, observation_sha256=hashlib.sha256(args.observation.read_bytes()).hexdigest(),
                  jpeg_quality=args.jpeg_quality, jpeg_sha256=hashes, original_metadata=str(args.metadata), endpoints=[])
    reference = None
    try:
        for index, entry in enumerate(args.server):
            label, url = entry.split('=', 1)
            transport = transport_module.HttpPolicyTransport(url, timeout_ms=args.timeout_ms, policy_path=args.policy_path or config.get('service', {}).get('endpoint', '/v1/robot-policy'),
                            api_key=key, max_response_bytes=32*1024*1024)
            session = uuid.uuid4().hex
            endpoint = dict(label=label, url=url, session_id=session, samples=[])
            report['endpoints'].append(endpoint)
            try:
                hello = dict(protocol_version=2, message_type='hello', session_id=session, request_id=1,
                             robot_layout=dep['robot_layout'], camera_names=dep['camera_names'], action_space='eef', arm_command_mode='eef')
                ack = transport.exchange(hello).response
                if ack.get('message_type') != 'hello_ack' or ack.get('session_id') != session or ack.get('request_id') != 1:
                    raise ValueError('Invalid hello acknowledgement')
                if ack.get('model_id') != dep['model_id']:
                    raise ValueError('Server model_id differs from benchmark config')
                endpoint['hello'] = dict(model_id=ack.get('model_id'), action_rate_hz=ack.get('action_rate_hz'))
                first = None
                for i in range(args.warmup+args.repeats):
                    rid=i+2
                    payload=dict(request, session_id=session, request_id=rid, timestamp=time.time(), client_monotonic=time.monotonic())
                    started=time.perf_counter(); exchange=transport.exchange(payload); elapsed=(time.perf_counter()-started)*1000
                    response=exchange.response
                    if response.get('error'):
                        raise RuntimeError('Policy response error code: '+str(response.get('error_code')))
                    if response.get('session_id') != session or response.get('request_id') != rid:
                        raise ValueError('Response identity mismatch')
                    values=wire_vector(response)
                    if i < args.warmup:
                        continue
                    if first is None:
                        first=values.copy()
                        np.save(args.output_dir/f'endpoint_{index}_first_wire.npy', first)
                        if reference is None: reference=first
                    endpoint['samples'].append(dict(request_id=rid, rtt_ms=elapsed, request_bytes=exchange.request_bytes,
                        response_bytes=exchange.response_bytes, server_timing=response.get('server_timing'),
                        repeat_max_abs=float(np.max(abs(values-first))),
                        **compare_wire(values,reference)))
                    np.save(args.output_dir/f'endpoint_{index}_last_wire.npy', values)
                    endpoint['rtt']=statistics([s['rtt_ms'] for s in endpoint['samples']])
                    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                rtt=endpoint['rtt']
                endpoint['legacy24_timing_gate']=dict(p99_below_400ms=rtt['p99_ms']<400, p99_plus_50ms_below_7step_lead=rtt['p99_ms']+50<7/15*1000,
                    all_samples_below_8step_alignment=rtt['max_ms']<8/15*1000,
                    note='Timing criterion only; not robot/control readiness or quality approval')
                print(json.dumps(dict(label=label, rtt=rtt, legacy24_timing_gate=endpoint['legacy24_timing_gate'])),flush=True)
            finally:
                transport.close()  # HTTP connection teardown ends its protocol session.
    finally:
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
