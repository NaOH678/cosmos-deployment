#!/usr/bin/env python3
"""Screen first possible early splice using fixed historical returned actions.

This does not simulate new observations or a closed-loop trajectory. Each
candidate independently uses the ORIGINAL old plan from the recorded trace.
Only the splice-kinematics gate is reproduced; other runtime validators and
controller tracking are not simulated. No network, ROS, model or hardware.
"""
import argparse
import json
import math
from pathlib import Path

from audit_cosmos_async_trace import dispatched_kinematics


def audit(trace, limits, bridge_max_steps=0):
    rows = [json.loads(line) for line in Path(trace).read_text().splitlines() if line]
    session = next(r for r in rows if r['event'] == 'session_start')
    if session['action_smoothing_method'] != 'none':
        raise ValueError('This replay requires recorded client smoothing=none')
    rate = session['action_rate_hz']
    raw = {(r['generation'], r['request_id']): r for r in rows
           if r['event'] == 'policy_action_chunk' and r['stage'] == 'server_output'}
    responses = {(r['generation'], r['request_id']): r for r in rows
                 if r['event'] == 'policy_response_ready'}
    dispatched = [r for r in rows if r['event'] == 'action_dispatch']
    result = []
    for activation in rows:
        if activation['event'] != 'pending_chunk_activate' or not activation['prefetched']:
            continue
        key = (activation['generation'], activation['request_id'])
        item = dict(request_id=key[1], original_chunk_id=activation['chunk_id'])
        result.append(item)
        if key not in raw or key not in responses:
            item['unavailable'] = 'missing response or raw action snapshot'
            continue
        snapshot, response = raw[key], responses[key]
        # response_ready is after client validation/store, so a conservative
        # readiness timestamp; cannot activate at server computation finish.
        ready_at = response['monotonic_time']
        candidates = [d for d in dispatched if d['scheduled_due_at'] >= ready_at]
        if not candidates:
            item['unavailable'] = 'no later dispatched old-plan endpoint'
            continue
        committed = candidates[0]
        at = committed['scheduled_due_at']
        earlier = [d for d in dispatched if d['scheduled_due_at'] < at]
        if not earlier or committed['chunk_id'] == activation['chunk_id']:
            item['unavailable'] = 'no recorded old-plan endpoint before original activation'
            continue
        current = earlier[-1]
        if current['chunk_id'] != committed['chunk_id'] or abs(at-current['scheduled_due_at']-1/rate) > .1/rate:
            item['unavailable'] = 'noncontiguous old-plan segment'
            continue
        skip = math.ceil((at-snapshot['observation_created_at'])*rate-1e-9)
        actions = snapshot['actions']
        item.update(candidate_activation_at=at, candidate_skip=skip,
                    original_activation_at=activation['schedule_start_at'],
                    activation_advance_ms=(activation['schedule_start_at']-at)*1000,
                    first_changed_waypoint_due_at=at+1/rate,
                    observation_age_ms=(at-snapshot['observation_created_at'])*1000)
        if skip > len(actions)-session['open_loop_horizon']:
            item['kinematics_pass'] = False
            item['rejection'] = 'full aligned horizon unavailable'
            continue
        synthetic = [dict(event='session_start', action_rate_hz=rate)]
        for index, record in enumerate((current, committed)):
            synthetic.append(dict(event='action_dispatch', chunk_id=1, chunk_action_index=index,
                                  scheduled_due_at=index/rate, arm_eef=record['arm_eef'], hand_deg=record['hand_deg']))
        for index in (1, 2):
            action = actions[skip+index]
            synthetic.append(dict(event='action_dispatch', chunk_id=1, chunk_action_index=index+1,
                                  scheduled_due_at=(index+1)/rate,
                                  arm_eef={'right': action['arm_action_right']['ee_pos']+action['arm_action_right']['ee_quat']},
                                  hand_deg={'right': action['hand_action_right']}))
        screened = dispatched_kinematics(synthetic, limits)['chunks']['1']
        item['kinematics_pass'] = not bool(screened['exceedances'])
        item['metrics'] = screened['maximum']
        item['violations'] = screened['violations']
        if bridge_max_steps:
            from wuji_data_pipeline.deployment_protocol import bridge_early_action_window

            def recorded_action(record):
                return {'arm_action_right': {'ee_pos': record['arm_eef']['right'][:3],
                                             'ee_quat': record['arm_eef']['right'][3:7]},
                        'hand_action_right': record['hand_deg']['right']}

            committed_action = recorded_action(committed)
            window = [committed_action, *actions[skip+1:skip+session['open_loop_horizon']]]
            try:
                bridged, steps, metrics = bridge_early_action_window(
                    window, previous_action=recorded_action(current), rate_hz=rate,
                    max_bridge_steps=bridge_max_steps, arm_sides=('right',), hand_sides=('right',),
                    limits={f'max_{key}': value for key, value in limits.items()})
                assert bridged[0] == committed_action
                assert bridged[steps:] == window[steps:]
                item['with_bridge'] = dict(kinematics_pass=True, bridge_steps=steps,
                                           bridge_duration_ms=steps/rate*1000, metrics=metrics,
                                           committed_exact=True, endpoint_and_tail_exact=True)
            except ValueError as exc:
                item['with_bridge'] = dict(kinematics_pass=False, rejection=str(exc))
    return dict(trace=str(Path(trace).resolve()), limits=limits, candidates=result,
                kinematics_pass_count=sum(r.get('kinematics_pass') is True for r in result),
                kinematics_reject_count=sum(r.get('kinematics_pass') is False for r in result),
                note=__doc__.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bridge-max-steps', type=int, default=0)
    a = p.parse_args()
    limits = dict(speed_m_s=.4, acceleration_m_s2=5., rotation_deg_s=120., hand_deg_s=200.)
    a.output.write_text(json.dumps(audit(a.trace, limits, a.bridge_max_steps), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
