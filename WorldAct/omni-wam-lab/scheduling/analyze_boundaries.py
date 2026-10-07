"""Read-only trace analysis; reports target changes, never inferred grasp events."""
import argparse
import json
import math
from pathlib import Path


def summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    def percentile(q):
        p = (len(values) - 1) * q
        lo = int(p)
        return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (p-lo)
    return {"count": len(values), "median": percentile(.5), "p95": percentile(.95), "max": values[-1]}


def sub(a, b):
    return [x-y for x,y in zip(a,b)]


def norm(a):
    return math.sqrt(sum(x*x for x in a))


def analyze(path):
    events = [json.loads(line) for line in Path(path).open()]
    rows = []
    for e in events:
        if e['event'] != 'policy_splice_snapshot' or e['status'] != 'activated':
            continue
        if not e.get('current'):
            continue
        raw = e['aligned_actions']['actions']
        final = e['final_actions']['actions']
        anchor = e['current']['action']['arm_action_right']['ee_pos']
        first = raw[0]['arm_action_right']['ee_pos']
        end = raw[-1]['arm_action_right']['ee_pos']
        reset = sub(first,anchor)
        travel = sub(end,first)
        denom = norm(reset)*norm(travel)
        rows.append({
            'chunk_id':e['chunk_id'], 'request_id':e['request_id'],
            'skip':e['skipped_actions'],
            'raw_boundary_distance_m':norm(reset),
            'reset_vs_chunk_travel_cosine':sum(a*b for a,b in zip(reset,travel))/denom if denom>1e-12 else None,
            'index1_arm_change_by_postprocessing_m':norm(sub(final[1]['arm_action_right']['ee_pos'],raw[1]['arm_action_right']['ee_pos'])),
            'index1_hand_change_by_postprocessing_deg':max(abs(x) for x in sub(final[1]['hand_action_right'],raw[1]['hand_action_right'])),
            'observation_age_at_activation_ms':1000*(e['schedule_start_at']-e['observation_created_at']),
        })
    gaps = [e['ros_publish_interval_ms'] for e in events if e['event']=='command_publish' and (e.get('ros_publish_interval_ms') or 0)>100]
    keys=['raw_boundary_distance_m','index1_arm_change_by_postprocessing_m','index1_hand_change_by_postprocessing_deg','observation_age_at_activation_ms']
    return {'trace':str(Path(path).resolve()),'limitations':'Targets only. Negative cosine is not a failed grasp or necessarily an undesirable retry. No counterfactual inference: changing observation timing changes model outputs.',
            'publish_gaps_above100ms':summary(gaps),'boundaries':len(rows),
            'negative_reset_vs_travel_count':sum(r['reset_vs_chunk_travel_cosine'] is not None and r['reset_vs_chunk_travel_cosine']<0 for r in rows),
            'metrics':{k:summary([r[k] for r in rows]) for k in keys},'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('traces',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    report=[analyze(t) for t in a.traces]
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    for r in report:
        print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
