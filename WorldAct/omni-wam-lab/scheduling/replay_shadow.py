"""Partial-evidence replay. Missing commitments remain missing; this is not a simulator."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
from shadow_gate import Sample, Limits, decide


def action(a):
    arm=a['arm_action_right']
    return tuple(arm['ee_pos']+arm['ee_quat']+[math.radians(x) for x in a['hand_action_right']])


def replay(directory):
    root=Path(directory)
    trace=next((root/'client').glob('deployment_trace*.jsonl'))
    events=[json.loads(l) for l in trace.open()]
    commands=[Sample(e['published_at'],tuple(e['arm_eef']['right'])+tuple(math.radians(x) for x in e['hand_deg']['right'])) for e in events if e['event']=='command_publish']
    commands.sort(key=lambda s:s.time); ct=[s.time for s in commands]
    states=[]
    for l in next((root/'client').glob('deployment_state*.jsonl')).open():
        d=json.loads(l)
        if d.get('event')!='snapshot':continue
        streams=d['streams'];arm=streams.get('right.arm_actual_eef');hand=streams.get('right.hand_state')
        if not arm or not hand or hand['data'].get('unit')!='radian':continue
        t=d['monotonic_ns']/1e9
        # Use logged source ages, not receive time as a surrogate for freshness.
        if arm.get('age_ms') is None or hand.get('age_ms') is None:continue
        effective=t-max(arm['age_ms'],hand['age_ms'])/1000
        vals=tuple(arm['data']['position_m']+arm['data']['quaternion_xyzw']+hand['data']['position'])
        states.append((t,Sample(effective,vals)))
    st=[x[0] for x in states];rows=[]
    for e in events:
        if e['event']!='policy_splice_snapshot' or e['status']!='activated':continue
        now=e['activation_callback_at'];obs=e['observation_created_at'];rate=e['action_rate_hz']
        # Explicit hypothesis: output row0 corresponds to observation + 1/model_rate.
        cand=[Sample(obs+(i+1)/rate,action(a)) for i,a in enumerate(e['pending_actions']['actions'])]
        ci=bisect_left(ct,now);si=bisect_right(st,now)
        command=commands[ci-1] if ci else None; measured=states[si-1][1] if si else None
        current=e.get('current'); exhausted=e.get('replaced_remaining_actions')==0
        old_valid=current['due_at'] if current and exhausted else None
        d=decide(now=now,observation_at=obs,candidate=cand,executed=commands[:ci],measured=measured,command=command,committed_until=None,old_valid_until=old_valid)
        d.update(chunk_id=e['chunk_id'],request_id=e['request_id'])
        rows.append(d)
    reasons=Counter(r for d in rows for r in d['reasons'])
    return dict(run=root.name,experimental_limits=asdict(Limits()),
                timing_hypothesis='Model output row i at observation_created_at + (i+1)/rate; source image timestamps are recorded separately, and true learned action phase is not known.',
                missing_evidence=['No request-time commitment deadline/full future plan snapshot. Do not invent it from post-hoc execution.','No grasp contact/phase ground truth. Full27D compatibility cannot prove semantic phase.'],
                accepted=sum(d['can_join'] for d in rows),rejected=sum(not d['can_join'] for d in rows),reason_counts=dict(reasons),
                known_exhausted_old_plan_count=sum(d['old_plan_budget_s']==0 for d in rows),unknown_budget_count=sum(d['old_plan_budget_s'] is None for d in rows),
                valid_online_acceptance_estimate=False,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    reports=[replay(r) for r in a.runs];Path(a.output).write_text(json.dumps(reports,indent=2)+'\n')
    for r in reports:print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
