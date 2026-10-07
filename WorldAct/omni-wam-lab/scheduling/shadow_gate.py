"""Experimental offline gate. No robot I/O; thresholds are hypotheses, not safety limits.
Actions: xyz metres, quaternion xyzw, 20 hand joints radians. Times: monotonic seconds.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Sample:
    time: float
    action: tuple


@dataclass(frozen=True)
class Limits:
    position_m: float = .025
    rotation_rad: float = math.radians(12)
    hand_rad: float = math.radians(10)
    state_age_s: float = .10
    match_time_s: float = .025
    max_commit_s: float = .60
    history_steps: int = 3


def differences(a, b):
    if len(a) != 27 or len(b) != 27 or not all(math.isfinite(x) for x in (*a,*b)):
        raise ValueError('expected finite 27D actions')
    qa,qb=a[3:7],b[3:7]
    n=math.sqrt(sum(x*x for x in qa)*sum(x*x for x in qb))
    if n<1e-10:
        raise ValueError('zero quaternion')
    return (math.dist(a[:3],b[:3]),2*math.acos(min(1,abs(sum(x*y for x,y in zip(qa,qb))/n))),max(abs(x-y) for x,y in zip(a[7:],b[7:])))


def compatible(a,b,limits):
    d=differences(a,b)
    return all(x<=y for x,y in zip(d,(limits.position_m,limits.rotation_rad,limits.hand_rad))),d


def decide(*, now, observation_at, candidate, executed, measured, command,
           committed_until, old_valid_until, limits=Limits()):
    """Return decision only. committed_until=None means unavailable evidence, never inferred.

    History matches timestamp only, not spatial nearest neighbor. No retiming search.
    A finite prior commitment is honored; validity horizon does not authorize extrapolation.
    """
    reasons=[];metrics={}
    if not math.isfinite(now) or not math.isfinite(observation_at) or observation_at>now:
        reasons.append('invalid_clock')
    for samples in (candidate,executed):
        if any(not math.isfinite(s.time) for s in samples) or any(a.time>=b.time for a,b in zip(samples,samples[1:])):
            reasons.append('nonmonotonic_samples')
    if committed_until is None:
        reasons.append('missing_commitment_record')
    elif not math.isfinite(committed_until) or committed_until-observation_at>limits.max_commit_s:
        reasons.append('invalid_or_excess_commitment')
    elif now<committed_until:
        reasons.append('committed_prefix_active')
    if measured is None or command is None:
        reasons.append('missing_measured_or_command_state')
    elif not 0<=now-measured.time<=limits.state_age_s or not 0<=now-command.time<=limits.state_age_s:
        reasons.append('stale_or_future_state')
    else:
        ok,d=compatible(measured.action,command.action,limits);metrics['tracking_residual']=d
        if not ok:reasons.append('tracking_residual')
    history=[s for s in candidate if observation_at<=s.time<=now][-limits.history_steps:]
    if len(history)<limits.history_steps:
        reasons.append('insufficient_candidate_history')
    else:
        distances=[]
        for s in history:
            prior=min(executed,key=lambda e:abs(e.time-s.time),default=None)
            if prior is None or abs(prior.time-s.time)>limits.match_time_s:
                reasons.append('missing_time_matched_executed_history');break
            ok,d=compatible(s.action,prior.action,limits);distances.append(d)
            if not ok:reasons.append('prefix_disagreement')
        metrics['prefix_residuals']=distances
    future=next((s for s in candidate if s.time>now),None)
    if future is None:
        reasons.append('candidate_exhausted')
    elif command is not None:
        ok,d=compatible(future.action,command.action,limits);metrics['entry_residual']=d
        if not ok:reasons.append('entry_disagreement')
    budget=None if old_valid_until is None else max(0.,old_valid_until-now)
    if reasons and budget==0:reasons.append('old_plan_budget_exhausted')
    return {'can_join':not reasons,'reasons':sorted(set(reasons)),
            'old_plan_budget_s':budget,'commit_wait_s':None if committed_until is None else max(0,committed_until-now),
            'metrics':metrics,'experimental_only':True}
