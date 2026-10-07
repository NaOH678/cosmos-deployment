"""Review candidate only: image-free, network-thread diagnostic snapshots.
Not installed/imported by production. Publication state is best-effort, not atomic.
"""
import copy
import time


def waypoint(w):
    return None if w is None else {'sequence':w.sequence,'due_at':w.due_at,'action':copy.deepcopy(w.action)}


def emit_schedule_snapshot(node, stage, request_id, generation, observation_created_at):
    writer=getattr(node,'_trace_writer',None)
    if writer is None or not getattr(node,'trace_policy_chunk_enabled',False):
        return
    started=time.monotonic()
    # Existing order: prefetch -> plan lock, NEVER latest while either is held.
    # Publisher does not take prefetch for every interpolation update; do not
    # pretend this lock creates an atomic publication-state snapshot.
    with node._prefetch_lock:
        stream_generation=node._stream_generation
        current0=node._interpolation_current
        start0=node._interpolation_current_start_at
        queue=node._action_plan.snapshot()
        current1=node._interpolation_current
        start1=node._interpolation_current_start_at
    plan_captured=time.monotonic()
    # Data callbacks replace references in _latest; no images are copied.
    with node._latest_lock:
        latest={k:v for k,v in node._latest.items() if k.startswith(('arm_eef_','arm_state_','hand_state_'))}
    state_captured=time.monotonic()
    measured={}
    for key,(source_stamp,value) in latest.items():
        if key.startswith('arm_eef_'):
            data={'eef':list(value),'unit':'metre_xyzw'}
        else:
            data={'position':list(value.position),'velocity':list(value.velocity),
                  'effort':list(value.effort),'position_velocity_unit':('degree' if key.startswith('arm_state_') else 'radian'), 'effort_unit':'source_message_unconverted'}
        measured[key]={'source_timestamp_ros':float(source_stamp),'data':data}
    next_point=queue[0] if queue else None
    method=getattr(node,'action_interpolation_method','none')
    protected=bool(method!='none' and current0 is not None and next_point is not None
                   and start0<=plan_captured<next_point.due_at)
    payload={
        'stage':stage,'request_id':request_id,'request_generation':generation,
        'stream_generation':stream_generation,'observation_created_at':observation_created_at,
        'plan_capture_monotonic':plan_captured,'state_capture_monotonic':state_captured,
        'source_clock_ros':node.get_clock().now().nanoseconds*1e-9,
        'snapshot_consistency':'best_effort_nonatomic_publisher',
        'publisher_change_detected':current0 is not current1 or start0!=start1,
        'queued_plan':[waypoint(w) for w in queue],
        'queued_plan_semantics':'planned_replaceable_tail_not_committed_prefix',
        'interpolation':{'method':method,'segment_start_at':start0,'from':waypoint(current0),'to':waypoint(next_point)},
        'protected_endpoint_observed':waypoint(next_point) if protected else None,
        'protected_endpoint_semantics':'current_segment_endpoint_only; recheck against publisher trace; not a request-time commitment horizon',
        'measured':measured,
    }
    payload['capture_compute_ms']=(time.monotonic()-started)*1000
    writer.record('policy_schedule_snapshot',**payload)
