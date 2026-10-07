import threading
import unittest
from types import SimpleNamespace as N
from diagnostic_snapshot_candidate import emit_schedule_snapshot


class DiagnosticTests(unittest.TestCase):
    def node(self):
        self.events=[]
        point=N(sequence=2,due_at=1e20,action={'hand_action_right':[1.]})
        node=N(_trace_writer=N(record=lambda *a,**k:self.events.append((a,k))),trace_policy_chunk_enabled=True,
               _prefetch_lock=threading.Lock(),_latest_lock=threading.Lock(),_stream_generation=3,
               _interpolation_current=N(sequence=1,due_at=0,action={}),_interpolation_current_start_at=0,
               _action_plan=N(snapshot=lambda:(point,)),action_interpolation_method='linear_slerp',
               _latest={'arm_state_right':(10,N(position=[90],velocity=[0],effort=[1])),
                        'hand_state_right':(11,N(position=[1.],velocity=[],effort=[])),
                        'camera_head':(12,object())},get_clock=lambda:N(now=lambda:N(nanoseconds=12_000_000_000)))
        return node,point
    def test_disabled_does_not_snapshot(self):
        node,point=self.node();node.trace_policy_chunk_enabled=False
        node._action_plan.snapshot=lambda:1/0
        emit_schedule_snapshot(node,'send',1,3,0)
        self.assertFalse(self.events)
    def test_image_free_units_and_semantics(self):
        node,point=self.node();emit_schedule_snapshot(node,'send',1,3,0)
        d=self.events[0][1]
        self.assertNotIn('camera_head',d['measured'])
        self.assertEqual(d['measured']['arm_state_right']['data']['position_velocity_unit'],'degree')
        self.assertIn('not_committed',d['queued_plan_semantics'])
        self.assertEqual(d['protected_endpoint_observed']['sequence'],2)
        point.action['hand_action_right'][0]=9
        self.assertEqual(d['queued_plan'][0]['action']['hand_action_right'][0],1)
    def test_write_outside_locks(self):
        node,_=self.node()
        def record(*a,**k):
            self.assertFalse(node._prefetch_lock.locked());self.assertFalse(node._latest_lock.locked())
        node._trace_writer.record=record
        emit_schedule_snapshot(node,'response',1,3,0)


if __name__=='__main__':unittest.main()
