import unittest
from shadow_gate import Sample,decide


def act(x=0,h=0):return (x,0,0,0,0,0,1)+tuple([h]*20)


class GateTests(unittest.TestCase):
    def args(self):
        samples=[Sample(t,act()) for t in (.1,.2,.3,.4)]
        return dict(now=.31,observation_at=0,candidate=samples,executed=samples[:3],measured=Sample(.30,act()),command=Sample(.30,act()),committed_until=.3,old_valid_until=.5)
    def test_accept_identical(self):self.assertTrue(decide(**self.args())['can_join'])
    def test_wrong_phase_same_position(self):
        a=self.args();a['candidate']=[Sample(s.time,act(h=.5)) for s in a['candidate']]
        self.assertIn('prefix_disagreement',decide(**a)['reasons'])
    def test_no_spatial_nearest_search(self):
        a=self.args();a['executed']=[Sample(.1,act(.2)),Sample(.2,act(.2)),Sample(.3,act(.2)),Sample(.9,act())]
        self.assertIn('prefix_disagreement',decide(**a)['reasons'])
    def test_hand_arm_mismatch(self):
        a=self.args();a['command']=Sample(.3,act(h=.5))
        self.assertIn('tracking_residual',decide(**a)['reasons'])
    def test_stale(self):
        a=self.args();a['measured']=Sample(.1,act())
        self.assertIn('stale_or_future_state',decide(**a)['reasons'])
    def test_normal_reverse_allowed(self):
        a=self.args();s=[Sample(t,act(x)) for t,x in zip((.1,.2,.3,.4),(.01,.02,.01,0))]
        a.update(candidate=s,executed=s[:3],measured=Sample(.3,act(.01)),command=Sample(.3,act(.01)))
        self.assertTrue(decide(**a)['can_join'])
    def test_commitment_protected(self):
        a=self.args();a['committed_until']=.4
        self.assertIn('committed_prefix_active',decide(**a)['reasons'])
    def test_missing_commitment_not_fabricated(self):
        a=self.args();a['committed_until']=None
        self.assertFalse(decide(**a)['can_join'])
    def test_starvation_reported(self):
        a=self.args();a.update(old_valid_until=.3,measured=None)
        self.assertIn('old_plan_budget_exhausted',decide(**a)['reasons'])


if __name__=='__main__':unittest.main()
