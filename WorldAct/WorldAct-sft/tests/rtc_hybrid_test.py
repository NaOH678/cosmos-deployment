import numpy as np
import pytest
from cosmos_framework.inference.robot_policy.rtc_hybrid_http import METHOD, validate_prefix


def payload(hard):
    a = np.zeros((hard + 2, 27), dtype=np.float32)
    a[:, 6] = 1
    return dict(method=METHOD, committed_steps=hard, actions=a.tolist())


def test_hybrid_accepts_sixteen_step_budget_and_soft_tail():
    actions, weights, hard = validate_prefix(payload(16))
    assert hard == 16 and actions.shape == (18, 27)
    np.testing.assert_array_equal(weights[:16], 1)
    assert 1 > weights[16] > weights[17] > 0


def test_hybrid_rejects_wrong_method_budget_and_quaternion():
    for field, value in [("method", "identity_jacobian_soft_prefix_v1"), ("committed_steps", 17)]:
        p = payload(16); p[field] = value
        with pytest.raises(ValueError): validate_prefix(p)
    p = payload(16); p["actions"][0][6] = 0
    with pytest.raises(ValueError): validate_prefix(p)
    a, w, h = validate_prefix(dict(method=METHOD, committed_steps=0, actions=[]))
    assert a.shape == (0, 27) and len(w) == h == 0
