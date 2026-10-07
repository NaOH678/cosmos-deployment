from pathlib import Path
import copy
import numpy as np
import pytest
import yaml
from wuji_data_pipeline.deployment_protocol import blend_action_prefix, PendingActionChunk


def inputs():
    actions = [{**{f"arm_action_{s}": {"ee_pos": [.4 + i * .0002, .1, .2],
                 "ee_quat": [0., 0., 0., 1.], "zsp": [0., 0., 1.]}
                 for s in ("left", "right")},
                **{f"hand_action_{s}": [10. + i * .1] * 20
                   for s in ("left", "right")}} for i in range(32)]
    kwargs = dict(anchor_poses={s: np.array([.4, .1, .2, 0., 0., 0., 1.]) for s in ("left", "right")},
                  anchor_hands={s: np.full(20, 40.) for s in ("left", "right")},
                  anchor_zsp={s: np.array([0., 0., 1.]) for s in ("left", "right")}, blend_steps=8)
    return actions, kwargs


def test_independent_hand2_preserves_arm_xyz_quaternion_zsp_and_tail_exactly():
    actions, kwargs = inputs()
    original = copy.deepcopy(actions)
    legacy = blend_action_prefix(actions, **kwargs)
    trial = blend_action_prefix(actions, **kwargs, hand_blend_steps=2)
    assert actions == original
    for index, (old, new) in enumerate(zip(legacy, trial)):
        for side in ("left", "right"):
            assert old[f"arm_action_{side}"] == new[f"arm_action_{side}"]
            if index >= 1:
                assert new[f"hand_action_{side}"] is actions[index][f"hand_action_{side}"]
    np.testing.assert_array_equal(trial[0]["hand_action_right"], np.full(20, 25.))


def test_default_hand_blend_retains_legacy_arithmetic():
    actions, kwargs = inputs()
    result = blend_action_prefix(actions, **kwargs)
    for index in range(8):
        f = (index + 1) / 8
        alpha = f * f * (3 - 2 * f)
        expected = ((1-alpha) * kwargs["anchor_hands"]["right"].astype(np.float32)
                    + alpha * np.asarray(actions[index]["hand_action_right"], dtype=np.float32))
        np.testing.assert_array_equal(result[index]["hand_action_right"], expected)


def test_independent_hand_option_rejects_joint_and_excessive_length():
    actions, kwargs = inputs()
    with pytest.raises(ValueError, match="EEF"):
        blend_action_prefix(actions, **kwargs, hand_blend_steps=2, action_mode="joint")
    with pytest.raises(ValueError, match="between"):
        blend_action_prefix(actions, **kwargs, hand_blend_steps=9)


def test_hand2_profile_changes_only_independent_hand_length():
    folder = Path(__file__).parents[1] / "config"
    base = yaml.safe_load((folder / "cosmos_protocol_v2_50k_retrain_cloud_legacy24_blend8.yaml").read_text())
    trial = yaml.safe_load((folder / "cosmos_protocol_v2_50k_retrain_cloud_legacy24_blend8_hand2.yaml").read_text())
    assert trial["deployment"].pop("boundary_hand_blend_steps") == 2
    assert trial == base


def test_node_activation_consumes_independent_hand_option():
    from test_deployment import _prefetch_node
    from wuji_data_pipeline.deployment_node import DeploymentNode
    node = _prefetch_node()
    actions, kwargs = inputs()
    node.boundary_blend_steps = 8
    node.boundary_hand_blend_steps = 2
    node._active_chunk_id = 1
    node._last_applied_pose = kwargs["anchor_poses"]
    node._last_applied_hand = kwargs["anchor_hands"]
    node._last_applied_zsp = kwargs["anchor_zsp"]
    node._pending_chunk = PendingActionChunk(tuple(actions), 30., 1., 1., True)
    assert DeploymentNode._activate_pending_chunk(node, 1.)
    first = node._action_plan.pop_due(1.)
    second = node._action_plan.pop_due(1. + 1/30)
    np.testing.assert_array_equal(first.action["hand_action_right"], np.full(20, 25.))
    assert second.action["hand_action_right"] == actions[1]["hand_action_right"]
