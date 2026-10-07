"""Offline configuration contract; no ROS node, network or robot is started."""
from pathlib import Path

import yaml

from wuji_data_pipeline.deployment_protocol import calculate_prefetch_lead


def test_cloud_legacy24_restores_verified_historical_timing():
    path = Path(__file__).parents[1] / "config/cosmos_protocol_v2_50k_retrain_cloud_legacy24.yaml"
    config = yaml.safe_load(path.read_text())["deployment"]
    assert config["open_loop_horizon"] == 24
    assert (config["prefetch_min_lead_actions"], config["prefetch_initial_lead_actions"],
            config["prefetch_max_lead_actions"]) == (3, 7, 7)
    assert config["action_rate_hz"] == 15
    assert config["publish_rate_hz"] == 120
    assert config["action_interpolation_method"] == "linear_slerp"
    assert config["action_smoothing_method"] == "none"
    assert config["boundary_blend_steps"] == 1
    assert config["initial_blend_steps"] == 0
    assert not config["early_splice_enabled"]
    assert config["early_splice_bridge_max_steps"] == 0
    assert config["policy_http_expected_chunk_size"] == (
        config["open_loop_horizon"] + config["prefetch_max_lead_actions"] + 1
    )
    assert config["diagnostic_trace_enabled"]
    assert config["diagnostic_policy_chunk_enabled"]
    assert config["state_action_trace_rate_hz"] == 120


def test_historical_lead_cap_cannot_hide_a_slow_server():
    lead, p99 = calculate_prefetch_lead(
        [350.0] * 49 + [600.0], action_rate_hz=15, safety_actions=2,
        minimum_actions=3, maximum_actions=7, initial_actions=7,
        minimum_samples=5,
    )
    # Adaptation caps at seven even though this server cannot satisfy either
    # lead or alignment budget. Pre-enable full-RTT verification is necessary.
    assert lead == 7
    assert p99 == 600.0
    assert p99 / 1000 > lead / 15
    assert p99 / 1000 > (32 - 24) / 15


def test_blend8_trial_changes_exactly_one_parameter_and_keeps_tail():
    import numpy as np
    from wuji_data_pipeline.deployment_protocol import blend_action_prefix
    directory = Path(__file__).parents[1] / "config"
    base = yaml.safe_load((directory / "cosmos_protocol_v2_50k_retrain_cloud_legacy24.yaml").read_text())
    trial = yaml.safe_load((directory / "cosmos_protocol_v2_50k_retrain_cloud_legacy24_blend8.yaml").read_text())
    assert trial["deployment"]["boundary_blend_steps"] == 8
    trial["deployment"]["boundary_blend_steps"] = 1
    assert trial == base
    actions = [{"arm_action_right": {"ee_pos": [.13 + i * .001, 0., 0.],
                                      "ee_quat": [0., 0., 0., 1.]},
                "hand_action_right": [0.] * 20} for i in range(24)]
    blended = blend_action_prefix(actions,
        anchor_poses={"right": np.array([0., 0., 0., 0., 0., 0., 1.])},
        anchor_hands={"right": np.zeros(20)}, anchor_zsp={"right": None},
        blend_steps=8, sides=("right",))
    assert len(blended) == 24
    assert all(left is right for left, right in zip(blended[8:], actions[8:]))
    np.testing.assert_allclose(blended[7]["arm_action_right"]["ee_pos"],
                               actions[7]["arm_action_right"]["ee_pos"])
