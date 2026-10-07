import numpy as np

from wuji_retargeting import Retargeter


def _config(hard_flexion_min=-0.05):
    return {
        "optimizer": {"type": "AdaptiveOptimizerAnalytical"},
        "retarget": {"hard_flexion_min": hard_flexion_min},
    }


def test_hard_flexion_min_tightens_only_pip_and_dip_bounds():
    baseline = Retargeter.from_config(_config(None), "right").optimizer
    constrained = Retargeter.from_config(_config(), "right").optimizer

    baseline_lower = np.asarray(baseline.opt.get_lower_bounds())
    constrained_lower = np.asarray(constrained.opt.get_lower_bounds())
    flex_indices = constrained._flex_idx
    other_indices = np.setdiff1d(np.arange(constrained.num_joints), flex_indices)

    np.testing.assert_allclose(constrained_lower[flex_indices], -0.05)
    np.testing.assert_allclose(
        constrained_lower[other_indices], baseline_lower[other_indices]
    )


def test_warm_start_is_clipped_to_active_optimizer_bounds():
    optimizer = Retargeter.from_config(_config(), "right").optimizer
    warm_start = np.full(optimizer.num_joints, -1.0)

    clipped = optimizer._get_init_qpos(warm_start)
    lower_bounds = np.asarray(optimizer.opt.get_lower_bounds())

    np.testing.assert_array_less(lower_bounds - 1e-12, clipped)
    np.testing.assert_allclose(clipped[optimizer._flex_idx], -0.05)
