"""The configured FK displacement scale must put the target on the noise's scale.

The bug this exists to prevent: FK was ported from PointFlow, which sets
``pointflow_displacement_scale`` to the measured std of its own displacement
data, and the FK port kept the config default (1.0) instead.  Nothing failed --
the config resolved, training ran, the loss fell -- but the RF forward process
``x_sigma = sigma*eps + (1-sigma)*target`` then mixes a unit-variance Gaussian
with a target 13x smaller, so the branch spends nearly all of its training
learning to predict the noise, and its one-step estimate stalls at ~23% of the
zero-motion baseline however long it trains.

There is no runtime symptom to assert on, so this asserts the *invariant* the
bug violated: ``eps`` is unit-variance, therefore ``target`` must be too.

Skipped, not failed, when the annotation tree is absent: the scale is a property
of the data selection, and a checkout without the data cannot check it.

    python cosmos_framework/configs/base/experiment/action/posttrain_config/fk_displacement_scale_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

# Import order matters: the experiment config reads these at import time.
DEFAULTS = {
    "EDGE_DROID_MODEL_PATH": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid",
    "BASE_CHECKPOINT_PATH": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid-dcp",
    "WAN_VAE_PATH": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth",
    "SINGLERIGHTHAND_RAW_ROOT": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100",
    "SINGLERIGHTHAND_CACHE_ROOT": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache",
    "FK_ANNOTATION_ROOT": "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/sandwich_fk21",
}

# The target std only has to be *comparable* to the unit-variance noise for the
# two terms to train together; requiring an exact match would make this fail on
# every data-selection change for no reason.  A 2x band catches the ported
# default (a 13x error) and every plausible drift of the selection, while
# tolerating a deliberate, mild reweighting.
TOLERANCE = 2.0


def main() -> int:
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault(
        "SINGLERIGHTHAND_EPISODE_ALLOWLIST", str(REPO / "examples/pointflow_sandwich_10_episodes.txt")
    )
    os.environ.setdefault(
        "SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT",
        os.path.join(DEFAULTS["SINGLERIGHTHAND_CACHE_ROOT"], "vae_window_latents"),
    )

    from omegaconf import OmegaConf

    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_singlerighthand_edge import (
        action_policy_singlerighthand_edge as cfg,
    )

    # Resolve exactly as the model reads it, rather than reading the source line.
    rf = OmegaConf.to_container(cfg["model"]["config"], resolve=True)["rectified_flow_training_config"]
    configured = float(rf["fk_displacement_scale"])
    print(f"配置值 fk_displacement_scale = {configured}")

    root = Path(os.environ["FK_ANNOTATION_ROOT"])
    if not root.is_dir():
        print(f"⏭  跳过：没有 FK 标注目录 {root}")
        return 0

    import scan_fk_displacement_scale as scan

    episodes = [
        line.strip()
        for line in Path(os.environ["SINGLERIGHTHAND_EPISODE_ALLOWLIST"]).read_text().split()
        if line.strip()
    ]
    found = [e for e in episodes if (root / e / "annotations" / "wuji_fk21.npz").is_file()]
    if not found:
        print(f"⏭  跳过：{root} 下没有 allowlist 里任何一个 episode 的标注")
        return 0

    import numpy as np

    elements = []
    for episode in found:
        with np.load(root / episode / "annotations" / "wuji_fk21.npz", allow_pickle=True) as store:
            positions = np.asarray(store["positions"])[:, scan.HAND_INDEX]
        elements.extend(d.reshape(-1) for d in scan.windows(positions))
    measured = float(np.concatenate(elements).std())
    ratio = configured / measured

    print(f"实测 per-element std        = {measured:.6f} m  ({len(found)} 个 episode)")
    print(f"比值 configured / measured  = {ratio:.3f}   （容差 {TOLERANCE}x）")

    if not (1 / TOLERANCE <= ratio <= TOLERANCE):
        print()
        print("❌ FK 的干净目标和单位方差噪声不在同一尺度上。")
        print("   这不会让任何东西报错，只会让分支几乎全程在学预测噪声 —— 典型症状是")
        print("   训练 loss 一路降到极小，而 ADE 卡在零运动基线的 ~20% 再也下不去。")
        print(f"   把 fk_displacement_scale 改成约 {measured:.6f}（或重跑 tools/scan_fk_displacement_scale.py）。")
        return 1

    print("✅ 目标与噪声同尺度")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
