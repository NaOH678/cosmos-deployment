"""Small-data simulation fitting with the established cluster/FK local-RoPE model."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_singlerighthand_edge import (
    action_policy_singlerighthand_edge,
)
from cosmos_framework.data.generator.action.datasets.sim_pointfk_dataset import get_sim_pointfk_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

config = copy.deepcopy(action_policy_singlerighthand_edge)
config["job"]["name"] = "action_policy_sim_pointfk_edge"
for split in ["train", "val"]:
    loader = config["dataloader_" + split]
    loader["dataset_name"] = "sim_pointfk_" + split
    loader["dataloader"]["datasets"] = {
        "sim_pointfk": dict(
            ratio=1,
            dataset=L(get_sim_pointfk_sft_dataset)(
                bundle="${oc.env:SIM_POINTFK_BUNDLE}",
                selection_root="${oc.env:SIM_POINTFK_SELECTION_ROOT}",
                split=split,
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                cfg_dropout_rate=0.1 if split == "train" else 0.0,
                max_action_dim=64,
                iterable_shuffle=split == "train",
                use_image_augmentation=False,
                cached_video_only=split == "train",
            ),
        )
    }
config["dataloader_train"]["dataloader"].update(num_workers=4, in_order=True)
config["trainer"].update(run_validation_on_start=True, validation_iter=500)
config["trainer"]["callbacks"]["pointflow_eval"].update(val_episodes=2, val_windows=4, max_points=1024)
config["trainer"]["callbacks"]["fk_rollout"]["enabled"] = False
ConfigStore.instance().store(group="experiment", package="_global_", name="action_policy_sim_pointfk_edge", node=config)
