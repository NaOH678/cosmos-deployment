"""Single-task, 52-joint, three-camera Bench2Dex Edge SFT."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_singlerighthand_edge import (
    action_policy_singlerighthand_edge,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_bench2dex_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

action_policy_bench2dex_edge = copy.deepcopy(action_policy_singlerighthand_edge)
cfg = action_policy_bench2dex_edge
cfg["job"]["name"] = "action_policy_bench2dex_edge"
cfg["dataloader_train"]["dataset_name"] = "action_bench2dex"
cfg["dataloader_train"]["dataloader"]["datasets"] = {
    "bench2dex": {
        "ratio": 1,
        "dataset": L(get_action_bench2dex_sft_dataset)(
            cache_root="${oc.env:BENCH2DEX_CACHE_ROOT}",
            vae_window_latent_root="${oc.env:BENCH2DEX_LATENT_ROOT,null}",
            vae_path="${oc.env:WAN_VAE_PATH,null}",
            action_stats_path="${oc.env:BENCH2DEX_ACTION_STATS_PATH,null}",
            action_stats_sha256="${oc.env:BENCH2DEX_ACTION_STATS_SHA256,null}",
            fps=20.0,
            chunk_length=32,
            split="train",
            split_seed=42,
            split_val_ratio=0.1,
            sample_stride=1,
            mode="wam",
            use_state=True,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            shuffle_block_size=256,
            viewpoint="concat_view",
            resolution="480",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            format_prompt_as_json=True,
        ),
    }
}
ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_bench2dex_edge",
    node=cfg,
)
