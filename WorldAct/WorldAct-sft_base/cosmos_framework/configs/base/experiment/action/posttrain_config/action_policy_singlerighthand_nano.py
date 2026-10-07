# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano-Policy-DROID SFT for the single-right-hand sandwich data."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.gpu_video_augmentation import GPUVideoAugmentation
from cosmos_framework.configs.base.defaults.reasoner import create_qwen2_tokenizer_with_download
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_singlerighthand_raw_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

action_policy_singlerighthand_nano = copy.deepcopy(action_policy_droid_nano)
action_policy_singlerighthand_nano["job"].update(
    project="cosmos3_action",
    group="action_sft",
    name="action_policy_singlerighthand_nano",
    wandb_mode="disabled",
)

model_config = action_policy_singlerighthand_nano["model"]["config"]
model_config["tokenizer"]["encode_exact_durations"] = [33]
model_config["max_num_tokens_after_packing"] = -1
model_config["rectified_flow_training_config"]["loss_scale"] = 10.0
model_config["vlm_config"]["tokenizer"] = L(create_qwen2_tokenizer_with_download)(
    pretrained_model_name="${oc.env:NANO_DROID_MODEL_PATH}",
    config_variant="hf",
)
model_config["vlm_config"]["pretrained_weights"].update(
    enabled=False,
    backbone_path="",
    credentials_path="",
    enable_gcs_patch_in_boto3=False,
)

# Nano-Policy-DROID already contains trained action heads. Load them from the
# converted DCP and warm-start only the training-time EMA copy from net.
action_policy_singlerighthand_nano["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
action_policy_singlerighthand_nano["checkpoint"]["enable_gcs_patch_in_boto3"] = False

action_policy_singlerighthand_nano["trainer"]["callbacks"]["gpu_video_augmentation"] = L(GPUVideoAugmentation)(
    enabled="${oc.env:COSMOS_GPU_VIDEO_AUGMENTATION,false}",
    crop_ratio=0.95,
    brightness=0.3,
    contrast=0.4,
    saturation=0.5,
    hue=0.08,
    chunk_size=8,
)

action_policy_singlerighthand_nano["dataloader_train"]["dataset_name"] = "action_singlerighthand"
dataloader = action_policy_singlerighthand_nano["dataloader_train"]["dataloader"]
dataloader.update(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=3,
)
dataloader["datasets"] = {
    "singlerighthand": {
        "ratio": 1,
        "dataset": L(get_action_singlerighthand_raw_sft_dataset)(
            root="${oc.env:SINGLERIGHTHAND_RAW_ROOT}",
            cache_root="${oc.env:SINGLERIGHTHAND_CACHE_ROOT}",
            fps=15.0,
            chunk_length=32,
            split="train",
            split_seed=42,
            split_val_ratio=0.03,
            sample_stride=1,
            mode="wam",
            use_state=True,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            shuffle_block_size=256,
            use_image_augmentation=False,
            use_precomputed_video="${oc.env:SINGLERIGHTHAND_USE_VIDEO_CACHE,false}",
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
    name="action_policy_singlerighthand_nano",
    node=action_policy_singlerighthand_nano,
)


__all__ = ["action_policy_singlerighthand_nano"]
