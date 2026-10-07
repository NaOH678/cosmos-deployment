# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge-DROID policy SFT for the single-right-hand sandwich data."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.gpu_video_augmentation import GPUVideoAugmentation
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_singlerighthand_raw_sft_dataset,
)
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L

action_policy_singlerighthand_edge = copy.deepcopy(action_policy_droid_nano)
action_policy_singlerighthand_edge["job"].update(
    project="cosmos3_action",
    group="action_sft",
    name="action_policy_singlerighthand_edge",
    wandb_mode="disabled",
)

model_config = copy.deepcopy(EDGE_MODEL_CONFIG)
model_config["tokenizer"]["encode_exact_durations"] = [33]
model_config["max_num_tokens_after_packing"] = -1
model_config["rectified_flow_training_config"]["loss_scale"] = 10.0
model_config["vlm_config"]["tokenizer"] = L(build_processor_lazy)(tokenizer_type="${oc.env:EDGE_DROID_MODEL_PATH}")
model_config["vlm_config"]["pretrained_weights"].update(
    enabled=False,
    backbone_path="",
    credentials_path="",
    enable_gcs_patch_in_boto3=False,
)
action_policy_singlerighthand_edge["model"]["config"] = model_config

# Edge uses an additional K-normalization parameter on the generation pathway.
optimizer_keys = action_policy_singlerighthand_edge["optimizer"]["keys_to_select"]
if "k_norm_und_for_gen" not in optimizer_keys:
    optimizer_keys.append("k_norm_und_for_gen")

# Edge-DROID already contains trained action heads. Preserve all of them when
# loading the converted DCP; only EMA is warm-started from the regular network.
action_policy_singlerighthand_edge["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
action_policy_singlerighthand_edge["checkpoint"]["enable_gcs_patch_in_boto3"] = False

action_policy_singlerighthand_edge["trainer"]["callbacks"]["gpu_video_augmentation"] = L(GPUVideoAugmentation)(
    enabled="${oc.env:COSMOS_GPU_VIDEO_AUGMENTATION,false}",
    crop_ratio=0.95,
    brightness=0.3,
    contrast=0.4,
    saturation=0.5,
    hue=0.08,
    chunk_size=8,
)

action_policy_singlerighthand_edge["dataloader_train"]["dataset_name"] = "action_singlerighthand"
dataloader = action_policy_singlerighthand_edge["dataloader_train"]["dataloader"]
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
            # Precomputed per-window VAE latents.  Without this every step re-runs
            # the frozen Wan2.2 VAE over every pixel frame of every window.  The
            # dataset validates the manifest's fps/chunk_length/sample_stride
            # against its own, so a stale cache is refused rather than misread.
            #
            # Only the window cache: the whole-episode cache is not equivalent
            # (see SingleRightHandRawDataset._read_window_latent).
            vae_window_latent_root="${oc.env:SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT,}",
            # Legacy whole-episode cache: a whole-episode encode at source FPS that is
            # resampled onto the window's 15 Hz lattice.  Measured to differ from a
            # fresh encode by about one frame of window shift, and it changes the
            # identity of the conditioning latent (index 0).  Used only when the
            # window cache above is unset; prefer the window cache.
            vae_latent_root="${oc.env:SINGLERIGHTHAND_VAE_LATENT_ROOT,}",
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
    name="action_policy_singlerighthand_edge",
    node=action_policy_singlerighthand_edge,
)


__all__ = ["action_policy_singlerighthand_edge"]
