# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_singlerighthand_edge import (
    action_policy_singlerighthand_edge,
)
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_singlerighthand_nano import (
    action_policy_singlerighthand_nano,
)


def test_nano_recipe_is_independent_from_edge_and_base_recipes():
    assert action_policy_singlerighthand_nano["job"]["name"] == "action_policy_singlerighthand_nano"
    assert action_policy_singlerighthand_nano["model"]["config"]["vlm_config"]["model_name"] == (
        "Qwen/Qwen3-VL-8B-Instruct"
    )
    assert action_policy_singlerighthand_nano["checkpoint"]["keys_to_skip_loading"] == ["net_ema."]

    assert action_policy_singlerighthand_edge["job"]["name"] == "action_policy_singlerighthand_edge"
    assert action_policy_droid_nano["job"]["name"] == "action_policy_droid_nano"
    assert "action2llm" in action_policy_droid_nano["checkpoint"]["keys_to_skip_loading"]


def test_nano_recipe_uses_local_policy_tokenizer_and_single_hand_data():
    tokenizer = action_policy_singlerighthand_nano["model"]["config"]["vlm_config"]["tokenizer"]
    assert tokenizer._get_node("pretrained_model_name")._value() == "${oc.env:NANO_DROID_MODEL_PATH}"
    assert tokenizer["config_variant"] == "hf"

    dataloader = action_policy_singlerighthand_nano["dataloader_train"]["dataloader"]
    dataset = dataloader["datasets"]["singlerighthand"]["dataset"]
    assert dataset["chunk_length"] == 32
    assert dataset["use_image_augmentation"] is False
    assert dataset._get_node("use_precomputed_video")._value() == ("${oc.env:SINGLERIGHTHAND_USE_VIDEO_CACHE,false}")
