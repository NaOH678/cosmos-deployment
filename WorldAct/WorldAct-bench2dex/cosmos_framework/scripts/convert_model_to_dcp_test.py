# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import json

from cosmos_framework.scripts.convert_model_to_dcp import (
    _redirect_processor_to_local,
    _redirect_video_vae_to_file,
    _redirect_video_vae_to_local,
)


def test_redirect_processor_to_local_for_edge_bundle(tmp_path):
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_class": "Cosmos3EdgeProcessor"}), encoding="utf-8"
    )
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    tokenizer_config = {
        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
        "config_variant": "hf",
        "repository": None,
        "revision": None,
        "subdir": "",
        "tokenizer_type": "nvidia/Cosmos3-Edge-Policy-DROID",
    }
    model_dict = {"config": {"vlm_config": {"tokenizer": tokenizer_config}}}

    assert _redirect_processor_to_local(model_dict, tmp_path)
    assert tokenizer_config == {
        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
        "config_variant": "hf",
        "tokenizer_type": str(tmp_path),
    }


def test_redirect_processor_to_local_leaves_incomplete_bundle_unchanged(tmp_path):
    tokenizer_config = {
        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
        "config_variant": "hf",
        "tokenizer_type": "nvidia/Cosmos3-Edge-Policy-DROID",
    }
    original = copy.deepcopy(tokenizer_config)
    model_dict = {"config": {"vlm_config": {"tokenizer": tokenizer_config}}}

    assert not _redirect_processor_to_local(model_dict, tmp_path)
    assert tokenizer_config == original


def test_redirect_video_vae_to_local(tmp_path):
    vae_path = tmp_path / "vae" / "Wan2.2_VAE.pth"
    vae_path.parent.mkdir()
    vae_path.write_bytes(b"stub")
    tokenizer_config = {
        "bucket_name": "remote-bucket",
        "vae_path": "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
    }
    model_dict = {"config": {"tokenizer": tokenizer_config}}

    assert _redirect_video_vae_to_local(model_dict, tmp_path)
    assert tokenizer_config["vae_path"] == str(vae_path)
    assert tokenizer_config["bucket_name"] == ""


def test_redirect_video_vae_to_local_leaves_missing_bundle_unchanged(tmp_path):
    tokenizer_config = {
        "bucket_name": "",
        "vae_path": "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
    }
    original = copy.deepcopy(tokenizer_config)
    model_dict = {"config": {"tokenizer": tokenizer_config}}

    assert not _redirect_video_vae_to_local(model_dict, tmp_path)
    assert tokenizer_config == original


def test_redirect_video_vae_to_explicit_file(tmp_path):
    vae_path = tmp_path / "shared" / "Wan2.2_VAE.pth"
    vae_path.parent.mkdir()
    vae_path.write_bytes(b"stub")
    tokenizer_config = {
        "bucket_name": "remote-bucket",
        "object_store_credential_path_pretrained": "credentials.secret",
        "vae_path": "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
    }
    model_dict = {"config": {"tokenizer": tokenizer_config}}

    assert _redirect_video_vae_to_file(model_dict, vae_path)
    assert tokenizer_config["vae_path"] == str(vae_path)
    assert tokenizer_config["bucket_name"] == ""
    assert tokenizer_config["object_store_credential_path_pretrained"] == ""
