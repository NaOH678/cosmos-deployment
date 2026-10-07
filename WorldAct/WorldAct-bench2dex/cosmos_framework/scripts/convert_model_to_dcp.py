# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a Hugging Face model to a DCP checkpoint."""

from cosmos_framework.inference.common.init import init_script

init_script(
    default_env={
        "COSMOS_DEVICE": "cpu",
    }
)

import math
import shutil
from typing import Annotated

import pydantic
import torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import (
    CheckpointOverrides,
    ResolvedFilePath,
    ResolvedPath,
)
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

_AVAE_REGISTRY_URI = "s3://bucket/pretrained/tokenizers/audio/avae"


def _redirect_avae_to_local(hf_path):
    """Point the AVAE registry entry at hf_path/sound_tokenizer/.

    Pre-seeds the CheckpointDirHf._path cache so the registered AVAE checkpoint
    resolves to the local sibling directory instead of fetching the pinned
    revision of nvidia/Cosmos3-Nano from the HF Hub during hydra instantiation.
    """
    sound_tokenizer_dir = hf_path / "sound_tokenizer"
    if not sound_tokenizer_dir.is_dir():
        return
    register_checkpoints()
    avae = _CHECKPOINTS.get(_AVAE_REGISTRY_URI)
    if avae is not None:
        avae.hf._path = str(sound_tokenizer_dir)


def _redirect_processor_to_local(model_dict: dict, hf_path) -> bool:
    """Use processor files bundled with a local checkpoint when compatible."""
    from cosmos_framework.inference.inference import (
        _bundled_processor_serves_node,
        _checkpoint_has_processor_files,
        _point_tokenizer_node_at_dir,
    )

    tokenizer_config = model_dict.get("config", {}).get("vlm_config", {}).get("tokenizer")
    if not isinstance(tokenizer_config, dict):
        return False
    if not _checkpoint_has_processor_files(hf_path):
        return False
    if not _bundled_processor_serves_node(tokenizer_config, hf_path):
        return False
    return _point_tokenizer_node_at_dir(tokenizer_config, hf_path)


def _redirect_video_vae_to_local(model_dict: dict, hf_path) -> bool:
    """Use a checkpoint-bundled Wan VAE instead of the download registry."""
    tokenizer_config = model_dict.get("config", {}).get("tokenizer")
    if not isinstance(tokenizer_config, dict):
        return False
    configured_path = tokenizer_config.get("vae_path")
    if not isinstance(configured_path, str) or not configured_path:
        return False

    file_name = configured_path.rsplit("/", 1)[-1]
    for candidate in (hf_path / "vae" / file_name, hf_path / file_name):
        if candidate.is_file():
            return _redirect_video_vae_to_file(model_dict, candidate)
    return False


def _redirect_video_vae_to_file(model_dict: dict, vae_path) -> bool:
    """Point the model's video tokenizer at an explicit local Wan VAE file."""
    tokenizer_config = model_dict.get("config", {}).get("tokenizer")
    if not isinstance(tokenizer_config, dict) or not vae_path.is_file():
        return False
    tokenizer_config["vae_path"] = str(vae_path)
    tokenizer_config["bucket_name"] = ""
    tokenizer_config["object_store_credential_path_pretrained"] = ""
    return True


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Hugging Face checkpoint."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output DCP checkpoint directory."""
    video_vae_path: ResolvedFilePath | None = None
    """Optional local Wan2.2_VAE.pth used while instantiating the model."""


def convert_model_to_dcp(args: Args):
    print("Loading model...")
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    hf_path = checkpoint_config.download_checkpoint()
    _redirect_avae_to_local(hf_path)
    model_dict = checkpoint_config.load_model_config_dict()
    if _redirect_processor_to_local(model_dict, hf_path):
        print(f"Using checkpoint-bundled processor files at {hf_path}")
    if args.video_vae_path is not None:
        if not _redirect_video_vae_to_file(model_dict, args.video_vae_path):
            raise ValueError("Checkpoint model config does not expose a video tokenizer")
        print(f"Using explicit local video VAE at {args.video_vae_path}")
    elif _redirect_video_vae_to_local(model_dict, hf_path):
        print(f"Using checkpoint-bundled video VAE at {model_dict['config']['tokenizer']['vae_path']}")
    hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
    hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path, config=hf_config)
    state_dict = get_model_state_dict(hf_model.model)

    # Match transformers default max shard size = 5GB.
    max_shard_size = 5 * 1024**3
    model_size = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor))
    thread_count = math.ceil(model_size / max_shard_size)

    print("Saving model...")
    storage_writer = FileSystemWriter(args.output_path / "model", thread_count=thread_count)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
    # ``checkpoint.json`` only exists for DCP-format source repos (e.g. Cosmos3-Nano);
    # safetensors/diffusers-layout repos (e.g. Cosmos3-Edge) don't ship it. Copy when present.
    source_checkpoint_json = hf_path / "checkpoint.json"
    if source_checkpoint_json.exists():
        shutil.copy(source_checkpoint_json, args.output_path / "checkpoint.json")
    hf_config.save_pretrained(args.output_path / "model")
    print(f"Saved checkpoint to {args.output_path}")


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_dcp(args)


if __name__ == "__main__":
    main()
