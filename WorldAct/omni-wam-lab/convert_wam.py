"""Lossless key conversion to Omni, checked against the local Edge component schema."""

import argparse
import json
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from cosmos_framework.scripts._convert_model_to_diffusers import (
    _remap_language_model_key,
    _remap_time_embedder_key,
)


def map_key(key):
    key = key.removeprefix("model.net.")
    if key.startswith("language_model."):
        return _remap_language_model_key(key.removeprefix("language_model."))
    for src, dst in [
        ("vae2llm.", "proj_in."),
        ("llm2vae.", "proj_out."),
        ("action2llm.", "action_proj_in."),
        ("llm2action.", "action_proj_out."),
    ]:
        if key.startswith(src):
            return dst + key.removeprefix(src)
    if key.startswith("time_embedder."):
        return "time_embedder." + _remap_time_embedder_key(
            key.removeprefix("time_embedder.")
        )
    if key == "action_modality_embed":
        return key
    raise ValueError(f"Unmapped weight: {key}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--base-package", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source_cfg = json.loads((a.source / "config.json").read_text())["model"]["config"]
    base_cfg = json.loads((a.base_package / "config.json").read_text())["model"][
        "config"
    ]
    for name in [
        "state_ch",
        "max_action_dim",
        "num_embodiment_domains",
        "action_gen",
        "sound_gen",
        "joint_attn_implementation",
        "video_temporal_causal",
        "enable_input_bias",
        "latent_downsample_factor",
    ]:
        if source_cfg[name] != base_cfg[name]:
            raise ValueError(f"Base component configuration differs: {name}")
    for name, value in source_cfg["diffusion_expert_config"].items():
        if name.startswith("_") or name == "load_weights_from_pretrained":
            continue
        if base_cfg["diffusion_expert_config"].get(name) != value:
            raise ValueError(f"Diffusion config differs: {name}")
    for name in ["qk_norm_for_text", "use_und_k_norm_for_gen", "base_config"]:
        left = source_cfg["vlm_config"]["model_instance"]["config"][name]
        right = base_cfg["vlm_config"]["model_instance"]["config"][name]
        if left != right:
            raise ValueError(f"Language config differs: {name}")
    base = a.base_package / "cosmos3-edge-droid"
    index = json.loads((a.source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    template = json.loads(
        (
            base / "transformer/diffusion_pytorch_model.safetensors.index.json"
        ).read_text()
    )["weight_map"]
    mapped = {map_key(k): k for k in index}
    if len(mapped) != len(index) or set(mapped) != set(template):
        raise ValueError(
            f"Weight coverage differs: missing={set(template) - set(mapped)}, extra={set(mapped) - set(template)}"
        )
    shapes = {}
    for shard in set(template.values()):
        with safe_open(base / "transformer" / shard, framework="pt") as f:
            shapes.update({k: f.get_slice(k).get_shape() for k in f.keys()})
    out = a.output / "transformer"
    out.mkdir(parents=True, exist_ok=True)
    output_map = {}
    nbytes = 0
    for i, shard in enumerate(sorted(set(index.values()))):
        tensors = {}
        with safe_open(a.source / shard, framework="pt") as f:
            for key in f.keys():
                dest = map_key(key)
                tensor = f.get_tensor(key)
                if list(tensor.shape) != shapes[dest]:
                    raise ValueError(f"Shape mismatch: {key}")
                tensors[dest] = tensor.contiguous()
                nbytes += tensor.numel() * tensor.element_size()
        filename = f"diffusion_pytorch_model-{i + 1:05d}-of-{len(set(index.values())):05d}.safetensors"
        save_file(tensors, out / filename, metadata={"format": "pt"})
        with safe_open(out / filename, framework="pt") as f:
            for key, tensor in tensors.items():
                if not torch.equal(tensor, f.get_tensor(key)):
                    raise ValueError(f"Serialization changed weight: {key}")
        output_map.update({k: filename for k in tensors})
    (out / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": nbytes}, "weight_map": output_map}, indent=2
        )
    )
    cfg = json.loads((base / "transformer/config.json").read_text())
    cfg.update(
        backbone_type="cosmos3_edge_nemotron_dense",
        temporal_compression_factor=4,
        position_embedding_type="unified_3d_mrope",
        joint_attn_implementation=source_cfg["joint_attn_implementation"],
        vision_temporal_position_mode=source_cfg["diffusion_expert_config"][
            "vision_temporal_position_mode"
        ],
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in ["vae", "text_tokenizer", "scheduler"]:
        target = a.output / name
        if not target.exists():
            target.symlink_to((base / name).resolve(), target_is_directory=True)
    shutil.copy2(a.base_package / "model_index.json", a.output / "model_index.json")
    (a.output / "conversion.json").write_text(
        json.dumps(
            {
                "source": str(a.source.resolve()),
                "base_components": str(base.resolve()),
                "weight_count": len(output_map),
                "all_weights_serialized_exactly": True,
                "base_transformer_weights_used": False,
                "ema_manifest": json.loads((a.source / "wam_export.json").read_text()),
            },
            indent=2,
        )
    )
    print(
        f"Converted and checked {len(output_map)} tensors; no base transformer weights used"
    )


if __name__ == "__main__":
    main()
