"""Task 7: full Cosmos3VFMNetwork entry with real one-layer MoT and Sonata."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cosmos_framework.data.generator.action.pointflow_source import PointFlowSource
from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq
from cosmos_framework.data.pointflow_batch import build_pointflow_batch
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, PackedAttentionMoT
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import MultiModalRotaryEmbedding
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class OneLayerLanguageModel(nn.Module):
    """Validation harness: real PackedAttentionMoT, random weights, no base checkpoint."""

    def __init__(self, hidden=64):
        super().__init__()
        if hidden == 2048:
            config = Nemotron3DenseVLTextConfig(num_hidden_layers=1)
        else:
            config = Nemotron3DenseVLTextConfig(
                hidden_size=hidden,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=hidden // 4,
                mrope_section=[4, 2, 2],
            )
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, hidden)
        self.layer = PackedAttentionMoT(
            config,
            layer_idx=0,
            layer_types=LayerTypes("nemotron_dense"),
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
            use_und_k_norm_for_gen=True,
        )
        self.rotary = MultiModalRotaryEmbedding(config)

    def forward(self, pack, attention_mask, position_ids, **kwargs):
        hidden = get_all_seq(pack)
        cos, sin = self.rotary(hidden, position_ids[:, None])
        result, _ = self.layer(pack, attention_mask, (from_all_seq(cos[0], pack), from_all_seq(sin[0], pack)))
        return result, {}


def make_network(hidden=64):
    language = OneLayerLanguageModel(hidden)
    config = Cosmos3VFMNetworkConfig(
        vlm_config=language.config,
        vision_gen=True,
        action_gen=True,
        latent_channel_size=4,
        latent_patch_size=1,
        latent_downsample_factor=8,
        action_dim=32,
        num_embodiment_domains=32,
        enable_fps_modulation=True,
    )
    return Cosmos3VFMNetwork(language, config)


def make_sequence(point):
    b = len(point.labeled)
    clean = GenerationDataClean(
        batch_size=b,
        is_image_batch=False,
        pointflow=point,
        x0_tokens_vision=[torch.randn(1, 4, 9, 2, 2) for _ in range(b)],
        x0_tokens_action=[torch.randn(32, 32) for _ in range(b)],
        fps_vision=torch.full((b,), point.timing.fps),
        fps_action=torch.full((b,), point.timing.fps),
    )
    plans = [
        SequencePlan(
            has_text=True, has_vision=True, has_action=True, has_point=bool(flag), condition_frame_indexes_vision=[0]
        )
        for flag in point.has_point
    ]
    return pack_input_sequence(
        plans,
        [[1, 2]] * b,
        clean,
        torch.full((b,), 500.0),
        {"eos_token_id": 3, "start_of_generation": 4, "end_of_generation": 5},
        enable_fps_modulation=True,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Task 7 real Sonata verification requires CUDA; run pointflow_branch_test.py on CPU")
    torch.manual_seed(args.seed)
    source = PointFlowSource(args.manifest, timing=PointFlowTiming(), max_points=256, seed=args.seed)
    name, row = next((name, row) for name, row in source.entries.items() if row is not None)
    point = source.load(name, np.arange(0, 65, 2), 30, row["video_size_wh"])
    # Match the synthetic 2x2 latent grid (8 pixel stride) with an explicit resize.
    from cosmos_framework.data.generator.action.pointflow_source import resize_pointflow_metadata

    resize_pointflow_metadata(point, row["video_size_wh"], (16, 16), (16, 16))
    batch = build_pointflow_batch([point, None, point])
    sequence = make_sequence(batch)
    sequence.to_cuda()
    net = make_network(2048).cuda().eval()
    base_weight = net.vae2llm.weight.detach().clone()
    net.install_pointflow(args.checkpoint, timing=batch.timing)
    torch.testing.assert_close(base_weight, net.vae2llm.weight)
    noise = torch.randn_like(batch.displacement, requires_grad=True)
    sigma = torch.tensor([0.2, 0.5, 0.8], device="cuda")
    output = net(sequence, pointflow_displacement=noise, pointflow_sigma=sigma)
    velocity = output["preds_pointflow"]
    assert velocity.shape == noise.shape and torch.isfinite(velocity).all()
    loss = velocity.square().mean()
    loss.backward()
    for label, parameter in [
        ("noise", noise),
        ("sonata", net.pointflow_branch.geometry.backbone.embedding.stem.linear.weight),
        ("codec", net.pointflow_branch.codec.motion_encoder[0].weight),
        ("attention", net.language_model.layer.q_proj_moe_gen.weight),
        ("video", net.vae2llm.weight),
        ("action", net.action2llm.fc.weight),
    ]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, (
            label
        )
    report = {
        "task": 7,
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "velocity_shape": list(velocity.shape),
        "point_hidden_shape": list(output["point_hidden"].shape),
        "cluster_offsets": output["pointflow_cluster_offsets"].tolist(),
        "video_outputs": len(output["preds_vision"]),
        "action_outputs": len(output["preds_action"]),
        "base_weight_preserved": True,
        "gradient": "point prediction reaches Sonata, Codec, attention, video/action inputs",
        "scope": "real Cosmos3VFMNetwork + real one-layer attention + pretrained Sonata; synthetic video/action latents; no Cosmos base checkpoint, joint loss or FSDP",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
