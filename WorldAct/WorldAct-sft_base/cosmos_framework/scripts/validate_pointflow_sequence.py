"""Task 4 CUDA check: real geometry -> Cosmos Edge attention -> point velocity."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.data.pointflow_dataset import PointFlowWindowDataset, collate_pointflow_windows
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, PackedAttentionMoT
from cosmos_framework.model.generator.pointflow_codec import PointFlowCodec
from cosmos_framework.model.generator.pointflow_geometry import SonataGeometryEncoder
from cosmos_framework.model.generator.pointflow_sequence import attach_point_tokens, point_positions
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import MultiModalRotaryEmbedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires a GPU; use pointflow_sequence_test for CPU verification")
    torch.manual_seed(args.seed)
    timing = PointFlowTiming()
    samples = []
    for episode, start, budget in [
        ("episode_0013_20260731_133649", 596, 512),
        ("episode_0015_20260730_170501", 1120, 256),
        ("episode_0014_20260731_133743", 596, 256),
    ]:
        samples.append(
            PointFlowWindowDataset(args.data_root, [episode], timing=timing, max_points=budget, seed=args.seed)[start]
        )
    inputs = {k: v.cuda() for k, v in collate_pointflow_windows(samples)["inputs"].items()}
    encoder = SonataGeometryEncoder(args.checkpoint, stage=args.stage).cuda().eval()
    codec = PointFlowCodec(timing=timing, geometry_dim=encoder.output_dim).cuda()
    geometry = encoder(inputs)
    noise = torch.randn(timing.steps, len(inputs["point_ids"]), 3, device="cuda", requires_grad=True)
    sigma = torch.tensor([0.2, 0.5, 0.8], device="cuda")
    tokens = codec(geometry, noise, sigma)

    # Synthetic video/action content; actual Cosmos packing, Edge mRoPE and MoT attention.
    # 2 text + 9 video (one spatial location each) + 32 action tokens per sample.
    def idx(values):
        return torch.tensor(values, device="cuda", dtype=torch.long)

    base = PackedSequence(
        sample_lens=[43] * 3,
        split_lens=[2, 41] * 3,
        attn_modes=["causal", "full"] * 3,
        sequence_length=129,
        text_ids=idx([1, 2] * 3),
        text_indexes=idx([b * 43 + i for b in range(3) for i in range(2)]),
        position_ids=torch.zeros(3, 129, device="cuda"),
        vision=ModalityData(sequence_indexes=idx([b * 43 + i for b in range(3) for i in range(2, 11)])),
        action=ModalityData(sequence_indexes=idx([b * 43 + i for b in range(3) for i in range(11, 43)])),
    )
    for b in range(3):
        base.position_ids[:, b * 43 : b * 43 + 2] = torch.arange(2, device="cuda")[None]
        base.position_ids[0, b * 43 + 2 : b * 43 + 11] = 2 + torch.arange(9, device="cuda") * 1.6
        base.position_ids[0, b * 43 + 11 : b * 43 + 43] = 2 + torch.arange(32, device="cuda") * 0.4
    # Explicit synthetic affine for this probe, NOT a claimed raw-video transform.
    affine = torch.tensor([[[1 / 32, 0, 0], [0, 1 / 32, 7]]] * 3, device="cuda")
    positions = point_positions(
        geometry["cluster_uv"],
        geometry["cluster_batch"],
        affine,
        torch.full((3,), 2.0, device="cuda"),
        torch.arange(1, 9, device="cuda") * timing.steps_per_token / timing.fps,
    )
    seq = attach_point_tokens(base, tokens, positions)
    hidden = torch.randn(seq.sequence_length, 2048, device="cuda")
    hidden[seq.point.sequence_indexes] = seq.point.tokens
    pack, meta, _ = build_packed_sequence(
        "two_way",
        packed_sequence=hidden,
        attn_modes=seq.attn_modes,
        split_lens=seq.split_lens,
        sample_lens=seq.sample_lens,
        packed_und_token_indexes=seq.text_indexes,
        packed_gen_token_indexes=seq.point.sequence_indexes,
        num_heads=16,
        head_dim=128,
        num_layers=1,
    )
    modalities = torch.zeros(seq.sequence_length, dtype=torch.long, device="cuda")
    modalities[seq.action.sequence_indexes] = 1
    modalities[seq.point.sequence_indexes] = 2
    meta.pointflow_modalities = modalities
    config = Nemotron3DenseVLTextConfig()
    layer = PackedAttentionMoT(
        config,
        layer_idx=0,
        layer_types=LayerTypes("nemotron_dense"),
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
    ).cuda()
    rotary = MultiModalRotaryEmbedding(config).cuda()
    cos, sin = rotary(hidden, seq.position_ids[:, None])
    output, _ = layer(pack, meta, (from_all_seq(cos[0], pack), from_all_seq(sin[0], pack)))
    point_hidden = seq.point.hidden(get_all_seq(output))
    velocity = codec.decode(geometry, inputs, point_hidden, noise, sigma)
    assert velocity.shape == noise.shape and torch.isfinite(velocity).all()
    loss = velocity.square().mean() + tokens["anchor_tokens"].square().mean()
    loss.backward()
    for grad in (
        noise.grad,
        layer.q_proj_moe_gen.weight.grad,
        encoder.backbone.embedding.stem.linear.weight.grad,
        codec.motion_encoder[0].weight.grad,
        codec.decoder[-1].weight.grad,
    ):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
    report = {
        "task": 4,
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "point_offsets": inputs["point_offsets"].tolist(),
        "cluster_offsets": geometry["cluster_offsets"].tolist(),
        "anchor_shape": list(tokens["anchor_tokens"].shape),
        "noisy_token_shape": list(tokens["noisy_tokens"].shape),
        "velocity_shape": list(velocity.shape),
        "cosmos_attention": "PASS: real PackedAttentionMoT, random one-layer weights",
        "sample_lens": seq.sample_lens,
        "point_hidden_shape": list(point_hidden.shape),
        "video_action": "synthetic tokens; no base checkpoint or joint training",
        "position_transform": "explicit synthetic affine, not dataset transform validation",
        "loss": "synthetic gradient probe, not training loss",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
