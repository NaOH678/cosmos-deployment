"""Full Cosmos3VFMNetwork entry with the FK branch installed, one-layer MoT, no checkpoint.

The gate the branch-level smoke test cannot reach: ``install_fk`` on a real
network, a real packed sequence, a forward that produces ``preds_fk``, and a
backward whose gradient reaches the encoder, the decoder, the shared attention,
and the video/action input projections.

Why it needs no base checkpoint: FK has no pretrained part -- the index
embeddings, the position MLP and the decoder are all new -- so unlike the
PointFlow gate there is nothing to load and nothing to accidentally destroy.
``base_weight_preserved`` is still asserted, because ``install_fk`` runs after
materialization and must not disturb the network it is attaching to.

Run on a CUDA node from the worktree root:

    PYTHONPATH=. <venv>/bin/python cosmos_framework/scripts/validate_fk_network.py \
        --annotation-root /mnt/.../raw_data/sandwich_fk21 --output /tmp/fk_network
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cosmos_framework.data.fk_batch import build_fk_batch
from cosmos_framework.data.fk_window import FKTiming
from cosmos_framework.data.generator.action.fk_source import FKSource
from cosmos_framework.data.generator.sequence_packing import SequencePlan, pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, PackedAttentionMoT
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import MultiModalRotaryEmbedding
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

EPISODES = ("episode_0013_20260731_133649", "episode_0014_20260731_133743")
KEYPOINTS = 21


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


def make_sequence(fk):
    """One sample per slot, video and action kept simple and synthetic.

    The FK payload is the only thing under test; the video/action latents exist so
    the sequence has the two-way shape FK requires and so its gradient can be
    checked for collateral damage.
    """
    b = len(fk.labeled)
    clean = GenerationDataClean(
        batch_size=b,
        is_image_batch=False,
        fk=fk,
        x0_tokens_vision=[torch.randn(1, 4, 9, 2, 2) for _ in range(b)],
        x0_tokens_action=[torch.randn(32, 32) for _ in range(b)],
        fps_vision=torch.full((b,), fk.timing.fps),
        fps_action=torch.full((b,), fk.timing.fps),
    )
    # ``SequencePlan`` in this worktree has no ``has_point``: v1 packs FK from
    # ``gen_data_clean.fk`` unconditionally, so there is no per-sample flag to set.
    plans = [
        SequencePlan(has_text=True, has_vision=True, has_action=True, condition_frame_indexes_vision=[0])
        for _ in range(b)
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
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This validates the real network and needs CUDA; run fk_branch_test.py on CPU instead")
    torch.manual_seed(args.seed)

    timing = FKTiming()
    source = FKSource(args.annotation_root, timing=timing)
    frame_ids = np.arange(0, 2 * (timing.steps + 1), 2)
    batch = build_fk_batch([source.load(name, frame_ids) for name in EPISODES], batch_size=len(EPISODES))
    print(f"FK 批: anchor {tuple(batch.inputs['anchor_xyz'].shape)}  displacement {tuple(batch.displacement.shape)}")

    sequence = make_sequence(batch)
    sequence.to_cuda()
    net = make_network(2048).cuda().eval()
    # install_fk runs after materialization; it must attach without disturbing
    # what is already there.
    base_weight = net.vae2llm.weight.detach().clone()
    net.install_fk(timing=batch.timing)
    torch.testing.assert_close(base_weight, net.vae2llm.weight)
    print("✅ install_fk 完成，且未扰动基础网络权重")

    # float32 on purpose, exactly as fk_add_noise produces it -- the branch casts.
    noisy = torch.randn_like(batch.displacement, requires_grad=True)
    # Spread over the sigma range but sized FROM the batch: a hard-coded list here
    # silently couples this script to the number of episodes above.
    sigma = torch.linspace(0.2, 0.8, len(batch.labeled), device="cuda")
    output = net(sequence, fk_displacement=noisy, fk_sigma=sigma)

    velocity = output["preds_fk"]
    assert velocity.shape == noisy.shape, f"preds_fk {tuple(velocity.shape)} != {tuple(noisy.shape)}"
    assert torch.isfinite(velocity).all()
    print(f"✅ 前向: preds_fk {tuple(velocity.shape)}  fk_hidden {tuple(output['fk_hidden'].shape)}")

    loss = velocity.square().mean()
    loss.backward()
    checks = [
        ("noise", noisy),
        ("fk_encoder", net.fk_branch.encoder.index_embedding.weight),
        ("fk_position", net.fk_branch.encoder.xyz_encoder[0].weight),
        ("fk_motion", net.fk_branch.motion_encoder[0].weight),
        ("fk_decoder", net.fk_branch.decoder[0].weight),
        ("attention", net.language_model.layer.q_proj_moe_gen.weight),
        ("video", net.vae2llm.weight),
        ("action", net.action2llm.fc.weight),
    ]
    for label, parameter in checks:
        grad = parameter.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, f"{label} 没拿到有效梯度"
        print(f"   ✅ {label:12s} grad_sum = {float(grad.abs().sum()):.6g}")

    report = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "preds_fk_shape": list(velocity.shape),
        "fk_hidden_shape": list(output["fk_hidden"].shape),
        "point_offsets": output["fk_point_offsets"].tolist(),
        "video_outputs": len(output["preds_vision"]),
        "action_outputs": len(output["preds_action"]),
        "base_weight_preserved": True,
        "gradient": "FK prediction reaches encoder, position MLP, motion MLP, decoder, attention, video/action inputs",
        "scope": (
            "real Cosmos3VFMNetwork + real one-layer attention + learned FK branch; "
            "synthetic video/action latents; no Cosmos base checkpoint, joint loss or FSDP"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
