"""Task 4: real Cosmos MoT attention, ragged packing, and mRoPE invariants."""

import unittest
from types import SimpleNamespace

import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData, ModalitySpan
from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, PackedAttentionMoT
from cosmos_framework.model.generator.pointflow_attention import pairwise_point_attention, temporal_rotary
from cosmos_framework.model.generator.pointflow_sequence import attach_point_tokens, point_positions
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
    apply_rotary_pos_emb_partial,
)


def base_sequence(device="cpu"):
    # Three samples: 2 text, 2 video, 1 action each; sample 1 has no points.
    def idx(x):
        return torch.tensor(x, dtype=torch.long, device=device)

    return PackedSequence(
        sample_lens=[5] * 3,
        split_lens=[2, 3] * 3,
        attn_modes=["causal", "full"] * 3,
        sequence_length=15,
        text_ids=idx([1, 2] * 3),
        text_indexes=idx([0, 1, 5, 6, 10, 11]),
        position_ids=torch.arange(15, device=device).float()[None].expand(3, -1).clone(),
        ce_loss_indexes=idx([1, 6, 11]),
        vision=ModalityData(
            sequence_indexes=idx([2, 3, 7, 8, 12, 13]),
            mse_loss_indexes=idx([3, 8, 13]),
            spans=[ModalitySpan(2 + 5 * b, 2, b, 0, 2, (1, 1, 2)) for b in range(3)],
        ),
        action=ModalityData(sequence_indexes=idx([4, 9, 14])),
    )


def build(sequence, hidden):
    pack, meta, _ = build_packed_sequence(
        "two_way",
        packed_sequence=hidden,
        attn_modes=sequence.attn_modes,
        split_lens=sequence.split_lens,
        sample_lens=sequence.sample_lens,
        packed_und_token_indexes=sequence.text_indexes,
        packed_gen_token_indexes=torch.empty(0, dtype=torch.long, device=hidden.device),
        num_heads=4,
        head_dim=16,
        num_layers=1,
    )
    ids = torch.zeros(sequence.sequence_length, dtype=torch.long, device=hidden.device)
    ids[sequence.action.sequence_indexes] = 1
    if sequence.point is not None:
        ids[sequence.point.sequence_indexes] = 2
    meta.pointflow_modalities = ids
    return pack, meta


class SequenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.config = Nemotron3DenseVLTextConfig(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, mrope_section=[4, 2, 2]
        )

    def test_ragged_positions_remap_and_gradient(self):
        base = base_sequence()
        noisy = torch.randn(2, 3, 64, requires_grad=True)
        anchor = torch.randn(3, 64, requires_grad=True)
        encoded = dict(noisy_tokens=noisy, anchor_tokens=anchor, cluster_offsets=torch.tensor([2, 2, 3]))
        uv = torch.tensor([[32.0, 64.0], [64.0, 96.0], [0.0, 0.0]])
        affine = torch.tensor([[[1 / 32, 0, 0], [0, 1 / 32, 7]]] * 3)
        positions = point_positions(
            uv, torch.tensor([0, 0, 2]), affine, torch.tensor([2.0, 7.0, 12.0]), torch.tensor([4 / 15, 8 / 15])
        )
        torch.testing.assert_close(positions[:, 0, 0], torch.tensor([2.0, 3.6, 5.2]))
        torch.testing.assert_close(positions[0, 0, 1:], torch.tensor([9.0, 1.0]))
        seq = attach_point_tokens(base, encoded, positions)
        self.assertEqual(seq.sample_lens, [11, 5, 8])
        self.assertEqual(seq.text_indexes.tolist(), [0, 1, 11, 12, 16, 17])
        self.assertEqual(seq.vision.mse_loss_indexes.tolist(), [3, 14, 19])
        self.assertEqual([s.sequence_start for s in seq.vision.spans], [2, 13, 18])
        self.assertEqual(base.sample_lens, [5] * 3)
        hidden = torch.randn(seq.sequence_length, 64)
        hidden[seq.point.sequence_indexes] = seq.point.tokens
        torch.testing.assert_close(seq.point.hidden(hidden), noisy)
        pack, meta = build(seq, hidden)
        rotary = MultiModalRotaryEmbedding(self.config)
        cos, sin = rotary(hidden, seq.position_ids[:, None])
        layer = PackedAttentionMoT(
            self.config,
            layer_idx=0,
            layer_types=LayerTypes("nemotron_dense"),
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
            use_und_k_norm_for_gen=True,
        )
        output, _ = layer(pack, meta, (from_all_seq(cos[0], pack), from_all_seq(sin[0], pack)))
        result = get_all_seq(output)
        self.assertEqual(result.shape, hidden.shape)
        result.square().mean().backward()
        self.assertGreater(noisy.grad.abs().sum().item(), 0)
        self.assertGreater(anchor.grad.abs().sum().item(), 0)
        changed = hidden.detach().clone()
        changed[16:] += 100
        other, _ = layer(from_all_seq(changed, pack), meta, (from_all_seq(cos[0], pack), from_all_seq(sin[0], pack)))
        torch.testing.assert_close(result[:16], get_all_seq(other)[:16])

    def test_pairwise_rope_and_single_softmax(self):
        seq = base_sequence()
        pack, meta = build(seq, torch.randn(15, 64))
        meta.pointflow_modalities[3] = 2
        q = torch.randn(15, 4, 16)
        k = torch.randn(15, 2, 16)
        v = torch.randn(15, 2, 16)
        rotary = MultiModalRotaryEmbedding(self.config)

        def rotated(pos):
            c, s = rotary(q, pos[:, None])
            c, s = c[0], s[0]
            qr, kr = apply_rotary_pos_emb_partial(q, k, c, s, unsqueeze_dim=1)
            qt, kt = temporal_rotary(q, k, c, s, [4, 2, 2], apply_rotary_pos_emb_partial)
            return qr, kr, qt, kt

        qr, kr, qt, kt = rotated(seq.position_ids)
        changed = seq.position_ids.clone()
        changed[1:, 3] += 17.3
        qr2, kr2, qt2, kt2 = rotated(changed)
        torch.testing.assert_close(qt, qt2)
        torch.testing.assert_close(kt, kt2)
        self.assertFalse(torch.allclose(kr[3], kr2[3]))
        kn = kr.clone()
        kn[seq.text_indexes] *= 1.7
        result = get_all_seq(
            pairwise_point_attention(
                *[from_all_seq(x, pack) for x in (qr, kr, v, qt, kt)],
                meta,
                chunk_size=2,
                normalized_k=from_all_seq(kn, pack),
            )
        )
        # Independent scalar query/head oracle: one softmax, GQA, causal, sample boundaries.
        expected = torch.zeros(15, 4, 16)
        for i in range(15):
            start = i // 5 * 5
            legal = range(start, i + 1) if i - start < 2 else range(start, start + 5)
            for h in range(4):
                scores = []
                values = []
                for j in legal:
                    pair = {int(meta.pointflow_modalities[i]), int(meta.pointflow_modalities[j])} == {1, 2}
                    a, b = (qt, kt) if pair else (qr, kn if i - start >= 2 else kr)
                    scores.append(torch.dot(a[i, h], b[j, h // 2]) / 4)
                    values.append(v[j, h // 2])
                expected[i, h] = torch.stack(scores).softmax(0) @ torch.stack(values)
        torch.testing.assert_close(result, expected.flatten(-2), atol=1e-6, rtol=1e-5)

    def test_vfm_forward_point_hook(self):
        from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork

        base = base_sequence()
        noisy = torch.randn(2, 3, 64, requires_grad=True)
        seq = attach_point_tokens(
            base,
            dict(noisy_tokens=noisy, anchor_tokens=None, cluster_offsets=torch.tensor([2, 2, 3])),
            torch.zeros(3, 3, 3),
        )
        layer = PackedAttentionMoT(
            self.config,
            layer_idx=0,
            layer_types=LayerTypes("nemotron_dense"),
            qk_norm_for_text=False,
            qk_norm_for_diffusion=True,
            use_und_k_norm_for_gen=True,
        )
        rotary = MultiModalRotaryEmbedding(self.config)

        def language(pack, attention_mask, position_ids, **kwargs):
            cos, sin = rotary(get_all_seq(pack), position_ids[:, None])
            output, _ = layer(pack, attention_mask, (from_all_seq(cos[0], pack), from_all_seq(sin[0], pack)))
            return output, {}

        # Run the actual VFM forward with lightweight synthetic modality encoders.
        model = SimpleNamespace(
            config=SimpleNamespace(
                joint_attn_implementation="two_way", vision_gen=False, action_gen=False, sound_gen=False
            ),
            video_temporal_causal=False,
            parallel_dims=None,
            pad_for_cuda_graphs=False,
            attention_io_layout="sequence_sharded",
            num_heads=4,
            head_dim=16,
            num_hidden_layers=1,
            natten_parameter_list=None,
            predict_text_tokens=False,
            language_model=language,
            _encode_text=lambda packed: (torch.zeros(packed.sequence_length, 64), torch.float32),
        )
        output = Cosmos3VFMNetwork.forward(model, seq)
        self.assertEqual(output["point_hidden"].shape, (2, 3, 64))
        output["point_hidden"].square().mean().backward()
        self.assertGreater(noisy.grad.abs().sum().item(), 0)
        model.video_temporal_causal = True
        with self.assertRaises(ValueError):
            Cosmos3VFMNetwork.forward(model, seq)

    def test_empty_points(self):
        seq = base_sequence()
        encoded = dict(noisy_tokens=torch.empty(2, 0, 64), anchor_tokens=None, cluster_offsets=torch.tensor([0, 0, 0]))
        out = attach_point_tokens(seq, encoded, torch.empty(3, 0, 3))
        self.assertEqual(out.sample_lens, seq.sample_lens)
        torch.testing.assert_close(out.position_ids, seq.position_ids)
        self.assertEqual(out.point.hidden(torch.zeros(15, 64)).shape, (2, 0, 64))


if __name__ == "__main__":
    unittest.main()
