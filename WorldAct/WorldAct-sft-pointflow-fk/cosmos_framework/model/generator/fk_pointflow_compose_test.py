"""FK and PointFlow tokens must compose: the four-modality packing gate.

Both ``attach_*_tokens`` were written while their modality was the only extra one,
so each remaps vision/action/sound *and itself* -- and nothing stopped there from
being correct, because the other payload was always ``None``.

Once both are present that is no longer true.  Whichever attaches second inserts
tokens into every sample, shifting every later index, so the payload already on
the sequence must be remapped too.  If it is not, the earlier modality keeps its
pre-attach ``sequence_indexes`` and

* ``forward`` scatters its tokens onto positions owned by another modality -- the
  write succeeds, so there is no shape error, only wrong activations;
* ``all_gen_indexes`` routes those stale positions instead of its own.

Neither symptom raises, and the loss still falls.  Hence this test.

Also checked, because PointFlow derives its own positions from them
(``pointflow_branch.video_aligned_point_positions`` reads ``sequence.position_ids``
at the vision indexes): the video positions must survive whichever order runs.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cosmos_framework.data.generator.sequence_packing import PackedSequence  # noqa: E402
from cosmos_framework.data.generator.sequence_packing.modality import (  # noqa: E402
    ModalityData,
    ModalitySpan,
)
from cosmos_framework.model.generator.fk_sequence import attach_fk_tokens  # noqa: E402
from cosmos_framework.model.generator.pointflow_sequence import attach_point_tokens  # noqa: E402

BLOCKS = 8
DIM = 4
TEXT, N_VISION, N_ACTION = 4, 9, 33
SAMPLE = TEXT + N_VISION + N_ACTION
FK_POINTS = [3, 2]
POINT_CLUSTERS = [4, 3]
N_SAMPLES = 2


def _offsets(counts):
    ends, total = [], 0
    for n in counts:
        total += n
        ends.append(total)
    return torch.tensor(ends)


def _encoded(counts):
    total = sum(counts)
    return {
        "point_offsets": _offsets(counts),
        "cluster_offsets": _offsets(counts),
        "noisy_tokens": torch.randn(BLOCKS, total, DIM),
        "anchor_tokens": torch.randn(total, DIM),
    }


def _positions(counts):
    return torch.randn(BLOCKS + 1, sum(counts), 3)


def _modality(indexes, spans, n_items):
    idx = torch.cat(indexes)
    return ModalityData(
        sequence_indexes=idx,
        timesteps=torch.zeros(len(idx)),
        mse_loss_indexes=idx.clone(),
        spans=spans,
        token_shapes=[(1, 3, 3)] * n_items,
        tokens=[torch.zeros(1, 3, 3, DIM) for _ in range(n_items)],
        condition_mask=[],
        noisy_frame_indexes=[],
        domain_id=[0] * n_items,
        raw_action_dim=[None] * n_items,
    )


def build_sequence():
    """Two samples of text + vision + action, with distinctive positions."""
    text_idx, vision_idx, action_idx = [], [], []
    vision_spans, action_spans = [], []
    position_ids = torch.zeros(3, SAMPLE * N_SAMPLES)
    for b in range(N_SAMPLES):
        base = b * SAMPLE
        t = torch.arange(TEXT) + base
        v = torch.arange(N_VISION) + base + TEXT
        a = torch.arange(N_ACTION) + base + TEXT + N_VISION
        text_idx.append(t)
        vision_idx.append(v)
        action_idx.append(a)
        position_ids[0, t] = 100 + b
        position_ids[0, v] = torch.arange(N_VISION, dtype=torch.float32) * 1.6 + b
        position_ids[1, v] = 3.0
        position_ids[2, v] = 4.0
        position_ids[0, a] = torch.arange(N_ACTION, dtype=torch.float32)
        vision_spans.append(ModalitySpan(int(v[0]), N_VISION, b, 0, N_VISION, (1, 3, 3)))
        action_spans.append(ModalitySpan(int(a[0]), N_ACTION, b, 0, N_ACTION, (4,)))

    sequence = PackedSequence(
        sample_lens=[SAMPLE] * N_SAMPLES,
        split_lens=[TEXT, SAMPLE - TEXT] * N_SAMPLES,
        attn_modes=["causal", "full"] * N_SAMPLES,
        sequence_length=SAMPLE * N_SAMPLES,
        text_ids=torch.zeros(1, SAMPLE * N_SAMPLES, dtype=torch.long),
        text_indexes=torch.cat(text_idx),
        position_ids=position_ids,
        vision=_modality(vision_idx, vision_spans, N_SAMPLES),
        action=_modality(action_idx, action_spans, N_SAMPLES),
    )
    return sequence, torch.cat(vision_idx), position_ids


def attach(sequence, *, fk_first):
    e_fk, e_pt = _encoded(FK_POINTS), _encoded(POINT_CLUSTERS)
    p_fk, p_pt = _positions(FK_POINTS), _positions(POINT_CLUSTERS)
    if fk_first:
        attached = attach_fk_tokens(sequence, e_fk, p_fk)
        attached = attach_point_tokens(attached, e_pt, p_pt)
    else:
        attached = attach_point_tokens(sequence, e_pt, p_pt)
        attached = attach_fk_tokens(attached, e_fk, p_fk)
    return attached


def expected_length():
    fk_content = sum(n * (1 + BLOCKS) for n in FK_POINTS)
    pt_content = sum(k * (1 + BLOCKS) for k in POINT_CLUSTERS)
    return SAMPLE * N_SAMPLES + fk_content + pt_content


def check_packing(order, fk_first):
    base, base_vision_idx, base_positions = build_sequence()
    seq = attach(base, fk_first=fk_first)
    fk_idx, pt_idx = seq.fk.sequence_indexes, seq.point.sequence_indexes

    assert seq.sequence_length == expected_length(), order
    assert sum(seq.sample_lens) == seq.sequence_length, order
    assert all(sum(seq.split_lens[2 * b : 2 * b + 2]) == n for b, n in enumerate(seq.sample_lens)), order
    assert seq.attn_modes == ["causal", "full"] * len(seq.sample_lens), order
    assert seq.position_ids.shape[1] == seq.sequence_length, order

    # The point of this test: no modality's indexes may collide with another's.
    new_idx = torch.cat([fk_idx, pt_idx])
    assert len(set(new_idx.tolist())) == len(new_idx), order
    assert 0 <= int(new_idx.min()) and int(new_idx.max()) < seq.sequence_length, order
    for name, payload in (("vision", seq.vision), ("action", seq.action)):
        overlap = set(payload.sequence_indexes.tolist()) & set(new_idx.tolist())
        assert not overlap, f"{order}: {name} indexes collide with the new modalities: {sorted(overlap)}"

    # PointFlow reads position_ids at the vision indexes to build its own; the video
    # positions must therefore survive both attaches unchanged.
    assert torch.allclose(seq.position_ids[:, seq.vision.sequence_indexes], base_positions[:, base_vision_idx]), (
        f"{order}: vision positions were rewritten"
    )
    assert seq.text_indexes.tolist()[:TEXT] == list(range(TEXT)), order


def check_fk_payload_remapped_when_second():
    """The regression proper: FK attaching first, PointFlow second, FK must shift."""
    base, _, _ = build_sequence()
    fk_only = attach_fk_tokens(base, _encoded(FK_POINTS), _positions(FK_POINTS))
    before = fk_only.fk.sequence_indexes.tolist()

    both = attach_point_tokens(fk_only, _encoded(POINT_CLUSTERS), _positions(POINT_CLUSTERS))
    after = both.fk.sequence_indexes.tolist()

    # PointFlow appends to the end of each sample, so anything at or after sample
    # 0's new end shifts by the length it inserted there; FK tokens inside sample 0
    # sit before that point and must not move.
    boundary = fk_only.sample_lens[0]
    inserted = both.sample_lens[0] - fk_only.sample_lens[0]
    assert inserted > 0
    expected = [i if i < boundary else i + inserted for i in before]
    assert after == expected, (
        f"FK indexes were not re-indexed through the PointFlow attach: expected {expected}, got {after}"
    )
    assert len(set(after)) == len(after)
    assert max(after) < both.sequence_length


def check_attention_mode_survives_the_second_attach():
    """PointTokenPayload carries ``attention_mode``; a re-index must not drop it.

    The payload is replaced wholesale by ``carried_tokens``, so a field that is not
    explicitly passed through is silently reset to its default -- and this one
    selects which attention rule ran, which the model records but never re-derives.
    """
    from cosmos_framework.model.generator.pointflow_attention import default_attention_mode

    base, _, _ = build_sequence()
    point_only = attach_point_tokens(base, _encoded(POINT_CLUSTERS), _positions(POINT_CLUSTERS))
    assert point_only.point.attention_mode == default_attention_mode()

    both = attach_fk_tokens(point_only, _encoded(FK_POINTS), _positions(FK_POINTS))
    assert both.point.attention_mode == point_only.point.attention_mode, (
        "FK's attach dropped PointFlow's attention_mode"
    )


def check_single_modality_unchanged():
    """With no PointFlow payload, FK's indexes must be exactly as they always were."""
    base, _, _ = build_sequence()
    seq = attach_fk_tokens(base, _encoded(FK_POINTS), _positions(FK_POINTS))
    assert seq.point is None
    assert seq.sequence_length == SAMPLE * N_SAMPLES + sum(n * (1 + BLOCKS) for n in FK_POINTS)
    assert max(seq.fk.sequence_indexes.tolist()) < seq.sequence_length

    base2, _, _ = build_sequence()
    seq2 = attach_point_tokens(base2, _encoded(POINT_CLUSTERS), _positions(POINT_CLUSTERS))
    assert seq2.fk is None
    assert max(seq2.point.sequence_indexes.tolist()) < seq2.sequence_length


if __name__ == "__main__":
    check_packing("fk_first", fk_first=True)
    check_packing("pointflow_first", fk_first=False)
    check_fk_payload_remapped_when_second()
    check_attention_mode_survives_the_second_attach()
    check_single_modality_unchanged()
    print("fk_pointflow_compose_test: OK")
