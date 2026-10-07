"""FK tokens attached to finalized, two-way Cosmos sequences.

The counterpart of ``pointflow_sequence.py``.  Two differences matter:

* there is no cluster/point split, so a keypoint's token *is* its own row -- no
  gather step, no per-point relative offset to carry through;
* the mRoPE position carries **time only**.  The spatial axes are produced by the
  same routine Cosmos uses for action (``grid_h = grid_w = 1``), which yields
  zero for both.  That is deliberate, not an omission: FK's uv would have to be
  projected onto a canvas this pipeline does not otherwise use, and a wrong
  spatial prior is worse than none.  See docs/fk_modality_design.md §1.4(a).
"""

from dataclasses import dataclass, replace

import torch

from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.sequence_span_remap import SpanRemapper

# Cosmos's FPS modulation reference point; the video path asserts the latent
# timeline it implies, so a change here without a change there would be caught.
BASE_FPS = 24.0


@dataclass
class FKTokenPayload:
    tokens: torch.Tensor
    sequence_indexes: torch.Tensor
    noisy_indexes: torch.Tensor  # [blocks, total_keypoints], global sequence indices

    def to_cuda(self):
        self.tokens = self.tokens.cuda()
        self.sequence_indexes = self.sequence_indexes.cuda()
        self.noisy_indexes = self.noisy_indexes.cuda()

    def hidden(self, sequence):
        return sequence[self.noisy_indexes]


def video_temporal_origins(sequence: PackedSequence) -> torch.Tensor:
    """One mRoPE temporal origin per sample: its first vision latent's time.

    Read from the packed sequence rather than derived from frame counts, so the
    FK tokens sit on the timeline the video actually got.
    """
    if sequence.vision is None:
        raise ValueError("FK requires a video grid")
    origins = []
    start = 0
    for length in sequence.sample_lens:
        ids = sequence.vision.sequence_indexes
        ids = ids[(ids >= start) & (ids < start + length)]
        if not len(ids):
            raise ValueError("Every FK batch slot requires a video item")
        origins.append(sequence.position_ids[0, ids[0]])
        start += length
    return torch.stack(origins).float()


def fk_positions(sequence: PackedSequence, point_batch, timing, *, base_fps=BASE_FPS) -> torch.Tensor:
    """``[1+blocks, sumN, 3]`` in ``(time, height, width)`` order.

    The temporal axis lands on the video's latent lattice: ``grid_t = blocks + 1``
    at Cosmos FPS gives ``origin + k * steps_per_token / fps * base_fps / tcf``,
    which is the same spacing the video latents use, so block ``b`` shares its
    timestamp with video latent ``b``.  Height and width come out zero because
    ``grid_h = grid_w = 1`` -- the action convention for a modality with no pixel
    position.
    """
    origins = video_temporal_origins(sequence).to(point_batch.device)
    ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=timing.blocks + 1,
        grid_h=1,
        grid_w=1,
        temporal_offset=0,
        fps=timing.fps,
        base_fps=base_fps,
        temporal_compression_factor=timing.steps_per_token,
    )
    ids = ids.to(device=point_batch.device).float()  # [3, blocks+1]
    times = ids[0] + origins[point_batch][:, None]  # [sumN, blocks+1]
    zero = torch.zeros_like(times)
    # [1+blocks, sumN, 3]: anchor first, then the blocks in chronological order.
    return torch.stack((times, zero, zero), dim=-1).permute(1, 0, 2).contiguous()


def attach_fk_tokens(sequence: PackedSequence, encoded, positions) -> PackedSequence:
    """Append each sample's FK tokens to its full split and remap the sequence.

    No padding: every sample contributes the same token count, but the offsets
    stay ragged so the shared packing/attention code needs no special case.
    """
    if sequence.fk is not None:
        raise ValueError("FK tokens are already attached")
    if sequence.attn_modes != ["causal", "full"] * len(sequence.sample_lens):
        raise ValueError("FK packing requires one causal/full pair per sample")
    if sum(sequence.sample_lens) != sequence.sequence_length:
        raise ValueError("Sample lengths do not match the sequence length")
    if any(sum(sequence.split_lens[2 * b : 2 * b + 2]) != n for b, n in enumerate(sequence.sample_lens)):
        raise ValueError("Splits must respect sample boundaries")

    noisy, anchor = encoded["noisy_tokens"], encoded["anchor_tokens"]
    blocks, points, _ = noisy.shape
    # Exclusive cumulative ends, one per sample: offsets[0] is the FIRST sample's
    # count, not zero.  Only a negative entry is malformed here; callers are
    # allowed an empty sample, which build_fk_batch represents as a repeated end.
    offsets = encoded["point_offsets"].tolist()
    if offsets != sorted(offsets) or not offsets or offsets[-1] != points or offsets[0] < 0:
        raise ValueError("Invalid FK point offsets")
    if positions.shape != (blocks + 1, points, 3) or not torch.isfinite(positions).all():
        raise ValueError("Expected finite FK positions [blocks+1,N,3]")

    device = noisy.device
    remap = torch.empty(sequence.sequence_length, device=device, dtype=torch.long)
    noisy_indices = torch.empty((blocks, points), device=device, dtype=torch.long)
    contents, indices, pos_chunks, lengths, splits = [], [], [], [], []
    old_start = new_start = pstart = 0
    for b, (n, pend) in enumerate(zip(sequence.sample_lens, offsets, strict=True)):
        count = pend - pstart
        # Anchor first, then the noisy blocks flattened block-major.
        content = torch.cat((anchor[pstart:pend], noisy[:, pstart:pend].reshape(-1, noisy.shape[-1])))
        target = new_start + n
        idx = torch.arange(target, target + len(content), device=device)
        noisy_indices[:, pstart:pend] = idx[count:].reshape(blocks, count)
        remap[old_start : old_start + n] = torch.arange(new_start, target, device=device)
        pos_chunks.append(
            torch.cat(
                (
                    sequence.position_ids[:, old_start : old_start + n].to(device).float(),
                    positions[:, pstart:pend].reshape(-1, 3).T.to(device).float(),
                ),
                dim=1,
            )
        )
        contents.append(content)
        indices.append(idx)
        lengths.append(n + len(content))
        splits.extend((sequence.split_lens[2 * b], sequence.split_lens[2 * b + 1] + len(content)))
        old_start += n
        new_start += n + len(content)
        pstart = pend

    def mapped(index):
        return None if index is None else remap[index.to(device)]

    remap_spans = SpanRemapper(sequence.sample_lens, lengths)

    def modality(data):
        if data is None:
            return None
        return replace(
            data,
            sequence_indexes=mapped(data.sequence_indexes),
            mse_loss_indexes=mapped(data.mse_loss_indexes),
            spans=remap_spans(data.spans),
        )

    def carried_tokens(payload):
        """Re-index an already-attached extra modality through this attach's remap.

        FK and PointFlow can both be present.  Whichever attaches second inserts
        tokens into every sample, which shifts every later index -- so the payload
        that is already on the sequence has to be remapped here just like
        vision/action/sound are.  Without it the earlier modality keeps its
        pre-attach ``sequence_indexes`` and the model scatters its tokens onto the
        wrong positions, with no shape error to catch it.
        """
        if payload is None:
            return None
        return replace(
            payload,
            sequence_indexes=remap[payload.sequence_indexes.to(device)],
            noisy_indexes=remap[payload.noisy_indexes.to(device)],
        )

    return replace(
        sequence,
        sample_lens=lengths,
        split_lens=splits,
        sequence_length=new_start,
        text_indexes=mapped(sequence.text_indexes),
        ce_loss_indexes=mapped(sequence.ce_loss_indexes),
        position_ids=torch.cat(pos_chunks, dim=1),
        vision=modality(sequence.vision),
        action=modality(sequence.action),
        sound=modality(sequence.sound),
        point=carried_tokens(sequence.point),
        fk=FKTokenPayload(torch.cat(contents), torch.cat(indices), noisy_indices),
    )
