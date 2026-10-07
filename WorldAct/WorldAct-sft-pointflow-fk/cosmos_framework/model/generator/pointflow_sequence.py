"""Ragged point tokens attached to finalized, two-way Cosmos sequences."""

from dataclasses import dataclass, replace

import torch

from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence
from cosmos_framework.model.generator.pointflow_attention import default_attention_mode
from cosmos_framework.model.generator.sequence_span_remap import SpanRemapper


@dataclass
class PointTokenPayload:
    tokens: torch.Tensor
    sequence_indexes: torch.Tensor
    noisy_indexes: torch.Tensor  # [blocks, total_clusters], global sequence indices
    # Must name the rule that actually ran, not the rule that was intended:
    # ``legacy_mrope`` is the default because the pairwise reference attention is
    # opt-in (see pointflow_attention.reference_attention_enabled).
    attention_mode: str = "legacy_mrope"

    def to_cuda(self):
        self.tokens = self.tokens.cuda()
        self.sequence_indexes = self.sequence_indexes.cuda()
        self.noisy_indexes = self.noisy_indexes.cuda()

    def hidden(self, sequence):
        return sequence[self.noisy_indexes]


def point_positions(cluster_uv, cluster_batch, uv_to_patch, temporal_origin, block_seconds, *, spatial_origin=None):
    """Return [1+blocks, K, 3] in (time, height, width) order.

    uv_to_patch[B,2,3] is the *actual* composed affine from tracker pixel
    centers to video patch coordinates, in (width,height) order. The caller
    must include crop/resize/padding/view offsets; no guessed image transform.
    temporal_origin[B] comes from the first video latent's mRoPE position.
    block_seconds are future result timestamps relative to the anchor.
    """
    device = cluster_uv.device
    affine = uv_to_patch.to(device=device, dtype=torch.float32)
    origin = temporal_origin.to(device=device, dtype=torch.float32)
    seconds = block_seconds.to(device=device, dtype=torch.float32)
    batch = cluster_batch
    if affine.shape != (len(origin), 2, 3) or cluster_uv.shape != (len(batch), 2):
        raise ValueError("Expected affine[B,2,3], origin[B], UV[K,2], cluster_batch[K]")
    if seconds.ndim != 1 or not torch.isfinite(seconds).all() or (seconds <= 0).any():
        raise ValueError("Block times must be finite, positive and chronological")
    if len(seconds) > 1 and (seconds.diff() <= 0).any():
        raise ValueError("Block times must increase")
    homogeneous = torch.cat((cluster_uv.float(), torch.ones_like(cluster_uv[:, :1])), -1)
    wh = torch.einsum("kij,kj->ki", affine[batch], homogeneous)
    if spatial_origin is not None:
        wh = wh + spatial_origin.to(device)[batch]  # (width,height), e.g. non-reset video mRoPE
    times = torch.cat((seconds.new_zeros(1), seconds)) * (24.0 / 4.0)
    result = torch.cat(
        ((origin[batch][None] + times[:, None])[..., None], wh.flip(-1)[None].expand(len(times), -1, -1)),
        -1,
    )
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite point positions")
    return result


def attach_point_tokens(sequence: PackedSequence, encoded, positions, *, attention_mode=None):
    """Copy/remap sequence metadata, appending each sample's points to its full split.

    No padding of K. Preserve original modality payload tensors and their gradients.
    Noisy token order within each sample is block-major; gather restores [blocks,K,D].
    """
    attention_mode = attention_mode or default_attention_mode()
    if attention_mode not in ("legacy_mrope", "pairwise_point_mrope", "local_pointflow_fk_mrope"):
        raise ValueError("Unknown point attention mode")
    if sequence.point is not None:
        raise ValueError("Point tokens already attached")
    offsets = encoded["cluster_offsets"].tolist()
    if sequence.attn_modes != ["causal", "full"] * len(offsets):
        raise ValueError("Point packing requires one causal/full pair per sample")
    if len(sequence.sample_lens) != len(offsets) or sum(sequence.sample_lens) != sequence.sequence_length:
        raise ValueError("Sample lengths do not match")
    if any(sum(sequence.split_lens[2 * b : 2 * b + 2]) != n for b, n in enumerate(sequence.sample_lens)):
        raise ValueError("Splits must respect sample boundaries")
    noisy, anchor = encoded["noisy_tokens"], encoded["anchor_tokens"]
    blocks, clusters, _ = noisy.shape
    if offsets != sorted(offsets) or not offsets or offsets[-1] != clusters or offsets[0] < 0:
        raise ValueError("Invalid cluster offsets")
    if positions.shape != (blocks + 1, clusters, 3) or not torch.isfinite(positions).all():
        raise ValueError("Expected finite point positions [blocks+1,K,3]")
    device = noisy.device
    remap = torch.empty(sequence.sequence_length, device=device, dtype=torch.long)
    noisy_indices = torch.empty((blocks, clusters), device=device, dtype=torch.long)
    contents, indices, pos_chunks, lengths, splits = [], [], [], [], []
    old_start = new_start = cstart = 0
    for b, (n, cend) in enumerate(zip(sequence.sample_lens, offsets, strict=True)):
        count = cend - cstart
        parts = [noisy[:, cstart:cend].reshape(-1, noisy.shape[-1])]
        if anchor is not None:
            parts.insert(0, anchor[cstart:cend])
        content = torch.cat(parts)
        pstart = new_start + n
        idx = torch.arange(pstart, pstart + len(content), device=device)
        noisy_indices[:, cstart:cend] = idx[count if anchor is not None else 0 :].reshape(blocks, count)
        remap[old_start : old_start + n] = torch.arange(new_start, pstart, device=device)
        pos_chunks.extend(
            (
                sequence.position_ids[:, old_start : old_start + n].to(device).float(),
                positions[0 if anchor is not None else 1 :, cstart:cend].reshape(-1, 3).T.to(device).float(),
            )
        )
        contents.append(content)
        indices.append(idx)
        lengths.append(n + len(content))
        splits.extend((sequence.split_lens[2 * b], sequence.split_lens[2 * b + 1] + len(content)))
        old_start += n
        new_start += n + len(content)
        cstart = cend

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
        fk=carried_tokens(sequence.fk),
        point=PointTokenPayload(torch.cat(contents), torch.cat(indices), noisy_indices, attention_mode),
    )
