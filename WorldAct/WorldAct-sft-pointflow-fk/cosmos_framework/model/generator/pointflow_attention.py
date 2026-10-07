"""Correctness-first CP=1 two-way attention with pair-specific Edge mRoPE.

Each query has ONE softmax across all legal keys. Query chunks bound temporary
score memory, but this reference implementation is quadratic and not a fused kernel.
"""

import os

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq


def reference_attention_enabled():
    """Whether the pairwise reference attention is the rule that actually runs.

    The reference implementation materializes a ``[heads, chunk, n]`` score tensor
    per query chunk and keeps every one of them in the autograd graph, so a packed
    batch with point tokens costs far more memory than a training step can hold.
    It is therefore off by default and the generator falls back to
    ``dispatch_attention_fn``, which gives every modality the plain full mRoPE.
    That is the design's ``legacy_mrope`` ablation: it matches the design for
    video-point, point-point and text, and differs only in rotating the spatial
    channels of action-point pairs as well.  Both this module's caller and the
    mode recorded on the point payload must read the flag here, so the metadata
    cannot claim a rule that did not run.
    """
    return os.environ.get("POINTFLOW_REFERENCE_ATTENTION", "false").lower() == "true"


def default_attention_mode():
    if os.environ.get("POINTFLOW_FK_LOCAL_ROPE", "false").lower() == "true":
        return "local_pointflow_fk_mrope"
    return "pairwise_point_mrope" if reference_attention_enabled() else "legacy_mrope"


def temporal_rotary(q, k, cos, sin, sections, apply_rotary):
    """Neutralize Edge's interleaved H/W frequencies (both rotary halves)."""
    half = cos.shape[-1] // 2
    if len(sections) != 3 or sum(sections) != half:
        raise ValueError("mRoPE sections do not match rotary dimension")
    spatial = torch.zeros(half, dtype=torch.bool, device=cos.device)
    spatial[1 : sections[1] * 3 : 3] = True
    spatial[2 : sections[2] * 3 : 3] = True
    spatial = spatial.repeat(2)
    return apply_rotary(q, k, cos.masked_fill(spatial, 1), sin.masked_fill(spatial, 0), unsqueeze_dim=1)


def pairwise_point_attention(q_pack, k_pack, v_pack, qt_pack, kt_pack, meta, *, normalized_k=None, chunk_size=128):
    """Action=1, point=2, other=0; only action/point pairs use temporal QK."""
    if meta.is_three_way or meta.control_stream_token_ranges is not None or q_pack["is_sharded"]:
        raise ValueError("Point reference attention only supports CP=1 two-way without controls")
    q, k, v, qt, kt = [get_all_seq(p) for p in (q_pack, k_pack, v_pack, qt_pack, kt_pack)]
    kn = k if normalized_k is None else get_all_seq(normalized_k)
    groups = q.shape[1] // k.shape[1]
    k, v, kt, kn = [x.repeat_interleave(groups, dim=1) for x in (k, v, kt, kn)]
    ids = meta.pointflow_modalities.to(q.device)
    if meta.attn_modes != ["causal", "full"] * len(meta.sample_lens) or len(ids) != len(q):
        raise ValueError("Invalid point attention layout")
    outputs = []
    start = 0
    for b, n in enumerate(meta.sample_lens):
        causal = meta.split_lens[2 * b]
        keys = torch.arange(n, device=q.device)
        for row in range(0, n, chunk_size):
            end = min(row + chunk_size, n)
            queries = torch.arange(row, end, device=q.device)
            qs = q[start + row : start + end].transpose(0, 1)
            # Text keeps original K; generation uses diffusion-normalized text K.
            scores = qs @ k[start : start + n].transpose(0, 1).transpose(-1, -2)
            if normalized_k is not None:
                gen_scores = qs @ kn[start : start + n].transpose(0, 1).transpose(-1, -2)
                scores = torch.where((queries >= causal)[None, :, None], gen_scores, scores)
            query_ids, key_ids = ids[start + row : start + end], ids[start : start + n]
            pair = ((query_ids[:, None] == 1) & (key_ids[None] == 2)) | (
                (query_ids[:, None] == 2) & (key_ids[None] == 1)
            )
            temporal_scores = qt[start + row : start + end].transpose(0, 1) @ kt[start : start + n].transpose(
                0, 1
            ).transpose(-1, -2)
            scores = torch.where(pair[None], temporal_scores, scores) * (q.shape[-1] ** -0.5)
            allowed = (queries[:, None] >= causal) | ((keys[None] < causal) & (keys[None] <= queries[:, None]))
            weights = scores.masked_fill(~allowed[None], -torch.inf).softmax(-1)
            outputs.append((weights @ v[start : start + n].transpose(0, 1)).transpose(0, 1).to(q.dtype))
        start += n
    return from_all_seq(torch.cat(outputs).flatten(-2), q_pack)
