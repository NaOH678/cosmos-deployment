"""Local Point/FK four-axis RoPE, with exact shared-normalizer attention.

Two implementations: partitioned 128-d attention, or lifted 160-d attention
with one varlen call for all generative queries. Both preserve the causal text
branch. No NxN score matrix or repeated KV heads; metadata is prepared once.
"""

from dataclasses import dataclass

import torch

DEPTH_PAIRS = (49, 50, 52, 53, 55, 56, 58, 59)
DEPTH_CHANNELS = DEPTH_PAIRS + tuple(i + 64 for i in DEPTH_PAIRS)


class _UniqueRows(torch.autograd.Function):
    """Gather unique rows without an atomic scatter-add in backward.

    Only for indexes constructed by make_partition (unique within each group).
    Separate uses still accumulate through autograd, including overlapping key
    groups. For a full permutation, inverse gives a gather-only backward.
    """

    @staticmethod
    def forward(ctx, values, indexes, inverse):
        ctx.save_for_backward(indexes, inverse)
        ctx.input_shape = values.shape
        # Keep the gather visible to Inductor; the custom backward below
        # avoids index's general scatter-add for these unique row indexes.
        return values[indexes]

    @staticmethod
    def backward(ctx, grad):
        indexes, inverse = ctx.saved_tensors
        if inverse is not None:
            return grad.index_select(0, inverse), None, None
        return grad.new_zeros(ctx.input_shape).index_copy_(0, indexes, grad), None, None


def _unique_rows(values, indexes):
    return _UniqueRows.apply(values, indexes, None)


@dataclass
class Group:
    indexes: torch.Tensor
    offsets: torch.Tensor
    max_length: int


@dataclass
class AttentionPartition:
    causal: Group
    other_queries: Group
    geometry: Group
    other_keys: Group
    all_keys: Group
    mixed_geometry: Group
    only_geometry: Group
    geometry_lookup: torch.Tensor
    depth: torch.Tensor  # packed geometry-token order, inverse camera depth
    output_order: torch.Tensor  # causal, other, mixed-geometry, only-geometry
    output_restore: torch.Tensor  # inverse permutation back to packed token order
    generation: Group
    stream_order: torch.Tensor  # causal then generation
    stream_restore: torch.Tensor


def make_partition(sample_lens, split_lens, geometry_indexes, depth):
    """Geometry must be in the full split. Empty slots retain sample isolation."""
    if len(split_lens) != 2 * len(sample_lens):
        raise ValueError("Local RoPE requires one causal/full pair per sample")
    device = geometry_indexes.device
    geo = geometry_indexes.detach().cpu().tolist()
    if len(set(geo)) != len(geo) or any(i < 0 or i >= sum(sample_lens) for i in geo):
        raise ValueError("Invalid or duplicate geometric token indexes")
    if depth.shape != (len(geo),) or not torch.isfinite(depth).all():
        raise ValueError("Expected one finite depth coordinate per geometry token")
    ordered = sorted(range(len(geo)), key=lambda i: geo[i])
    lookup = set(geo)
    groups = [[] for _ in range(8)]
    cursor = 0
    for b, length in enumerate(sample_lens):
        c, f = split_lens[2 * b : 2 * b + 2]
        if c + f != length:
            raise ValueError("Split crosses sample boundary")
        causal = list(range(cursor, cursor + c))
        full = list(range(cursor + c, cursor + length))
        if lookup.intersection(causal):
            raise ValueError("Geometric tokens must be in the full split")
        local_geo = [i for i in full if i in lookup]
        other = [i for i in full if i not in lookup]
        for group, ids in zip(
            groups,
            [
                causal,
                other,
                local_geo,
                causal + other,
                causal + full,
                local_geo if causal + other else [],
                [] if causal + other else local_geo,
                full,
            ],
            strict=True,
        ):
            group.append(ids)
        cursor += length

    def build(rows):
        lengths = [len(row) for row in rows]
        return Group(
            torch.tensor([i for row in rows for i in row], device=device, dtype=torch.long),
            torch.tensor([0] + lengths, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32),
            max(lengths, default=0),
        )

    built = [build(rows) for rows in groups]
    lookup_tensor = torch.full((sum(sample_lens),), -1, device=device, dtype=torch.long)
    lookup_tensor[built[2].indexes] = torch.arange(len(geo), device=device)
    output_order = torch.cat([built[i].indexes for i in (0, 1, 5, 6)])
    stream_order = torch.cat((built[0].indexes, built[7].indexes))
    return AttentionPartition(
        *built[:7],
        lookup_tensor,
        depth[ordered],
        output_order,
        output_order.argsort(),
        built[7],
        stream_order,
        stream_order.argsort(),
    )


def rotate_geometry(raw, cos3, sin3, depth, frequencies):
    """Replace only eight H/W rotation pairs; all time channels remain intact."""
    if raw.shape[-1] != 128 or cos3.shape[-1] != 128:
        raise ValueError("Local RoPE currently supports Edge head_dim=128 only")
    cos, sin = cos3.clone(), sin3.clone()
    phase = depth.float()[:, None] * frequencies.float()[None]
    indexes = list(DEPTH_PAIRS) + [i + 64 for i in DEPTH_PAIRS]
    cos[:, indexes] = phase.cos().repeat(1, 2).to(cos.dtype)
    sin[:, indexes] = phase.sin().repeat(1, 2).to(sin.dtype)
    a, b = raw.chunk(2, dim=-1)
    return (raw * cos[:, None] + torch.cat((-b, a), -1) * sin[:, None]).to(raw.dtype)


def fused_partition_attention(q3, k3, v, q4, k4, partition, *, normalized_k=None):
    """BSHD fused kernels; q4/k4 contain ONLY geometric tokens in sorted order.

    Raw and diffusion-normalized text K are deliberately separate. All kernels
    are Flash2 varlen, including a shared-normalizer two-group backward.
    """
    from cosmos_framework.model.attention.flash2 import flash2_attention as attention
    from cosmos_framework.model.attention.flash2.two_group import two_group_varlen
    from cosmos_framework.model.attention.masks import CausalType

    p = partition
    kgen = k3 if normalized_k is None else normalized_k
    outputs = []

    def attend(q, k, values, qgroup, kgroup, *, causal=False, lse=False):
        return attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            values.unsqueeze(0),
            cumulative_seqlen_Q=qgroup.offsets,
            cumulative_seqlen_KV=kgroup.offsets,
            max_seqlen_Q=qgroup.max_length,
            max_seqlen_KV=kgroup.max_length,
            is_causal=causal,
            causal_type=CausalType.DontCare if causal else None,
            return_lse=lse,
        )

    c, o, g, other, all_keys = p.causal, p.other_queries, p.geometry, p.other_keys, p.all_keys
    mixed, only = p.mixed_geometry, p.only_geometry
    # These query groups partition the full sequence. Gather them together so
    # backward needs one inverse permutation, rather than three zero-filled
    # full-size gradients followed by additions.
    qc, qo, qm, _ = _UniqueRows.apply(q3, p.output_order, p.output_restore).split(
        [len(c.indexes), len(o.indexes), len(mixed.indexes), len(only.indexes)], dim=0
    )
    if len(c.indexes):
        out = attend(qc, _unique_rows(k3, c.indexes), _unique_rows(v, c.indexes), c, c, causal=True)
        outputs.append(out[0])
    if len(o.indexes):
        # all_keys is the identity permutation of the packed sequence.
        out = attend(qo, kgen, v, o, all_keys)
        outputs.append(out[0])
    geometric_v = _unique_rows(v, g.indexes) if len(g.indexes) else None
    if len(mixed.indexes):
        # In ordinary training every geometric query has non-geometric keys.
        mixed_q4 = q4 if len(mixed.indexes) == len(g.indexes) else _unique_rows(q4, p.geometry_lookup[mixed.indexes])
        out = two_group_varlen(
            qm,
            _unique_rows(kgen, other.indexes),
            _unique_rows(v, other.indexes),
            mixed_q4,
            k4,
            geometric_v,
            mixed.offsets,
            other.offsets,
            g.offsets,
            mixed.max_length,
            other.max_length,
            g.max_length,
        )
        outputs.append(out)
    if len(only.indexes):
        # Flash2 returns a special LSE for empty KV. Never merge that branch:
        # these samples have just one legal key group and need no normalization merge.
        only_q4 = q4 if len(only.indexes) == len(g.indexes) else _unique_rows(q4, p.geometry_lookup[only.indexes])
        out = attend(only_q4, k4, geometric_v, only, g)
        outputs.append(out[0])
    # Restore once, with a gather-only backward, instead of repeatedly copying
    # the entire output and creating zero-filled gradients for each group.
    if not outputs:
        return q3.new_zeros((*q3.shape[:-1], v.shape[-1]))
    return _UniqueRows.apply(torch.cat(outputs, dim=0), p.output_restore, p.output_order)


def lifted_partition_attention(q3, k3, v, q4, k4, partition, *, normalized_k=None):
    """Exact local score replacement via 128+16+16 channels, one gen call.

    Requires q4/k4 to equal the geometric rows of q3/kgen outside
    DEPTH_CHANNELS, as guaranteed by rotate_geometry in the production path.
    Causal text uses the original 128-d Q/K/V and unmodified normalization.
    """
    from cosmos_framework.model.attention.flash2 import flash2_attention as attention
    from cosmos_framework.model.attention.masks import CausalType

    p = partition
    c, full, g = p.causal, p.generation, p.geometry
    kgen = k3 if normalized_k is None else normalized_k
    outputs = []

    def attend(q, k, values, qgroup, kgroup, causal=False):
        return attention(
            q[None],
            k[None],
            values[None],
            cumulative_seqlen_Q=qgroup.offsets,
            cumulative_seqlen_KV=kgroup.offsets,
            max_seqlen_Q=qgroup.max_length,
            max_seqlen_KV=kgroup.max_length,
            is_causal=causal,
            causal_type=CausalType.DontCare if causal else None,
            scale=128**-0.5,
        )[0]

    if len(c.indexes):
        outputs.append(
            attend(_unique_rows(q3, c.indexes), _unique_rows(k3, c.indexes), _unique_rows(v, c.indexes), c, c, True)
        )
    if len(full.indexes):
        if len(g.indexes):
            channels = list(DEPTH_CHANNELS)
            q_extra = torch.cat((q4[..., channels], _unique_rows(q3, g.indexes)[..., channels]), dim=-1)
            k_extra = torch.cat((k4[..., channels], -_unique_rows(kgen, g.indexes)[..., channels]), dim=-1)
            q_tail = q3.new_zeros((*q3.shape[:-1], 32)).index_copy(0, g.indexes, q_extra)
            k_tail = kgen.new_zeros((*kgen.shape[:-1], 32)).index_copy(0, g.indexes, k_extra)
            q_lifted = torch.cat((q3, q_tail), dim=-1)
            k_lifted = torch.cat((kgen, k_tail), dim=-1)
            v_lifted = torch.nn.functional.pad(v, (0, 32))
            out = attend(_unique_rows(q_lifted, full.indexes), k_lifted, v_lifted, full, p.all_keys)
            outputs.append(out[..., :128])
        else:
            outputs.append(attend(_unique_rows(q3, full.indexes), kgen, v, full, p.all_keys))
    if not outputs:
        return q3.new_zeros((*q3.shape[:-1], v.shape[-1]))
    return _UniqueRows.apply(torch.cat(outputs, dim=0), p.stream_restore, p.stream_order)


def packed_local_attention(q3, k3, v, raw_q, raw_k, cos, sin, metadata, *, normalized_k=None):
    from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq

    p = metadata.local_geometry_partition
    ids = p.geometry.indexes
    cos3, sin3 = get_all_seq(cos)[ids], get_all_seq(sin)[ids]
    q4 = rotate_geometry(_unique_rows(get_all_seq(raw_q), ids), cos3, sin3, p.depth, metadata.local_depth_frequencies)
    k4 = rotate_geometry(_unique_rows(get_all_seq(raw_k), ids), cos3, sin3, p.depth, metadata.local_depth_frequencies)
    implementation = getattr(metadata, "local_geometry_implementation", "partition")
    attention_fn = lifted_partition_attention if implementation == "lifted160" else fused_partition_attention
    out = attention_fn(
        get_all_seq(q3),
        get_all_seq(k3),
        get_all_seq(v),
        q4,
        k4,
        p,
        normalized_k=None if normalized_k is None else get_all_seq(normalized_k),
    )
    return from_all_seq(out.flatten(-2), q3)
