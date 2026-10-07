"""Independent dense oracle for production partitions, including backward."""

import pytest
import torch

from cosmos_framework.model.generator.pointflow_fk_attention import (
    DEPTH_CHANNELS,
    DEPTH_PAIRS,
    _unique_rows,
    _UniqueRows,
    fused_partition_attention,
    lifted_partition_attention,
    make_partition,
    rotate_geometry,
)


def test_unique_rows_overlapping_uses_and_noncontiguous_gradient():
    torch.manual_seed(17)
    x = torch.randn(9, 3, 4, dtype=torch.float64, requires_grad=True)
    a, b = torch.tensor([7, 1, 4]), torch.tensor([4, 0, 7, 3])
    wa = torch.randn(3, 4, 3, dtype=x.dtype).transpose(1, 2)
    wb = torch.randn(4, 4, 3, dtype=x.dtype).transpose(1, 2)
    actual = (_unique_rows(x, a) * wa).sum() + (_unique_rows(x, b) * wb).sum()
    reference = (x[a] * wa).sum() + (x[b] * wb).sum()
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(reference, x)[0])
    empty = _unique_rows(x, torch.empty(0, dtype=torch.long))
    torch.testing.assert_close(torch.autograd.grad(empty.sum(), x)[0], torch.zeros_like(x))


def test_unique_rows_inverse_permutation_backward():
    torch.manual_seed(17)
    x = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    order = torch.randperm(9)
    restore = order.argsort()
    y = _UniqueRows.apply(x, restore, order)
    weight = torch.randn_like(y)
    torch.testing.assert_close(y, x[restore])
    actual = torch.autograd.grad((y * weight).sum(), x)[0]
    expected = torch.autograd.grad((x[restore] * weight).sum(), x)[0]
    torch.testing.assert_close(actual, expected)


def dense_kernel(
    q, k, v, *, cumulative_seqlen_Q, cumulative_seqlen_KV, is_causal=False, return_lse=False, scale=None, **kwargs
):
    outputs, lses = [], []
    for a, b, c, d in zip(
        cumulative_seqlen_Q[:-1],
        cumulative_seqlen_Q[1:],
        cumulative_seqlen_KV[:-1],
        cumulative_seqlen_KV[1:],
        strict=True,
    ):
        qi, ki, vi = q[0, a:b].transpose(0, 1), k[0, c:d].transpose(0, 1), v[0, c:d].transpose(0, 1)
        ki = ki.repeat_interleave(qi.shape[0] // ki.shape[0], 0)
        vi = vi.repeat_interleave(qi.shape[0] // vi.shape[0], 0)
        scores = (qi @ ki.transpose(-1, -2)) * (qi.shape[-1] ** -0.5 if scale is None else scale)
        if is_causal:
            scores = scores.masked_fill(torch.ones_like(scores, dtype=torch.bool).triu(1), -torch.inf)
        outputs.append((scores.softmax(-1) @ vi).transpose(0, 1))
        lses.append(scores.logsumexp(-1).transpose(0, 1))
    out = torch.cat(outputs)[None]
    return (out, torch.cat(lses)[None]) if return_lse else out


def merge_kernel(outputs, lse_tensors, **kwargs):
    lse = torch.logaddexp(*lse_tensors)
    return sum(o * (l - lse).exp()[..., None] for o, l in zip(outputs, lse_tensors, strict=True)), lse


def install_cpu_kernels(monkeypatch):
    import cosmos_framework.model.attention as attention
    import cosmos_framework.model.attention.flash2 as flash2
    import cosmos_framework.model.attention.flash2.two_group as two_group

    def dense_two_group(q0, k0, v0, q1, k1, v1, cu_q, cu_k0, cu_k1, *lengths):
        parts = [
            dense_kernel(
                q[None], k[None], v[None], cumulative_seqlen_Q=cu_q, cumulative_seqlen_KV=cu_k, return_lse=True
            )
            for q, k, v, cu_k in [(q0, k0, v0, cu_k0), (q1, k1, v1, cu_k1)]
        ]
        return merge_kernel([p[0] for p in parts], [p[1] for p in parts])[0][0]

    monkeypatch.setattr(attention, "attention", dense_kernel)
    monkeypatch.setattr(attention, "merge_attentions", merge_kernel)
    monkeypatch.setattr(flash2, "flash2_attention", dense_kernel)
    monkeypatch.setattr(two_group, "two_group_varlen", dense_two_group)


def oracle(q, k, v, q4, k4, p, kn):
    n, h, dim = q.shape
    kt = k.repeat_interleave(h // k.shape[1], 1)
    vt = v.repeat_interleave(h // v.shape[1], 1)
    nt = kn.repeat_interleave(h // kn.shape[1], 1)
    scores = torch.einsum("ihd,jhd->hij", q, kt) / dim**0.5
    full = torch.cat((p.other_queries.indexes, p.geometry.indexes))
    scores[:, full] = torch.einsum("ihd,jhd->hij", q[full], nt) / dim**0.5
    ids = p.geometry.indexes
    four = torch.einsum("ihd,jhd->hij", q4, k4.repeat_interleave(h // k4.shape[1], 1)) / dim**0.5
    scores[:, ids[:, None], ids[None]] = four
    allowed = torch.zeros(n, n, dtype=torch.bool, device=q.device)
    for i in range(len(p.all_keys.offsets) - 1):
        start, end = p.all_keys.offsets[i : i + 2]
        c = p.causal.offsets[i + 1] - p.causal.offsets[i]
        allowed[start : start + c, start : start + c] = torch.ones(
            int(c), int(c), dtype=torch.bool, device=q.device
        ).tril()
        allowed[start + c : end, start:end] = True
    return torch.einsum("hij,jhd->ihd", scores.masked_fill(~allowed[None], -torch.inf).softmax(-1), vt)


def four_axis_from_channels(q, k, q_depth, k_depth, p):
    """Share the unchanged 112 channels, preserving the real derivative graph."""
    channels = torch.tensor(DEPTH_CHANNELS, device=q.device)
    return (
        q[p.geometry.indexes].index_copy(-1, channels, q_depth),
        k[p.geometry.indexes].index_copy(-1, channels, k_depth),
    )


@pytest.mark.parametrize("shared_k", [False, True])
@pytest.mark.parametrize(
    "splits, ids",
    [
        ([2, 5, 3, 4], [4, 5, 12]),
        ([2, 5, 3, 4], []),
        ([0, 7, 0, 7], list(range(14))),
        ([0, 7, 3, 4], list(range(7)) + [12]),
    ],
)
def test_lifted160_matches_dense_scores_and_root_gradients(monkeypatch, splits, ids, shared_k):
    install_cpu_kernels(monkeypatch)
    torch.manual_seed(17)
    q, k, v, qd, kd, kn = [
        torch.randn(shape, dtype=torch.float64, requires_grad=True)
        for shape in [(14, 4, 128), (14, 2, 128), (14, 2, 128), (len(ids), 4, 16), (len(ids), 2, 16), (14, 2, 128)]
    ]
    if shared_k:
        kn = k
    p = make_partition([7, 7], splits, torch.tensor(ids, dtype=torch.long), torch.ones(len(ids)))
    q4, k4 = four_axis_from_channels(q, kn, qd, kd, p)
    actual = lifted_partition_attention(q, k, v, q4, k4, p, normalized_k=None if shared_k else kn)
    ref = oracle(q, k, v, q4, k4, p, kn)
    torch.testing.assert_close(actual, ref, atol=2e-12, rtol=2e-12)
    weight = torch.randn_like(ref)
    inputs = [q, k, v] + ([] if shared_k else [kn]) + ([qd, kd] if ids else [])
    a = torch.autograd.grad((actual * weight).sum(), inputs, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad((ref * weight).sum(), inputs, allow_unused=True)
    for x, y, base in zip(a, b, inputs, strict=True):
        torch.testing.assert_close(
            torch.zeros_like(base) if x is None else x,
            torch.zeros_like(base) if y is None else y,
            atol=2e-12,
            rtol=2e-12,
        )


@pytest.mark.parametrize("ids", [[4, 5, 12], [], list(range(2, 7)) + list(range(10, 14))])
@pytest.mark.parametrize("shared_k", [False, True])
def test_partition_backward_and_sample_isolation(monkeypatch, ids, shared_k):
    install_cpu_kernels(monkeypatch)
    torch.manual_seed(17)
    q, k, v, q4, k4, kn = [
        torch.randn(shape, dtype=torch.float64, requires_grad=True)
        for shape in [(14, 4, 128), (14, 2, 128), (14, 2, 128), (len(ids), 4, 128), (len(ids), 2, 128), (14, 2, 128)]
    ]
    p = make_partition([7, 7], [2, 5, 3, 4], torch.tensor(ids, dtype=torch.long), torch.ones(len(ids)))
    if shared_k:
        kn = k
    out = fused_partition_attention(q, k, v, q4, k4, p, normalized_k=None if shared_k else kn)
    ref = oracle(q, k, v, q4, k4, p, kn)
    torch.testing.assert_close(out, ref, atol=2e-12, rtol=2e-12)
    weight = torch.randn_like(out)
    args = [q, k, v] + ([] if shared_k else [kn]) + ([q4, k4] if ids else [])
    g1 = torch.autograd.grad((out * weight).sum(), args, retain_graph=True)
    g2 = torch.autograd.grad((ref * weight).sum(), args)
    for a, b in zip(g1, g2, strict=True):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    altered = v.detach().clone()
    altered[7:] += 100
    same = fused_partition_attention(q, k, altered, q4, k4, p, normalized_k=None if shared_k else kn)
    torch.testing.assert_close(out[:7], same[:7])


def test_depth_rotation_preserves_other_pairs():
    torch.manual_seed(2)
    raw = torch.randn(5, 4, 128)
    phase = torch.randn(5, 64).repeat(1, 2)
    freq = torch.logspace(torch.log10(torch.tensor(0.125)), torch.log10(torch.tensor(4.0)), 8)
    a = rotate_geometry(raw, phase.cos(), phase.sin(), torch.arange(5.0), freq)
    b = rotate_geometry(raw, phase.cos(), phase.sin(), torch.arange(5.0) + 1, freq)
    unchanged = [i for i in range(128) if i % 64 not in DEPTH_PAIRS]
    torch.testing.assert_close(a[:, :, unchanged], b[:, :, unchanged], atol=0, rtol=0)
    assert (a - b).abs().max() > 0.1


@pytest.mark.parametrize("implementation", ["partition", "lifted160"])
def test_four_modality_network_no_future_leak(monkeypatch, implementation):
    import numpy as np

    import cosmos_framework.model.generator.pointflow_branch as branch
    from cosmos_framework.data.fk_batch import build_fk_batch
    from cosmos_framework.data.fk_batch_test import fk_sample
    from cosmos_framework.data.pointflow_batch import build_pointflow_batch
    from cosmos_framework.data.pointflow_batch_test import point_sample
    from cosmos_framework.model.generator.pointflow_branch_test import GeometrySurrogate
    from cosmos_framework.scripts.validate_pointflow_network import make_network, make_sequence

    monkeypatch.delenv("POINTFLOW_REFERENCE_ATTENTION", raising=False)
    monkeypatch.setattr(branch, "SonataGeometryEncoder", GeometrySurrogate)
    install_cpu_kernels(monkeypatch)
    sample = point_sample(3)
    sample["inputs"]["anchor_xyz"][:] = 0.3
    pf = build_pointflow_batch([sample])
    seq = make_sequence(pf)
    fk = fk_sample()
    fk["inputs"]["anchor_uv"] = np.ones((21, 2), np.float32)
    seq.fk_data = build_fk_batch([fk])
    net = make_network(2048).eval()
    net.install_pointflow("surrogate", timing=pf.timing, content_dim=16, decoder_dim=16)
    net.install_fk(timing=seq.fk_data.timing, content_dim=16, decoder_dim=16)
    net.install_local_geometry_rope(implementation=implementation)
    point_state = torch.randn_like(pf.displacement, requires_grad=True)
    fk_state = torch.randn_like(seq.fk_data.displacement, requires_grad=True)
    kwargs = dict(
        pointflow_displacement=point_state,
        pointflow_sigma=torch.tensor([0.4]),
        fk_displacement=fk_state,
        fk_sigma=torch.tensor([0.4]),
    )
    out = net(seq, **kwargs)
    (out["preds_pointflow"].square().mean() + out["preds_fk"].square().mean()).backward()
    assert point_state.grad.abs().sum() > 0 and fk_state.grad.abs().sum() > 0
    pf.displacement.fill_(99)
    pf.valid.fill_(False)
    seq.fk_data.displacement.fill_(88)
    seq.fk_data.valid.fill_(False)
    with torch.no_grad():
        other = net(seq, **kwargs)
    for key in ["preds_pointflow", "preds_fk"]:
        torch.testing.assert_close(out[key], other[key])
    if implementation == "lifted160":
        # Exercise real packing and rotations, not only synthetic lifted inputs.
        net.config.local_geometry_rope["implementation"] = "partition"
        with torch.no_grad():
            partition_out = net(seq, **kwargs)
        for key in ["preds_pointflow", "preds_fk"]:
            torch.testing.assert_close(out[key], partition_out[key], atol=1e-5, rtol=1e-5)
