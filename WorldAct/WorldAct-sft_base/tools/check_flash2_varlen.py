# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate + benchmark flash2 varlen vs natten (the only sm80 varlen backend).

Upstream bans flash2 varlen "due to instability"; COSMOS_FLASH2_VARLEN=1 lifts
the ban locally (flash2/checks.py). Before training with it, this script checks
on training-like packed shapes:

  1. forward allclose vs natten (bf16 tolerances),
  2. backward grad allclose vs natten,
  3. NaN/Inf cleanliness and forward determinism,
  4. fwd+bwd wall time, natten vs flash2.

Run:  COSMOS_FLASH2_VARLEN=1 python tools/check_flash2_varlen.py
"""

import os

os.environ.setdefault("COSMOS_FLASH2_VARLEN", "1")

import torch

from cosmos_framework.model.attention.frontend import attention
from cosmos_framework.model.attention.masks import CausalType

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def make_pack(segments, heads_q, heads_kv, head_dim, seed=0):
    total = sum(segments)
    cu = torch.tensor([0] + list(torch.tensor(segments).cumsum(0)), device=DEVICE, dtype=torch.int32)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn(1, total, heads_q, head_dim, device=DEVICE, dtype=DTYPE, generator=g)
    k = torch.randn(1, total, heads_kv, head_dim, device=DEVICE, dtype=DTYPE, generator=g)
    v = torch.randn(1, total, heads_kv, head_dim, device=DEVICE, dtype=DTYPE, generator=g)
    return q, k, v, cu, max(segments)


def run(backend, q, k, v, cu, max_len, causal, grad_out=None):
    q, k, v = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = attention(
        q,
        k,
        v,
        is_causal=causal,
        causal_type=CausalType.DontCare if causal else None,
        cumulative_seqlen_Q=cu,
        cumulative_seqlen_KV=cu,
        max_seqlen_Q=max_len,
        max_seqlen_KV=max_len,
        backend=backend,
    )
    if grad_out is None:
        return out, None
    grads = torch.autograd.grad(out, (q, k, v), grad_out)
    return out, grads


def check_case(name, segments, heads_q, heads_kv, head_dim, causal):
    q, k, v, cu, max_len = make_pack(segments, heads_q, heads_kv, head_dim)
    g = torch.Generator(device=DEVICE).manual_seed(1)
    grad_out = torch.randn(1, sum(segments), heads_q, head_dim, device=DEVICE, dtype=DTYPE, generator=g)

    out_n, grads_n = run("natten", q, k, v, cu, max_len, causal, grad_out)
    out_f, grads_f = run("flash2", q, k, v, cu, max_len, causal, grad_out)
    out_f2, _ = run("flash2", q, k, v, cu, max_len, causal)

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()

    nan_free = not (out_f.isnan().any() or any(t.isnan().any() for t in grads_f))
    deterministic_fwd = torch.equal(out_f, out_f2)
    print(
        f"{name}: fwd_rel={rel(out_f, out_n):.2e} "
        f"grad_rel=[{', '.join(f'{rel(a, b):.2e}' for a, b in zip(grads_f, grads_n, strict=True))}] "
        f"nan_free={nan_free} deterministic_fwd={deterministic_fwd}"
    )
    assert nan_free, f"{name}: flash2 produced NaN/Inf"
    assert rel(out_f, out_n) < 2e-2, f"{name}: forward mismatch vs natten"
    for a, b, tag in zip(grads_f, grads_n, "qkv", strict=True):
        assert rel(a, b) < 5e-2, f"{name}: grad_{tag} mismatch vs natten"


def bench_case(name, segments, heads, head_dim, causal, iters=10):
    q, k, v, cu, max_len = make_pack(segments, heads, heads, head_dim)
    results = {}
    for backend in ("natten", "flash2"):
        for _ in range(3):
            run(backend, q, k, v, cu, max_len, causal, torch.ones_like(q))
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run(backend, q, k, v, cu, max_len, causal, torch.ones_like(q))
        end.record()
        torch.cuda.synchronize()
        results[backend] = start.elapsed_time(end) / iters
    speedup = results["natten"] / results["flash2"]
    print(f"{name}: natten={results['natten']:.1f}ms flash2={results['flash2']:.1f}ms speedup={speedup:.2f}x")


def main():
    torch.manual_seed(0)
    print("== correctness (vs natten, bf16) ==")
    check_case("mha-causal mixed-lengths", [13700, 13596, 257, 1, 3], 16, 16, 128, causal=True)
    check_case("mha-full mixed-lengths", [13700, 13596, 257, 1, 3], 16, 16, 128, causal=False)
    check_case("gqa-causal mixed-lengths", [8192, 4096, 17, 1], 16, 8, 128, causal=True)
    check_case("mha-causal single-segment", [16384], 16, 16, 128, causal=True)
    print("== benchmark (fwd+bwd, 8 x 13600 packed, causal) ==")
    bench_case("bench mha 16h d128", [13600] * 8, 16, 128, causal=True)


if __name__ == "__main__":
    main()
