"""Two disjoint varlen key groups with one softmax, using only FlashAttention2.

The two groups may use different rotations of the same query. Both Flash2
backward calls must receive the *global* output and LSE: weighting independent
attention gradients would omit the derivative of the softmax normalizer.
"""

import torch


class _TwoGroupVarlen(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q0, k0, v0, q1, k1, v1, cu_q, cu_k0, cu_k1, max_q, max_k0, max_k1):
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        tensors = (q0, k0, v0, q1, k1, v1)
        if any(t.ndim != 3 or t.shape[-1] != 128 for t in tensors):
            raise ValueError("Local geometry Flash2 currently requires [tokens, heads, 128] Q/K/V")
        if q0.shape != q1.shape:
            raise ValueError("Both key groups must share the query layout")
        scale = q0.shape[-1] ** -0.5

        def attend(q, k, v, cu_k, max_k):
            out, lse, _ = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                max_q,
                max_k,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=False,
                return_attn_probs=True,
            )
            return out, lse

        a, la = attend(q0, k0, v0, cu_k0, max_k0)
        b, lb = attend(q1, k1, v1, cu_k1, max_k1)
        # LSE is float32 [heads, total_queries]. No attention-score matrix is
        # materialized. Callers exclude queries with an empty key group.
        lse = torch.logaddexp(la, lb)
        weight_b = torch.sigmoid(lb - la).transpose(0, 1).unsqueeze(-1)
        out = (a.float() + weight_b * (b.float() - a.float())).to(a.dtype)
        ctx.save_for_backward(*tensors, out, lse, cu_q, cu_k0, cu_k1)
        ctx.lengths = (max_q, max_k0, max_k1)
        ctx.scale = scale
        ctx.deterministic = torch.are_deterministic_algorithms_enabled()
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_backward

        # Unpack exactly once: non-reentrant activation checkpointing consumes
        # each saved tensor on first access. Never overwrite upstream buffers.
        q0, k0, v0, q1, k1, v1, out, lse, cu_q, cu_k0, cu_k1 = ctx.saved_tensors
        max_q, max_k0, max_k1 = ctx.lengths
        grads = []
        for q, k, v, cu_k, max_k in ((q0, k0, v0, cu_k0, max_k0), (q1, k1, v1, cu_k1, max_k1)):
            dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
            _wrapped_flash_attn_varlen_backward(
                dout.contiguous(),
                q,
                k,
                v,
                out,
                lse,
                dq,
                dk,
                dv,
                cu_q,
                cu_k,
                max_q,
                max_k,
                0.0,
                ctx.scale,
                False,
                -1,
                -1,
                0.0,
                None,
                ctx.deterministic,
            )
            grads.extend((dq, dk, dv))
        return (*grads, None, None, None, None, None, None)


def two_group_varlen(q0, k0, v0, q1, k1, v1, cu_q, cu_k0, cu_k1, max_q, max_k0, max_k1):
    """Noncausal, dropout-free attention; each query has keys in both groups."""
    return _TwoGroupVarlen.apply(q0, k0, v0, q1, k1, v1, cu_q, cu_k0, cu_k1, max_q, max_k0, max_k1)
