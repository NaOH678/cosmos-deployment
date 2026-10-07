"""Small CUDA forward/backward correctness check; not a throughput benchmark."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("COSMOS_FLASH2_VARLEN", "1")
os.environ.setdefault("I4_ATTN_BACKENDS", "flash2")
import torch
from torch.utils.checkpoint import checkpoint

from cosmos_framework.model.generator.pointflow_fk_attention import (
    fused_partition_attention,
    lifted_partition_attention,
    make_partition,
)
from cosmos_framework.model.generator.pointflow_fk_attention_test import four_axis_from_channels, oracle


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint", action="store_true", help="Exercise non-reentrant activation recomputation")
    parser.add_argument("--selective", action="store_true", help="Use the production per-op checkpoint wrapper")
    parser.add_argument("--save-ops-regex", nargs="+", default=["mm", "_flash_attn.*forward"])
    parser.add_argument("--compile", action="store_true", help="Compile with Inductor, fullgraph=True and dynamic=True")
    parser.add_argument("--implementation", choices=["partition", "lifted160"], default="partition")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    results = []
    for lengths, splits, ids in [
        ([77, 61], [5, 72, 3, 58], [9, 20, 45, 90, 120]),
        ([77, 61], [5, 72, 3, 58], []),
        ([77, 61], [0, 77, 0, 61], list(range(138))),
        ([77, 61], [0, 77, 3, 58], list(range(77)) + [90, 120]),
    ]:
        n = sum(lengths)
        depth_dim = 16 if args.implementation == "lifted160" else 128
        tensors = [
            torch.randn(s, device=device, dtype=torch.bfloat16, requires_grad=True)
            for s in [
                (n, 4, 128),
                (n, 2, 128),
                (n, 2, 128),
                (len(ids), 4, depth_dim),
                (len(ids), 2, depth_dim),
                (n, 2, 128),
            ]
        ]
        q, k, v, q4, k4, kn = tensors
        p = make_partition(
            lengths, splits, torch.tensor(ids, device=device, dtype=torch.long), torch.ones(len(ids), device=device)
        )

        def run(q, k, v, q4, k4, kn):
            if args.implementation == "lifted160":
                q4, k4 = four_axis_from_channels(q, kn, q4, k4, p)
                return lifted_partition_attention(q, k, v, q4, k4, p, normalized_k=kn)
            return fused_partition_attention(q, k, v, q4, k4, p, normalized_k=kn)

        def checked(*inputs):
            return checkpoint(run, *inputs, use_reentrant=False) if args.checkpoint else run(*inputs)

        if args.selective:
            from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
            from cosmos_framework.model.generator.mot.parallelize_unified_mot import _apply_selective_ac

            class AttentionCheck(torch.nn.Module):
                def forward(self, *inputs):
                    return run(*inputs)

            checked = _apply_selective_ac(
                AttentionCheck(),
                ActivationCheckpointingConfig(mode="selective", save_ops_regex=args.save_ops_regex),
            )
        execute = torch.compile(checked, fullgraph=True, dynamic=True) if args.compile else checked
        out = execute(*tensors)
        exact = [t.detach().double().requires_grad_() for t in tensors]
        if args.implementation == "lifted160":
            eq4, ek4 = four_axis_from_channels(exact[0], exact[5], exact[3], exact[4], p)
            ref = oracle(*exact[:3], eq4, ek4, p, exact[5])
        else:
            ref = oracle(*exact[:5], p, exact[5])
        weight = torch.randn_like(out)
        used = [0, 1, 2, 5] + ([3, 4] if ids else [])
        actual_grads = torch.autograd.grad((out * weight).sum(), [tensors[i] for i in used], allow_unused=True)
        if args.checkpoint or args.selective or args.compile:
            plain = [t.detach().clone().requires_grad_() for t in tensors]
            plain_out = run(*plain)
            plain_grads = torch.autograd.grad((plain_out * weight).sum(), [plain[i] for i in used], allow_unused=True)
            torch.testing.assert_close(
                out, plain_out, atol=0.008 if args.compile else 0, rtol=0.01 if args.compile else 0
            )
            for a, b in zip(actual_grads, plain_grads, strict=True):
                assert (a is None) == (b is None)
                if a is not None:
                    torch.testing.assert_close(a, b, atol=0.002, rtol=0.01)
        ref_grads = torch.autograd.grad((ref * weight.double()).sum(), [exact[i] for i in used], allow_unused=True)
        errors = []
        for a, b in zip(actual_grads, ref_grads, strict=True):
            if a is None or b is None:
                assert a is None and (b is None or b.count_nonzero() == 0)
                continue
            assert torch.isfinite(a).all()
            errors.append(float((a.double() - b).abs().max()))
            torch.testing.assert_close(a.double(), b, atol=0.025, rtol=0.035)
        error = float((out.double() - ref).abs().max().detach())
        torch.testing.assert_close(out.double(), ref, atol=0.015, rtol=0.025)
        results.append(dict(tokens=n, geometry=len(ids), output_max_error=error, gradient_max_error=max(errors)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        device=torch.cuda.get_device_name(),
        dtype="bfloat16",
        seed=args.seed,
        activation_checkpointing=args.checkpoint or args.selective,
        checkpoint_mode="selective" if args.selective else "full" if args.checkpoint else "none",
        save_ops_regex=args.save_ops_regex if args.selective else [],
        torch_compile=args.compile,
        backend="flash2_varlen_only",
        implementation=args.implementation,
        cases=results,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        timing="Correctness check only; no throughput claim.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
