"""Small actual cuDNN test: invalid middle padding must not affect output."""

import json
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "source"))
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl  # noqa: E402
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata  # noqa: E402


def main():
    torch.manual_seed(0)
    attention = CuDNNAttentionImpl(
        num_heads=2, head_size=128, softmax_scale=128**-0.5, backend_explicit=True
    )
    q = torch.randn(2, 32, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 172, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones((2, 172), device="cuda", dtype=torch.bool)
    mask[1, 19:140] = False
    metadata = AttentionMetadata(attn_mask=mask)
    a = attention.forward_cuda(q, k, v, metadata)
    k[:, 19:140][1] = 1000
    v[:, 19:140][1] = -1000
    b = attention.forward_cuda(q, k, v, metadata)
    assert torch.equal(a, b), (a - b).abs().max().item()
    c = attention.forward_cuda(
        q[1:2],
        torch.cat([k[1:2, :19], k[1:2, 140:]], dim=1),
        torch.cat([v[1:2, :19], v[1:2, 140:]], dim=1),
    )
    out = {
        "invalid_padding_sentinel_max_diff": (a - b).abs().max().item(),
        "masked_vs_trimmed_bf16_max_diff": (b[1:2] - c).abs().max().item(),
        "backend": "explicit CuDNNAttentionImpl",
        "q_shape": list(q.shape),
        "text_lengths": [140, 19],
        "gen_tokens": 32,
    }
    (root / "padding_test.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
