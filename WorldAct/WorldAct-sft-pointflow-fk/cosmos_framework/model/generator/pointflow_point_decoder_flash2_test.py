"""Flash2 point refinement preserves ragged attention and its gradients."""

import copy

import pytest
import torch
from torch.nn import functional as F

from cosmos_framework.model.generator.pointflow_point_decoder import PointDecoderBlock, _varlen_layout


def test_layout_skips_empty_samples():
    cu, maximum = _varlen_layout([0, 3, 3, 8, 8], 8, "cpu")
    assert cu.tolist() == [0, 3, 8]
    assert maximum == 5
    for offsets in ([3, 2, 8], [3, 9], [3, 7]):
        with pytest.raises(ValueError):
            _varlen_layout(offsets, 8, "cpu")
    cu, maximum = _varlen_layout([0, 0], 0, "cpu")
    assert cu.tolist() == [0] and maximum == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Flash2 requires CUDA")
@pytest.mark.parametrize("offsets", [[0, 17, 17, 54, 54], [1024, 2048]])
def test_flash2_matches_reference_values_gradients_and_isolation(offsets):
    torch.manual_seed(42)
    module = PointDecoderBlock(256, 4).cuda().bfloat16()
    reference = copy.deepcopy(module).float()
    x = torch.randn(offsets[-1], 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    actual = module._attention(x, offsets)
    qkv = reference.qkv(xr).reshape(len(x), 3, 4, 64).permute(1, 2, 0, 3)
    outputs, start = [], 0
    for end in offsets:
        if end > start:
            q, k, v = qkv[:, :, start:end].unbind(0)
            outputs.append(F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(end - start, 256))
        start = end
    expected = torch.cat(outputs)
    torch.testing.assert_close(actual.float(), expected, atol=0.006, rtol=0.03)
    upstream = torch.randn_like(actual) / len(x)
    actual.backward(upstream)
    expected.backward(upstream.float())
    torch.testing.assert_close(x.grad.float(), xr.grad, atol=0.0001, rtol=0.05)
    for a, b in zip(module.qkv.parameters(), reference.qkv.parameters(), strict=True):
        torch.testing.assert_close(a.grad.float(), b.grad, atol=0.001, rtol=0.05)
    first_end = next(end for end in offsets if end > 0)
    changed = x.detach().clone()
    changed[first_end:] += 3
    with torch.no_grad():
        isolated = module._attention(changed, offsets)
    torch.testing.assert_close(isolated[:first_end], actual[:first_end], atol=0, rtol=0)
