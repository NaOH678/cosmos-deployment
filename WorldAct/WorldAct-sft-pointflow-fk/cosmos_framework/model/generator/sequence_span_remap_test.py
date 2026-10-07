"""CPU metadata mapping agrees with the previous dense tensor map."""

from dataclasses import replace

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.modality import ModalitySpan
from cosmos_framework.model.generator.sequence_span_remap import SpanRemapper


@pytest.mark.parametrize("extra", [[6, 0, 3], [0, 0, 0], [2, 7, 1]])
def test_span_mapping_matches_dense_reference(extra):
    old = [5, 3, 7]
    new = [n + k for n, k in zip(old, extra, strict=True)]
    dense = torch.cat([torch.arange(n) + sum(new[:i]) for i, n in enumerate(old)])
    mapper = SpanRemapper(old, new)
    for start in range(sum(old)):
        for length in range(sum(old) - start + 1):
            span = ModalitySpan(start, length, 0, 0, length, (1, 1, length))
            expected_start = int(dense[start])
            if length and int(dense[start + length - 1]) != expected_start + length - 1:
                with pytest.raises(ValueError, match="crosses sample boundaries"):
                    mapper([span])
            else:
                assert mapper([span]) == [replace(span, sequence_start=expected_start)]
                assert span.sequence_start == start


def test_empty_samples_and_invalid_index():
    mapper = SpanRemapper([0, 2, 0, 3], [1, 4, 2, 3])
    assert [mapper.index(i) for i in range(5)] == [1, 2, 7, 8, 9]
    assert mapper.index(-1) == 9
    for i in [-6, 5]:
        with pytest.raises(IndexError):
            mapper.index(i)


@pytest.mark.parametrize("fk_first", [False, True])
def test_composed_payloads_keep_values_and_gradients(fk_first):
    from cosmos_framework.model.generator.fk_pointflow_compose_test import (
        FK_POINTS,
        POINT_CLUSTERS,
        _encoded,
        _positions,
        build_sequence,
        check_packing,
    )
    from cosmos_framework.model.generator.fk_sequence import attach_fk_tokens
    from cosmos_framework.model.generator.pointflow_sequence import attach_point_tokens

    check_packing("optimized", fk_first=fk_first)
    base, _, _ = build_sequence()
    point, fk = _encoded(POINT_CLUSTERS), _encoded(FK_POINTS)
    for encoded in (point, fk):
        for key in ("noisy_tokens", "anchor_tokens"):
            encoded[key].requires_grad_()
    operations = [(attach_fk_tokens, fk, FK_POINTS), (attach_point_tokens, point, POINT_CLUSTERS)]
    if not fk_first:
        operations.reverse()
    for attach, encoded, counts in operations:
        base = attach(base, encoded, _positions(counts))
    hidden = torch.zeros(base.sequence_length, point["noisy_tokens"].shape[-1])
    for payload in (base.point, base.fk):
        hidden = hidden.index_copy(0, payload.sequence_indexes, payload.tokens)
    for payload, encoded in ((base.point, point), (base.fk, fk)):
        torch.testing.assert_close(payload.hidden(hidden), encoded["noisy_tokens"])
    hidden.square().sum().backward()
    for encoded in (point, fk):
        for key in ("noisy_tokens", "anchor_tokens"):
            torch.testing.assert_close(encoded[key].grad, 2 * encoded[key].detach())
