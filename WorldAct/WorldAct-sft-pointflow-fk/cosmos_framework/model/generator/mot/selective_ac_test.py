"""Selective checkpoint shape filtering and gradients, without CUDA."""

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.checkpoint import CheckpointPolicy

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.toml_config.sft_config import ActivationCheckpointingConfig as TomlACConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import _apply_selective_ac, _selective_ac_policy


def test_projection_shapes_exclude_mlp_and_other_matmuls():
    policy = _selective_ac_policy(
        ActivationCheckpointingConfig(mode="selective", save_ops_regex=[], save_mm_shapes=[[2048, 2048], [2048, 1024]])
    )
    for width_in, width_out, expected in [
        (2048, 2048, True),
        (2048, 1024, True),
        (2048, 9216, False),
        (9216, 2048, False),
    ]:
        x = torch.empty(13, width_in, device="meta")
        # Match actual nn.Linear's transposed (noncontiguous) weight operand.
        weight = torch.empty(width_out, width_in, device="meta").t()
        actual = policy(None, torch.ops.aten.mm.default, x, weight)
        assert actual == (CheckpointPolicy.MUST_SAVE if expected else CheckpointPolicy.MUST_RECOMPUTE)
    assert policy(None, torch.ops.aten.bmm.default) == CheckpointPolicy.MUST_RECOMPUTE
    assert policy(None, torch.ops.aten.addmm.default) == CheckpointPolicy.MUST_RECOMPUTE


def test_empty_shapes_preserve_defaults_and_names_remain_additive():
    x = torch.empty(3, 4, device="meta")
    w = torch.empty(4, 4, device="meta")
    assert (
        _selective_ac_policy(ActivationCheckpointingConfig())(None, torch.ops.aten.mm.default, x, w)
        == CheckpointPolicy.MUST_RECOMPUTE
    )
    assert (
        _selective_ac_policy(ActivationCheckpointingConfig(save_ops_regex=["mm"]))(
            None, torch.ops.aten.mm.default, x, w
        )
        == CheckpointPolicy.MUST_SAVE
    )
    for shapes in ([[2]], [[2, -1]], [[2, 3, 4]]):
        with pytest.raises(ValueError, match="save_mm_shapes"):
            _selective_ac_policy(ActivationCheckpointingConfig(save_mm_shapes=shapes))
    config = TomlACConfig(mode="selective", save_mm_shapes=[[2048, 1024]])
    assert config.save_mm_shapes == [[2048, 1024]]
    structured = OmegaConf.structured(ActivationCheckpointingConfig())
    merged = OmegaConf.merge(structured, OmegaConf.from_dotlist(["save_mm_shapes=[[2048,2048],[2048,1024]]"]))
    assert OmegaConf.to_object(merged).save_mm_shapes == [[2048, 2048], [2048, 1024]]


def test_selective_projection_parameter_and_input_gradients():
    torch.manual_seed(17)
    module = torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(8, 32, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 8, bias=False),
    ).double()
    x = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    inputs = [x, *module.parameters()]
    expected = module(x)
    expected_grads = torch.autograd.grad(expected.square().mean(), inputs)
    wrapped = _apply_selective_ac(
        module, ActivationCheckpointingConfig(mode="selective", save_ops_regex=[], save_mm_shapes=[[8, 8]])
    )
    actual = wrapped(x)
    actual_grads = torch.autograd.grad(actual.square().mean(), inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
