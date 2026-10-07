from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.robot_policy.rtc_batch_cfg import duplicate_single_sample, install
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def data():
    return GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        raw_state_vision=[torch.ones(2)],
        x0_tokens_vision=[torch.ones(2)],
        fps_vision=torch.tensor([15]),
        x0_tokens_action=[torch.ones(33, 64)],
        action_domain_id=[torch.tensor(1)],
    )


def test_duplicate_condition_batch_preserves_original():
    original = data()
    copied = duplicate_single_sample(original)
    assert original.batch_size == 1 and len(original.x0_tokens_vision) == 1
    assert copied.batch_size == 2
    assert copied.x0_tokens_vision[0] is copied.x0_tokens_vision[1]
    torch.testing.assert_close(copied.fps_vision, torch.tensor([15, 15]))
    with pytest.raises(ValueError):
        duplicate_single_sample(copied)


def test_batched_cfg_accumulates_both_branch_gradients_on_same_latent():
    calls = []

    def velocity(**kw):
        calls.append(kw)
        assert kw["noise_x"][0] is kw["noise_x"][1]
        assert kw["text_tokens"] == [[1], [2]]
        assert kw["gen_data_clean"].batch_size == 2
        return [2 * kw["noise_x"][0] + 1, 4 * kw["noise_x"][1] - 1]

    model = SimpleNamespace(
        parallel_dims=None,
        config=SimpleNamespace(sound_gen=False, joint_attn_implementation="two_way"),
        _get_velocity=velocity,
    )
    install(model)
    x = torch.tensor([0.2, 0.4], requires_grad=True)
    c, u = model._rtc_batched_cfg(
        net=None,
        noise_x=[x],
        timestep=torch.ones(1, 1),
        cond_tokens=[[1]],
        uncond_tokens=[[2]],
        sequence_plans=[object()],
        gen_data_clean=data(),
        skip_text_tokens_for_cfg=False,
        has_noisy_actions=True,
    )
    combined = u[0] + 3 * (c[0] - u[0])
    torch.testing.assert_close(torch.autograd.grad(combined.sum(), x)[0], torch.full_like(x, -2))
    assert len(calls) == 1
