"""Experimental single-GPU batched CFG; both branches retain full autograd."""

from dataclasses import fields, replace

import torch


def duplicate_single_sample(data):
    if data.batch_size != 1 or data.num_vision_items_per_sample is not None:
        raise ValueError("RTC batched CFG supports one single-view latent sample only")
    updates = {"batch_size": 2}
    for field in fields(data):
        value = getattr(data, field.name)
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Unexpected batch layout for {field.name}")
            updates[field.name] = value * 2
        elif isinstance(value, torch.Tensor):
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(f"Unexpected tensor batch layout for {field.name}")
            updates[field.name] = torch.cat([value, value], dim=0)
    return replace(data, **updates)


def install(model):
    dims = model.parallel_dims
    if dims is not None and (dims.cfgp_enabled or dims.cp_enabled or dims.dp_shard_enabled):
        raise ValueError("Batched RTC CFG requires single GPU")
    if model.config.sound_gen or model.config.joint_attn_implementation != "two_way":
        raise ValueError("Batched RTC CFG supports the tested vision/action two-way model only")

    def batched(
        *,
        net,
        noise_x,
        timestep,
        cond_tokens,
        uncond_tokens,
        sequence_plans,
        gen_data_clean,
        skip_text_tokens_for_cfg,
        has_noisy_actions,
    ):
        if len(noise_x) != 1 or skip_text_tokens_for_cfg:
            raise ValueError("Batched CFG requires single sample and explicit unconditional text")
        data = duplicate_single_sample(gen_data_clean)
        # Reuse the SAME differentiable input in both branches. Autograd sums
        # conditional and unconditional contributions to the original latent.
        result = model._get_velocity(
            net=net,
            noise_x=noise_x * 2,
            timestep=timestep.repeat(2, 1),
            text_tokens=cond_tokens + uncond_tokens,
            sequence_plans=sequence_plans * 2,
            gen_data_clean=data,
            skip_text_tokens=False,
            packed_sequence_template=None,
            memory=None,
            has_noisy_actions=has_noisy_actions,
        )
        if len(result) != 2:
            raise RuntimeError("Batched CFG returned an invalid number of branches")
        return [result[0]], [result[1]]

    model._rtc_batched_cfg = batched
