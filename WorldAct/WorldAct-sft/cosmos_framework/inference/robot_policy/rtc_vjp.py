"""RTC pseudoinverse guidance ported to Cosmos' decreasing-sigma convention.

Reference: Physical-Intelligence/real-time-chunking-kinetix src/model.py,
FlowPolicy.realtime_action. x_clean=x-sigma*v, hence corrected v=v-g*J.T*error.
The model, CFG and joint video/action dependencies remain in the Jacobian.
Conditioning is fixed; only unconditioned latent coordinates are variables.
"""

import math
import time

import torch

METHOD = "rtc_vjp_soft_prefix_v1"


def prefix_weights(start, end, total, *, device=None, dtype=torch.float32):
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= total:
        raise ValueError("invalid RTC prefix bounds")
    index = torch.arange(total, device=device, dtype=dtype)
    w = ((start - 1 - index) / (end - start + 1) + 1).clamp(0, 1)
    w = w * torch.expm1(w) / math.expm1(1.0)
    return torch.where(index < end, w, 0)


def guided_velocity(velocity_fn, latent, sigma, target, weights, *, free_mask=None, max_guidance=10.0, profile=False):
    """Exact reverse-mode VJP. No identity/stop-gradient/finite-difference fallback.

    target and weights have latent shape; weights zero outside action prefix.
    Caller maps Cosmos timestep/1000 to sigma. Returned tensors are detached
    so sampling never backpropagates through multiple denoising steps.
    """
    if not 0 < sigma <= 1.0 or not math.isfinite(max_guidance) or max_guidance <= 0:
        raise ValueError("invalid sigma or RTC guidance cap")
    if latent.shape != target.shape or latent.shape != weights.shape:
        raise ValueError("RTC latent/target/weights shape mismatch")
    with torch.inference_mode(False), torch.enable_grad():
        x = latent.detach().clone().requires_grad_(True)
        if free_mask is None:
            free_mask = torch.ones_like(x)
        free_mask = free_mask.to(device=x.device, dtype=x.dtype).detach()
        if free_mask.shape != x.shape:
            raise ValueError("RTC free-coordinate mask shape mismatch")
        model_input = x * free_mask + latent.detach() * (1 - free_mask)
        if profile and x.is_cuda:
            torch.cuda.synchronize(x.device)
        forward_started = time.perf_counter()
        v = velocity_fn(model_input)
        if profile and x.is_cuda:
            torch.cuda.synchronize(x.device)
        forward_ms = (time.perf_counter() - forward_started) * 1000
        if not v.requires_grad:
            raise RuntimeError("RTC model velocity is detached; exact VJP unavailable")
        clean = model_input - sigma * v
        error = ((target - clean) * weights).detach()
        backward_started = time.perf_counter()
        correction = torch.autograd.grad(clean, x, grad_outputs=error, create_graph=False, retain_graph=False)[0]
        if profile and x.is_cuda:
            torch.cuda.synchronize(x.device)
        backward_ms = (time.perf_counter() - backward_started) * 1000
        gain = (
            max_guidance if sigma == 1.0 else min(((1 - sigma) ** 2 + sigma**2) / (sigma * (1 - sigma)), max_guidance)
        )
        corrected = ((v - gain * correction) * free_mask).detach()
        if not torch.isfinite(corrected).all():
            raise RuntimeError("RTC VJP produced nonfinite velocity")
        return corrected, dict(
            **(dict(forward_ms=forward_ms, backward_ms=backward_ms) if profile else {}),
            gain=gain,
            correction_norm=float(correction.norm().detach()),
            error_norm=float(error.norm()),
            jacobian_effect_norm=float((correction - error * free_mask).norm().detach()),
        )


def checkpoint_decoder_layers(model, *, every=1):
    """Recompute each transformer layer for backward; preserve the exact VJP.

    No CPU offload, gradients for weights, or changed model parameters.
    Non-reentrant checkpoint supports nested SequencePack inputs and outputs.
    """
    from torch.utils.checkpoint import checkpoint

    from cosmos_framework.model.generator.mot.unified_mot import MoTDecoderLayer

    if type(every) is not int or every < 0:
        raise ValueError("checkpoint interval must be nonnegative (zero disables recomputation)")
    count = 0
    wrapped = 0
    for module in model.modules():
        if isinstance(module, MoTDecoderLayer) and not getattr(module, "_rtc_checkpointed", False):
            if every == 0 or count % every != 0:
                count += 1
                continue
            original = module.forward

            def run(*args, _forward=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)

            module.forward = run
            module._rtc_checkpointed = True
            count += 1
            wrapped += 1
    if not count:
        raise RuntimeError("RTC could not locate MoT decoder layers for recomputation")
    return wrapped


def deterministic_attention_backward():
    """Opt-in exact-worker setting for repeatable FlashAttention gradients.

    The default attention backend allows nondeterministic BF16 accumulation.
    Install before compilation; only the caller's worker process is affected.
    """
    from cosmos_framework.model.generator.mot import attention as mot_attention
    from cosmos_framework.model.generator.mot import inference_text_kv_memory

    for module in (mot_attention, inference_text_kv_memory):
        original = module.attention

        def call(*args, _attention=original, **kwargs):
            kwargs["deterministic"] = True
            return _attention(*args, **kwargs)

        module.attention = call


def configure_exact_vjp_backend(*, deterministic=True):
    """Install the measured single-GPU exact-VJP backend before model loading.

    Keep text KV reuse under the model's existing two-way-attention/layout
    checks. Compile language forward/backward, no CUDA Graph, no activation
    checkpointing by default. Caller must freeze model parameters and execute
    generation under no_grad (not inference_mode), enabling grad only for VJP.
    This is process-local and is never installed by the ordinary HTTP service.
    """
    import os

    from cosmos_framework.inference.args import OmniSetupOverrides

    os.environ["COSMOS_FLASH2_VARLEN"] = "1"
    original = OmniSetupOverrides.build_setup

    def setup(self, *args, **kwargs):
        self.cfgp_size = self.cp_size = self.dp_shard_size = self.dp_replicate_size = 1
        self.use_cuda_graphs = False
        self.use_torch_compile = True
        self.compiled_region = "language"
        return original(self, *args, **kwargs)

    OmniSetupOverrides.build_setup = setup
    if deterministic:
        deterministic_attention_backward()
    return dict(
        method=METHOD,
        compiled_region="language",
        cuda_graphs=False,
        text_kv="model_guarded",
        deterministic_backward=deterministic,
        activation_checkpointing=False,
        flash2_varlen=True,
    )


def skip_zero_video_output_backward(model):
    """For ACTION-ONLY RTC error, omit the zero-cotangent video decoder branch.

    This is valid only when RTC weights are zero on all vision outputs.
    It does NOT detach visual hidden states or the action-to-video-input VJP:
    action predictions still differentiate through the complete joint backbone.
    Video velocity values and video generation are unchanged.
    """
    original = model.denoise

    def denoise(*args, **kwargs):
        output = original(*args, **kwargs)
        if torch.is_grad_enabled():
            output = dict(output)
            output["preds_vision"] = [value.detach() for value in output["preds_vision"]]
        return output

    model.denoise = denoise
