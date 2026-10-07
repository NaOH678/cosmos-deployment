"""Opt-in single-GPU soft-prefix guidance HTTP experiment.

Identity Jacobian approximation, NOT the exact RTC VJP. Never clamps output
commands. The client executes its committed prefix and consumes only the suffix.
"""
from dataclasses import replace
import os
import threading
import numpy as np

METHOD = "identity_jacobian_soft_prefix_v1"
_context = threading.local()


def validate_prefix(payload):
    if not isinstance(payload, dict) or payload.get("method") != METHOD:
        raise ValueError("RTC experiment requires a versioned prefix on every request")
    hard = payload.get("committed_steps")
    if type(hard) is not int or not 0 <= hard <= 12:
        raise ValueError("RTC prefix count must be an integer in [0,12]")
    values = np.asarray(payload.get("actions"), dtype=np.float32)
    if hard == 0:
        if values.shape != (0,):
            raise ValueError("bootstrap must have an empty RTC prefix")
        return values.reshape(0, 27), np.empty(0, dtype=np.float32), hard
    if (values.ndim != 2 or values.shape[1] != 27
            or not hard <= len(values) <= hard + 2 or not np.isfinite(values).all()):
        raise ValueError("RTC prefix must contain finite 27D EEF/hand actions")
    if not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1.0, atol=0.02):
        raise ValueError("RTC prefix quaternion is not normalized")
    weights = np.ones(len(values), dtype=np.float32)
    for j in range(hard, len(values)):
        w = (len(values) - j) / (len(values) - hard + 1)
        weights[j] = w * np.expm1(w) / np.expm1(1.0)
    return values, weights, hard


def install():
    import torch
    from cosmos_framework.inference.robot_policy import adapters
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    original_forward = UniPCSampler.forward
    base_adapter = adapters.SingleRightHandCosmosAdapter

    def forward(self, velocity_fn, noise, *args, **kwargs):
        prefix = getattr(_context, "prefix", None)
        if prefix is None or len(prefix[0]) == 0:
            return original_forward(self, velocity_fn, noise, *args, **kwargs)
        values, weights, _ = prefix
        def guided(xs, timestep):
            if len(xs) != 1 or xs[0].ndim != 1 or xs[0].numel() <= 33 * 64:
                raise ValueError("RTC supports the validated flat [vision|33x64 action] layout only")
            sigma = float(timestep.item()) / 1000.0
            if not 0 < sigma <= 1.001:
                raise ValueError("RTC unexpected diffusion timestep")
            x = xs[0]
            v = velocity_fn(xs, timestep)[0]
            action = (x - sigma * v)[-33 * 64:].reshape(33, 64)
            target = torch.as_tensor(values, device=x.device, dtype=x.dtype)
            weight = torch.as_tensor(weights, device=x.device, dtype=x.dtype)
            # Leave vision, condition state row zero and padded dimensions alone.
            correction = torch.zeros_like(v)
            correction[-33 * 64:].reshape(33, 64)[1:1+len(values), :27] = (
                target - action[1:1+len(values), :27]) * weight[:, None]
            gain = min(10.0, ((1-sigma)**2 + sigma**2) / max(sigma*(1-sigma), 1e-6))
            _context.calls += 1
            return [v - gain * correction]
        return original_forward(self, guided, noise, *args, **kwargs)

    class PrefixAdapter(base_adapter):
        def __init__(self, config):
            m = config.model
            if (m.native_action_dim != 27 or m.native_chunk_size != 32
                    or m.max_action_dim != 64 or m.sampler != "unipc"
                    or config.deployment.action_space != "eef"
                    or not config.deployment.model_id.endswith("-rtc-approx-v1")):
                raise ValueError("RTC service config/layout/model ID mismatch")
            super().__init__(config)
            if getattr(self.model.config, "sound_gen", False):
                raise ValueError("RTC action-tail layout does not support sound")

        def infer(self, observation):
            prefix = validate_prefix(observation.get("rtc_prefix"))
            _context.prefix = prefix
            _context.calls = 0
            try:
                output = super().infer(observation)
                if prefix[2] and _context.calls != self.config.model.num_steps:
                    raise RuntimeError("RTC guidance did not run at every sampler step")
                return replace(output, rtc_prefix_guidance=dict(
                    method=METHOD, committed_steps=prefix[2],
                    soft_steps=len(prefix[0])-prefix[2], sampler_calls=_context.calls))
            finally:
                _context.prefix = None
    UniPCSampler.forward = forward
    adapters.SingleRightHandCosmosAdapter = PrefixAdapter


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("RTC experiment is single GPU only")
    install()
    from cosmos_framework.inference.robot_policy.cfg_http import main as serve
    serve()

if __name__ == "__main__":
    main()
