"""Opt-in single-GPU soft-prefix guidance HTTP experiment.

Hybrid RTC: exact VJP at sampler index 2; identity approximation elsewhere. Never clamps output
commands. The client executes its committed prefix and consumes only the suffix.
"""
from dataclasses import replace
import os
import threading
import numpy as np

METHOD = "hybrid_vjp_step2_soft_prefix_v1"
_context = threading.local()


def validate_prefix(payload):
    if not isinstance(payload, dict) or payload.get("method") != METHOD:
        raise ValueError("RTC experiment requires a versioned prefix on every request")
    hard = payload.get("committed_steps")
    if type(hard) is not int or not 0 <= hard <= 16:
        raise ValueError("RTC prefix count must be an integer in [0,16]")
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
            from .rtc_vjp import guided_velocity
            target = torch.zeros_like(x)
            mask = torch.zeros_like(x)
            target[-33*64:].reshape(33,64)[1:1+len(values), :27] = torch.as_tensor(values, device=x.device, dtype=x.dtype)
            mask[-33*64:].reshape(33,64)[1:1+len(values), :27] = torch.as_tensor(weights, device=x.device, dtype=x.dtype)[:,None]
            index = _context.calls
            if index == 2:
                result, metrics = guided_velocity(lambda z: velocity_fn([z], timestep)[0], x, sigma, target, mask, free_mask=_context.free_mask, profile=True)
            else:
                import time
                torch.cuda.synchronize(); started = time.perf_counter()
                with torch.no_grad():
                    v = velocity_fn(xs, timestep)[0]
                    correction = (target-(x-sigma*v))*mask*_context.free_mask
                    gain = min(10., ((1-sigma)**2+sigma**2)/max(sigma*(1-sigma),1e-6))
                    result = (v-gain*correction)*_context.free_mask
                torch.cuda.synchronize()
                metrics = dict(forward_ms=(time.perf_counter()-started)*1000, backward_ms=0.)
            if not torch.isfinite(result).all():
                raise RuntimeError("Nonfinite hybrid RTC velocity")
            _context.metrics.append(dict(step_index=index, exact_vjp=index==2, sigma=sigma, **metrics))
            _context.calls += 1
            return [result]
        return original_forward(self, guided, noise, *args, **kwargs)

    class PrefixAdapter(base_adapter):
        def __init__(self, config):
            m = config.model
            if (m.native_action_dim != 27 or m.native_chunk_size != 32
                    or m.max_action_dim != 64 or m.sampler != "unipc"
                    or config.deployment.action_space != "eef"
                    or not config.deployment.model_id.endswith("-rtc-hybrid-step2-v1")):
                raise ValueError("RTC service config/layout/model ID mismatch")
            self._rtc_autograd_enabled = True
            self._rtc_warmed = False
            super().__init__(config)
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            prepare_original = self.model._prepare_inference_data
            def prepare(*args, **kwargs):
                result = prepare_original(*args, **kwargs)
                _context.free_mask = 1-result[6][0].float()
                _context.free_mask[-33*64:].reshape(33,64)[:,27:] = 0
                return result
            self.model._prepare_inference_data = prepare
            if getattr(self.model.config, "sound_gen", False):
                raise ValueError("RTC action-tail layout does not support sound")

        def _infer_native(self, images, state):
            if not self._rtc_warmed:
                if getattr(_context, "prefix", None) is not None:
                    raise RuntimeError("Hybrid RTC must warm up before serving requests")
                # Compile both normal and mixed-autograd paths before HTTP readiness.
                native, _ = super()._infer_native(images, state)
                for hard in (16, 15):
                    values=native[:hard+2].copy()
                    weights=np.ones(len(values),dtype=np.float32)
                    for j in range(hard,len(values)):
                        w=(len(values)-j)/(len(values)-hard+1)
                        weights[j]=w*np.expm1(w)/np.expm1(1.)
                    _context.prefix=(values, weights, hard)
                    try:
                        for _ in range(3):
                            _context.calls=0; _context.metrics=[]
                            _, ms=super()._infer_native(images,state)
                            if _context.calls != 4:
                                raise RuntimeError("Hybrid warmup did not execute four steps")
                            print({"hybrid_warmup_prefix":hard,"model_ms":ms},flush=True)
                    finally:
                        _context.prefix=None
                self._rtc_warmed=True
            return super()._infer_native(images,state)

        def infer(self, observation):
            prefix = validate_prefix(observation.get("rtc_prefix"))
            _context.prefix = prefix
            _context.calls = 0
            _context.metrics = []
            try:
                output = super().infer(observation)
                if prefix[2] and _context.calls != self.config.model.num_steps:
                    raise RuntimeError("RTC guidance did not run at every sampler step")
                return replace(output, rtc_prefix_guidance=dict(
                    method=METHOD, committed_steps=prefix[2],
                    soft_steps=len(prefix[0])-prefix[2], sampler_calls=_context.calls, steps=_context.metrics))
            finally:
                _context.prefix = None
    UniPCSampler.forward = forward
    adapters.SingleRightHandCosmosAdapter = PrefixAdapter


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("RTC experiment is single GPU only")
    from .rtc_vjp import configure_exact_vjp_backend
    configure_exact_vjp_backend()
    install()
    from cosmos_framework.inference.robot_policy.cfg_http import main as serve
    serve()

if __name__ == "__main__":
    main()
