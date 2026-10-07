"""Capture initial conditioning/noise and per-step CFG velocity for one request.

Usage: [torchrun --standalone --nproc-per-node=2] THIS_SCRIPT
       profile_cosmos_inference.py [profiler options]
Use absolute paths. Set CFG_BENCH_NO_TEXT_KV=1 and/or
CFG_BENCH_NO_COMPILE=1 or CFG_BENCH_STATIC=1 for controlled comparisons.
CFG_BENCH_ALL_STATIC=1 forces all compiled heads and blocks to static shapes. CUDA Graph stays disabled.
Tensor copies invalidate timings; this script is ONLY a numerical diagnostic.
"""

import os
import sys
import runpy
from pathlib import Path


def main():
    args = sys.argv[1:]

    def opt(k):
        return args[args.index(k) + 1]

    sys.path.insert(0, opt("--inference-repo"))
    import torch

    if os.environ.get("CFG_BENCH_ALL_STATIC") == "1":
        original_compile = torch.compile

        def static_compile(*a, **kw):
            kw["dynamic"] = False
            return original_compile(*a, **kw)

        torch.compile = static_compile
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.inference.robot_policy.adapters import (
        SingleRightHandCosmosAdapter,
    )

    rank = int(os.environ.get("RANK", 0))
    out = Path(opt("--output-dir")) / f"rank{rank}"
    active = False
    captures = {}
    call = 0
    steps = 0
    original_prepare = OmniMoTModel._prepare_inference_data
    original_velocity = OmniMoTModel._get_velocity
    original_infer = SingleRightHandCosmosAdapter._infer_native

    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().clone()
        if isinstance(x, (list, tuple)):
            return [cpu(v) for v in x]
        return x

    def prepare(self, *a, **kw):
        result = original_prepare(self, *a, **kw)
        if active:
            captures["prepare"] = {str(i): cpu(result[i]) for i in (2, 3, 4, 5, 6, 7)}
        return result

    def velocity(self, *a, **kw):
        nonlocal steps
        if active:
            key = f"velocity_{steps}"
            steps += 1
            captures[key] = {
                k: cpu(kw[k])
                for k in ("noise_x", "timestep", "text_tokens", "skip_text_tokens")
            }
        result = original_velocity(self, *a, **kw)
        if active:
            captures[key]["result"] = cpu(result)
        return result

    def infer(self, *a, **kw):
        nonlocal active, call
        call += 1
        active = call == int(opt("--warmup")) + 1
        result = original_infer(self, *a, **kw)
        if active:
            out.mkdir(parents=True, exist_ok=True)
            torch.save(captures, out / "diagnostic.pt")
        active = False
        return result

    OmniMoTModel._prepare_inference_data = prepare
    OmniMoTModel._get_velocity = velocity
    SingleRightHandCosmosAdapter._infer_native = infer
    # Apply the existing CFG layout without modifying production defaults.
    from cosmos_framework.inference.args import OmniSetupOverrides

    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if os.environ.get("CFG_BENCH_NO_TEXT_KV") == "1":
        OmniMoTModel._can_reuse_inference_text_kv = lambda *a, **kw: False
    original_setup = OmniSetupOverrides.build_setup

    def setup(self, *a, **kw):
        self.cfgp_size = world
        self.cp_size = 1
        self.dp_shard_size = 1
        self.dp_replicate_size = world
        self.use_cuda_graphs = False
        if os.environ.get("CFG_BENCH_NO_COMPILE") == "1":
            self.use_torch_compile = False
        if os.environ.get("CFG_BENCH_STATIC") == "1":
            self.compile_dynamic = False
        return original_setup(self, *a, **kw)

    OmniSetupOverrides.build_setup = setup
    args[args.index("--output-dir") + 1] = str(out)
    # Run the sibling profiler with the requested arguments. Its reported timings
    # include diagnostic copies and MUST NOT be interpreted as benchmark latency.
    script = args.pop(0)
    sys.argv = [script] + args
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
