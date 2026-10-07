#!/usr/bin/env python3
"""Isolated single/dual CFG replay wrapper; no HTTP or robot control.

[torchrun --standalone --nproc-per-node=2] THIS_SCRIPT
    benchmark_cosmos_acceleration.py [benchmark options]
Use absolute paths. CFG_COMPILE_REGION=language selects eager encode/decode heads.
CFG_ALLOW_LOCAL_KV=1 enables experimental independent request-local text caches
on the two CFG ranks. CFG_SERIAL_WARMUP=1 executes both branches on each rank
for the first request to align warmup order. Production defaults are untouched.
"""

import json
import os
from pathlib import Path
import runpy
import sys


def main():
    args = sys.argv[1:]

    def option(name):
        return args[args.index(name) + 1]

    sys.path.insert(0, option("--inference-repo"))
    import torch
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.robot_policy.adapters import (
        SingleRightHandCosmosAdapter,
    )
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world not in (1, 2):
        raise ValueError("This experiment supports exactly one or two ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    original_setup = OmniSetupOverrides.build_setup

    def setup(self, *a, **kw):
        self.cfgp_size = world
        self.cp_size = 1
        self.dp_shard_size = 1
        self.dp_replicate_size = world
        self.use_cuda_graphs = False
        if os.environ.get("CFG_COMPILE_REGION"):
            self.compiled_region = os.environ["CFG_COMPILE_REGION"]
        return original_setup(self, *a, **kw)

    OmniSetupOverrides.build_setup = setup
    if os.environ.get("CFG_ALLOW_LOCAL_KV") == "1":

        def reuse(
            self,
            sequence_plans,
            gen_data_clean,
            *,
            reuse_pack_templates,
            has_velocity_postprocess,
        ):
            if has_velocity_postprocess or not reuse_pack_templates:
                return False
            if self.parallel_dims is not None and (
                self.parallel_dims.cp_enabled or self.parallel_dims.dp_shard_enabled
            ):
                return False
            if (
                self.config.joint_attn_implementation != "two_way"
                or self.config.video_temporal_causal
            ):
                return False
            if self.config.sound_gen and any(plan.has_sound for plan in sequence_plans):
                return False
            return (
                gen_data_clean.batch_size == 1
                and len(sequence_plans) == 1
                and gen_data_clean.num_vision_items_per_sample is None
            )

        OmniMoTModel._can_reuse_inference_text_kv = reuse
    original_infer = SingleRightHandCosmosAdapter._infer_native
    original_cfg = OmniMoTModel._run_classifier_free_guidance
    calls = 0
    serial = False

    def infer(self, *a, **kw):
        nonlocal calls, serial
        calls += 1
        serial = os.environ.get("CFG_SERIAL_WARMUP") == "1" and calls == 1
        try:
            return original_infer(self, *a, **kw)
        finally:
            serial = False

    def cfg(
        self, cond_tokens, uncond_tokens, skip_text_tokens_for_cfg, single_velocity_fn
    ):
        if serial:
            return single_velocity_fn(cond_tokens, False), single_velocity_fn(
                uncond_tokens, skip_text_tokens_for_cfg
            )
        return original_cfg(
            self,
            cond_tokens,
            uncond_tokens,
            skip_text_tokens_for_cfg,
            single_velocity_fn,
        )

    SingleRightHandCosmosAdapter._infer_native = infer
    OmniMoTModel._run_classifier_free_guidance = cfg
    output = Path(option("--output-dir")) / f"rank{rank}"
    args[args.index("--output-dir") + 1] = str(output)
    script = args.pop(0)
    sys.argv = [script] + args
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        report = output / "report.json"
        if report.exists():
            data = json.loads(report.read_text())
            data["cfg_experiment"] = {
                "world_size": world,
                "rank": rank,
                "compile_region": os.environ.get("CFG_COMPILE_REGION", "all"),
                "local_kv": os.environ.get("CFG_ALLOW_LOCAL_KV") == "1",
                "serial_warmup": os.environ.get("CFG_SERIAL_WARMUP") == "1",
                "cuda_graphs": False,
            }
            report.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
