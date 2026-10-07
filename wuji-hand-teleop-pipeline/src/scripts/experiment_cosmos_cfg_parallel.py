"""Isolated offline CFG experiment; never starts HTTP or robot control.

Run with: [torchrun --standalone --nproc-per-node=2] THIS_SCRIPT
    profile_cosmos_inference.py [profiler arguments with absolute paths]
Required CFG_REAL_INPUT points to recorded NPZ with RGB head/right_wrist and
native 27D state (hand radians). Optional CFG_SERIAL_WARMUP=1 executes both
branches in the same order on the first call per rank; CFG_ALLOW_LOCAL_KV=1
allows each CFG rank its own request-local text cache (experimental).
CFG_COMPILE_REGION=language leaves VFM encode/decode heads uncompiled.
Uses sibling diagnose_cosmos_cfg.py. First measured timing includes diagnostic
copies and must be excluded. These options do not alter production defaults."""

import os
import sys
import runpy
import json
import hashlib
from pathlib import Path
import numpy as np


def main():
    args = sys.argv[1:]

    def opt(k):
        return args[args.index(k) + 1]

    sys.path.insert(0, opt("--inference-repo"))
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.inference.robot_policy.adapters import (
        SingleRightHandCosmosAdapter,
    )
    from cosmos_framework.inference.args import OmniSetupOverrides

    original_setup = OmniSetupOverrides.build_setup

    def setup(self, *a, **kw):
        if os.environ.get("CFG_COMPILE_REGION"):
            self.compiled_region = os.environ["CFG_COMPILE_REGION"]
        return original_setup(self, *a, **kw)

    OmniSetupOverrides.build_setup = setup
    original_infer = SingleRightHandCosmosAdapter._infer_native
    original_cfg = OmniMoTModel._run_classifier_free_guidance
    calls = 0
    serial = False
    sample = np.load(os.environ["CFG_REAL_INPUT"], allow_pickle=False)
    images = {k: sample[k].copy() for k in ("head", "right_wrist")}
    state = sample["state"].copy()

    def infer(self, _images, _state):
        nonlocal calls, serial
        calls += 1
        serial = os.environ.get("CFG_SERIAL_WARMUP") == "1" and calls == 1
        try:
            return original_infer(self, images, state)
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
    sys.argv = [str(Path(__file__).with_name("diagnose_cosmos_cfg.py"))] + args
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        rank = int(os.environ.get("RANK", "0"))
        report = Path(opt("--output-dir")) / f"rank{rank}" / "report.json"
        if report.exists():
            data = json.loads(report.read_text())
            data["input_shapes"] = {k: list(v.shape) for k, v in images.items()}
            data["input_sha256"] = hashlib.sha256(
                b"".join(images[k].tobytes() for k in sorted(images)) + state.tobytes()
            ).hexdigest()
            data["input_npz_sha256"] = hashlib.sha256(
                Path(os.environ["CFG_REAL_INPUT"]).read_bytes()
            ).hexdigest()
            data["input_npz"] = os.environ["CFG_REAL_INPUT"]
            data["experiment"] = {
                k: os.environ.get(k, "0")
                for k in (
                    "CFG_SERIAL_WARMUP",
                    "CFG_ALLOW_LOCAL_KV",
                    "CFG_BENCH_NO_TEXT_KV",
                    "CFG_COMPILE_REGION",
                )
            }
            data["note"] = (
                "Real recorded RGB/native state. First measured request copies diagnostic tensors; exclude timing index0. No robot or HTTP."
            )
            report.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
