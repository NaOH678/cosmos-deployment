"""Offline-only vLLM-Omni WAM benchmark. No robot network/control code."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_adapter import extract_actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--eager", action="store_true")
    p.add_argument(
        "--compile-granularity", choices=("regional", "full"), default="regional"
    )
    p.add_argument("--static", action="store_true")
    p.add_argument("--quantization", choices=("fp8",), default=None)
    p.add_argument("--first-frame-only", action="store_true")
    a = p.parse_args()
    os.environ["WAM_FIRST_FRAME_ONLY"] = "1" if a.first_frame_only else "0"
    if a.repeat < 1 or a.warmup < 0:
        p.error("Invalid repeat/warmup")
    root = Path(__file__).resolve().parents[1]
    upstream = json.loads((root / "upstream.json").read_text())
    sys.path.insert(0, str(root.parent / "WorldAct-sft"))
    sys.path.insert(0, upstream["source"])
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(root), upstream["source"], str(root.parent / "WorldAct-sft")]
    )
    from wam_adapter import load_packet
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    m, frame, _, _ = load_packet(a.packet)
    a.output.mkdir(parents=True, exist_ok=True)
    os.environ["WAM_RUNTIME_REPORT"] = str(a.output.resolve() / "runtime.json")
    report = {
        "upstream": upstream,
        "model": str(a.model.resolve()),
        "packet": str(a.packet.resolve()),
        "input_metadata": m,
        "precision": "bfloat16",
        "quantization": a.quantization,
        "first_frame_only": a.first_frame_only,
        "eager": a.eager,
        "compile_granularity": a.compile_granularity,
        "compile_dynamic": not a.static,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "samples": [],
        "status": "started",
        "numerical_parity_verified": False,
    }
    engine = Omni(
        model=str(a.model.resolve()),
        model_class_name="Cosmos3OmniDiffusersPipeline",
        dtype="bfloat16",
        quantization=a.quantization,
        enforce_eager=a.eager,
        diffusion_compile_granularity=a.compile_granularity,
        diffusion_compile_dynamic=not a.static,
        cfg_parallel_size=1,
        tensor_parallel_size=1,
        model_config={"guardrails": False},
        profiler_config={"profiler": "torch", "torch_profiler_dir": str(a.output.resolve()/"traces"), "torch_profiler_record_shapes": True, "torch_profiler_with_stack": False, "torch_profiler_with_memory": True},
    )
    try:
        for i in range(a.warmup + a.repeat + 2):
            sp = OmniDiffusionSamplingParams(
                seed=m["seed"],
                num_inference_steps=m["num_inference_steps"],
                guidance_scale=m["guidance_scale"],
                num_frames=33,
                height=frame.shape[1],
                width=frame.shape[2],
                extra_args={
                    "wam_packet": str(a.packet.resolve()),
                    "use_system_prompt": False,
                    "frame_rate": m["fps"],
                    "guardrails": False,
                },
            )
            if i == a.warmup + a.repeat:
                engine.start_profile(profile_prefix="warm_fp8")
            started = time.perf_counter()
            result = engine.generate(
                {"prompt": m["prompt"], "modalities": ["video"]}, sp
            )
            if not isinstance(result, (list, dict)) and not hasattr(
                result, "multimodal_output"
            ):
                result = list(result)
            elapsed = (time.perf_counter() - started) * 1000
            actions = extract_actions(result)
            if actions is None:
                raise RuntimeError(f"No actions in Omni output: {type(result)}")
            if hasattr(actions, "detach"):
                actions = actions.detach().float().cpu().numpy()
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape == (1, 32, 27):
                actions = actions[0]
            if actions.shape != (32, 27) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid action output: {actions.shape}")
            np.save(a.output / f"actions_{i:03d}.npy", actions)
            report["samples"].append(
                {"index": i, "warmup": i < a.warmup, "profiled": i >= a.warmup+a.repeat, "end_to_end_ms": elapsed}
            )
            (a.output / "report.json").write_text(json.dumps(report, indent=2))
            print(f"request {i}: {elapsed:.2f} ms", flush=True)
        engine.stop_profile()
        timings = [s["end_to_end_ms"] for s in report["samples"] if not s["warmup"] and not s["profiled"]]
        report.update(
            status="completed",
            p50_ms=float(np.percentile(timings, 50)),
            p95_ms=float(np.percentile(timings, 95)),
        )
    finally:
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        engine.close()


if __name__ == "__main__":
    main()
