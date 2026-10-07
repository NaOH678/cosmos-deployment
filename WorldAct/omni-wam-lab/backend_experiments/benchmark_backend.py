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

def install_compile_patch():
    import torch
    if getattr(torch, "_wam_compile_patched", False):
        return
    torch._wam_compile_patched = True
    _torch_compile = torch.compile
    def _compile_with_mode(*args, **kwargs):
        fn = args[0] if args else kwargs.get("model")
        if getattr(fn, "__qualname__", "") == "Cosmos3GenDecoderLayer.forward":
            kwargs.pop("options", None)
            kwargs["mode"] = os.environ["WAM_COMPILE_MODE"]
            print("WAM_GEN_COMPILE:", kwargs["mode"], "dynamic=", kwargs.get("dynamic"), flush=True)
        compiled = _torch_compile(*args, **kwargs)
        if getattr(fn, "__qualname__", "") == "Cosmos3GenDecoderLayer.forward" and os.environ.get("WAM_GRAPH_CLONE") == "1":
            import functools
            @functools.wraps(compiled)
            def durable_output(*call_args, **call_kwargs):
                return compiled(*call_args, **call_kwargs).clone()
            return durable_output
        return compiled
    torch.compile = _compile_with_mode
    import atexit
    _graph_counts = {"capture_begin": 0, "replay": 0}
    def _count_graph_method(name):
        original = getattr(torch.cuda.CUDAGraph, name)
        def counted(self, *args, **kwargs):
            _graph_counts[name] += 1
            if _graph_counts[name] == 1:
                print(f"WAM_GRAPH_AUDIT: actual {name} observed", flush=True)
            return original(self, *args, **kwargs)
        return counted
    for _method in _graph_counts:
        setattr(torch.cuda.CUDAGraph, _method, _count_graph_method(_method))
    def _write_graph_counts():
        if os.environ.get("WAM_GRAPH_AUDIT"):
            Path(os.environ["WAM_GRAPH_AUDIT"] + f".{os.getpid()}.json").write_text(json.dumps(_graph_counts))
    atexit.register(_write_graph_counts)
    print("WAM_COMPILE_MODE_PATCH:", os.environ["WAM_COMPILE_MODE"], flush=True)

if os.environ.get("WAM_COMPILE_MODE"):
    install_compile_patch()



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--graph-clone", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    p.add_argument("--attention", choices=("CUDNN_ATTN", "FLASH_ATTN", "TORCH_SDPA"), default="CUDNN_ATTN")
    p.add_argument(
        "--compile-granularity", choices=("regional", "full"), default="regional"
    )
    p.add_argument("--static", action="store_true")
    p.add_argument("--quantization", choices=("fp8",), default=None)
    p.add_argument("--first-frame-only", action="store_true")
    a = p.parse_args()
    os.environ["WAM_GRAPH_CLONE"] = "1" if a.graph_clone else "0"
    os.environ["WAM_COMPILE_MODE"] = a.compile_mode
    install_compile_patch()
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = a.attention
    os.environ["WAM_FIRST_FRAME_ONLY"] = "1" if a.first_frame_only else "0"
    if a.repeat < 1 or a.warmup < 0:
        p.error("Invalid repeat/warmup")
    root = Path(__file__).resolve().parents[1]
    upstream = json.loads((root / "upstream.json").read_text())
    sys.path.insert(0, str(root.parent / "WorldAct-sft"))
    sys.path.insert(0, upstream["source"])
    fa4 = str(root / "backend_experiments/fa4_deps")
    if a.attention == "FLASH_ATTN":
        sys.path.insert(0, fa4)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        ([fa4] if a.attention == "FLASH_ATTN" else []) + [str(root), upstream["source"], str(root.parent / "WorldAct-sft")]
    )
    from wam_adapter import load_packet
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    packets = [Path(x["packet"]) for x in json.loads(a.corpus.read_text())["observations"]] if a.corpus else [a.packet]
    m, frame, _, _ = load_packet(packets[0])
    a.output.mkdir(parents=True, exist_ok=True)
    os.environ["WAM_RUNTIME_REPORT"] = str(a.output.resolve() / "runtime.json")
    os.environ["WAM_GRAPH_AUDIT"] = str(a.output.resolve() / "graph_audit")
    report = {
        "upstream": upstream,
        "model": str(a.model.resolve()),
        "packet": str(packets[0].resolve()),
        "corpus": str(a.corpus.resolve()) if a.corpus else None,
        "input_metadata": m,
        "precision": "bfloat16",
        "quantization": a.quantization,
        "attention": a.attention,
        "first_frame_only": a.first_frame_only,
        "eager": a.eager,
        "compile_granularity": a.compile_granularity,
        "compile_dynamic": not a.static,
        "compile_mode": a.compile_mode,
        "graph_clone": a.graph_clone,
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
        profiler_config={"profiler": "torch", "torch_profiler_dir": str(a.output.resolve()/"traces"), "torch_profiler_record_shapes": True} if a.profile else None,
    )
    try:
        for i in range(a.warmup + a.repeat + (2 if a.profile else 0)):
            packet = packets[i % len(packets)]
            m, frame, _, _ = load_packet(packet)
            sp = OmniDiffusionSamplingParams(
                seed=m["seed"],
                num_inference_steps=m["num_inference_steps"],
                guidance_scale=m["guidance_scale"],
                num_frames=33,
                height=frame.shape[1],
                width=frame.shape[2],
                extra_args={
                    "wam_packet": str(packet.resolve()),
                    "use_system_prompt": False,
                    "frame_rate": m["fps"],
                    "guardrails": False,
                },
            )
            if a.profile and i == a.warmup + a.repeat:
                engine.start_profile(profile_prefix="backend")
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
                {"index": i, "packet": str(packet), "warmup": i < a.warmup, "profiled": i >= a.warmup + a.repeat, "end_to_end_ms": elapsed}
            )
            (a.output / "report.json").write_text(json.dumps(report, indent=2))
            print(f"request {i}: {elapsed:.2f} ms", flush=True)
        if a.profile:
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
