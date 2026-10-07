#!/usr/bin/env python3
"""Standalone Cosmos GPU benchmark. Does not start HTTP, ROS or robot control.

Run the same script/config on both GPUs for an actual hardware comparison.
Synthetic images measure execution cost, not task success or policy quality.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    worktrees = Path(__file__).resolve().parents[3]
    world = worktrees / "WorldAct"
    run = world / "model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-repo", type=Path, default=world / "WorldAct-sft")
    parser.add_argument("--config", type=Path, default=world / "WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml")
    parser.add_argument("--checkpoint-dir", type=Path, default=run / "iter_000040000")
    parser.add_argument("--model-config-file", type=Path, default=run / "config.deploy.yaml")
    parser.add_argument("--model-package", type=Path, default=world / "models/cosmos3-edge-droid")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/cosmos-inference-profile"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--torch-profile", action="store_true", help="One additional instrumented inference after normal timings")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")
    for key in ("inference_repo", "config", "checkpoint_dir", "model_config_file", "model_package", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EDGE_DROID_MODEL_PATH"] = str(args.model_package)
    os.chdir(args.inference_repo)
    sys.path.insert(0, str(args.inference_repo))

    import torch
    import cosmos_framework

    actual = Path(cosmos_framework.__file__).resolve()
    if actual != args.inference_repo / "cosmos_framework/__init__.py":
        raise RuntimeError(f"Unexpected imported source: {actual}")
    report = {"source": str(actual), "python": sys.executable, "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "packages": {}, "status": "environment_only", "note": "Synthetic fixed inputs; no robot or HTTP. Profiled timings are separate from baseline."}
    for name in ("flash-attn", "triton", "transformers", "natten"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    def gpu_snapshot():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used", "--format=csv"], capture_output=True, text=True, timeout=10)
            return {"returncode": result.returncode, "output": result.stdout.strip(), "error": result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"error": str(error)}
    report["gpu_before"] = gpu_snapshot()
    def save():
        path = args.output_dir / "report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {path}", flush=True)
    if not report["cuda_available"] or args.environment_only:
        if not report["cuda_available"]:
            report["status"] = "cuda_unavailable"
        save()
        print(json.dumps(report, indent=2))
        return 0 if args.environment_only else 2

    from cosmos_framework.inference.common.init import init_script
    init_script()
    import numpy as np
    from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig, load_robot_policy_config
    from cosmos_framework.inference.robot_policy.adapters import create_model_adapter

    raw = load_robot_policy_config(args.config).model_dump()
    raw["model"].update(service_mode="full", checkpoint_path=str(args.checkpoint_dir),
                         config_file=str(args.model_config_file), warmup=False,
                         output_dir=str(args.output_dir / "model_output"))
    config = RobotPolicyConfig.model_validate(raw)
    if config.deployment.action_space != "eef":
        raise ValueError("This benchmark's synthetic state requires an EEF profile")
    report["gpu"] = torch.cuda.get_device_name(0)
    report["capability"] = torch.cuda.get_device_capability(0)
    report["model_config_sha256"] = hashlib.sha256(args.model_config_file.read_bytes()).hexdigest()
    report["sampling"] = {k: getattr(config.model, k) for k in ("sampler", "guidance", "num_steps", "shift", "seed", "resolution", "use_ema_weights")}
    started = time.perf_counter()
    adapter = create_model_adapter(config)
    report["load_s"] = time.perf_counter() - started
    rng = np.random.default_rng(0)
    images = {name: rng.integers(0, 256, tuple(config.image_preprocessing.warmup_shapes[name]), dtype=np.uint8)
              for name in config.deployment.camera_names}
    state = np.zeros(config.model.native_action_dim, dtype=np.float32)
    state[:3] = [.57, -.22, .276]
    state[6] = 1.0
    report["input_shapes"] = {name: list(frame.shape) for name, frame in images.items()}
    report["input_sha256"] = hashlib.sha256(b"".join(images[name].tobytes() for name in sorted(images)) + state.tobytes()).hexdigest()
    for _ in range(args.warmup):
        adapter._infer_native(images, state)
    timings = []
    for index in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        actions, model_ms = adapter._infer_native(images, state)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - started) * 1000
        timings.append({"model_ms": model_ms, "adapter_ms": total_ms})
        print(f"{index+1}/{args.repeats}: model={model_ms:.2f} ms, adapter={total_ms:.2f} ms", flush=True)
    if not np.isfinite(actions).all():
        raise RuntimeError("Non-finite policy output")
    np.save(args.output_dir / "actions.npy", actions)
    report["actions_shape"] = list(actions.shape)
    report["actions_sha256"] = hashlib.sha256(actions.tobytes()).hexdigest()
    report["timings"] = timings
    report["median_ms"] = {key: statistics.median(row[key] for row in timings) for key in timings[0]}
    report["gpu_after"] = gpu_snapshot()
    report["status"] = "baseline_complete"
    save()
    if args.torch_profile:
        # Instrument only a separate run; profiler overhead is not a latency result.
        originals = []
        calls = {}
        def wrap_phase(obj, name):
            original = getattr(obj, name, None)
            if original is None:
                return
            originals.append((obj, name, original, name in vars(obj)))
            @functools.wraps(original)
            def wrapped(*positional, **keywords):
                calls[name] = calls.get(name, 0) + 1
                with torch.profiler.record_function("cosmos::" + name):
                    return original(*positional, **keywords)
            setattr(obj, name, wrapped)
        wrap_phase(adapter, "_build_batch")
        for name in ("generate_samples_from_batch", "_prepare_inference_data", "_encode_vision_x0_tokens", "_pack_input_sequence", "_get_velocity"):
            wrap_phase(adapter.model, name)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                with torch.profiler.record_function("cosmos_adapter_inference"):
                    adapter._infer_native(images, state)
        finally:
            for obj, name, original, was_instance_attribute in reversed(originals):
                if was_instance_attribute:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)
        profiler.export_chrome_trace(str(args.output_dir / "torch_trace.json"))
        averages = profiler.key_averages()
        for kind, sort_by in (("gpu", "self_cuda_time_total"), ("cpu", "self_cpu_time_total")):
            (args.output_dir / f"operators_{kind}.txt").write_text(averages.table(sort_by=sort_by, row_limit=60))
        report["profile_phase_calls"] = calls
        # CPU ranges and GPU annotations can share a name. Retain both with
        # their device type; a name-keyed dict silently overwrites CPU ranges.
        report["profile_phase_inclusive_ms"] = [
            {"name": event.key, "device_type": str(event.device_type),
             "cpu": event.cpu_time_total / 1000,
             "device": event.device_time_total / 1000}
            for event in averages if event.key.startswith("cosmos::")
        ]
        report["profile_note"] = "Phase times include descendants and profiler overhead; do not sum nested phases or compare to unprofiled latency."
        report["status"] = "profile_complete"
        save()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
