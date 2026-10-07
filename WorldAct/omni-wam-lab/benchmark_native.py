"""Offline native baseline using the deployed single-GPU CFG backend."""

import argparse
import json
import time
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    a = p.parse_args()
    if a.warmup < 1 or a.repeat < 1:
        p.error("warmup and repeat must be positive")
    from cosmos_framework.inference.common.init import init_script

    init_script()
    from cosmos_framework.inference.robot_policy.cfg_http import install_backend
    from cosmos_framework.inference.robot_policy.config import (
        load_robot_policy_config,
        RobotPolicyConfig,
    )
    from cosmos_framework.inference.robot_policy.adapters import (
        SingleRightHandCosmosAdapter,
    )

    mode = install_backend(1)
    raw = load_robot_policy_config(a.config).model_dump()
    raw["model"].update(warmup=False, use_cuda_graphs=False)
    config = RobotPolicyConfig.model_validate(raw)
    adapter = SingleRightHandCosmosAdapter(config)
    with np.load(a.observation, allow_pickle=False) as z:
        images = {k: z[k].copy() for k in config.deployment.camera_names}
        state = z["state"].copy()
    a.output.mkdir(parents=True, exist_ok=True)
    result = {
        "config": str(a.config.resolve()),
        "model": raw["model"],
        "observation": str(a.observation.resolve()),
        "runs": [],
    }
    for i in range(a.warmup + a.repeat):
        mode["serial_warmup"] = i == 0
        started = time.perf_counter()
        actions, ms = adapter._infer_native(images, state)
        elapsed = (time.perf_counter() - started) * 1000
        np.save(a.output / f"actions_{i:03d}.npy", actions)
        result["runs"].append({"index": i, "warmup": i < a.warmup, "model_ms": ms, "end_to_end_ms": elapsed})
        print(f"native {i}: model={ms:.2f} ms, end_to_end={elapsed:.2f} ms", flush=True)
    for key in ("model_ms", "end_to_end_ms"):
        values = [r[key] for r in result["runs"] if not r["warmup"]]
        result[key] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    (a.output / "report.json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
