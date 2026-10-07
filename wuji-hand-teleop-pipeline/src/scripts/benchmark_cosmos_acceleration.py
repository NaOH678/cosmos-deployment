#!/usr/bin/env python3
"""Offline sampling-step benchmark; never starts HTTP or robot control.

Launch with the inference environment's Python. NPZ keys: head, right_wrist
(RGB uint8), state (xyz metres, xyzw, 20 hand angles). Declare hand units.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--hand-units", choices=("degrees", "radians"), required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 3, 2])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if min(args.steps) < 1 or args.repeats < 2 or args.warmup < 1:
        parser.error("positive steps/warmup and at least two repeats required")
    for name in ("inference_repo", "config", "model_package", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.observations = [path.resolve() for path in args.observations]
    args.steps = list(dict.fromkeys(args.steps))
    if 4 in args.steps:
        args.steps = [4] + [step for step in args.steps if step != 4]
    sys.path.insert(0, str(args.inference_repo))
    os.environ["EDGE_DROID_MODEL_PATH"] = str(args.model_package.resolve())
    os.chdir(args.inference_repo)
    import numpy as np
    import torch
    from cosmos_framework.inference.common.init import init_script
    from cosmos_framework.inference.robot_policy.config import load_robot_policy_config
    from cosmos_framework.inference.robot_policy.adapters import create_model_adapter

    init_script()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_robot_policy_config(args.config)
    if config.deployment.action_space != "eef" or config.model.service_mode != "full":
        raise ValueError("Use a full EEF deployment config")
    config.model.warmup = False
    config.model.use_cuda_graphs = False
    config.model.output_dir = str(args.output_dir / "model_output")
    adapter = create_model_adapter(config)
    report = {
        "note": "Offline real recorded inputs; no task-success evaluation, HTTP or robot.",
        "config": str(args.config),
        "source": str(args.inference_repo),
        "gpu": torch.cuda.get_device_name(),
        "guidance": config.model.guidance,
        "seed": config.model.seed,
        "hand_input_units": args.hand_units,
        "results": [],
    }
    for sample in args.observations:
        with np.load(sample, allow_pickle=False) as data:
            images = {
                name: data[name].copy() for name in config.deployment.camera_names
            }
            state = data["state"].astype(np.float32).copy()
        for name, image in images.items():
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{sample}: {name} must be RGB uint8 HxWx3")
        if state.shape != (27,) or not np.isfinite(state).all():
            raise ValueError(f"invalid state: {sample}")
        if args.hand_units == "degrees":
            state[7:] = np.deg2rad(state[7:])
        outputs = {}
        for steps in args.steps:
            config.model.num_steps = steps
            for _ in range(args.warmup):
                adapter._infer_native(images, state)
            timings = []
            first = None
            repeat_max = 0.0
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                actions, model_ms = adapter._infer_native(images, state)
                torch.cuda.synchronize()
                timings.append([model_ms, (time.perf_counter() - started) * 1000])
                if not np.isfinite(actions).all():
                    raise RuntimeError("nonfinite actions")
                if first is None:
                    first = actions.copy()
                repeat_max = max(repeat_max, float(np.max(np.abs(actions - first))))
            outputs[steps] = actions.copy()
            np.save(args.output_dir / f"{sample.stem}_steps{steps}.npy", actions)
            row = {
                "sample": str(sample),
                "input_rgb_shapes": {
                    name: list(image.shape) for name, image in images.items()
                },
                "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                "steps": steps,
                "repeats": args.repeats,
                "timings_model_adapter_ms": timings,
                "median_model_adapter_ms": np.median(timings, axis=0).tolist(),
                "p95_model_adapter_ms": np.percentile(timings, 95, axis=0).tolist(),
                "repeat_max_abs": repeat_max,
                "max_adjacent_xyz_m": float(
                    np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).max()
                ),
            }
            if 4 in outputs:
                diff = np.linalg.norm(actions[:, :3] - outputs[4][:, :3], axis=1)
                row["vs4_xyz_median_m"] = float(np.median(diff))
                row["vs4_xyz_mean_max_m"] = [float(diff.mean()), float(diff.max())]
            report["results"].append(row)
            (args.output_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k != "timings_model_adapter_ms"}
                ),
                flush=True,
            )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
