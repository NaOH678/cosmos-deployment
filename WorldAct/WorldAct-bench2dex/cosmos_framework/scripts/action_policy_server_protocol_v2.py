# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Serve the single-right-hand Cosmos action policy over Wuji protocol-v2."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import os
from pathlib import Path
from typing import Any

from cosmos_framework.inference.robot_policy.adapters import create_model_adapter
from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig, load_robot_policy_config
from cosmos_framework.inference.robot_policy.server import create_http_server
from cosmos_framework.utils import log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Protocol-v2 deployment YAML")
    parser.add_argument("--model-id", help="Override deployment.model_id for controlled debugging")
    parser.add_argument("--checkpoint-path", help="Override model.checkpoint_path")
    parser.add_argument("--model-config-file", help="Override model.config_file with the matching frozen config")
    parser.add_argument("--service-mode", choices=("full", "hold", "small_motion"), help="Override model.service_mode")
    parser.add_argument("--host", help="Override service.host")
    parser.add_argument("--port", type=int, help="Override service.port")
    parser.add_argument("--guidance", type=float, help="Override model.guidance")
    parser.add_argument("--num-steps", type=int, help="Override model.num_steps")
    parser.add_argument("--shift", type=float, help="Override model.shift")
    parser.add_argument(
        "--trajectory-smoothing",
        choices=("none", "binomial5"),
        help="Override model.trajectory_smoothing",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip the startup model warmup")
    return parser.parse_args()


def _apply_overrides(config: RobotPolicyConfig, args: argparse.Namespace) -> RobotPolicyConfig:
    raw: dict[str, Any] = config.model_dump()
    overrides = {
        ("deployment", "model_id"): args.model_id,
        ("model", "checkpoint_path"): args.checkpoint_path,
        ("model", "config_file"): args.model_config_file,
        ("model", "service_mode"): args.service_mode,
        ("service", "host"): args.host,
        ("service", "port"): args.port,
        ("model", "guidance"): args.guidance,
        ("model", "num_steps"): args.num_steps,
        ("model", "shift"): args.shift,
        ("model", "trajectory_smoothing"): args.trajectory_smoothing,
    }
    for (section, field), value in overrides.items():
        if value is not None:
            raw[section][field] = value
    if args.no_warmup:
        raw["model"]["warmup"] = False
    return RobotPolicyConfig.model_validate(raw)


def main() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("protocol-v2 policy serving currently requires WORLD_SIZE=1")

    args = _parse_args()
    config = _apply_overrides(load_robot_policy_config(args.config), args)
    api_key = config.auth.load_api_key()
    adapter = create_model_adapter(config)
    server = create_http_server(config, adapter, api_key)
    log.info(
        f"Robot policy ready endpoint=http://{config.service.host}:{config.service.port}"
        f"{config.service.endpoint} model_id={config.deployment.model_id} "
        f"mode={config.model.service_mode} cameras={','.join(config.deployment.camera_names)} "
        f"action_space={config.deployment.action_space} "
        f"weights={'ema' if config.model.use_ema_weights else 'regular'} "
        f"sampler={config.model.sampler} guidance={config.model.guidance} "
        f"num_steps={config.model.num_steps} shift={config.model.shift} "
        f"trajectory_smoothing={config.model.trajectory_smoothing}"
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        log.info("Robot policy shutdown requested")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
