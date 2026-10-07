"""Experimental one/two-rank CFG HTTP server, isolated from the default entry.

Run using torchrun --standalone --nproc-per-node=2 -m
cosmos_framework.inference.robot_policy.cfg_http --config ... --warmup-npz ...
The HTTP listener is rank zero only. No robot or ROS commands are issued.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np


def install_backend(world: int):
    """Install the experimentally verified backend only in this process."""
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    original_setup = OmniSetupOverrides.build_setup
    original_cfg = OmniMoTModel._run_classifier_free_guidance
    state = {"serial_warmup": False}

    def setup(self, *a, **kw):
        self.cfgp_size = world
        self.cp_size = self.dp_shard_size = 1
        self.dp_replicate_size = world
        self.use_cuda_graphs = False
        self.compiled_region = "language"
        return original_setup(self, *a, **kw)

    def reuse(self, sequence_plans, gen_data_clean, *, reuse_pack_templates, has_velocity_postprocess):
        if has_velocity_postprocess or not reuse_pack_templates:
            return False
        if self.parallel_dims is not None and (self.parallel_dims.cp_enabled or self.parallel_dims.dp_shard_enabled):
            return False
        if self.config.joint_attn_implementation != "two_way" or self.config.video_temporal_causal:
            return False
        if self.config.sound_gen and any(plan.has_sound for plan in sequence_plans):
            return False
        return (
            gen_data_clean.batch_size == 1
            and len(sequence_plans) == 1
            and gen_data_clean.num_vision_items_per_sample is None
        )

    def cfg(self, cond_tokens, uncond_tokens, skip_text_tokens_for_cfg, single_velocity_fn):
        if state["serial_warmup"]:
            return single_velocity_fn(cond_tokens, False), single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg)
        return original_cfg(self, cond_tokens, uncond_tokens, skip_text_tokens_for_cfg, single_velocity_fn)

    OmniSetupOverrides.build_setup = setup
    OmniMoTModel._can_reuse_inference_text_kv = reuse
    OmniMoTModel._run_classifier_free_guidance = cfg
    return state


def action_digest(actions):
    return hashlib.sha256(np.asarray(actions, dtype=np.float32).tobytes()).hexdigest()


def validate_ack(ack, sequence, actions):
    if not isinstance(ack, dict) or ack.get("sequence") != sequence or ack.get("kind") != "complete":
        raise RuntimeError("CFG peer returned an invalid acknowledgement")
    if ack.get("action_sha256") != action_digest(actions):
        raise RuntimeError("CFG ranks returned different actions")


def fail_process(reason):
    print(f"CFG_FATAL: {reason}", file=sys.stderr, flush=True)
    os._exit(70)  # torchrun terminates the peer; never reuse a desynchronized pair.


class Control:
    def __init__(self, group):
        self.group = group

    def exchange(self, payload=None, source=0):
        import torch.distributed as dist

        bucket = [payload]
        dist.broadcast_object_list(bucket, src=source, group=self.group)
        return bucket[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup-npz", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--request-watchdog-s", type=float, default=30.0)
    args = parser.parse_args()
    if args.request_watchdog_s <= 0:
        parser.error("watchdog timeout must be positive")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world not in (1, 2):
        parser.error("exactly one or two ranks required")
    import torch
    import torch.distributed as dist

    from cosmos_framework.inference.common.init import init_script
    from cosmos_framework.inference.robot_policy.adapters import SingleRightHandCosmosAdapter
    from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig, load_robot_policy_config
    from cosmos_framework.inference.robot_policy.server import create_http_server

    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "60")
    init_script()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world == 2:
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(seconds=60))
        # Idle worker waits here indefinitely; the per-request watchdog bounds
        # live requests. A short Gloo timeout would kill an idle ready service.
        control = Control(dist.new_group(backend="gloo", timeout=timedelta(days=7)))
    else:
        control = None
    mode = install_backend(world)
    raw = load_robot_policy_config(args.config).model_dump()
    raw["service"].update(host="127.0.0.1", port=args.port, max_inflight_inferences=1)
    raw["model"].update(warmup=False, use_cuda_graphs=False)
    config = RobotPolicyConfig.model_validate(raw)
    if config.deployment.action_space != "eef" or config.model.service_mode not in ("full", "small_motion"):
        raise ValueError("experimental CFG HTTP supports the validated EEF model only")
    if rank:
        os.environ.pop("COSMOS_RECORDING_DIR", None)
    api_key = config.auth.load_api_key() if rank == 0 else None

    class Adapter(SingleRightHandCosmosAdapter):
        dispatch = False
        sequence = 0

        def _infer_native(self, images, state):
            if not self.dispatch or world == 1:
                return super()._infer_native(images, state)
            self.sequence += 1
            watchdog = threading.Timer(args.request_watchdog_s, fail_process, args=("request watchdog expired",))
            watchdog.daemon = True
            watchdog.start()
            try:
                control.exchange({"kind": "infer", "sequence": self.sequence, "images": images, "state": state})
                actions, model_ms = super()._infer_native(images, state)
                validate_ack(control.exchange(source=1), self.sequence, actions)
                return actions, model_ms
            except Exception as exc:
                self._ready = False
                fail_process(type(exc).__name__)
            finally:
                watchdog.cancel()

    adapter = Adapter(config)
    with np.load(args.warmup_npz, allow_pickle=False) as warmup:
        images = {name: warmup[name].copy() for name in config.deployment.camera_names}
        native_state = warmup["state"].copy()
    for index in range(3):
        mode["serial_warmup"] = index == 0
        adapter._infer_native(images, native_state)
    mode["serial_warmup"] = False
    if world == 2:
        pids = [None] * world
        dist.all_gather_object(pids, os.getpid(), group=control.group)
    else:
        pids = [os.getpid()]

    if rank == 1:
        previous_sequence = 0
        while True:
            request = control.exchange()
            if request.get("kind") == "stop":
                control.exchange({"kind": "stopped"}, source=1)
                break
            if request.get("kind") != "infer" or request.get("sequence") != previous_sequence + 1:
                fail_process("invalid request sequence")
            previous_sequence = request["sequence"]
            try:
                actions, _ = adapter._infer_native(request["images"], request["state"])
                control.exchange(
                    {"kind": "complete", "sequence": previous_sequence, "action_sha256": action_digest(actions)},
                    source=1,
                )
            except Exception as exc:
                fail_process(type(exc).__name__)
        dist.destroy_process_group()
        return

    latent_recorder = None
    if os.environ.get("COSMOS_VIDEO_LATENT_DIR"):
        from cosmos_framework.inference.robot_policy.latent_recording import install
        latent_recorder = install(adapter, Path(os.environ["COSMOS_VIDEO_LATENT_DIR"]) / str(time.time_ns()))
    adapter.dispatch = True
    server = create_http_server(config, adapter, api_key)
    status = {
        "ready": True,
        "rank_pids": pids,
        "world_size": world,
        "port": args.port,
        "compiled_region": "language",
        "local_text_kv": True,
        "serial_warmup": True,
        "cuda_graphs": False,
        "source": str(Path(__file__).resolve()),
        "recording_dir": str(adapter._recorder.directory) if adapter._recorder else None,
        "started_wall_ns": time.time_ns(),
    }
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    args.status_file.write_text(json.dumps(status, indent=2) + "\n")
    print(f"CFG_READY rank0_pid={os.getpid()} world={world} port={args.port}", flush=True)

    def shutdown(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        server.application._executor.shutdown(wait=True, cancel_futures=True)
        server.server_close()
        if latent_recorder is not None:
            latent_recorder.close()
        status.update(ready=False, stopped_wall_ns=time.time_ns())
        args.status_file.write_text(json.dumps(status, indent=2) + "\n")
        try:
            if world == 2:
                control.exchange({"kind": "stop"})
                if control.exchange(source=1).get("kind") != "stopped":
                    fail_process("missing stop acknowledgement")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
