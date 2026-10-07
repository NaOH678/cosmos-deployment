"""Single-GPU Omni WAM backend for the existing protocol-v2 HTTP service."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from cosmos_framework.inference.robot_policy.adapters import ModelAdapter, SingleRightHandCosmosAdapter
from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig, load_robot_policy_config
from cosmos_framework.inference.robot_policy.recording import AsyncRequestRecorder
from cosmos_framework.inference.robot_policy.server import create_http_server
from cosmos_framework.inference.robot_policy.wam_packet import build_first_frame_packet

LOG = logging.getLogger(__name__)


def validate_export(config, model_path):
    """Reject a stale export rather than serving different weights under a model id."""
    manifest = json.loads((model_path / "conversion.json").read_text())
    provenance = manifest["ema_manifest"]
    expected = Path(config.model.checkpoint_path).resolve()
    if Path(provenance["checkpoint"]).resolve() != expected:
        raise ValueError(f"Omni export checkpoint mismatch: expected {expected}")
    config_path = Path(config.model.config_file).resolve()
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != provenance["config_sha256"]:
        raise ValueError("Omni export frozen config hash mismatch")
    if not provenance["ema"] or not config.model.use_ema_weights:
        raise ValueError("This exported Omni backend requires EMA weights")
    if config.model.sampler != "unipc" or config.model.max_action_dim != 64:
        raise ValueError("Omni WAM currently supports UniPC and padded action dimension 64")
    if config.deployment.action_space != "eef" or config.model.service_mode != "full":
        raise ValueError("Omni WAM currently supports full, right-hand EEF policy only")
    if config.service.max_inflight_inferences != 1:
        raise ValueError("Omni policy requires max_inflight_inferences=1")
    if config.model.checkpoint_sha256:
        from cosmos_framework.inference.robot_policy.adapters import _verify_checkpoint_sha256

        _verify_checkpoint_sha256(str(expected), config.model.checkpoint_sha256)
    return manifest


class OmniWamAdapter(SingleRightHandCosmosAdapter):
    """Keep native preprocessing, postprocessing and recorder; replace prediction."""

    def __init__(self, config, lab_root, model_path, quantization="fp8"):
        ModelAdapter.__init__(self, config)
        self._ready = False
        self._lock = threading.Lock()
        self._inference_timing = threading.local()
        self._request_context = threading.local()
        self._recorder = None
        self._last_packet = None
        self.input_video_key = "video"
        self._first_frame_packet = os.environ.get("WAM_CPU_FIRST_FRAME_PACKET", "1") != "0"
        self._warmup_image_shapes = {
            k: tuple(config.image_preprocessing.warmup_shapes[k]) for k in config.deployment.camera_names
        }
        self._unwarmed_image_shapes = set()
        self.provenance = validate_export(config, model_path)
        upstream = json.loads((lab_root / "upstream.json").read_text())
        source = Path(upstream["source"])
        native = Path(__file__).resolve().parents[3]
        for root in (native, source, lab_root):
            sys.path.insert(0, str(root))
        os.environ["PYTHONPATH"] = os.pathsep.join(map(str, (lab_root, source, native)))
        os.environ["WAM_FIRST_FRAME_ONLY"] = "1"
        os.environ.setdefault("WAM_FULL_GEN_GRAPH", "1")
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        self.sampling_params_cls = OmniDiffusionSamplingParams
        self.quantization = quantization
        self.engine = Omni(
            model=str(model_path),
            model_class_name="Cosmos3OmniDiffusersPipeline",
            dtype="bfloat16",
            quantization=None if quantization == "none" else quantization,
            enforce_eager=False,
            diffusion_compile_granularity="regional",
            diffusion_compile_dynamic=True,
            cfg_parallel_size=1,
            tensor_parallel_size=1,
            model_config={"guardrails": False},
        )
        try:
            # Ready covers the configured camera shapes. Different live shapes may
            # need a separate graph capture; never claim arbitrary shapes are warm.
            images = {k: np.zeros(self._warmup_image_shapes[k], dtype=np.uint8) for k in config.deployment.camera_names}
            state = np.zeros(27, dtype=np.float32)
            state[6] = 1.0
            for _ in range(5):
                self._infer_native(images, state)
            self._warmup_packet_shape = tuple(self._last_packet["first_frame"].shape)
            LOG.info(
                "Omni warmup completed: camera_shapes=%s packet_shape=%s full_gen_graph=%s",
                self._warmup_image_shapes,
                self._warmup_packet_shape,
                os.environ["WAM_FULL_GEN_GRAPH"],
            )
            self._recorder = AsyncRequestRecorder.from_environment(config)
            self._ready = True
        except BaseException:
            self.engine.close()
            raise

    def _make_packet(self, images, state):
        if getattr(self, "_first_frame_packet", os.environ.get("WAM_CPU_FIRST_FRAME_PACKET", "1") != "0"):
            return build_first_frame_packet(self.config, images, state)
        return self._make_native_packet(images, state)

    def _make_native_packet(self, images, state):
        batch = self._build_batch(images, state)
        # Preserve the native prompt, frame composition, state and domain exactly.
        return {
            "metadata": {
                "schema": "wam-native-v1",
                "prompt": batch["prompt"][0],
                "domain_id": int(batch["domain_id"][0]),
                "fps": self.config.deployment.action_rate_hz,
                "raw_action_dim": 27,
                "num_frames": 33,
                "num_inference_steps": self.config.model.num_steps,
                "guidance_scale": self.config.model.guidance,
                "flow_shift": self.config.model.shift,
                "seed": self.config.model.seed,
                "action_condition_indexes": [0],
                "action_start_frame_offset": 0,
            },
            "first_frame": batch["video"][0][0][:, 0].contiguous().numpy(),
            "action": batch["action"][0][0].numpy(),
            "image_size": batch["image_size"][0].cpu().numpy(),
        }

    def _generate(self, packet, *, flush=False):
        from wam_adapter import extract_actions

        metadata = packet["metadata"]
        frame = packet["first_frame"]
        sp = self.sampling_params_cls(
            seed=metadata["seed"],
            num_inference_steps=metadata["num_inference_steps"],
            guidance_scale=metadata["guidance_scale"],
            num_frames=33,
            height=frame.shape[1],
            width=frame.shape[2],
            extra_args={
                "wam_observation": packet,
                "use_system_prompt": False,
                "frame_rate": metadata["fps"],
                "guardrails": False,
                "wam_capture_metadata": getattr(self._request_context, "metadata", None),
                "wam_flush_latents": flush,
            },
        )
        result = self.engine.generate({"prompt": metadata["prompt"], "modalities": ["video"]}, sp, use_tqdm=False)
        actions = extract_actions(result)
        if actions is None:
            raise ValueError("Omni returned no actions")
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape == (1, 32, 27):
            actions = actions[0]
        if actions.shape != (32, 27) or not np.isfinite(actions).all():
            raise ValueError("Omni returned invalid 32x27 actions")
        return actions

    def _infer_native(self, images_rgb, right_state):
        with self._lock:
            if self._ready:
                shapes = tuple((key, tuple(images_rgb[key].shape)) for key in self.config.deployment.camera_names)
                if (
                    any(shape != self._warmup_image_shapes[key] for key, shape in shapes)
                    and shapes not in self._unwarmed_image_shapes
                ):
                    self._unwarmed_image_shapes.add(shapes)
                    LOG.warning(
                        "Observation shape differs from startup warmup; first request for this shape may capture a graph: actual=%s configured=%s",
                        shapes,
                        self._warmup_image_shapes,
                    )
            started = time.perf_counter()
            packet = self._make_packet(images_rgb, right_state)
            prep_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            actions = self._generate(packet)
            elapsed = (time.perf_counter() - started) * 1000
            self._last_packet = packet
            self._inference_timing.value = {"batch_preprocess_ms": prep_ms, "omni_generate_ms": elapsed}
            return actions, elapsed

    def infer(self, observation):
        if observation.get("rtc_prefix") is not None:
            raise ValueError("Omni backend does not implement RTC prefix guidance")
        self._request_context.metadata = {
            **{k: observation.get(k) for k in ("session_id", "request_id", "timestamp", "client_monotonic")},
            "model_id": self.config.deployment.model_id,
            "model_config": self.config.model.config_file,
            "action_rate_hz": self.config.deployment.action_rate_hz,
            "backend": "vllm-omni",
            "quantization": self.quantization,
            "latent_format": "omni_cosmos3_vae_normalized_BCTHW",
        }
        try:
            return super().infer(observation)
        finally:
            self._request_context.metadata = None

    def close(self):
        self._ready = False
        self.close_recording()
        try:
            if os.environ.get("COSMOS_VIDEO_LATENT_DIR") and self._last_packet is not None:
                with self._lock:
                    self._generate(self._last_packet, flush=True)
        finally:
            self.engine.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--omni-root", type=Path, required=True)
    parser.add_argument("--omni-model", type=Path, required=True)
    parser.add_argument("--omni-quantization", choices=("fp8", "none"), default="fp8")
    parser.add_argument("--check-only", action="store_true")
    for flag in ("checkpoint-path", "model-config-file", "model-id", "service-mode", "host", "trajectory-smoothing"):
        parser.add_argument("--" + flag)
    parser.add_argument("--port", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--shift", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_robot_policy_config(args.config).model_dump()
    mapping = {
        "checkpoint_path": ("model", "checkpoint_path"),
        "model_config_file": ("model", "config_file"),
        "model_id": ("deployment", "model_id"),
        "service_mode": ("model", "service_mode"),
        "host": ("service", "host"),
        "port": ("service", "port"),
        **{key: ("model", key) for key in ("num_steps", "guidance", "shift", "trajectory_smoothing")},
    }
    for key, (section, field) in mapping.items():
        if getattr(args, key) is not None:
            raw[section][field] = getattr(args, key)
    config = RobotPolicyConfig.model_validate(raw)
    lab, model = args.omni_root.resolve(), args.omni_model.resolve()
    validate_export(config, model)
    upstream = json.loads((lab / "upstream.json").read_text())
    for path in (
        lab / "wam_adapter.py",
        Path(upstream["source"]) / "vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py",
    ):
        if not path.is_file():
            raise ValueError(f"Missing Omni adapter source: {path}")
    if args.check_only:
        print("Omni source/export/config checks passed; no GPU or server started.")
        return
    key = config.auth.load_api_key()
    recording = os.environ.get("COSMOS_RECORDING_DIR")
    audit_root = Path(recording) if recording else config.model.output_dir / "omni_runtime"
    audit_root.mkdir(parents=True, exist_ok=True)
    os.environ["WAM_RUNTIME_REPORT"] = str((audit_root / "omni_runtime.json").resolve())
    os.environ.setdefault("WAM_FULL_GEN_AUDIT", str((audit_root / "omni_graph_audit.json").resolve()))
    logging.basicConfig(level=logging.INFO)
    adapter = OmniWamAdapter(config, lab, model, args.omni_quantization)
    (audit_root / "omni_service.json").write_text(
        json.dumps(
            {
                "backend": "vllm-omni",
                "quantization": args.omni_quantization,
                "model": str(model),
                "model_id": config.deployment.model_id,
                "conversion": adapter.provenance,
                "upstream": upstream,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (
                        Path(__file__).resolve(),
                        Path(__file__).with_name("wam_packet.py").resolve(),
                        lab / "wam_adapter.py",
                        *((lab / "wam_gen_graph.py",) if (lab / "wam_gen_graph.py").is_file() else ()),
                        Path(upstream["source"]) / "vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py",
                    )
                },
                "first_frame_only": True,
                "cpu_first_frame_packet": adapter._first_frame_packet,
                "full_gen_graph": os.environ.get("WAM_FULL_GEN_GRAPH") == "1",
                "warmup_camera_shapes": adapter._warmup_image_shapes,
                "warmup_packet_shape": adapter._warmup_packet_shape,
                "warmup_scope": "Configured camera shapes only; different live shapes may need graph capture.",
                "config": config.model_dump(mode="json"),
                "request_recording_dir": str(adapter._recorder.directory) if adapter._recorder else None,
                "video_latent_dir": os.environ.get("COSMOS_VIDEO_LATENT_DIR"),
            },
            indent=2,
        )
    )
    server = None

    def shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    try:
        server = create_http_server(config, adapter, key)
        print(
            f"Omni policy ready: http://{config.service.host}:{config.service.port}; model={config.deployment.model_id}; quantization={args.omni_quantization}",
            flush=True,
        )
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if server is not None:
                server.application._executor.shutdown(wait=True, cancel_futures=True)
                server.server_close()
        finally:
            adapter.close()


if __name__ == "__main__":
    main()
