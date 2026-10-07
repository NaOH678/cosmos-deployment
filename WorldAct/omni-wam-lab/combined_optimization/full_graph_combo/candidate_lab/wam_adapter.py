"""Offline-only adapter: consumes native packets, never converts to DROID actions."""

import json
import atexit
from collections.abc import Mapping
import os
from pathlib import Path
import numpy as np
import torch


def load_packet(path):
    if isinstance(path, Mapping):
        meta = dict(path["metadata"])
        frame = np.asarray(path["first_frame"])
        action = np.asarray(path["action"])
        size = np.asarray(path["image_size"])
    else:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
            frame = z["first_frame"].copy()
            action = z["action"].copy()
            size = z["image_size"].copy()
    if meta["schema"] != "wam-native-v1" or meta["action_condition_indexes"] != [0]:
        raise ValueError("Invalid WAM conditioning packet")
    if frame.dtype != np.uint8 or size.shape != (4,) or not np.isfinite(size).all():
        raise ValueError("Invalid camera frame or image-size metadata")
    if action.shape != (33, 64) or frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("Expected 33x64 action and CHW frame")
    if not np.isfinite(action).all() or np.any(action[1:]) or np.any(action[:, 27:]):
        raise ValueError(
            "Packet must contain current state only, with zero future/padding"
        )
    return meta, frame, action, size


def build_inputs(path):
    from vllm_omni.diffusion.models.cosmos3.utils import RoboLabPolicyInputs

    m, frame, action, size = load_packet(path)
    # Match the upstream/native VAE input normalization exactly.
    input_frames = (
        1 if os.environ.get("WAM_FIRST_FRAME_ONLY") == "1" else m["num_frames"]
    )
    video = torch.zeros((1, 3, input_frames, *frame.shape[1:]), dtype=torch.uint8)
    video[0, :, 0] = torch.from_numpy(frame)
    return RoboLabPolicyInputs(
        prompt=m["prompt"],
        video_tensor=video.float() / 127.5 - 1.0,
        action_tensor=torch.from_numpy(action),
        action_condition_indexes=[0],
        action_start_frame_offset=m["action_start_frame_offset"],
        raw_action_dim=27,
        domain_id=m["domain_id"],
        fps=m["fps"],
        height=frame.shape[1],
        width=frame.shape[2],
        image_size=torch.from_numpy(size),
        num_frames=m["num_frames"],
        num_inference_steps=m["num_inference_steps"],
        guidance_scale=m["guidance_scale"],
        flow_shift=m["flow_shift"],
        seed=m["seed"],
        history_length=1,
        action_space="wam_absolute_27",
        observation={"_wam_native": True},
    )


def finish_output(output):
    if os.environ.get("WAM_COMPILE_DIAGNOSTICS") == "1":
        from torch._dynamo.utils import counters

        print(
            "WAM_DYNAMO_COUNTERS "
            + json.dumps({k: dict(v) for k, v in counters.items()}, default=str),
            flush=True,
        )
    action = output.output["payload"]["actions"]
    if tuple(action.shape) != (1, 33, 27) or not torch.isfinite(action).all():
        raise ValueError("Invalid raw WAM output")
    output.output["payload"]["actions"] = action[:, 1:].contiguous()
    output.output["metadata"].pop("internal", None)
    output.output["metadata"]["actions"]["representation"] = "wam_absolute_27"
    return output


def native_or_torch_noise(seed, shape, *, generator, device, dtype):
    if seed is None:
        from diffusers.utils.torch_utils import randn_tensor

        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    # Native arch_invariant_rand restarts RandomState(seed) per modality.
    noise = np.random.RandomState(int(seed)).standard_normal(shape).astype(np.float32)
    tensor = torch.from_numpy(noise)
    if len(shape) == 3:  # native action initialization rounds to model BF16
        tensor = tensor.to(torch.bfloat16)
    return tensor.to(device=device, dtype=dtype)


def record_runtime(transformer):
    """Record actual loaded dtypes and implementations once per worker."""
    from collections import Counter

    target = os.environ.get("WAM_RUNTIME_REPORT")
    if not target or getattr(transformer, "_wam_runtime_recorded", False):
        return
    methods = Counter()
    attention = Counter()
    compiled = []
    for name, module in transformer.named_modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            methods[type(method).__module__ + "." + type(method).__name__] += 1
        impl = getattr(module, "impl", None)
        if impl is not None:
            attention[type(impl).__module__ + "." + type(impl).__name__] += 1
        if hasattr(module.forward, "_torchdynamo_orig_callable"):
            compiled.append(name)
    data = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "parameter_dtypes": dict(
            Counter(str(p.dtype) for p in transformer.parameters())
        ),
        "quant_methods": dict(methods),
        "attention_implementations": dict(attention),
        "compiled_forward_modules": compiled,
    }
    Path(target).write_text(json.dumps(data, indent=2))
    transformer._wam_runtime_recorded = True


_latent_recorder = None


def capture_latents(latents, metadata):
    global _latent_recorder
    directory = os.environ.get("COSMOS_VIDEO_LATENT_DIR")
    if not directory or metadata is None:
        return
    if _latent_recorder is None:
        from cosmos_framework.inference.robot_policy.latent_recording import (
            LatentRecorder,
        )
        import time

        _latent_recorder = LatentRecorder(Path(directory) / f"omni_{time.time_ns()}")
        atexit.register(flush_latents)
    _latent_recorder.submit(latents, metadata)


def flush_latents():
    global _latent_recorder
    if _latent_recorder is not None:
        _latent_recorder.close()
        _latent_recorder = None


def extract_actions(value):
    if isinstance(value, list):
        for item in value:
            found = extract_actions(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        if "actions" in value:
            return value["actions"]
        for key in ["payload", "multimodal_output", "output"]:
            if key in value:
                found = extract_actions(value[key])
                if found is not None:
                    return found
    for attr in ["multimodal_output", "output"]:
        if hasattr(value, attr):
            found = extract_actions(getattr(value, attr))
            if found is not None:
                return found
    return None

