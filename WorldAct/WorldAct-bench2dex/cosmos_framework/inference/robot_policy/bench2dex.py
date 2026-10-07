"""Bench2Dex RPC policy: raw RGB/current qpos in, absolute radian commands out."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from cosmos_framework.inference.robot_policy.adapters import (
    SingleRightHandCosmosAdapter,
)
from cosmos_framework.utils.bench2dex_contract import (
    CAMERAS,
    JOINT_NAMES,
    ROBOT,
    VIEW_DESCRIPTION,
    compose_rgb,
    permutation,
)
from cosmos_framework.utils.bench2dex_normalization import load_bench2dex_normalizer


def load_policy_action_normalizer(options):
    """Use the checkpoint's stats contract; a deployment override may only relocate the same file."""
    from omegaconf import OmegaConf

    config_file = options.get("config_file")
    override = options.get("action_stats_path")
    if not config_file or Path(config_file).suffix not in {".yaml", ".yml"}:
        if override:
            raise ValueError("Normalized inference requires the training config.yaml")
        return None
    cfg = OmegaConf.load(config_file)
    prefix = "dataloader_train.dataloader.datasets.bench2dex.dataset"
    path = OmegaConf.select(cfg, f"{prefix}.action_stats_path")
    digest = OmegaConf.select(cfg, f"{prefix}.action_stats_sha256")
    if not path:
        if override or digest:
            raise ValueError("Cannot attach normalized stats to an unnormalized checkpoint config")
        return None
    if not digest:
        raise ValueError("Normalized training config must pin action_stats_sha256")
    return load_bench2dex_normalizer(override or path, digest)[0]


class Bench2DexPolicy(SingleRightHandCosmosAdapter):
    def __init__(self, options):
        manifest = json.loads(Path(options["manifest"]).read_text())
        if (
            manifest.get("schema") != "cosmos_bench2dex_joint_v1"
            or manifest.get("robot_key") != ROBOT
            or manifest["joint_names"] != list(JOINT_NAMES)
            or manifest["cameras"] != list(CAMERAS)
        ):
            raise ValueError("Incompatible training manifest")
        self.device = "cuda"
        self.view_width = int(manifest["view_width"])
        self.expected_runtime_names = (
            json.loads(Path(options["runtime_joint_names"]).read_text()) if options.get("runtime_joint_names") else None
        )
        if self.expected_runtime_names is not None:
            permutation(self.expected_runtime_names)
        self.runtime_names = None
        self.execution_horizon = int(options.get("execution_horizon", 4))
        if not 1 <= self.execution_horizon <= 32:
            raise ValueError("execution_horizon must be 1..32")
        self.seed = int(options.get("seed", 0))
        self.action_normalizer = load_policy_action_normalizer(options)
        model = SimpleNamespace(
            checkpoint_path=options["checkpoint_path"],
            checkpoint_sha256=None,
            config_file=options.get("config_file", "cosmos_framework/configs/base/config.py"),
            experiment="action_policy_bench2dex_edge",
            experiment_overrides=options.get("experiment_overrides", []),
            output_dir=Path(options.get("output_dir", "outputs/bench2dex_policy")),
            credential_path="credentials/gcp_checkpoint.secret",
            sampler="unipc",
            use_ema_weights=bool(options.get("use_ema_weights", True)),
            max_action_dim=64,
            native_action_dim=52,
            native_chunk_size=32,
            warmup=False,
            task=manifest["task_text"],
            domain_name="bench2dex_wuji",
            resolution="480",
            seed=self.seed,
            guidance=float(options.get("guidance", 3)),
            num_steps=int(options.get("num_steps", 4)),
            shift=5.0,
        )
        super().__init__(SimpleNamespace(model=model, deployment=SimpleNamespace(action_rate_hz=20.0)))

    def get_action(self, observation):
        names = observation.get("joint_names")
        if names is None:
            raise ValueError("Missing runtime joint_names: update Bench2Dex raw observation RPC metadata")
        names = list(names)
        if self.expected_runtime_names is not None and names != self.expected_runtime_names:
            raise ValueError("Runtime joint order differs from configured expected order")
        if names != self.runtime_names:
            self.to_model = permutation(names)
            self.to_runtime = permutation(JOINT_NAMES, names)
            self.runtime_names = names
        q = np.asarray(observation["joint_action"]["qpos"], np.float32)
        if q.shape != (52,) or not np.isfinite(q).all():
            raise ValueError("Expected finite 52D runtime qpos")
        images = {name: observation["observation"][name]["rgb"] for name in CAMERAS}
        actions, _ = self._infer_native(images, q[self.to_model])
        if actions.shape != (32, 52) or not np.isfinite(actions).all():
            raise ValueError("Invalid generated actions")
        return np.ascontiguousarray(actions[: self.execution_horizon, self.to_runtime])

    def reset_model(self):
        self.config.model.seed = self.seed

    def _build_batch(self, images_rgb: Mapping[str, np.ndarray], right_state: np.ndarray) -> dict[str, Any]:
        import torch

        from cosmos_framework.data.generator.action.action_processing import (
            ActionProcessingRecord,
            make_batched_action_processing_fields,
        )
        from cosmos_framework.data.generator.action.domain_utils import get_domain_id
        from cosmos_framework.data.generator.action.json_formatter import (
            ActionPromptJsonFormatter,
        )
        from cosmos_framework.data.generator.action.transforms import (
            build_sequence_plan_from_mode,
            find_closest_target_size,
            reflection_pad_to_target,
        )

        composed = torch.from_numpy(compose_rgb(images_rgb, self.view_width))
        target_frames = self.config.model.native_chunk_size + 1
        _, height, width = composed.shape
        video = torch.zeros((3, target_frames, height, width), dtype=torch.uint8)
        video[:, 0] = composed

        target_w, target_h = find_closest_target_size(height, width, self.config.model.resolution)
        padded: dict[str, Any] = {"video": video}
        reflection_pad_to_target(padded, ["video"], True, target_w, target_h)
        video = padded["video"]
        image_size = padded["image_size"]

        action = torch.zeros((target_frames, self.config.model.max_action_dim), dtype=torch.float32)
        state = torch.from_numpy(right_state)
        normalizer = getattr(self, "action_normalizer", None)
        if normalizer is not None:
            state = normalizer.normalize_action(state)
        action[0, : self.config.model.native_action_dim] = state
        sequence_plan = build_sequence_plan_from_mode(
            mode="wam",
            video_length=target_frames,
            action_length=target_frames,
            has_text=True,
        )
        prompt_data: dict[str, Any] = {
            "ai_caption": self.config.model.task,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.config.deployment.action_rate_hz),
            "image_size": image_size,
            "mode": "wam",
            "viewpoint": "concat_view",
            "additional_view_description": VIEW_DESCRIPTION,
        }
        formatted = ActionPromptJsonFormatter(caption_key="ai_caption")(prompt_data)["ai_caption"]
        prompt = json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)
        record = ActionProcessingRecord(
            raw_action_dim=self.config.model.native_action_dim, action_normalizer=normalizer
        )
        return {
            self.input_video_key: [[video]],
            "action": [[action]],
            **make_batched_action_processing_fields(record, batch_size=1),
            "mode": ["wam"],
            "ai_caption": [prompt],
            "prompt": [prompt],
            "conditioning_fps": [torch.tensor(self.config.deployment.action_rate_hz, dtype=torch.long)],
            "image_size": image_size.unsqueeze(0).to(device=self.device),
            "domain_id": [torch.tensor(get_domain_id(self.config.model.domain_name), dtype=torch.long)],
            "sequence_plan": [sequence_plan],
        }


def get_model(options):
    return Bench2DexPolicy(options)


def reset_model(model):
    model.reset_model()
