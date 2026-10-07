"""Portable Bench2Dex joint scaling shared by training and inference (no model imports)."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cosmos_framework.utils.bench2dex_contract import JOINT_NAMES

SCHEMA = "bench2dex_joint_normalization_v2"
METHOD = "action_quantile_scale_floor"
LEGACY_CONTRACT = ("bench2dex_joint_normalization_v1", "quantile_state_action_union_scale_floor")


def action_quantile_parameters(stats, min_scale=0.05):
    """Match Bench2Dex/PointFlow-FK float32 arithmetic; state statistics are diagnostic only."""
    low, high = (np.asarray(stats[k], dtype=np.float32) for k in ("q01", "q99"))
    if (
        low.shape != (52,)
        or high.shape != (52,)
        or not np.isfinite([low, high]).all()
        or np.any(high < low)
        or not np.isfinite(min_scale)
        or min_scale <= 0
    ):
        raise ValueError("Invalid 52D action quantile statistics")
    return (low + high) / 2, np.maximum((high - low) / 2, min_scale)


@dataclass(frozen=True)
class Bench2DexJointNormalizer:
    offset: tuple[float, ...]
    scale: tuple[float, ...]

    def normalize_action(self, action):
        if action.shape[-1] != len(JOINT_NAMES):
            raise ValueError("Normalize 52 real joints before channel padding")
        return (action - action.new_tensor(self.offset)) / action.new_tensor(self.scale)

    def denormalize_action(self, action):
        if action.shape[-1] != len(JOINT_NAMES):
            raise ValueError("Remove padded channels before denormalization")
        return action * action.new_tensor(self.scale) + action.new_tensor(self.offset)


def load_bench2dex_normalizer(path, expected_sha256=None):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Bench2Dex action stats SHA256 mismatch")
    stats = json.loads(payload)
    contract = (stats.get("schema"), stats.get("method"))
    if (
        contract not in {(SCHEMA, METHOD), LEGACY_CONTRACT}
        or stats.get("joint_names") != list(JOINT_NAMES)
        or stats.get("units") != "radian"
        or stats.get("forward_clamp") is not None
        or stats.get("fit_split") != "train"
    ):
        raise ValueError("Incompatible Bench2Dex action stats contract")
    offset, scale = (np.asarray(stats[k], dtype=np.float64) for k in ("offset", "scale"))
    if offset.shape != (52,) or scale.shape != (52,) or not np.isfinite([offset, scale]).all() or (scale <= 0).any():
        raise ValueError("Action stats require 52 finite offsets and positive scales")
    if contract == (SCHEMA, METHOD):
        if (
            not stats.get("source", "").startswith("training episodes only")
            or stats.get("frame_weighting") != "all_action_valid_frames_once"
            or stats.get("state_transform") != "same_as_action"
        ):
            raise ValueError("Incompatible action-only statistics sampling/state contract")
        resolved_offset, resolved_scale = action_quantile_parameters(stats["action"], stats["min_scale"])
        if not np.array_equal(offset.astype(np.float32), resolved_offset) or not np.array_equal(
            scale.astype(np.float32), resolved_scale
        ):
            raise ValueError("Saved offset/scale differs from action q01/q99")
        offset, scale = resolved_offset, resolved_scale
    # v1 remains readable for existing checkpoints; never reinterpret it using v2 math.
    return Bench2DexJointNormalizer(tuple(offset), tuple(scale)), stats, digest
