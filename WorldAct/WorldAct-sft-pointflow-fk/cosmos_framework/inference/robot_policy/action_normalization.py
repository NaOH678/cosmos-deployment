"""27D EEF normalization from checksum-pinned frozen training metadata.

State and action use (x-offset)/scale; quaternion channels are identity.
The adapter keeps ActionProcessingRecord.action_normalizer=None and explicitly
inverts the generated 27D actions once, before physical-space smoothing.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

NAMES = ["right_pos_x", "right_pos_y", "right_pos_z", "right_quat_x",
         "right_quat_y", "right_quat_z", "right_quat_w"] + [f"right_hand_{i}" for i in range(20)]

@dataclass(frozen=True)
class EEFActionNormalizer:
    offset: np.ndarray
    scale: np.ndarray
    sha256: str

    def _array(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] != 27 or not np.isfinite(values).all():
            raise ValueError("Normalization requires finite 27D unpadded state/action")
        return values

    def normalize(self, values):
        return (self._array(values) - self.offset) / self.scale

    def denormalize(self, values):
        return self._array(values) * self.scale + self.offset


def load_stats(path, expected_sha256):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if not expected_sha256 or digest != expected_sha256:
        raise ValueError("Action stats SHA256 mismatch with frozen training config")
    s = json.loads(payload)
    if (s.get("schema") != "singlerighthand_action_quantile_v1"
        or s.get("method") != "action_quantile_scale_floor"
        or s.get("arm_action_space") != "eef" or s.get("action_names") != NAMES
        or s.get("units") != ["meter"]*3 + ["unitless_quaternion"]*4 + ["radian"]*20
        or s.get("state_transform") != "same_as_action"
        or s.get("forward_clamp", "missing") is not None
        or s.get("preserved_dimensions") != [3, 4, 5, 6]
        or s.get("fit_split") != "train"):
        raise ValueError("Unsupported single-right-hand normalization contract")
    offset, scale = (np.asarray(s[k], dtype=np.float32) for k in ("offset", "scale"))
    if (offset.shape != (27,) or scale.shape != (27,)
        or not np.isfinite([offset, scale]).all() or np.any(scale <= 0)
        or np.any(offset[3:7] != 0) or np.any(scale[3:7] != 1)):
        raise ValueError("Invalid normalization offsets/scales or quaternion transform")
    return EEFActionNormalizer(offset, scale, digest)


def load_for_deployment(config):
    path = Path(config.model.config_file)
    if path.suffix not in {".yaml", ".yml"}:
        return None
    frozen = yaml.safe_load(path.read_text())
    train = frozen.get("dataloader_train", {}).get("dataloader", {}).get("datasets", {}).get("singlerighthand", {}).get("dataset", {})
    stats_path = train.get("action_stats_path")
    expected = train.get("action_stats_sha256")
    if not stats_path:
        if expected:
            raise ValueError("Frozen config has a stats digest but no stats path")
        return None
    if config.deployment.action_space != "eef":
        raise ValueError("Normalized checkpoint requires EEF deployment")
    # The deploy copy must resolve the training stats explicitly; never guess
    # a similarly named sidecar or silently fall back to unnormalized inference.
    stats_path = Path(stats_path).expanduser()
    if not stats_path.is_absolute():
        stats_path = path.parent / stats_path
    normalizer = load_stats(stats_path, expected)
    val = frozen.get("dataloader_val", {}).get("dataloader", {}).get("datasets", {}).get("singlerighthand", {}).get("dataset", {})
    if val.get("action_stats_sha256", expected) != expected:
        raise ValueError("Train/validation action stats digests disagree")
    return normalizer
