"""CPU-only contracts for the live Omni bridge; never controls a robot."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cosmos_framework.inference.robot_policy.config import load_robot_policy_config
from cosmos_framework.inference.robot_policy.omni_http import (
    OmniWamAdapter,
    validate_export,
)
from wam_adapter import load_packet

ROOT = Path(__file__).resolve().parent
CONFIG = (
    ROOT.parent.parent
    / "wuji-hand-teleop-pipeline/datasets/tianji_wuji/diagnostics/local5090_services/20261007T004610546291Z_4w_standalone/server.yaml"
)


class LiveOmniContracts(unittest.TestCase):
    def test_export_identity_and_wrong_checkpoint_rejected(self):
        config = load_robot_policy_config(CONFIG)
        validate_export(config, ROOT / "artifacts/4w-ema-omni")
        wrong = config.model_copy(deep=True)
        wrong.model.checkpoint_path = str(
            Path(config.model.checkpoint_path).with_name("iter_000020000")
        )
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            validate_export(wrong, ROOT / "artifacts/4w-ema-omni")

    def test_live_packet_preserves_native_preprocessing_and_new_state(self):
        adapter = OmniWamAdapter.__new__(OmniWamAdapter)
        adapter.config = load_robot_policy_config(CONFIG)
        adapter.input_video_key = "video"
        metadata = json.loads(
            str(
                np.load(ROOT / "fixtures/observation_01.npz", allow_pickle=False)[
                    "metadata"
                ].item()
            )
        )
        with np.load(metadata["source"], allow_pickle=False) as z:
            images = {name: z[name].copy() for name in ("head", "right_wrist")}
            state = z["state"].copy()
        original_to = torch.Tensor.to

        def cpu_to(tensor, *args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", cpu_to):
            batch = adapter._build_batch(images, state)
            packet = adapter._make_packet(images, state)
            changed = state.copy()
            changed[0] += 0.02
            next_packet = adapter._make_packet(images, changed)
        meta, frame, action, size = load_packet(packet)
        np.testing.assert_array_equal(frame, batch["video"][0][0][:, 0].numpy())
        np.testing.assert_array_equal(action, batch["action"][0][0].numpy())
        np.testing.assert_array_equal(size, batch["image_size"][0].numpy())
        self.assertEqual(meta["prompt"], batch["prompt"][0])
        np.testing.assert_array_equal(next_packet["action"][0, :27], changed)
        np.testing.assert_array_equal(packet["action"][0, :27], state)
        self.assertTrue(np.all(next_packet["action"][1:] == 0))
        bad = copy.deepcopy(packet)
        bad["action"][1, 0] = 1
        with self.assertRaisesRegex(ValueError, "current state only"):
            load_packet(bad)


if __name__ == "__main__":
    unittest.main()
