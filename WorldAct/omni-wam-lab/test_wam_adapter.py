from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from wam_adapter import load_packet, finish_output

ROOT = Path(__file__).resolve().parent


class WamAdapterTests(unittest.TestCase):
    def test_recorded_state_and_native_contract(self):
        m, frame, action, size = load_packet(ROOT / "fixtures/observation_01.npz")
        with np.load(m["source"], allow_pickle=False) as z:
            np.testing.assert_array_equal(action[0, :27], z["state"])
        self.assertEqual(m["domain_id"], 26)
        self.assertEqual(m["action_start_frame_offset"], 0)
        self.assertEqual(m["action_condition_indexes"], [0])
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.shape, (3, 736, 544))
        self.assertTrue(np.all(action[1:] == 0))
        self.assertEqual(
            (m["num_inference_steps"], m["guidance_scale"], m["flow_shift"]),
            (4, 3.0, 5.0),
        )

    def test_preprocessing_matches_native_adapter(self):
        from unittest.mock import patch
        from cosmos_framework.inference.robot_policy.adapters import (
            SingleRightHandCosmosAdapter,
        )
        from cosmos_framework.inference.robot_policy.config import (
            load_robot_policy_config,
        )

        config = (
            ROOT.parent
            / "WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml"
        )
        adapter = SingleRightHandCosmosAdapter.__new__(SingleRightHandCosmosAdapter)
        adapter.config = load_robot_policy_config(config)
        adapter.input_video_key = "video"
        m, frame, action, size = load_packet(ROOT / "fixtures/observation_01.npz")
        with np.load(m["source"], allow_pickle=False) as z:
            images = {name: z[name].copy() for name in ["head", "right_wrist"]}
            state = z["state"].copy()
        original_to = torch.Tensor.to

        def cpu_to(tensor, *args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return original_to(tensor, *args, **kwargs)

        # Only the final metadata device transfer is replaced; all native
        # composition, resizing, padding, prompt and state logic runs unchanged.
        with patch.object(torch.Tensor, "to", cpu_to):
            batch = adapter._build_batch(images, state)
        np.testing.assert_array_equal(frame, batch["video"][0][0][:, 0].numpy())
        np.testing.assert_array_equal(action, batch["action"][0][0].numpy())
        np.testing.assert_array_equal(size, batch["image_size"][0].numpy())
        self.assertEqual(m["prompt"], batch["prompt"][0])
        self.assertEqual(m["domain_id"], batch["domain_id"][0].item())

    def test_absolute_actions_unchanged_except_state_row(self):
        actions = torch.arange(33 * 27, dtype=torch.float32).reshape(1, 33, 27)
        output = SimpleNamespace(
            output={
                "payload": {"actions": actions},
                "metadata": {
                    "actions": {},
                    "internal": {"robolab_action_postprocess": "must remove"},
                },
            }
        )
        result = finish_output(output)
        self.assertTrue(
            torch.equal(result.output["payload"]["actions"], actions[:, 1:])
        )
        self.assertNotIn("internal", result.output["metadata"])

    def test_initial_noise_matches_native(self):
        from wam_adapter import native_or_torch_noise
        from cosmos_framework.utils.misc import arch_invariant_rand

        for shape, dtype in [
            ((1, 48, 9, 4, 4), torch.float32),
            ((1, 33, 64), torch.bfloat16),
        ]:
            expected = (
                arch_invariant_rand(shape[1:], dtype, "cpu", 0).float().unsqueeze(0)
            )
            actual = native_or_torch_noise(
                0, shape, generator=None, device="cpu", dtype=torch.float32
            )
            self.assertTrue(torch.equal(expected, actual))

    def test_invalid_output_rejected(self):
        output = SimpleNamespace(
            output={"payload": {"actions": torch.full((1, 33, 27), float("nan"))}}
        )
        with self.assertRaises(ValueError):
            finish_output(output)


if __name__ == "__main__":
    unittest.main()
