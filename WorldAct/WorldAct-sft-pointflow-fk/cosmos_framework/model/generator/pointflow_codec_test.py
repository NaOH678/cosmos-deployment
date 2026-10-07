"""Independent tests for chronological motion blocks and ragged point content."""

import unittest

import torch

from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.pointflow_codec import PointFlowCodec, motion_blocks, restore_motion
from cosmos_framework.model.generator.pointflow_geometry import summarize_geometry


def fixture(empty=False):
    mapping = torch.tensor([], dtype=torch.long) if empty else torch.tensor([0, 0, 1, 2])
    point_batch = torch.tensor([], dtype=torch.long) if empty else torch.tensor([0, 0, 0, 2])
    points, voxels = len(mapping), 0 if empty else 3
    inputs = {
        "original_to_voxel": mapping,
        "anchor_xyz": torch.randn(points, 3),
        "anchor_uv": torch.randn(points, 2),
        "point_batch": point_batch,
        "point_ids": torch.arange(points),
        "point_offsets": torch.tensor([0, 0, 0] if empty else [3, 3, 4]),
        "has_geometry": torch.tensor([False, False, False] if empty else [True, False, True]),
        "image_size_wh": torch.tensor([[640, 448]] * 3),
    }
    features = torch.randn(0 if empty else 2, 8, requires_grad=True)
    geometry = summarize_geometry(
        inputs,
        features,
        torch.tensor([], dtype=torch.long) if empty else torch.tensor([0, 0, 1]),
        torch.tensor([], dtype=torch.long) if empty else torch.tensor([0, 2]),
    )
    geometry["voxel_features"] = torch.randn(voxels, 4, requires_grad=True)
    return inputs, geometry


class CodecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.timing = PointFlowTiming(fps=15, steps=8, steps_per_token=4)
        self.codec = PointFlowCodec(
            timing=self.timing, geometry_dim=8, local_dim=4, hidden_dim=16, content_dim=12, decoder_dim=24
        )

    def test_time_order_roundtrip(self):
        x = torch.arange(8 * 3 * 3).reshape(8, 3, 3)
        blocks = motion_blocks(x, self.timing)
        torch.testing.assert_close(blocks[1, 2], x[4:8, 2].flatten())
        torch.testing.assert_close(restore_motion(blocks, self.timing), x)

    def test_ragged_isolation_anchor_and_gradient(self):
        inputs, geometry = fixture()
        noise = torch.randn(8, 4, 3, requires_grad=True)
        sigma = torch.tensor([0.2, 0.5, 0.8])
        tokens = self.codec(geometry, noise, sigma)
        self.assertEqual(tokens["noisy_tokens"].shape, (2, 2, 16))
        changed = noise.detach().clone()
        changed[:, 3] += 10  # alter sample 2 only; sample 1 is empty
        other = self.codec(geometry, changed, sigma)
        torch.testing.assert_close(tokens["noisy_tokens"][:, 0], other["noisy_tokens"][:, 0])
        torch.testing.assert_close(tokens["anchor_tokens"], other["anchor_tokens"])
        other_sigma = self.codec(geometry, noise, torch.zeros(3))
        torch.testing.assert_close(tokens["anchor_tokens"], other_sigma["anchor_tokens"])
        velocity = self.codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, sigma)
        self.assertEqual(velocity.shape, (8, 4, 3))
        # Points 0/1 share a voxel/cluster but retain different noisy skips.
        self.assertFalse(torch.allclose(velocity[:, 0], velocity[:, 1]))
        (velocity.square().mean() + tokens["anchor_tokens"].square().mean()).backward()
        for name, tensor in [
            ("noise", noise),
            ("geometry", geometry["cluster_features"]),
            ("local", geometry["voxel_features"]),
        ]:
            self.assertIsNotNone(tensor.grad, name)
            self.assertTrue(torch.isfinite(tensor.grad).all(), name)
            self.assertGreater(tensor.grad.abs().sum().item(), 0, name)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.codec.parameters()))

    def test_empty_and_validation(self):
        inputs, geometry = fixture(empty=True)
        noise = torch.empty(8, 0, 3)
        tokens = self.codec(geometry, noise, torch.zeros(3))
        out = self.codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, torch.zeros(3))
        self.assertEqual(out.shape, (8, 0, 3))
        (out.sum() + tokens["anchor_tokens"].sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.codec.parameters()))
        with self.assertRaises(ValueError):
            self.codec(geometry, noise, torch.tensor([0.0, float("nan"), 1.0]))
        with self.assertRaises(ValueError):
            self.codec(geometry, torch.empty(9, 0, 3), torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
