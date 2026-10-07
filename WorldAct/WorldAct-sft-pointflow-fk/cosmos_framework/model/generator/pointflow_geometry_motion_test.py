"""Geometry-motion pairing, permutation invariance and gradients before pooling."""

import unittest

import torch

from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.pointflow_codec import PointFlowCodec
from cosmos_framework.model.generator.pointflow_codec_test import fixture


class GeometryMotionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.inputs, self.geometry = fixture()
        self.noise = torch.randn(8, 4, 3)
        self.sigma = torch.tensor([0.2, 0.5, 0.8])

    def codec(self, **kwargs):
        return PointFlowCodec(
            timing=PointFlowTiming(steps=8),
            geometry_dim=8,
            local_dim=4,
            hidden_dim=16,
            content_dim=12,
            decoder_dim=24,
            **kwargs,
        )

    def test_motion_swap_changes_fused_tokens_only(self):
        # Points 0/1 share a voxel, but their relative XYZ distinguishes them.
        swapped = self.noise[:, [1, 0, 2, 3]]
        for enabled in (False, True):
            codec = self.codec(geometry_motion_fusion=enabled)
            first = codec(self.geometry, self.noise, self.sigma, inputs=self.inputs)
            other = codec(self.geometry, swapped, self.sigma, inputs=self.inputs)
            torch.testing.assert_close(first["anchor_tokens"], other["anchor_tokens"])
            torch.testing.assert_close(first["noisy_tokens"][:, 1], other["noisy_tokens"][:, 1])
            if enabled:
                self.assertGreater((first["noisy_tokens"] - other["noisy_tokens"]).abs().max().item(), 1e-5)
            else:
                torch.testing.assert_close(first["noisy_tokens"], other["noisy_tokens"])

    def test_joint_point_permutation_preserves_tokens(self):
        codec = self.codec(geometry_motion_fusion=True)
        first = codec(self.geometry, self.noise, self.sigma, inputs=self.inputs)
        permutation = torch.tensor([2, 0, 1, 3])
        geometry = dict(self.geometry)
        for key in ("original_to_cluster", "relative_xyz", "point_ids"):
            geometry[key] = geometry[key][permutation]
        inputs = dict(self.inputs, original_to_voxel=self.inputs["original_to_voxel"][permutation])
        other = codec(geometry, self.noise[:, permutation], self.sigma, inputs=inputs)
        torch.testing.assert_close(first["noisy_tokens"], other["noisy_tokens"])

    def test_encoder_gradient_and_sample_isolation(self):
        codec = self.codec(geometry_motion_fusion=True)
        noise = self.noise.requires_grad_()
        self.geometry["relative_xyz"].requires_grad_()
        first = codec(self.geometry, noise, self.sigma, inputs=self.inputs)
        changed = noise.detach().clone()
        changed[:, 3] += 10
        other = codec(self.geometry, changed, self.sigma, inputs=self.inputs)
        torch.testing.assert_close(first["noisy_tokens"][:, 0], other["noisy_tokens"][:, 0])
        first["noisy_tokens"].square().mean().backward()
        for tensor in (noise, self.geometry["voxel_features"], self.geometry["relative_xyz"]):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_empty_backward(self):
        inputs, geometry = fixture(empty=True)
        codec = self.codec(geometry_motion_fusion=True)
        noise = torch.empty(8, 0, 3)
        tokens = codec(geometry, noise, torch.zeros(3), inputs=inputs)
        velocity = codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, torch.zeros(3))
        self.assertEqual(velocity.shape, (8, 0, 3))
        (velocity.sum() + tokens["anchor_tokens"].sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters()))

    def test_point_decoder_backward(self):
        codec = self.codec(geometry_motion_fusion=True, point_blocks=1, point_block_dim=8, point_block_heads=2)
        tokens = codec(self.geometry, self.noise, self.sigma, inputs=self.inputs)
        velocity = codec.decode(self.geometry, self.inputs, tokens["noisy_tokens"], self.noise, self.sigma)
        self.assertEqual(velocity.shape, self.noise.shape)
        (velocity.square().mean() + tokens["anchor_tokens"].square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters()))

    def test_default_checkpoint_compatibility_and_input_guard(self):
        torch.manual_seed(7)
        default = self.codec()
        torch.manual_seed(7)
        disabled = self.codec(geometry_motion_fusion=False)
        disabled.load_state_dict(default.state_dict(), strict=True)
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
        first = default(self.geometry, self.noise, self.sigma)
        other = disabled(self.geometry, self.noise, self.sigma, inputs=self.inputs)
        torch.testing.assert_close(first["noisy_tokens"], other["noisy_tokens"], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "original_to_voxel"):
            self.codec(geometry_motion_fusion=True)(self.geometry, self.noise, self.sigma)


if __name__ == "__main__":
    unittest.main()
