"""Tests for the optional point-level decoder blocks and multi-resolution skips."""

import unittest

import torch

from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.pointflow_codec import PointFlowCodec
from cosmos_framework.model.generator.pointflow_codec_test import fixture
from cosmos_framework.model.generator.pointflow_geometry import voxel_level_features


class Point(dict):
    def __getattr__(self, name):
        return self[name]


def fake_levels():
    """Three-level hierarchy: 4 voxels -> 3 clusters -> 2 clusters."""
    fine = Point(feat=torch.randn(4, 8))
    mid = Point(feat=torch.randn(3, 16), pooling_parent=fine, pooling_inverse=torch.tensor([0, 0, 1, 2]))
    coarse = Point(feat=torch.randn(2, 32), pooling_parent=mid, pooling_inverse=torch.tensor([0, 1, 1]))
    return [fine, mid, coarse]


class VoxelLevelFeatureTests(unittest.TestCase):
    def test_gather_matches_chain(self):
        fine, mid, coarse = fake_levels()
        torch.testing.assert_close(voxel_level_features([fine], 0), fine.feat)
        torch.testing.assert_close(voxel_level_features([fine, mid], 1), mid.feat[[0, 0, 1, 2]])
        torch.testing.assert_close(voxel_level_features([fine, mid, coarse], 2), coarse.feat[[0, 0, 1, 1]])


class PointDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.timing = PointFlowTiming(fps=15, steps=8, steps_per_token=4)

    def make_codec(self, **kwargs):
        params = dict(timing=self.timing, geometry_dim=8, local_dim=4, hidden_dim=16, content_dim=12, decoder_dim=24)
        params.update(kwargs)
        return PointFlowCodec(**params)

    def test_zero_init_and_gradient(self):
        codec = self.make_codec(point_blocks=2, point_block_dim=16, point_block_heads=4)
        self.assertIsNone(codec.decoder)
        inputs, geometry = fixture()
        noise = torch.randn(8, 4, 3, requires_grad=True)
        sigma = torch.tensor([0.2, 0.5, 0.8])
        tokens = codec(geometry, noise, sigma)
        velocity = codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, sigma)
        self.assertEqual(velocity.shape, (8, 4, 3))
        self.assertEqual(velocity.abs().max().item(), 0.0)  # zero-initialised head
        (
            velocity.square().mean() + tokens["noisy_tokens"].square().mean() + tokens["anchor_tokens"].square().mean()
        ).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters()))
        self.assertIsNotNone(geometry["cluster_features"].grad)

    def test_attention_does_not_cross_samples(self):
        codec = self.make_codec(point_blocks=2, point_block_dim=16, point_block_heads=4)
        with torch.no_grad():
            codec.point_decoder.out[-1].weight.normal_(0, 0.1)
        inputs, geometry = fixture()
        noise = torch.randn(8, 4, 3)
        sigma = torch.tensor([0.2, 0.5, 0.8])
        tokens = codec(geometry, noise, sigma)
        velocity = codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, sigma)
        # Perturb the points of sample 2 only (index 3); sample 0 must be unaffected.
        changed = noise.clone()
        changed[:, 3] += 10
        other = codec.decode(geometry, inputs, codec(geometry, changed, sigma)["noisy_tokens"], changed, sigma)
        torch.testing.assert_close(velocity[:, :3], other[:, :3])
        self.assertFalse(torch.allclose(velocity[:, 3], other[:, 3]))

    def test_default_path_unchanged(self):
        codec = self.make_codec()
        self.assertIsNone(codec.point_decoder)
        inputs, geometry = fixture()
        noise = torch.randn(8, 4, 3)
        sigma = torch.tensor([0.2, 0.5, 0.8])
        tokens = codec(geometry, noise, sigma)
        velocity = codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, sigma)
        self.assertEqual(velocity.shape, (8, 4, 3))
        self.assertGreater(velocity.abs().max().item(), 0.0)

    def test_skip_features_concat_and_validation(self):
        codec = self.make_codec(skip_dims=(5,), point_blocks=1, point_block_dim=16, point_block_heads=4)
        inputs, geometry = fixture()
        geometry["voxel_skip_features"] = [torch.randn(3, 5, requires_grad=True)]
        noise = torch.randn(8, 4, 3, requires_grad=True)
        sigma = torch.tensor([0.2, 0.5, 0.8])
        tokens = codec(geometry, noise, sigma)
        velocity = codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, sigma)
        self.assertEqual(velocity.shape, (8, 4, 3))
        velocity.sum().backward()
        self.assertIsNotNone(geometry["voxel_skip_features"][0].grad)
        del geometry["voxel_skip_features"]
        with self.assertRaises(ValueError):
            codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, sigma)
        geometry["voxel_skip_features"] = [torch.randn(3, 5), torch.randn(3, 5)]
        with self.assertRaises(ValueError):
            codec.decode(geometry, inputs, tokens["noisy_tokens"], noise, sigma)


if __name__ == "__main__":
    unittest.main()
