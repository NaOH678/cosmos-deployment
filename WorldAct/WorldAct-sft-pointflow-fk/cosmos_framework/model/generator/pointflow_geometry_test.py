"""CPU tests of geometry aggregation independent of sparse kernels."""

import unittest

import torch

from cosmos_framework.model.generator.pointflow_geometry import SonataGeometryEncoder, summarize_geometry


class Point(dict):
    def __getattr__(self, name):
        return self[name]


class FakeHierarchy(torch.nn.Module):
    """Tests index plumbing only; it does not emulate Sonata's CUDA operations."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, data):
        assert data["offset"].tolist() == [2, 3]  # empty middle sample removed
        point = Point(feat=self.weight * torch.ones(3, 32), batch=torch.tensor([0, 0, 1]))
        for stage in range(1, 5):
            point = Point(
                feat=self.weight * torch.ones(2, SonataGeometryEncoder.channels[stage]),
                batch=torch.tensor([0, 1]),
                pooling_parent=point,
                pooling_inverse=torch.tensor([0, 0, 1]) if stage == 1 else torch.tensor([0, 1]),
            )
        return point


class GeometryTests(unittest.TestCase):
    def test_original_member_weighting_and_gradient(self):
        inputs = {
            "original_to_voxel": torch.tensor([0, 0, 1, 2]),
            "anchor_xyz": torch.tensor([[0.0, 0, 1], [0, 0, 1], [9, 0, 1], [20, 0, 1]]),
            "anchor_uv": torch.tensor([[0.0, 0], [0, 0], [9, 0], [20, 0]]),
            "point_batch": torch.tensor([0, 0, 0, 2]),
            "point_ids": torch.tensor([7, 8, 9, 7]),
            "point_offsets": torch.tensor([3, 3, 4]),
            "has_geometry": torch.tensor([True, False, True]),
        }
        features = torch.randn(2, 256, requires_grad=True)
        output = summarize_geometry(inputs, features, torch.tensor([0, 0, 1]), torch.tensor([0, 2]))
        torch.testing.assert_close(output["cluster_xyz"][:, 0], torch.tensor([3.0, 20]))
        self.assertEqual(output["cluster_offsets"].tolist(), [1, 1, 2])
        self.assertEqual(output["cluster_counts"].tolist(), [3, 1])
        torch.testing.assert_close(
            output["relative_xyz"] + output["cluster_xyz"][output["original_to_cluster"]], inputs["anchor_xyz"]
        )
        output["cluster_features"].sum().backward()
        self.assertIsNotNone(features.grad)
        with self.assertRaises(ValueError):
            summarize_geometry(inputs, features, torch.tensor([0, 0, 1]), torch.tensor([0, 1]))
        encoder = SonataGeometryEncoder.__new__(SonataGeometryEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.backbone = FakeHierarchy()
        encoder.stage, encoder.output_dim, encoder.frozen = 3, 256, False
        inputs.update(
            coord=torch.zeros(3, 3),
            feat=torch.zeros(3, 9),
            grid_coord=torch.zeros(3, 3, dtype=torch.long),
            voxel_offsets=torch.tensor([2, 2, 3]),
        )
        output = encoder(inputs)
        self.assertEqual(output["original_to_cluster"].tolist(), [0, 0, 0, 1])
        self.assertEqual(output["cluster_batch"].tolist(), [0, 2])
        output["cluster_features"].sum().backward()
        self.assertGreater(encoder.backbone.weight.grad.item(), 0)
        encoder.frozen = True
        encoder.train()
        self.assertFalse(encoder.backbone.training)
        self.assertFalse(encoder(inputs)["cluster_features"].requires_grad)


if __name__ == "__main__":
    unittest.main()
