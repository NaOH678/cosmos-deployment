"""Per-point token mode: every original point is its own cluster and token."""

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.data.pointflow_batch import build_pointflow_batch, pointflow_token_upper_bound
from cosmos_framework.data.pointflow_batch_test import point_sample
from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.pointflow_codec import PointFlowCodec
from cosmos_framework.model.generator.pointflow_geometry import summarize_per_point_geometry
from cosmos_framework.scripts.validate_pointflow_network import make_network, make_sequence


def shared_voxel_sample(n, voxels):
    """n points sharing fewer voxels, so point and voxel counts differ."""
    sample = point_sample(n)
    sample["inputs"]["original_to_voxel"] = np.arange(n) % voxels
    sample["inputs"]["voxel_representatives"] = np.arange(voxels)
    for key in ("coord", "feat", "grid_coord"):
        sample["inputs"][key] = sample["inputs"][key][:voxels]
    return sample


class PerPointSurrogate(nn.Module):
    """Stand-in for SonataGeometryEncoder(per_point=True): level-0 features per point."""

    output_dim = 32

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.per_point = kwargs.get("per_point", False)
        self.local = nn.Linear(9, 32)

    def forward(self, inputs):
        voxel_features = self.local(inputs["feat"])
        result = summarize_per_point_geometry(inputs, voxel_features)
        result["voxel_features"] = voxel_features
        return result


def test_summarize_per_point_geometry_contract():
    batch = build_pointflow_batch([shared_voxel_sample(4, 2), None, point_sample(3)])
    voxel_features = torch.randn(len(batch.inputs["coord"]), 32)
    geometry = summarize_per_point_geometry(batch.inputs, voxel_features)
    n = len(batch.inputs["point_ids"])
    assert geometry["original_to_cluster"].tolist() == list(range(n))
    assert geometry["cluster_counts"].tolist() == [1] * n
    assert geometry["cluster_batch"].tolist() == [0, 0, 0, 0, 2, 2, 2]
    assert geometry["cluster_offsets"].tolist() == [4, 4, 7]
    torch.testing.assert_close(geometry["cluster_xyz"], batch.inputs["anchor_xyz"])
    torch.testing.assert_close(geometry["cluster_uv"], batch.inputs["anchor_uv"])
    assert not geometry["relative_xyz"].any() and not geometry["relative_uv"].any()
    torch.testing.assert_close(geometry["cluster_features"], voxel_features[batch.inputs["original_to_voxel"]])
    assert geometry["cluster_features"].shape == (n, 32)


def test_per_point_tokens_do_not_pool_motion():
    """The cluster-mode failure this mode removes: one point's motion must not
    leak into another point's token through index_add pooling."""
    batch = build_pointflow_batch([shared_voxel_sample(4, 2), point_sample(3)])
    voxel_features = torch.randn(len(batch.inputs["coord"]), 32)
    geometry = summarize_per_point_geometry(batch.inputs, voxel_features)
    geometry["voxel_features"] = voxel_features
    codec = PointFlowCodec(timing=batch.timing, geometry_dim=32, hidden_dim=64, content_dim=16, decoder_dim=16)
    sigma = torch.tensor([0.3, 0.7])
    displacement = torch.randn_like(batch.displacement)
    encoded = codec.encode(geometry, displacement, sigma)
    assert encoded["noisy_tokens"].shape == (8, 7, 64)
    assert encoded["anchor_tokens"].shape == (7, 64)
    perturbed = displacement.clone()
    perturbed[:, 2] += 10.0  # one point, every future step
    other = codec.encode(geometry, perturbed, sigma)
    changed = (encoded["noisy_tokens"] - other["noisy_tokens"]).abs().amax(dim=(0, 2))
    assert changed[2] > 0
    assert torch.all(changed[[0, 1, 3, 4, 5, 6]] == 0)
    hidden = torch.randn(8, 7, 64)
    prediction = codec.decode(geometry, batch.inputs, hidden, displacement, sigma)
    assert prediction.shape == (32, 7, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="full VFM forward needs a GPU attention backend")
def test_per_point_forward_matches_point_topology(monkeypatch):
    import cosmos_framework.model.generator.pointflow_branch as branch

    monkeypatch.setattr(branch, "SonataGeometryEncoder", PerPointSurrogate)
    torch.manual_seed(5)
    batch = build_pointflow_batch([point_sample(3), None, point_sample(2)])
    sequence = make_sequence(batch)
    net = make_network().eval()
    net.install_pointflow("surrogate", timing=batch.timing, content_dim=16, decoder_dim=16, token_mode="per_point")
    assert net.pointflow_branch.geometry.per_point
    noise = torch.randn_like(batch.displacement)
    sigma = torch.tensor([0.2, 0.5, 0.8])
    output = net(sequence, pointflow_displacement=noise, pointflow_sigma=sigma)
    assert output["preds_pointflow"].shape == (32, 5, 3)
    # One token per point: cluster offsets equal the point offsets.
    assert output["pointflow_cluster_offsets"].tolist() == [3, 3, 5]
    output["preds_pointflow"].square().mean().backward()
    for parameter in (
        net.pointflow_branch.geometry.local.weight,
        net.pointflow_branch.codec.motion_encoder[0].weight,
        net.pointflow_branch.codec.decoder[0].weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_token_mode_validation(monkeypatch):
    import cosmos_framework.model.generator.pointflow_branch as branch

    monkeypatch.setattr(branch, "SonataGeometryEncoder", PerPointSurrogate)
    net = make_network().eval()
    with pytest.raises(ValueError, match="token mode"):
        net.install_pointflow("surrogate", timing=PointFlowTiming(), token_mode="per_voxel")


def test_attach_point_tokens_per_point_topology():
    """Sequence layer is cluster-agnostic: per-point geometry yields 9 tokens per point."""
    from cosmos_framework.model.generator.pointflow_sequence import attach_point_tokens, point_positions

    batch = build_pointflow_batch([point_sample(3), point_sample(2)])
    sequence = make_sequence(batch)
    n = len(batch.inputs["point_ids"])
    encoded = {
        "anchor_tokens": torch.randn(n, 16),
        "noisy_tokens": torch.randn(8, n, 16),
        "cluster_offsets": torch.tensor([3, 5]),
        "cluster_batch": batch.inputs["point_batch"],
    }
    positions = point_positions(
        batch.inputs["anchor_uv"],
        batch.inputs["point_batch"],
        batch.inputs["uv_to_video"],
        torch.zeros(2),
        torch.arange(1, 9, dtype=torch.float32) * 4 / 15,
    )
    packed = attach_point_tokens(sequence, encoded, positions)
    assert packed.point.tokens.shape == (n * 9, 16)
    assert packed.point.noisy_indexes.shape == (8, n)
    # Block-major order: token (block, point) gathers back block by block.
    dummy_hidden = torch.randn(packed.sequence_length, 16)
    hidden = packed.point.hidden(dummy_hidden)
    assert hidden.shape == (8, n, 16)


def test_upper_bound_counts_points_not_voxels():
    sample = shared_voxel_sample(4, 2)
    assert pointflow_token_upper_bound(sample) == 4 * 9
