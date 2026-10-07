"""Task 7 CPU integration: actual VFM forward/heads/attention, sparse encoder surrogate."""

import pytest
import torch
from torch import nn

from cosmos_framework.data.pointflow_batch import build_pointflow_batch
from cosmos_framework.data.pointflow_batch_test import point_sample
from cosmos_framework.model.generator.pointflow_geometry import summarize_geometry
from cosmos_framework.scripts.validate_pointflow_network import make_network, make_sequence


class GeometrySurrogate(nn.Module):
    output_dim = 8

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.features = nn.Linear(9, 8)
        self.local = nn.Linear(9, 32)

    def forward(self, inputs):
        features = self.features(inputs["feat"])
        result = summarize_geometry(inputs, features, torch.arange(len(features)), inputs["voxel_batch"])
        result["voxel_features"] = self.local(inputs["feat"])
        return result


def setup(monkeypatch):
    import cosmos_framework.model.generator.pointflow_branch as branch

    monkeypatch.setattr(branch, "SonataGeometryEncoder", GeometrySurrogate)
    torch.manual_seed(3)
    first, second = point_sample(3), point_sample(2)
    for sample in (first, second):
        sample["inputs"]["feat"][:] = 0.3
        sample["inputs"]["anchor_xyz"][:] = 0.2
    batch = build_pointflow_batch([first, None, second])
    sequence = make_sequence(batch)
    net = make_network().eval()
    base = net.vae2llm.weight.clone()
    net.install_pointflow("surrogate", timing=batch.timing, content_dim=16, decoder_dim=16)
    torch.testing.assert_close(base, net.vae2llm.weight)
    return net, sequence, batch


@pytest.mark.parametrize("fusion", ["false", "true"])
def test_vfm_forward_backward_and_future_label_isolation(monkeypatch, fusion):
    monkeypatch.setenv("POINTFLOW_GEOMETRY_MOTION_FUSION", fusion)
    net, sequence, batch = setup(monkeypatch)
    noise = torch.randn_like(batch.displacement, requires_grad=True)
    sigma = torch.tensor([0.2, 0.5, 0.8])
    output = net(sequence, pointflow_displacement=noise, pointflow_sigma=sigma)
    assert output["preds_pointflow"].shape == (32, 5, 3)
    assert output["pointflow_cluster_offsets"].tolist() == [3, 3, 5]
    assert len(output["preds_vision"]) == len(output["preds_action"]) == 3
    assert sequence.point is None  # original indices are not mutated
    output["preds_pointflow"].square().mean().backward()
    for parameter in [
        noise,
        net.pointflow_branch.geometry.features.weight,
        net.pointflow_branch.codec.motion_encoder[0].weight,
        net.language_model.layer.q_proj_moe_gen.weight,
        net.vae2llm.weight,
        net.action2llm.fc.weight,
    ]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    batch.valid[:] = False
    batch.displacement[:] = 999
    other = net(sequence, pointflow_displacement=noise, pointflow_sigma=sigma)
    torch.testing.assert_close(output["preds_pointflow"], other["preds_pointflow"])


def test_guards(monkeypatch):
    net, sequence, batch = setup(monkeypatch)
    with pytest.raises(ValueError, match="explicit noisy"):
        net(sequence)
    with pytest.raises(ValueError, match="already installed"):
        net.install_pointflow("unused", timing=batch.timing)
    with pytest.raises(ValueError, match="temporal causal"):
        net(sequence, video_temporal_causal=True)
    sequence.position_ids[0, sequence.vision.sequence_indexes] = 0
    with pytest.raises(ValueError, match="timeline"):
        net(sequence, pointflow_displacement=torch.zeros_like(batch.displacement), pointflow_sigma=torch.ones(3) * 0.5)


def test_points_outside_the_video_grid_are_rejected():
    """A canvas mismatch moves the point tokens off the video grid silently.

    Nothing downstream reads their (h, w): the loss never touches it and the
    overlays re-project the 3D labels, so without this check a run trains for
    thousands of steps on positions that no video token occupies.
    """
    from cosmos_framework.model.generator.pointflow_branch import check_points_inside_video_grid

    grid = [(17, 23)]
    batch = torch.zeros(400, dtype=torch.long)
    inside = torch.zeros(9, 400, 3)
    inside[..., 1] = 11.0  # height, within the 23-row grid
    inside[..., 2] = 8.0  # width, within the 17-column grid
    assert check_points_inside_video_grid(inside, batch, grid) == 1.0

    outside = inside.clone()
    outside[..., 1] = 25.8  # the measured y for a transposed canvas
    with pytest.raises(ValueError, match="inside the video token grid"):
        check_points_inside_video_grid(outside, batch, grid)

    # Too few tokens to act on: the statistic is meaningless at integration-test size.
    assert check_points_inside_video_grid(outside[:, :5], batch[:5], grid) is None


def test_pixel_center_affine_matches_video_origin(monkeypatch):
    from cosmos_framework.model.generator.pointflow_branch import video_aligned_point_positions

    net, sequence, batch = setup(monkeypatch)
    # This case pins the tracker->patch affine, so fix the anchor UV its comment
    # assumes rather than inheriting the shared fixture's per-field fill value.
    batch.inputs["anchor_uv"][:] = 0
    geometry = net.pointflow_branch.geometry(batch.inputs)
    batch.inputs["uv_to_video"][:] = torch.tensor([[2.0, 0, 0.5], [0, 2.0, 16.5]])
    positions = video_aligned_point_positions(sequence, batch.inputs, geometry, batch.timing, 8)
    # anchor UV=(0,0) -> video=(.5,16.5) -> patch=(-.375,1.625)
    torch.testing.assert_close(positions[0, 0, 1:], torch.tensor([1.625, -0.375]))
    torch.testing.assert_close(positions[1:, 0, 0] - positions[0, 0, 0], torch.arange(1, 9) * 1.6)


def test_empty_geometry_batch(monkeypatch):
    net, _, _ = setup(monkeypatch)
    batch = build_pointflow_batch([point_sample(0), None])
    sequence = make_sequence(batch)
    output = net(sequence, pointflow_displacement=torch.empty(32, 0, 3), pointflow_sigma=torch.ones(2) * 0.5)
    assert output["preds_pointflow"].shape == (32, 0, 3)
    assert len(output["preds_vision"]) == 2
    assert output["pointflow_cluster_offsets"].tolist() == [0, 0]
