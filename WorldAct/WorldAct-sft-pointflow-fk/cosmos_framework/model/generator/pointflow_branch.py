"""Learned PointFlow branch owned by the Cosmos generation network."""

import os

import torch
from torch import nn

from cosmos_framework.model.generator.pointflow_codec import PointFlowCodec
from cosmos_framework.model.generator.pointflow_geometry import SonataGeometryEncoder
from cosmos_framework.model.generator.pointflow_profiling import phase as pointflow_phase
from cosmos_framework.model.generator.pointflow_sequence import attach_point_tokens, point_positions

# Share of point tokens that must land inside the video token grid before a
# mismatch is treated as a data-plumbing bug rather than an edge effect.
POINT_GRID_MIN_INSIDE = 0.75
# Below this many clusters the statistic is too noisy to act on; the CPU
# integration tests build 5-cluster batches.
POINT_GRID_MIN_TOKENS = 256


def check_points_inside_video_grid(positions, cluster_batch, grids):
    """Fail when the point tokens and the video tokens disagree about the image.

    A point token's ``(h, w)`` feeds the same mRoPE axes as the video tokens, so it
    only means something inside the video token grid.  A data path that hands the
    dataset a different canvas than the manifest affine was authored for -- a video
    cache built at a transposed resolution, say -- leaves most points off the grid,
    where the position is a phase no real patch occupies.  Nothing else notices:
    the loss never reads it, and the visualisations re-project from the 3D labels
    rather than from these coordinates.

    Measured on the sandwich data: 0.969 inside when the canvases agree, 0.158 when
    they do not, so the threshold sits well clear of both.  Small point sets are
    skipped because the statistic is meaningless there (the CPU integration tests
    build 5-cluster batches).
    """
    tokens = int(cluster_batch.numel())
    if tokens < POINT_GRID_MIN_TOKENS:
        return None
    grid = torch.as_tensor(grids, device=positions.device, dtype=positions.dtype)
    bounds = grid[cluster_batch]  # [K,2] as (width, height)
    height, width = positions[..., 1], positions[..., 2]
    inside = (height >= 0) & (height < bounds[:, 1][None]) & (width >= 0) & (width < bounds[:, 0][None])
    fraction = float(inside.float().mean())
    if fraction < POINT_GRID_MIN_INSIDE:
        raise ValueError(
            f"only {fraction:.1%} of {tokens} point tokens fall inside the video token grid "
            f"{[tuple(int(v) for v in g) for g in grids]} (need {POINT_GRID_MIN_INSIDE:.0%}); the "
            f"PointFlow uv_to_video canvas and the video the model receives were built for different "
            f"geometry"
        )
    return fraction


def video_aligned_point_positions(sequence, inputs, geometry, timing, pixel_stride, *, check_grid=True):
    """Convert actual tracker->video affine to patch centers and reuse video origins."""
    if pixel_stride <= 0 or sequence.vision is None:
        raise ValueError("PointFlow requires a video grid and positive pixel stride")
    device = geometry["cluster_uv"].device
    affine = inputs["uv_to_video"].to(device=device, dtype=torch.float32).clone() / pixel_stride
    affine[:, :, 2] += 0.5 / pixel_stride - 0.5
    temporal, spatial, grids = [], [], []
    start = 0
    for length in sequence.sample_lens:
        payloads = {
            span.payload_index for span in sequence.vision.spans if start <= span.sequence_start < start + length
        }
        if len(payloads) != 1:
            raise ValueError("PointFlow requires exactly one composed video item per sample")
        # Vision token_shapes are (latent_t, patch_h, patch_w) for this payload, so
        # the token grid comes from metadata and needs no device sync.
        token_shape = sequence.vision.token_shapes[next(iter(payloads))]
        grids.append((token_shape[2], token_shape[1]))
        ids = sequence.vision.sequence_indexes
        ids = ids[(ids >= start) & (ids < start + length)]
        if not len(ids):
            raise ValueError("Each PointFlow batch slot requires a video item")
        pos = sequence.position_ids[:, ids]
        times = torch.unique(pos[0], sorted=True)
        expected = timing.steps // timing.steps_per_token + 1
        if len(times) != expected:
            raise ValueError("Video latent timeline does not match PointFlow blocks")
        target = times[0] + torch.arange(expected, device=times.device) * timing.steps_per_token / timing.fps * 6
        if not torch.allclose(times.float(), target.float(), atol=1e-4, rtol=0):
            raise ValueError("Video mRoPE must use Cosmos physical FPS modulation")
        temporal.append(pos[0, 0])
        spatial.append(pos[[2, 1], 0])
        start += length
    positions = point_positions(
        geometry["cluster_uv"],
        geometry["cluster_batch"],
        affine,
        torch.stack(temporal),
        torch.arange(1, timing.steps // timing.steps_per_token + 1, device=device)
        * timing.steps_per_token
        / timing.fps,
        spatial_origin=torch.stack(spatial),
    )
    if check_grid:
        geometry["point_grid_inside_fraction"] = check_points_inside_video_grid(
            positions, geometry["cluster_batch"], grids
        )
    return positions


class PointFlowBranch(nn.Module):
    def __init__(
        self,
        checkpoint,
        *,
        timing,
        hidden_dim,
        stage=3,
        freeze_geometry=False,
        content_dim=256,
        decoder_dim=256,
        token_mode="cluster",
    ):
        super().__init__()
        if token_mode not in ("cluster", "per_point"):
            raise ValueError(f"Unknown PointFlow token mode: {token_mode}")
        # Optional decode-side upgrades, both off by default (POINTFLOW_DECODE_*):
        # multi-resolution Sonata skip features and per-point transformer blocks.
        skip_levels = tuple(int(v) for v in os.environ.get("POINTFLOW_DECODE_SKIP_LEVELS", "").split(",") if v.strip())
        point_blocks = int(os.environ.get("POINTFLOW_DECODE_POINT_BLOCKS", "0"))
        point_block_dim = int(os.environ.get("POINTFLOW_DECODE_POINT_DIM", "256"))
        point_block_heads = int(os.environ.get("POINTFLOW_DECODE_POINT_HEADS", "4"))
        self.geometry = SonataGeometryEncoder(
            checkpoint,
            stage=stage,
            freeze=freeze_geometry,
            per_point=token_mode == "per_point",
            skip_levels=skip_levels,
        )
        self.codec = PointFlowCodec(
            timing=timing,
            geometry_dim=self.geometry.output_dim,
            hidden_dim=hidden_dim,
            content_dim=content_dim,
            decoder_dim=decoder_dim,
            skip_dims=tuple(SonataGeometryEncoder.channels[k] for k in skip_levels),
            point_blocks=point_blocks,
            point_block_dim=point_block_dim,
            point_block_heads=point_block_heads,
            geometry_motion_fusion=os.environ.get("POINTFLOW_GEOMETRY_MOTION_FUSION", "false").lower() == "true",
        )
        self.timing = timing

    def encode(self, sequence, noisy_displacement, sigma, *, pixel_stride):
        data = sequence.pointflow_data
        if data.timing != self.timing:
            raise ValueError("Model and PointFlow batch timing differ")
        if sequence.point is not None:
            raise ValueError("Point tokens are already attached")
        # No future labels/masks enter either learned encoder or positional encoding.
        with pointflow_phase("pf_sonata"):
            geometry = self.geometry(data.inputs)
        dtype = next(self.codec.parameters()).dtype
        geometry = {
            key: value.to(dtype) if key in ("cluster_features", "voxel_features") else value
            for key, value in geometry.items()
        }
        if "voxel_skip_features" in geometry:
            geometry["voxel_skip_features"] = [value.to(dtype) for value in geometry["voxel_skip_features"]]
        noisy = noisy_displacement.to(dtype)
        with pointflow_phase("pf_codec_encode"):
            encoded = self.codec(geometry, noisy, sigma, inputs=data.inputs)
        with pointflow_phase("pf_positions"):
            positions = video_aligned_point_positions(sequence, data.inputs, geometry, self.timing, pixel_stride)
        with pointflow_phase("pf_attach_tokens"):
            packed = attach_point_tokens(sequence, encoded, positions)
        return packed, geometry, noisy

    def decode(self, geometry, inputs, hidden, noisy_displacement, sigma):
        with pointflow_phase("pf_codec_decode"):
            return self.codec.decode(
                geometry, inputs, hidden.to(next(self.codec.parameters()).dtype), noisy_displacement, sigma
            )
