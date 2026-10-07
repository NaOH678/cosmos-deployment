"""PointFlow content tokens and original-point velocity decoding.

Decode is broadcast + static local features + a per-point MLP by default; an
optional PointDecoder (point_blocks > 0) inserts per-sample transformer blocks
so points re-differentiate with task context before the velocity head."""

import math

import torch
from torch import nn

from cosmos_framework.data.pointflow_window import PointFlowTiming
from cosmos_framework.model.generator.pointflow_point_decoder import PointDecoder
from cosmos_framework.model.generator.pointflow_profiling import phase as pointflow_phase


def motion_blocks(displacement, timing):
    """[H,N,3] -> [H/q,N,3q], preserving chronological XYZ triplets."""
    if displacement.ndim != 3 or displacement.shape[0] != timing.steps or displacement.shape[2] != 3:
        raise ValueError("Expected displacement [H,N,3] for configured timing")
    return (
        displacement.reshape(timing.steps // timing.steps_per_token, timing.steps_per_token, displacement.shape[1], 3)
        .permute(0, 2, 1, 3)
        .flatten(2)
    )


def restore_motion(blocks, timing):
    if (
        blocks.ndim != 3
        or blocks.shape[0] != timing.steps // timing.steps_per_token
        or blocks.shape[2] != 3 * timing.steps_per_token
    ):
        raise ValueError("Expected motion blocks [H/q,N,3q]")
    return (
        blocks.reshape(blocks.shape[0], blocks.shape[1], timing.steps_per_token, 3)
        .permute(0, 2, 1, 3)
        .reshape(timing.steps, blocks.shape[1], 3)
    )


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class PointFlowCodec(nn.Module):
    """Encode geometry + noisy displacement; decode per-point RF velocity.

    noisy_displacement and output velocity use the caller's model-space flow scale.
    Geometry XYZ remains in meters; fixed xyz_scale normalizes its content encoding.
    sigma is one diffusion time per sample, never a physical trajectory timestamp.
    Future GT validity masks are intentionally absent from this interface.
    """

    def __init__(
        self,
        *,
        timing=None,
        geometry_dim=256,
        local_dim=32,
        skip_dims=(),
        hidden_dim=2048,
        content_dim=256,
        decoder_dim=256,
        xyz_scale=1.0,
        include_anchor=True,
        geometry_motion_fusion=False,
        point_blocks=0,
        point_block_dim=256,
        point_block_heads=4,
    ):
        super().__init__()
        if not math.isfinite(xyz_scale) or xyz_scale <= 0:
            raise ValueError("xyz_scale must be finite and positive")
        if min(geometry_dim, local_dim, hidden_dim, content_dim, decoder_dim) < 1:
            raise ValueError("Feature dimensions must be positive")
        if any(d < 1 for d in skip_dims):
            raise ValueError("skip_dims must be positive")
        self.timing = timing or PointFlowTiming()
        self.hidden_dim, self.include_anchor = hidden_dim, include_anchor
        self.local_dim = local_dim
        self.skip_dims = tuple(skip_dims)
        self.geometry_motion_fusion = geometry_motion_fusion
        self.register_buffer("xyz_scale", torch.tensor(float(xyz_scale)))
        self.register_buffer("sigma_frequencies", torch.exp(-math.log(10000) * torch.arange(32) / 32))
        self.geometry_projection = nn.Linear(geometry_dim, content_dim)
        self.xyz_encoder = mlp(3, content_dim, content_dim)
        motion_dim = 3 * self.timing.steps_per_token
        if geometry_motion_fusion:
            motion_dim += local_dim + 3
        self.motion_encoder = mlp(motion_dim, content_dim, content_dim)
        self.point2llm = nn.Sequential(nn.LayerNorm(2 * content_dim), nn.Linear(2 * content_dim, hidden_dim))
        self.anchor_projection = nn.Sequential(nn.LayerNorm(content_dim), nn.Linear(content_dim, hidden_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.sigma_encoder = mlp(64, content_dim, hidden_dim)
        fused_dim = hidden_dim * 2 + local_dim + sum(self.skip_dims) + 5 + 3 * self.timing.steps_per_token
        if point_blocks > 0:
            self.decoder = None
            self.point_decoder = PointDecoder(
                fused_dim,
                3,
                3 * self.timing.steps_per_token,
                dim=point_block_dim,
                heads=point_block_heads,
                blocks=point_blocks,
            )
        else:
            self.decoder = mlp(fused_dim, decoder_dim, 3 * self.timing.steps_per_token)
            self.point_decoder = None

    def _sigma(self, sigma, batch_size):
        if sigma.shape != (batch_size,) or not torch.isfinite(sigma).all() or torch.any((sigma < 0) | (sigma > 1)):
            raise ValueError("sigma must contain one finite value in [0,1] per sample")
        phase = sigma[:, None].float() * self.sigma_frequencies[None] * 1000
        values = torch.cat((phase.cos(), phase.sin()), dim=-1)
        return self.sigma_encoder(values.to(self.sigma_encoder[0].weight.dtype))

    def _blocks(self, geometry, noisy_displacement):
        blocks = motion_blocks(noisy_displacement, self.timing)
        if blocks.shape[1] != len(geometry["point_ids"]) or not torch.isfinite(blocks).all():
            raise ValueError("Noisy state must be finite and cover every original point")
        return blocks

    def encode(self, geometry, noisy_displacement, sigma, *, inputs=None):
        blocks = self._blocks(geometry, noisy_displacement)
        sigma_features = self._sigma(sigma, len(geometry["has_geometry"]))
        shape = geometry["cluster_features"].shape
        g = self.geometry_projection(geometry["cluster_features"]) + self.xyz_encoder(
            (geometry["cluster_xyz"] / self.xyz_scale).to(geometry["cluster_features"].dtype)
        )
        motion_inputs = blocks
        if self.geometry_motion_fusion:
            if inputs is None:
                raise ValueError("Geometry-motion fusion requires original_to_voxel inputs")
            local = geometry["voxel_features"][inputs["original_to_voxel"]]
            if local.shape != (blocks.shape[1], self.local_dim):
                raise ValueError("Geometry-motion fusion requires one level-0 feature per original point")
            relative = geometry["relative_xyz"] / self.xyz_scale
            # Joint nonlinearity precedes pooling: preserve which geometry carries
            # which noisy motion. No future GT or validity masks enter this path.
            static = torch.cat((local.to(blocks.dtype), relative.to(blocks.dtype)), dim=-1)
            motion_inputs = torch.cat((blocks, static.unsqueeze(0).expand(len(blocks), -1, -1)), dim=-1)
        point_motion = self.motion_encoder(motion_inputs)
        pooled = point_motion.new_zeros((len(blocks), shape[0], point_motion.shape[-1]))
        pooled = pooled.index_add(1, geometry["original_to_cluster"], point_motion)
        pooled = pooled / geometry["cluster_counts"].clamp_min(1)[None, :, None]
        content = torch.cat((g.unsqueeze(0).expand(len(blocks), -1, -1), pooled), dim=-1)
        noisy_tokens = (
            self.point2llm(content) + self.modality_embedding + sigma_features[geometry["cluster_batch"]][None]
        )
        anchor_tokens = self.anchor_projection(g) + self.modality_embedding if self.include_anchor else None
        return {
            "anchor_tokens": anchor_tokens,
            "noisy_tokens": noisy_tokens,
            "cluster_offsets": geometry["cluster_offsets"],
            "cluster_batch": geometry["cluster_batch"],
        }

    def decode(self, geometry, inputs, point_hidden, noisy_displacement, sigma):
        """point_hidden [H/q,K,D] is supplied by a future Cosmos integration."""
        with pointflow_phase("pf_dec_inputs"):
            blocks = self._blocks(geometry, noisy_displacement)
            expected = (len(blocks), len(geometry["cluster_features"]), self.hidden_dim)
            if point_hidden.shape != expected:
                raise ValueError(f"Expected point_hidden shape {expected}")
            sigma_features = self._sigma(sigma, len(geometry["has_geometry"]))[geometry["point_batch"]]
            local = geometry["voxel_features"][inputs["original_to_voxel"]]
            if self.skip_dims:
                skips = geometry.get("voxel_skip_features")
                if skips is None or len(skips) != len(self.skip_dims):
                    raise ValueError("Geometry skip features do not match codec skip_dims")
                local = torch.cat([local] + [s[inputs["original_to_voxel"]] for s in skips], dim=-1)
            expected_local = self.local_dim + sum(self.skip_dims)
            if local.shape != (len(geometry["point_ids"]), expected_local):
                raise ValueError("Local point features have an unexpected shape")
            image_size = inputs["image_size_wh"][geometry["point_batch"]]
            if torch.any(image_size <= 0):
                raise ValueError("image_size_wh must be positive")
            relative = torch.cat(
                (geometry["relative_xyz"] / self.xyz_scale, geometry["relative_uv"] / image_size), dim=-1
            ).to(local.dtype)
            anchor = None
            point_offsets = None
            if self.point_decoder is not None:
                anchor = (inputs["anchor_xyz"] / self.xyz_scale).to(local.dtype)
                point_offsets = geometry["point_offsets"].tolist()
        # One serial pass per motion block: len(blocks) = timing.steps/ steps_per_token.
        with pointflow_phase("pf_dec_blocks"):
            outputs = []
            for block in range(len(blocks)):
                hidden = point_hidden[block, geometry["original_to_cluster"]]
                fused = torch.cat((hidden, local, relative, blocks[block], sigma_features), dim=-1)
                if self.point_decoder is not None:
                    outputs.append(self.point_decoder(fused, anchor, point_offsets))
                else:
                    outputs.append(self.decoder(fused))
        return restore_motion(torch.stack(outputs), self.timing)

    def forward(self, geometry, noisy_displacement, sigma, *, inputs=None):
        return self.encode(geometry, noisy_displacement, sigma, inputs=inputs)
