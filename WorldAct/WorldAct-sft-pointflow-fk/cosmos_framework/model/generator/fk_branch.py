"""Learned FK branch: 21 keypoints in, 32-step motion out.

The PointFlow codec's counterpart, and deliberately its mirror image in structure
so "FK is a peer modality" is visible in the code, not just asserted:

    PointFlow   g_j =  W_F . F_PTv3_j  +  MLP_xyz(X_j / s)
    FK          g_i =  W_idx . e_idx(i) +  MLP_xyz(p_i / s)

The left term is the local identity -- a PTv3 feature there, a learned per-joint
embedding here -- and the right term is where the thing is in camera-frame metres.
Adding two separately-encoded halves rather than concatenating them keeps identity
and position in different subspaces, which is what makes the position half
ablatable on its own.

Where it is simpler than PointFlow: one token per keypoint, so no pooling on the
way in and no gather or per-point relative offset on the way out.
"""

import math
from typing import Protocol

import torch
from torch import nn

from cosmos_framework.data.fk_window import FKTiming

KEYPOINTS = 21
BASE_FPS = 24.0


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


def motion_blocks(displacement, timing: FKTiming):
    """``[H,N,3]`` -> ``[H/q,N,3q]``, preserving chronological XYZ triplets.

    The 12 numbers per block are ordered, not a bag: a token stands for "these
    four steps, in this order", and a permutation would ask the model to
    reconstruct the order from the timestamps alone.
    """
    if displacement.ndim != 3 or displacement.shape[0] != timing.steps or displacement.shape[2] != 3:
        raise ValueError(f"Expected displacement [H,N,3] with H={timing.steps}, got {tuple(displacement.shape)}")
    return (
        displacement.reshape(timing.blocks, timing.steps_per_token, displacement.shape[1], 3)
        .permute(0, 2, 1, 3)
        .flatten(2)
    )


def restore_motion(blocks, timing: FKTiming):
    """Inverse of ``motion_blocks``: ``[H/q,N,3q]`` -> ``[H,N,3]``."""
    if blocks.ndim != 3 or blocks.shape[0] != timing.blocks or blocks.shape[2] != 3 * timing.steps_per_token:
        raise ValueError(f"Expected motion blocks [H/q,N,3q], got {tuple(blocks.shape)}")
    return (
        blocks.reshape(blocks.shape[0], blocks.shape[1], timing.steps_per_token, 3)
        .permute(0, 2, 1, 3)
        .reshape(timing.steps, blocks.shape[1], 3)
    )


class FKEncoder(Protocol):
    """What the rest of the branch needs from a keypoint encoder.

    ``index`` is the anatomical joint number, not a tracker ID, so an
    implementation is allowed -- expected -- to treat it as a fixed identity.
    """

    def __call__(self, xyz: torch.Tensor, index: torch.Tensor) -> torch.Tensor: ...


class MLPIndexEncoder(nn.Module):
    """``token_i = index_projection(e_index(i)) + MLP_xyz(p_i / xyz_scale)``."""

    def __init__(self, *, hidden_dim, keypoints=KEYPOINTS, index_dim=64, content_dim=256, xyz_scale=1.0):
        super().__init__()
        if not math.isfinite(xyz_scale) or xyz_scale <= 0:
            raise ValueError("xyz_scale must be finite and positive")
        self.register_buffer("xyz_scale", torch.tensor(float(xyz_scale)))
        self.index_embedding = nn.Embedding(keypoints, index_dim)
        self.index_projection = nn.Linear(index_dim, content_dim)
        self.xyz_encoder = mlp(3, content_dim, content_dim)

    def forward(self, xyz, index):
        if xyz.shape[:-1] != index.shape:
            raise ValueError(f"xyz {tuple(xyz.shape)} and index {tuple(index.shape)} disagree")
        if xyz.shape[-1] != 3:
            raise ValueError(f"expected [..., 3] xyz, got {tuple(xyz.shape)}")
        return self.index_projection(self.index_embedding(index)) + self.xyz_encoder(
            (xyz / self.xyz_scale).to(self.index_projection.weight.dtype)
        )


class FKBranch(nn.Module):
    """Encode keypoint geometry plus noised motion; decode per-keypoint velocity."""

    def __init__(
        self,
        *,
        timing: FKTiming | None = None,
        hidden_dim=2048,
        keypoints=KEYPOINTS,
        index_dim=64,
        content_dim=256,
        decoder_dim=256,
        xyz_scale=1.0,
        include_anchor=True,
    ):
        super().__init__()
        if min(hidden_dim, index_dim, content_dim, decoder_dim) < 1:
            raise ValueError("Feature dimensions must be positive")
        self.timing = timing or FKTiming()
        self.hidden_dim, self.keypoints, self.include_anchor = hidden_dim, keypoints, include_anchor
        self.encoder = MLPIndexEncoder(
            hidden_dim=hidden_dim,
            keypoints=keypoints,
            index_dim=index_dim,
            content_dim=content_dim,
            xyz_scale=xyz_scale,
        )
        self.register_buffer("sigma_frequencies", torch.exp(-math.log(10000) * torch.arange(32) / 32))
        self.motion_encoder = mlp(3 * self.timing.steps_per_token, content_dim, content_dim)
        self.point2llm = nn.Sequential(nn.LayerNorm(2 * content_dim), nn.Linear(2 * content_dim, hidden_dim))
        self.anchor_projection = nn.Sequential(nn.LayerNorm(content_dim), nn.Linear(content_dim, hidden_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.sigma_encoder = mlp(64, content_dim, hidden_dim)
        # No local/voxel term and no relative offset: a keypoint's token is its own
        # row, so there is nothing the decoder has to be told about where inside a
        # cluster the point sits.  What remains is the backbone hidden state, the
        # noised block, and the diffusion time -- and sigma is `hidden_dim` wide,
        # not a scalar, hence the doubled hidden term.
        self.decoder = mlp(
            2 * hidden_dim + 3 * self.timing.steps_per_token, decoder_dim, 3 * self.timing.steps_per_token
        )

    def _sigma(self, sigma, batch_size):
        if sigma.shape != (batch_size,) or not torch.isfinite(sigma).all() or torch.any((sigma < 0) | (sigma > 1)):
            # Name the mismatch explicitly.  The old message said only what sigma
            # should be, so a caller who passed the wrong NUMBER of values had to
            # go read the batch to find out what the right number was.
            raise ValueError(
                f"sigma must hold one finite value in [0,1] per sample: expected shape ({batch_size},) "
                f"for this batch, got {tuple(sigma.shape)}"
            )
        phase = sigma[:, None].float() * self.sigma_frequencies[None] * 1000
        values = torch.cat((phase.cos(), phase.sin()), dim=-1)
        return self.sigma_encoder(values.to(self.sigma_encoder[0].weight.dtype))

    def _blocks(self, n_points, noisy_displacement):
        # The noised state arrives float32 on purpose: ``FKBatch.to`` keeps metric
        # geometry out of the model dtype, and ``fk_add_noise`` builds xt from
        # float32 pieces.  The parameters are bf16 under training, so without this
        # cast the first Linear raises "mat1 and mat2 must have the same dtype" --
        # on GPU only, since the CPU unit tests run float32 throughout.
        # PointFlowBranch.encode does the same cast for the same reason.
        blocks = motion_blocks(noisy_displacement.to(self.motion_encoder[0].weight.dtype), self.timing)
        if blocks.shape[1] != n_points or not torch.isfinite(blocks).all():
            raise ValueError("Noisy state must be finite and cover every keypoint")
        return blocks

    def encode(self, batch, noisy_displacement, sigma):
        xyz = batch.inputs["anchor_xyz"]
        index = batch.inputs["point_ids"]
        point_batch = batch.inputs["point_batch"]
        sigma_features = self._sigma(sigma, len(batch.inputs["has_geometry"]))[point_batch]  # [N,hidden]
        g = self.encoder(xyz, index)  # [N,content]
        blocks = self._blocks(len(xyz), noisy_displacement)  # [blocks,N,3q]
        motion = self.motion_encoder(blocks)  # [blocks,N,content]
        content = torch.cat((g.unsqueeze(0).expand(len(blocks), -1, -1), motion), dim=-1)
        noisy_tokens = self.point2llm(content) + self.modality_embedding + sigma_features.unsqueeze(0)
        anchor_tokens = self.anchor_projection(g) + self.modality_embedding if self.include_anchor else None
        return {
            "anchor_tokens": anchor_tokens,
            "noisy_tokens": noisy_tokens,
            "point_offsets": batch.inputs["point_offsets"],
            "point_batch": point_batch,
        }

    def decode(self, point_hidden, noisy_displacement, sigma, point_batch, batch_size):
        """``point_hidden [blocks,N,D]`` comes back from the Cosmos backbone."""
        blocks = self._blocks(noisy_displacement.shape[1], noisy_displacement)
        expected = (len(blocks), noisy_displacement.shape[1], self.hidden_dim)
        if point_hidden.shape != expected:
            raise ValueError(f"Expected FK point hidden {expected}, got {tuple(point_hidden.shape)}")
        sigma_features = self._sigma(sigma, batch_size)[point_batch]
        outputs = [
            self.decoder(torch.cat((point_hidden[b], blocks[b], sigma_features), dim=-1)) for b in range(len(blocks))
        ]
        return restore_motion(torch.stack(outputs), self.timing)
