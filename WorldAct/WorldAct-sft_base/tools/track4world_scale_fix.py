"""Opt-in geometry scale restoration for Track4World; original model.py is untouched.

Record the actual divisor (s_k + epsilon), never just the last chunk's scalar.
This module fixes geometric units only. It does not reinterpret a learned flow
computed across differently normalized chunks as metric motion.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class ChunkScaleLedger:
    divisors: list[torch.Tensor] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    def normalize(self, camera_points: torch.Tensor):
        """Normalize [T,3,H,W] camera geometry and record its exact inverse."""
        if camera_points.ndim != 4 or camera_points.shape[1] != 3:
            raise ValueError("Expected camera points [T,3,H,W]")
        divisor = camera_points.norm(dim=1).mean().detach() + 1e-6
        if not torch.isfinite(divisor) or divisor <= 0:
            raise ValueError("Non-finite/invalid chunk scale")
        self.divisors.append(divisor)
        self.lengths.append(len(camera_points))
        return camera_points / divisor

    def restore(self, normalized_points: torch.Tensor):
        """Restore all concatenated chunks with their OWN recorded divisors."""
        if len(normalized_points) != sum(self.lengths):
            raise ValueError("Frame count does not match recorded chunk boundaries")
        per_frame = torch.cat([d.expand(n) for d, n in zip(self.divisors, self.lengths)])
        return normalized_points * per_frame.to(normalized_points).reshape((-1,) + (1,) * (normalized_points.ndim - 1))

    def normalize_geometry(self, camera_points, world_points, camera_poses):
        """Opt-in replacement for the geometry normalization block in forward_point.

        One ledger belongs to one inference request. Pass concatenated outputs to
        restore_geometry before exporting metric geometry. Learned flows are not
        accepted here because they may already mix different normalization units.
        """
        camera = self.normalize(camera_points)
        divisor = self.divisors[-1]
        poses = camera_poses.clone()
        poses[..., :3, 3] /= divisor
        return camera, world_points / divisor, poses

    def restore_geometry(self, camera_points, world_points, camera_poses):
        """Restore camera/world positions and translations, leaving rotations alone."""
        poses = camera_poses.clone()
        poses[..., :3, 3] = self.restore(poses[..., :3, 3])
        return self.restore(camera_points), self.restore(world_points), poses


def correct_cached_depth(depth, chunk_original_scale, last_original_scale, chunk_metric_ratio=1.0):
    """Undo legacy last-scale restoration and optionally change Metric calibration.

    cached Z = Z_k / (s_k + eps) * s_last.
    corrected Z = cached Z * (s_k + eps)/s_last * metric_ratio_k.
    Scales must come from the matching original inference. Rerun-derived scales
    make this a reconstruction with reproducibility uncertainty, not exact replay.
    """
    if chunk_original_scale <= 0 or last_original_scale <= 0 or chunk_metric_ratio <= 0:
        raise ValueError("Scales must be positive")
    return depth * ((chunk_original_scale + 1e-6) / last_original_scale) * chunk_metric_ratio
