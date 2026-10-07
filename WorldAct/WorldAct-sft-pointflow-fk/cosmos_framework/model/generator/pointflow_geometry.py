"""Sonata geometry boundary for packed PointFlow inputs (no future labels)."""

import torch
from torch import nn


def voxel_level_features(levels, k):
    """Per-voxel features of hierarchy level k (0 = the finest voxel resolution)."""
    level = levels[k]
    mapping = torch.arange(len(level.feat), device=level.feat.device)
    while "pooling_parent" in level:
        mapping = mapping[level.pooling_inverse]
        level = level.pooling_parent
    return levels[k].feat[mapping]


def summarize_geometry(inputs, features, voxel_to_cluster, cluster_batch):
    """Aggregate original members, not unweighted intermediate voxel centroids."""
    mapping = voxel_to_cluster[inputs["original_to_voxel"]]
    count = features.shape[0]
    if not torch.equal(cluster_batch[mapping], inputs["point_batch"]):
        raise ValueError("Cluster mapping crosses sample boundaries")
    members = torch.bincount(mapping, minlength=count)
    if count and torch.any(members == 0):
        raise ValueError("Output cluster has no original members")
    centers = {}
    for key in ("xyz", "uv"):
        source = inputs[f"anchor_{key}"]
        sums = source.new_zeros((count, source.shape[1])).index_add(0, mapping, source)
        centers[key] = sums / members.clamp_min(1).unsqueeze(1)
    batch_size = len(inputs["has_geometry"])
    return {
        "cluster_features": features,
        "original_to_cluster": mapping,
        "voxel_to_cluster": voxel_to_cluster,
        "cluster_batch": cluster_batch,
        "cluster_offsets": torch.bincount(cluster_batch, minlength=batch_size).cumsum(0),
        "cluster_counts": members,
        "cluster_xyz": centers["xyz"],
        "cluster_uv": centers["uv"],
        "relative_xyz": inputs["anchor_xyz"] - centers["xyz"][mapping],
        "relative_uv": inputs["anchor_uv"] - centers["uv"][mapping],
        "point_ids": inputs["point_ids"],
        "point_batch": inputs["point_batch"],
        "point_offsets": inputs["point_offsets"],
        "has_geometry": inputs["has_geometry"],
    }


def summarize_per_point_geometry(inputs, voxel_features):
    """One cluster per original point: the degenerate clustering for per-point tokens.

    ``voxel_features`` are the level-0 Sonata features; indexing them by
    ``original_to_voxel`` gives every original point its own token feature while
    the codec's cluster-agnostic pooling (``index_add`` with the identity mapping,
    ``counts == 1``) passes each point's own noisy motion through untouched.
    Relative offsets vanish because each point is its own cluster centre.
    """
    point_count = len(inputs["point_ids"])
    mapping = torch.arange(point_count, device=inputs["point_ids"].device)
    batch_size = len(inputs["has_geometry"])
    return {
        "cluster_features": voxel_features[inputs["original_to_voxel"]],
        "original_to_cluster": mapping,
        "voxel_to_cluster": inputs["voxel_representatives"],  # voxel -> its representative point's token
        "cluster_batch": inputs["point_batch"],
        "cluster_offsets": torch.bincount(inputs["point_batch"], minlength=batch_size).cumsum(0),
        "cluster_counts": inputs["point_ids"].new_ones(point_count),
        "cluster_xyz": inputs["anchor_xyz"],
        "cluster_uv": inputs["anchor_uv"],
        "relative_xyz": torch.zeros_like(inputs["anchor_xyz"]),
        "relative_uv": torch.zeros_like(inputs["anchor_uv"]),
        "point_ids": inputs["point_ids"],
        "point_batch": inputs["point_batch"],
        "point_offsets": inputs["point_offsets"],
        "has_geometry": inputs["has_geometry"],
    }


class SonataGeometryEncoder(nn.Module):
    """Use enc3 by default; preserve original sample IDs when excluding empty inputs.

    Construction loads a local checkpoint strictly on CPU. Move this module and
    batch['inputs'] to the same device before forward. Source features remain
    trainable unless freeze=True. GPU sparse kernels are required for nonempty data.

    With ``per_point=True`` the encoder emits one cluster per original point,
    using level-0 features, so each point becomes its own sequence token.
    """

    channels = (32, 64, 128, 256, 512)

    def __init__(self, checkpoint, *, stage=3, freeze=False, per_point=False, skip_levels=()):
        super().__init__()
        if stage not in range(5):
            raise ValueError("stage must be in 0..4")
        if any(k not in range(1, 5) for k in skip_levels):
            raise ValueError("skip_levels must be in 1..4")
        self.skip_levels = tuple(skip_levels)
        from cosmos_framework.auxiliary.sonata.model import PointTransformerV3

        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = data["config"]
        if not config.get("enc_mode") or tuple(config["enc_channels"]) != self.channels:
            raise ValueError("Expected Sonata small encoder configuration")
        self.backbone = PointTransformerV3(**config)
        self.backbone.load_state_dict(data["state_dict"], strict=True)
        self.stage, self.frozen = stage, freeze
        self.per_point = per_point
        self.output_dim = self.channels[0] if per_point else self.channels[stage]
        self.backbone.requires_grad_(not freeze)
        if freeze:
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, inputs):
        active = torch.nonzero(inputs["has_geometry"], as_tuple=True)[0]
        if len(active) == 0:
            if len(inputs["feat"]) or len(inputs["point_ids"]):
                raise ValueError("Empty geometry flags disagree with point data")
            features = inputs["feat"].new_empty((0, self.output_dim))
            # Preserve a zero gradient path for a caller's empty-batch loss.
            zero = sum((p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad), features.sum())
            features = features + zero
            empty = inputs["point_ids"].new_empty((0,))
            if self.per_point:
                result = summarize_per_point_geometry(inputs, inputs["feat"].new_empty((0, self.channels[0])) + zero)
            else:
                result = summarize_geometry(inputs, features, empty, empty)
            result["voxel_features"] = inputs["feat"].new_empty((0, self.channels[0])) + zero
            if self.skip_levels:
                result["voxel_skip_features"] = [
                    inputs["feat"].new_empty((0, self.channels[k])) + zero for k in self.skip_levels
                ]
            return result
        if inputs["feat"].device != next(self.parameters()).device:
            raise ValueError("Move encoder and packed inputs to the same device")
        counts = torch.diff(inputs["voxel_offsets"], prepend=inputs["voxel_offsets"].new_zeros(1))
        if not torch.equal(counts > 0, inputs["has_geometry"]):
            raise ValueError("Voxel offsets disagree with geometry flags")
        data = {key: inputs[key] for key in ("coord", "feat", "grid_coord")}
        data["offset"] = counts[active].cumsum(0)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            encoded = self.backbone(data)
        levels = [encoded]
        while "pooling_parent" in levels[-1]:
            levels.append(levels[-1].pooling_parent)
        levels.reverse()
        if len(levels) != len(self.channels):
            raise ValueError("Expected traceable five-stage encoder hierarchy")
        if self.per_point:
            result = summarize_per_point_geometry(inputs, levels[0].feat)
            result["voxel_features"] = levels[0].feat
            if self.skip_levels:
                result["voxel_skip_features"] = [voxel_level_features(levels, k) for k in self.skip_levels]
            return result
        selected = levels[self.stage]
        mapping = torch.arange(len(selected.feat), device=selected.feat.device)
        level = selected
        while "pooling_parent" in level:
            mapping = mapping[level.pooling_inverse]
            level = level.pooling_parent
        if len(mapping) != len(inputs["feat"]):
            raise ValueError("Incomplete voxel-to-cluster mapping")
        result = summarize_geometry(inputs, selected.feat, mapping, active[selected.batch.long()])
        result["voxel_features"] = levels[0].feat
        if self.skip_levels:
            result["voxel_skip_features"] = [voxel_level_features(levels, k) for k in self.skip_levels]
        return result
