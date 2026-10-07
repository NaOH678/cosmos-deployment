"""Dump one PointFlow selection as ``window.npz`` + ``encoding_enc0.npz``, on CPU.

``validate_pointflow_sonata.py`` writes the same two files, but only after a real
Sonata forward, which needs CUDA.  That is overkill when all we want is to *watch*
a selection: ``visualize_pointflow_motion.py`` reads the encoding only for
``point_ids``, ``original_to_cluster``, ``cluster_counts`` and ``cluster_uv`` --
never ``cluster_features``.

At ``stage=0`` the cluster assignment is exact without any forward pass.  The
encoder's level 0 is its own input, so ``SonataGeometryEncoder`` maps a point to
``original_to_voxel`` unchanged (``levels[0]`` carries no ``pooling_parent``, so
the mapping walk in ``pointflow_geometry.summarize_geometry`` never executes).
That identity is what makes a CPU dump honest at stage 0 and only at stage 0.

Cluster fields are written so the motion renderer can consume the directory; the
``cluster_features`` array is deliberately absent, because nothing computed it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cosmos_framework.data.pointflow_window import PointFlowTiming, prepare_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select-motion-fraction", type=float, default=0.0)
    parser.add_argument("--select-top-n", type=int, default=0)
    parser.add_argument("--min-voxel-members", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--steps-per-token", type=int, default=4)
    arguments = parser.parse_args()

    timing = PointFlowTiming(fps=arguments.fps, steps=arguments.steps, steps_per_token=arguments.steps_per_token)
    window = prepare_window(
        arguments.episode,
        arguments.start_frame,
        arguments.max_points,
        arguments.voxel_size,
        arguments.seed,
        timing=timing,
        allow_empty=True,
        select_motion_fraction=arguments.select_motion_fraction,
        select_top_n=arguments.select_top_n,
        min_voxel_members=arguments.min_voxel_members,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output / "window.npz", **{k: v for k, v in window.items() if k != "anchor_rgb"})

    # stage-0 identity: a cluster is a 2 cm voxel, and the encoder's level 0 keeps
    # one row per voxel, so the point -> cluster map is the point -> voxel map.
    mapping = np.asarray(window["original_to_voxel"])
    counts = np.bincount(mapping, minlength=int(mapping.max()) + 1 if len(mapping) else 0)
    if len(mapping) and not (counts > 0).all():
        raise ValueError("A voxel has no original members; stage-0 identity does not hold")
    centers = {}
    for key, source in (("cluster_xyz", window["anchor_xyz"]), ("cluster_uv", window["anchor_uv"])):
        sums = np.zeros((len(counts), source.shape[1]), dtype=np.float64)
        np.add.at(sums, mapping, source)
        centers[key] = sums / np.maximum(counts, 1)[:, None]
    np.savez_compressed(
        arguments.output / "encoding_enc0.npz",
        point_ids=window["point_ids"],
        original_to_cluster=mapping.astype(np.int64),
        cluster_counts=counts,
        cluster_xyz=centers["cluster_xyz"],
        cluster_uv=centers["cluster_uv"],
        note=np.asarray("synthetic stage-0 clustering (== voxelization); no Sonata forward"),
    )

    record = {
        "episode": arguments.episode.name,
        "start_frame": arguments.start_frame,
        "voxel_size": arguments.voxel_size,
        "max_points": arguments.max_points,
        "select_motion_fraction": arguments.select_motion_fraction,
        "select_top_n": arguments.select_top_n,
        "min_voxel_members": arguments.min_voxel_members,
        "original_points": int(len(window["point_ids"])),
        "stage0_clusters": int(len(counts)),
        "points_per_cluster": float(len(mapping) / max(len(counts), 1)),
    }
    (arguments.output / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
