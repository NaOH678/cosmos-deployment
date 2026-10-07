"""Validate and summarize one Tianji/Wuji LMDB episode."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .episode import load_episode
from .replay_core import split_trajectory, validate_trajectory
from .schema import RobotLayout


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect an LMDB episode")
    parser.add_argument("episode_dir")
    args = parser.parse_args(argv)
    arrays, metadata = load_episode(args.episode_dir)
    layout = RobotLayout.from_metadata(metadata)
    trajectories = split_trajectory(arrays["action"], layout, arrays.get("zsp"))
    validate_trajectory(trajectories)
    summary = {
        "episode": args.episode_dir,
        "steps": int(arrays["action"].shape[0]),
        "frame_rate": metadata.get("frame_rate"),
        "action_dim": int(arrays["action"].shape[1]),
        "qpos_dim": int(arrays["qpos"].shape[1]) if "qpos" in arrays else None,
        "eef_dim": int(arrays["eef"].shape[1]) if "eef" in arrays else None,
        "sides": list(layout.sides),
        "cameras": metadata.get("camera_names", []),
        "videos": metadata.get("videos", {}),
    }
    auxiliary_metadata = metadata.get("auxiliary_camera")
    if isinstance(auxiliary_metadata, dict):
        depth_streams = auxiliary_metadata.get("depth_streams", {}) or {}
        infrared_videos = (
            auxiliary_metadata.get("infrared_videos", {}) or {}
        )
        summary["auxiliary_camera"] = {
            "best_effort": auxiliary_metadata.get("best_effort", True),
            "depth_saved_frames": {
                name: int(info.get("saved_frames", 0))
                for name, info in depth_streams.items()
            },
            "infrared_saved_frames": {
                name: int(info.get("saved_frames", 0))
                for name, info in infrared_videos.items()
            },
            "writer": auxiliary_metadata.get("writer", {}),
            "initialization_error": auxiliary_metadata.get(
                "initialization_error"
            ),
            "finalize_error": auxiliary_metadata.get("finalize_error"),
        }
    diagnostic_metadata = metadata.get("teleop_diagnostics", {})
    if diagnostic_metadata.get("datasets"):
        tracker_available = arrays.get("teleop_tracker_available")
        tracker_valid = arrays.get("teleop_tracker_valid")
        tracker_error = arrays.get("teleop_tracker_alignment_error_s")
        summary["teleop_diagnostics"] = {
            "enabled": diagnostic_metadata.get("enabled", True),
            "non_blocking": diagnostic_metadata.get("non_blocking", True),
            "tracker_roles": diagnostic_metadata.get("tracker_roles", []),
            "tracker_available_rate": (
                float(np.mean(tracker_available))
                if tracker_available is not None and tracker_available.size
                else 0.0
            ),
            "tracker_valid_rate": (
                float(np.mean(tracker_valid))
                if tracker_valid is not None and tracker_valid.size
                else 0.0
            ),
            "tracker_alignment_error_mean_s": (
                float(np.mean(tracker_error[tracker_available.astype(bool)]))
                if tracker_error is not None
                and tracker_available is not None
                and np.any(tracker_available)
                else None
            ),
            "manus": {},
        }
        for side in layout.sides:
            available = arrays.get(f"teleop_manus_{side}_available")
            keypoints_valid = arrays.get(
                f"teleop_manus_{side}_keypoints_valid"
            )
            node_valid = arrays.get(f"teleop_manus_{side}_node_valid")
            alignment_error = arrays.get(
                f"teleop_manus_{side}_alignment_error_s"
            )
            available_mask = (
                available.astype(bool)
                if available is not None
                else np.zeros((arrays["action"].shape[0], 1), dtype=bool)
            )
            summary["teleop_diagnostics"]["manus"][side] = {
                "available_rate": (
                    float(np.mean(available))
                    if available is not None and available.size
                    else 0.0
                ),
                "keypoints_valid_rate": (
                    float(np.mean(keypoints_valid))
                    if keypoints_valid is not None and keypoints_valid.size
                    else 0.0
                ),
                "raw_node_valid_rate": (
                    float(np.mean(node_valid))
                    if node_valid is not None and node_valid.size
                    else 0.0
                ),
                "alignment_error_mean_s": (
                    float(np.mean(alignment_error[available_mask]))
                    if alignment_error is not None and np.any(available_mask)
                    else None
                ),
            }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
