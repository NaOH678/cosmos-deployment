#!/usr/bin/env python3
"""Export canonical Wuji 21-point FK keypoints in the Tianji Link_Base frame.

The output semantic order matches HaWoR/OpenPose/MediaPipe:
  0 wrist/palm,
  1:5 thumb, 5:9 index, 9:13 middle, 13:17 ring, 17:21 pinky.

For each Wuji finger, joint2, joint3, joint4 and tip are used.  Joint1 and
joint2 form the robot's compound finger-base joint; joint2 is the canonical
MCP/CMC representative.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
REPLAY_SCRIPT = HERE.with_name("replay_teleop.py")
DEFAULT_EPISODE = PROJECT_ROOT / "tianji_wuji_data"
DEFAULT_URDF = (
    PROJECT_ROOT
    / "marvin_wuji_d435_description"
    / "urdf"
    / "marvin_wuji_d435_complete.urdf"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "generated" / "tianji_wuji_fk21.npz"

SIDES = ("left", "right")
KEYPOINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def _load_replay_module():
    spec = importlib.util.spec_from_file_location("replay_teleop", REPLAY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REPLAY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"model has no body named {name!r}")
    return body_id


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"model has no joint named {name!r}")
    return joint_id


def fk21_sources(model: mujoco.MjModel, side: str) -> list[tuple[str, int]]:
    """Return (MuJoCo object kind, id) for one hand's canonical 21 points."""
    if side not in SIDES:
        raise ValueError(f"unsupported side: {side}")
    sources: list[tuple[str, int]] = [
        ("body", _body_id(model, f"{side}_hand_palm_link"))
    ]
    for finger in range(1, 6):
        for joint in (2, 3, 4):
            sources.append(
                (
                    "joint",
                    _joint_id(
                        model, f"{side}_hand_finger{finger}_joint{joint}"
                    ),
                )
            )
        sources.append(
            (
                "body",
                _body_id(model, f"{side}_hand_finger{finger}_tip_link"),
            )
        )
    if len(sources) != 21:
        raise AssertionError(f"expected 21 sources, got {len(sources)}")
    return sources


def positions_in_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_body_id: int,
    sources: list[tuple[str, int]],
) -> np.ndarray:
    """Read current FK points and express them in Link_Base coordinates."""
    base_position_world = data.xpos[base_body_id]
    rotation_world_base = data.xmat[base_body_id].reshape(3, 3)
    rotation_base_world = rotation_world_base.T
    result = np.empty((21, 3), dtype=np.float64)
    for index, (kind, object_id) in enumerate(sources):
        if kind == "joint":
            position_world = data.xanchor[object_id]
        else:
            position_world = data.xpos[object_id]
        result[index] = rotation_base_world @ (
            position_world - base_position_world
        )
    return result


def anchor_timestamps(episode_dir: Path, num_frames: int) -> np.ndarray:
    path = episode_dir / "sync_timestamps.json"
    if not path.is_file():
        return np.arange(num_frames, dtype=np.float64)
    records = json.loads(path.read_text())
    if len(records) != num_frames:
        raise ValueError(
            f"sync timestamp count {len(records)} != frame count {num_frames}"
        )
    return np.asarray(
        [float(record.get("anchor", index)) for index, record in enumerate(records)],
        dtype=np.float64,
    )


def export(args: argparse.Namespace) -> dict:
    replay = _load_replay_module()
    episode_dir = args.episode_dir.expanduser().resolve()
    trajectory, metadata = replay.load_episode(episode_dir)
    model = replay.load_model(args.urdf)
    data = mujoco.MjData(model)

    dataset_names = replay.dataset_joint_names(metadata)
    addresses, lower, upper = replay.model_joint_map(model, dataset_names)
    if trajectory.shape[1] != addresses.size:
        raise ValueError(
            f"trajectory width {trajectory.shape[1]} != mapping {addresses.size}"
        )

    base_body_id = _body_id(model, "Link_Base")
    sources = {side: fk21_sources(model, side) for side in SIDES}
    positions = np.empty(
        (trajectory.shape[0], len(SIDES), len(KEYPOINT_NAMES), 3),
        dtype=np.float32,
    )

    limit_violation = np.maximum(
        np.maximum(lower - trajectory, trajectory - upper), 0.0
    )
    for frame_index, frame in enumerate(trajectory):
        if args.clip_limits:
            frame = np.clip(frame, lower, upper)
        data.qpos[addresses] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(SIDES):
            positions[frame_index, side_index] = positions_in_base(
                model, data, base_body_id, sources[side]
            )

    active_sides = tuple(metadata.get("active_hand_sides", SIDES))
    side_is_observed = np.asarray(
        [side in active_sides for side in SIDES], dtype=np.bool_
    )
    timestamps = anchor_timestamps(episode_dir, trajectory.shape[0])

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positions=positions,
        timestamps=timestamps,
        side_names=np.asarray(SIDES),
        keypoint_names=np.asarray(KEYPOINT_NAMES),
        side_is_observed=side_is_observed,
        frame_rate=np.asarray(float(metadata.get("frame_rate", 30.0))),
        coordinate_frame=np.asarray("Link_Base"),
        units=np.asarray("metre"),
        source=np.asarray("/observations/qpos"),
        qpos_was_clipped=np.asarray(bool(args.clip_limits)),
    )

    side_ranges = {}
    for side_index, side in enumerate(SIDES):
        values = positions[:, side_index]
        side_ranges[side] = {
            "observed": bool(side_is_observed[side_index]),
            "min_xyz_m": values.min(axis=(0, 1)).astype(float).tolist(),
            "max_xyz_m": values.max(axis=(0, 1)).astype(float).tolist(),
        }
    summary = {
        "output": str(output),
        "shape": list(positions.shape),
        "coordinate_frame": "Link_Base",
        "units": "metre",
        "sides": list(SIDES),
        "keypoint_names": list(KEYPOINT_NAMES),
        "side_is_observed": side_is_observed.tolist(),
        "qpos_was_clipped": bool(args.clip_limits),
        "limit_violation_frames": int(
            np.count_nonzero(np.any(limit_violation > 1e-7, axis=1))
        ),
        "max_limit_violation_rad": float(limit_violation.max(initial=0.0)),
        "ranges": side_ranges,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--clip-limits",
        action="store_true",
        help="clip measured qpos to URDF limits before FK (default keeps raw feedback)",
    )
    return parser.parse_args()


def main() -> int:
    summary = export(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
