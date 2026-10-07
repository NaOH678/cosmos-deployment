#!/usr/bin/env python3
"""Extract Cosmos action state/targets from single-right-hand raw LMDB episodes."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-text", default="make a sandwich")
    parser.add_argument("--arm-action-space", choices=("eef", "joint"), default="eef")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_state_action(
    raw_action: np.ndarray,
    observed_eef: np.ndarray,
    observed_hand_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the active right side and build matching 27-D state/action arrays."""
    raw_action = np.asarray(raw_action, dtype=np.float32)
    observed_eef = np.asarray(observed_eef, dtype=np.float32)
    observed_hand_deg = np.asarray(observed_hand_deg, dtype=np.float32)
    if raw_action.ndim != 2 or raw_action.shape[1] != 54:
        raise ValueError(f"Expected action shape [N,54], got {raw_action.shape}")
    if observed_eef.shape != (raw_action.shape[0], 14):
        raise ValueError(f"Expected observed EEF shape {(raw_action.shape[0], 14)}, got {observed_eef.shape}")
    if observed_hand_deg.shape != (raw_action.shape[0], 40):
        raise ValueError(f"Expected observed hand shape {(raw_action.shape[0], 40)}, got {observed_hand_deg.shape}")

    state = np.concatenate(
        [observed_eef[:, 7:14], np.deg2rad(observed_hand_deg[:, 20:40])],
        axis=1,
    )
    action = raw_action[:, 27:54].copy()
    action[:, 7:] = np.deg2rad(action[:, 7:])
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action contains non-finite values")
    quat_norm = np.linalg.norm(state[:, 3:7], axis=1)
    if not np.allclose(quat_norm, 1.0, atol=2e-3):
        raise ValueError(f"Observed right EEF quaternion norm outside tolerance: {quat_norm.min()}..{quat_norm.max()}")
    return state.astype(np.float32, copy=False), action.astype(np.float32, copy=False)


def _layout_slice(layout_rows: list[dict[str, Any]], side: str, key: str) -> slice:
    row = next((item for item in layout_rows if item.get("side") == side), None)
    if row is None:
        raise ValueError(f"Missing {side!r} entry in layout")
    bounds = row.get(key)
    if (
        not isinstance(bounds, (list, tuple))
        or len(bounds) != 2
        or not all(isinstance(value, int) for value in bounds)
        or bounds[0] < 0
        or bounds[1] <= bounds[0]
    ):
        raise ValueError(f"Invalid {side}.{key} layout bounds: {bounds!r}")
    return slice(bounds[0], bounds[1])


def build_joint_state_action(
    qpos: np.ndarray,
    raw_action: np.ndarray,
    arm_joint_command: np.ndarray,
    robot_layout: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Build right-arm/hand state and joint targets in radians."""
    qpos = np.asarray(qpos, dtype=np.float32)
    raw_action = np.asarray(raw_action, dtype=np.float32)
    arm_joint_command = np.asarray(arm_joint_command, dtype=np.float32)
    qpos_layout = robot_layout.get("qpos_layout")
    action_layout = robot_layout.get("action_layout")
    if not isinstance(qpos_layout, list) or not isinstance(action_layout, list):
        raise ValueError("robot_layout must contain qpos_layout and action_layout lists")

    arm_dof_per_side = int(robot_layout.get("arm_dof_per_side", 0))
    hand_dof_per_side = int(robot_layout.get("hand_dof_per_side", 0))
    if arm_dof_per_side != 7 or hand_dof_per_side != 20:
        raise ValueError(
            f"Expected 7 arm + 20 hand joints per side, got {arm_dof_per_side} + {hand_dof_per_side}"
        )
    if qpos.ndim != 2 or raw_action.ndim != 2 or arm_joint_command.ndim != 2:
        raise ValueError(
            "qpos, raw_action, and arm_joint_command must be rank-2 arrays; "
            f"got {qpos.shape}, {raw_action.shape}, {arm_joint_command.shape}"
        )
    num_frames = len(qpos)
    if len(raw_action) != num_frames or len(arm_joint_command) != num_frames:
        raise ValueError(
            f"LMDB row mismatch: qpos={num_frames}, action={len(raw_action)}, "
            f"arm_joint_command={len(arm_joint_command)}"
        )
    expected_arm_dim = 2 * arm_dof_per_side
    if arm_joint_command.shape[1] != expected_arm_dim:
        raise ValueError(f"Expected arm_joint_command width {expected_arm_dim}, got {arm_joint_command.shape}")

    right_arm = _layout_slice(qpos_layout, "right", "arm")
    right_state_hand = _layout_slice(qpos_layout, "right", "hand")
    right_action_hand = _layout_slice(action_layout, "right", "hand")
    state = np.concatenate((qpos[:, right_arm], qpos[:, right_state_hand]), axis=1)
    action = np.concatenate(
        (
            arm_joint_command[:, arm_dof_per_side:],
            np.deg2rad(raw_action[:, right_action_hand]),
        ),
        axis=1,
    )
    expected_shape = (num_frames, 27)
    if state.shape != expected_shape or action.shape != expected_shape:
        raise ValueError(f"Expected state/action shape {expected_shape}, got {state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action contains non-finite values")
    return (
        np.ascontiguousarray(state, dtype=np.float32),
        np.ascontiguousarray(action, dtype=np.float32),
    )


def _read_lmdb_arrays(
    lmdb_dir: Path,
    *,
    arm_action_space: str,
    robot_layout: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "python-lmdb is required only for this preprocessing command. On this cluster use /usr/bin/python3."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="singlerighthand_lmdb_") as tmp:
        local_lmdb = Path(tmp) / "lmdb"
        shutil.copytree(lmdb_dir, local_lmdb)
        env = lmdb.open(str(local_lmdb), readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin(write=False) as txn:
                values = []
                keys = (
                    (b"/observations/qpos", b"action", b"/diagnostics/arm_joint_command")
                    if arm_action_space == "joint"
                    else (b"action", b"/observations/eef", b"/observations/hand_joint_deg")
                )
                for key in keys:
                    payload = txn.get(key)
                    if payload is None:
                        raise KeyError(f"Missing LMDB key {key.decode()} in {lmdb_dir}")
                    values.append(np.asarray(pickle.loads(payload), dtype=np.float32))
        finally:
            env.close()
    if arm_action_space == "joint":
        return build_joint_state_action(*values, robot_layout)
    return build_state_action(*values)


def main() -> None:
    args = _parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    episode_output = output_root / "episodes"
    episode_output.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(path for path in raw_root.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {raw_root}")

    manifest_rows = []
    for index, episode_dir in enumerate(episode_dirs):
        with (episode_dir / "meta_info.pkl").open("rb") as file:
            metadata = pickle.load(file)
        output_path = episode_output / f"{episode_dir.name}.npz"
        if output_path.exists() and not args.overwrite:
            with np.load(output_path) as cached:
                num_frames = int(cached["state"].shape[0])
        else:
            state, action = _read_lmdb_arrays(
                episode_dir / "lmdb",
                arm_action_space=args.arm_action_space,
                robot_layout=metadata["robot_layout"],
            )
            num_frames = len(state)
            temp_path = output_path.with_suffix(".npz.tmp")
            with temp_path.open("wb") as file:
                np.savez(file, state=state, action=action)
            temp_path.replace(output_path)

        expected_steps = int(metadata.get("num_steps", num_frames))
        if num_frames != expected_steps:
            raise ValueError(f"{episode_dir.name}: arrays={num_frames}, metadata num_steps={expected_steps}")
        for camera in ("head", "right_wrist"):
            if not (episode_dir / "videos" / f"{camera}.mp4").is_file():
                raise FileNotFoundError(episode_dir / "videos" / f"{camera}.mp4")
        manifest_rows.append(
            {
                "name": episode_dir.name,
                "num_frames": num_frames,
                "source_fps": float(metadata.get("frame_rate", 30.0)),
            }
        )
        print(f"[{index + 1:03d}/{len(episode_dirs):03d}] {episode_dir.name}: {num_frames} frames")

    if args.arm_action_space == "joint":
        state_layout = "right_arm_joint_rad_7 + right_hand_joint_rad_20"
        action_layout = "right_arm_target_joint_rad_7 + right_hand_target_joint_rad_20"
    else:
        state_layout = "right_eef_xyz_quat_xyzw + right_hand_joint_rad_20"
        action_layout = "right_eef_xyz_quat_xyzw + right_hand_target_rad_20"
    manifest = {
        "schema_version": 2,
        "raw_root": str(raw_root),
        "task_text": args.task_text,
        "arm_action_space": args.arm_action_space,
        "state_layout": state_layout,
        "action_layout": action_layout,
        "episodes": manifest_rows,
    }
    manifest_path = output_root / "manifest.json"
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_manifest.replace(manifest_path)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
