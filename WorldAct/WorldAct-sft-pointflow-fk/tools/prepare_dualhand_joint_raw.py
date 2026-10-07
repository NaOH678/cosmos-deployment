#!/usr/bin/env python3
"""Extract dual-hand joint state/actions from synchronized raw LMDB episodes."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ACTION_DIM = 54
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
SIDES = ("left", "right")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-text", default=None)
    parser.add_argument("--expected-episodes", type=int, default=98)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def build_state_action(
    qpos: np.ndarray,
    raw_action: np.ndarray,
    arm_joint_command: np.ndarray,
    robot_layout: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Build left-arm/hand + right-arm/hand state and joint targets in radians."""
    qpos = np.asarray(qpos, dtype=np.float32)
    raw_action = np.asarray(raw_action, dtype=np.float32)
    arm_joint_command = np.asarray(arm_joint_command, dtype=np.float32)

    qpos_layout = robot_layout.get("qpos_layout")
    action_layout = robot_layout.get("action_layout")
    if not isinstance(qpos_layout, list) or not isinstance(action_layout, list):
        raise ValueError("robot_layout must contain qpos_layout and action_layout lists")
    if list(robot_layout.get("sides", [])) != list(SIDES):
        raise ValueError(f"Expected robot_layout.sides={list(SIDES)}, got {robot_layout.get('sides')!r}")

    arm_dof_per_side = int(robot_layout.get("arm_dof_per_side", 0))
    hand_dof_per_side = int(robot_layout.get("hand_dof_per_side", 0))
    if arm_dof_per_side != 7 or hand_dof_per_side != 20:
        raise ValueError(
            f"Expected 7 arm + 20 hand joints per side, got {arm_dof_per_side} + {hand_dof_per_side}"
        )

    num_frames = len(qpos)
    if qpos.ndim != 2 or raw_action.ndim != 2 or arm_joint_command.ndim != 2:
        raise ValueError(
            "qpos, raw_action, and arm_joint_command must be rank-2 arrays; "
            f"got {qpos.shape}, {raw_action.shape}, {arm_joint_command.shape}"
        )
    if len(raw_action) != num_frames or len(arm_joint_command) != num_frames:
        raise ValueError(
            f"LMDB row mismatch: qpos={num_frames}, action={len(raw_action)}, "
            f"arm_joint_command={len(arm_joint_command)}"
        )
    if arm_joint_command.shape[1] != 2 * arm_dof_per_side:
        raise ValueError(
            f"Expected arm_joint_command width {2 * arm_dof_per_side}, got {arm_joint_command.shape}"
        )

    left_arm = _layout_slice(qpos_layout, "left", "arm")
    left_state_hand = _layout_slice(qpos_layout, "left", "hand")
    right_arm = _layout_slice(qpos_layout, "right", "arm")
    right_state_hand = _layout_slice(qpos_layout, "right", "hand")
    left_action_hand = _layout_slice(action_layout, "left", "hand")
    right_action_hand = _layout_slice(action_layout, "right", "hand")

    state = np.concatenate(
        (
            qpos[:, left_arm],
            qpos[:, left_state_hand],
            qpos[:, right_arm],
            qpos[:, right_state_hand],
        ),
        axis=1,
    )
    action = np.concatenate(
        (
            arm_joint_command[:, :arm_dof_per_side],
            np.deg2rad(raw_action[:, left_action_hand]),
            arm_joint_command[:, arm_dof_per_side:],
            np.deg2rad(raw_action[:, right_action_hand]),
        ),
        axis=1,
    )
    if state.shape != (num_frames, ACTION_DIM) or action.shape != state.shape:
        raise ValueError(
            f"Expected state/action shape {(num_frames, ACTION_DIM)}, got {state.shape}/{action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action contains NaN or Inf")
    return (
        np.ascontiguousarray(state, dtype=np.float32),
        np.ascontiguousarray(action, dtype=np.float32),
    )


def _read_lmdb_arrays(lmdb_dir: Path, robot_layout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "python-lmdb is required for this command. On this cluster run it with /usr/bin/python3."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="dualhand_joint_lmdb_") as tmp:
        local_lmdb = Path(tmp) / "lmdb"
        shutil.copytree(lmdb_dir, local_lmdb)
        env = lmdb.open(str(local_lmdb), readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin(write=False) as txn:
                arrays = []
                for key in (b"/observations/qpos", b"action", b"/diagnostics/arm_joint_command"):
                    payload = txn.get(key)
                    if payload is None:
                        raise KeyError(f"Missing LMDB key {key.decode()} in {lmdb_dir}")
                    arrays.append(np.asarray(pickle.loads(payload), dtype=np.float32))
        finally:
            env.close()
    return build_state_action(*arrays, robot_layout)


def _load_metadata(episode_dir: Path) -> dict[str, Any]:
    with (episode_dir / "meta_info.pkl").open("rb") as file:
        metadata = pickle.load(file)
    if not isinstance(metadata, dict):
        raise TypeError(f"{episode_dir.name}: meta_info.pkl must contain a dictionary")
    if metadata.get("active_arm_sides") != list(SIDES):
        raise ValueError(f"{episode_dir.name}: expected both active arms")
    if metadata.get("active_hand_sides") != list(SIDES):
        raise ValueError(f"{episode_dir.name}: expected both active hands")
    if metadata.get("camera_names") != list(CAMERA_NAMES):
        raise ValueError(
            f"{episode_dir.name}: expected cameras {list(CAMERA_NAMES)}, got {metadata.get('camera_names')!r}"
        )
    return metadata


def _validate_sources(episode_dir: Path, metadata: dict[str, Any]) -> tuple[int, float]:
    num_frames = int(metadata["num_steps"])
    source_fps = float(metadata["frame_rate"])
    if num_frames <= 0 or source_fps <= 0:
        raise ValueError(f"{episode_dir.name}: invalid num_steps/frame_rate")
    if not (episode_dir / "lmdb" / "data.mdb").is_file():
        raise FileNotFoundError(episode_dir / "lmdb" / "data.mdb")
    video_metadata = metadata.get("videos", {})
    for camera in CAMERA_NAMES:
        path = episode_dir / "videos" / f"{camera}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        camera_metadata = video_metadata.get(camera, {})
        if int(camera_metadata.get("num_frames", -1)) != num_frames:
            raise ValueError(
                f"{episode_dir.name}: {camera} metadata has {camera_metadata.get('num_frames')} frames, "
                f"expected {num_frames}"
            )
    return num_frames, source_fps


def _load_existing(output_path: Path, num_frames: int) -> None:
    with np.load(output_path, allow_pickle=False) as cached:
        state = np.asarray(cached["state"])
        action = np.asarray(cached["action"])
    expected = (num_frames, ACTION_DIM)
    if state.shape != expected or action.shape != expected or state.dtype != np.float32 or action.dtype != np.float32:
        raise ValueError(
            f"Invalid existing cache {output_path}: expected float32 {expected}, "
            f"got {state.dtype} {state.shape}/{action.dtype} {action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"Invalid existing cache {output_path}: contains NaN or Inf")


def main() -> None:
    args = _parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    episode_dirs = sorted(path for path in raw_root.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {raw_root}")
    if args.expected_episodes > 0 and len(episode_dirs) != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, found {len(episode_dirs)}")

    episode_output = output_root / "episodes"
    episode_output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    task_names = set()
    total_frames = 0
    for index, episode_dir in enumerate(episode_dirs, start=1):
        metadata = _load_metadata(episode_dir)
        num_frames, source_fps = _validate_sources(episode_dir, metadata)
        task_names.add(str(metadata.get("task_name", "")).strip())
        output_path = episode_output / f"{episode_dir.name}.npz"
        if output_path.exists() and not args.overwrite:
            _load_existing(output_path, num_frames)
        else:
            state, action = _read_lmdb_arrays(episode_dir / "lmdb", metadata["robot_layout"])
            if len(state) != num_frames:
                raise ValueError(f"{episode_dir.name}: LMDB rows={len(state)}, metadata num_steps={num_frames}")
            temp_path = output_path.with_suffix(".npz.tmp")
            with temp_path.open("wb") as file:
                np.savez(file, state=state, action=action)
            temp_path.replace(output_path)

        total_frames += num_frames
        manifest_rows.append(
            {
                "name": episode_dir.name,
                "num_frames": num_frames,
                "source_fps": source_fps,
                "videos": {camera: f"{episode_dir.name}/videos/{camera}.mp4" for camera in CAMERA_NAMES},
            }
        )
        print(f"[{index:03d}/{len(episode_dirs):03d}] {episode_dir.name}: {num_frames} frames", flush=True)

    task_names.discard("")
    if args.task_text is None and len(task_names) != 1:
        raise ValueError(f"Expected one common task_name, found {sorted(task_names)}; pass --task-text")
    task_text = args.task_text or next(iter(task_names))
    manifest = {
        "schema": "cosmos_dualhand_joint_v1",
        "schema_version": 1,
        "dataset_type": "dualhand_joint",
        "arm_action_space": "joint",
        "raw_root": str(raw_root),
        "task_text": task_text,
        "num_episodes": len(manifest_rows),
        "total_frames": total_frames,
        "state_dim": ACTION_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": "left_arm_joint_7 + left_hand_joint_20 + right_arm_joint_7 + right_hand_joint_20",
        "action_layout": (
            "left_arm_target_joint_7 + left_hand_target_joint_20 + "
            "right_arm_target_joint_7 + right_hand_target_joint_20"
        ),
        "units": {"state": "radian", "action": "radian"},
        "camera_names": list(CAMERA_NAMES),
        "episodes": manifest_rows,
    }
    manifest_path = output_root / "manifest.json"
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temp_manifest.replace(manifest_path)
    print(f"Wrote {manifest_path}: {len(manifest_rows)} episodes, {total_frames} frames")


if __name__ == "__main__":
    main()
