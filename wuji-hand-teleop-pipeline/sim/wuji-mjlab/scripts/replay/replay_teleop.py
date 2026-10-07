#!/usr/bin/env python3
"""Replay a Tianji + Wuji LMDB episode with the combined robot URDF.

This is intentionally independent from ROS and the teleoperation stack.  It
loads the measured 54-DoF ``/observations/qpos`` trajectory and writes it
directly to MuJoCo joint positions, making it a kinematic data/URDF replay
rather than a controller or dynamics test.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import pickle
import sys
import time
import xml.etree.ElementTree as ET

import lmdb
import mujoco
import numpy as np


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
DEFAULT_EPISODE = PROJECT_ROOT / "tianji_wuji_data"
DEFAULT_URDF = (
    PROJECT_ROOT
    / "marvin_wuji_d435_description"
    / "urdf"
    / "marvin_wuji_d435_complete.urdf"
)

ARM_JOINTS = {
    "left": [f"Joint{i}_L" for i in range(1, 8)],
    "right": [f"Joint{i}_R" for i in range(1, 8)],
}
HAND_JOINTS = {
    side: [
        f"{side}_hand_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    ]
    for side in ("left", "right")
}


def _load_pickle(path: Path):
    # Episodes are locally generated trusted artifacts.  Pickle must not be
    # used here with downloaded/untrusted episode directories.
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_episode(episode_dir: Path) -> tuple[np.ndarray, dict]:
    """Load the aggregate measured joint trajectory and metadata."""
    episode_dir = episode_dir.expanduser().resolve()
    meta_path = episode_dir / "meta_info.pkl"
    lmdb_path = episode_dir / "lmdb"
    if not lmdb_path.is_dir():
        raise FileNotFoundError(
            f"{episode_dir} must contain lmdb/"
        )
    metadata = _load_pickle(meta_path) if meta_path.is_file() else None
    env = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, max_readers=1
    )
    try:
        with env.begin() as txn:
            raw = txn.get(b"/observations/qpos")
            if raw is None:
                raise KeyError("LMDB has no aggregate /observations/qpos dataset")
            qpos = np.asarray(pickle.loads(raw), dtype=np.float64)
            if metadata is None:
                raw_metadata = (
                    txn.get(b"meta_info") or txn.get(b"__metadata__")
                )
                if raw_metadata is None:
                    raise FileNotFoundError(
                        f"{episode_dir} has neither meta_info.pkl nor "
                        "embedded LMDB metadata"
                    )
                metadata = pickle.loads(raw_metadata)
    finally:
        env.close()
    if not isinstance(metadata, dict):
        raise ValueError("episode metadata must be a dictionary")
    if qpos.ndim != 2 or not np.all(np.isfinite(qpos)):
        raise ValueError(f"qpos must be a finite 2-D array, got {qpos.shape}")
    expected_steps = int(metadata.get("num_steps", qpos.shape[0]))
    if qpos.shape[0] != expected_steps:
        raise ValueError(
            f"qpos has {qpos.shape[0]} frames, metadata says {expected_steps}"
        )
    return qpos, metadata


def dataset_joint_names(metadata: dict) -> list[str]:
    """Return joint names in the recorder's side-interleaved qpos order."""
    layout = metadata.get("robot_layout", {})
    sides = list(layout.get("sides", ["left", "right"]))
    arm_dof = int(layout.get("arm_dof_per_side", 7))
    hand_dof = int(layout.get("hand_dof_per_side", 20))
    if any(side not in ARM_JOINTS for side in sides):
        raise ValueError(f"unsupported dataset sides: {sides}")
    if arm_dof != 7 or hand_dof != 20:
        raise ValueError(
            "combined URDF mapping requires 7 arm + 20 hand joints per side; "
            f"metadata has {arm_dof} + {hand_dof}"
        )
    return [
        name
        for side in sides
        for name in (ARM_JOINTS[side] + HAND_JOINTS[side])
    ]


def resolve_urdf_resources(urdf_path: Path) -> str:
    """Resolve this ROS package's mesh URIs for MuJoCo's URDF importer."""
    urdf_path = urdf_path.expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    # MuJoCo's URDF importer strips mesh directories by default.  Keep the
    # absolute paths resolved below; left/right hand trees also contain equal
    # basenames in different directories, so basename-only lookup is unsafe.
    mujoco_extension = root.find("mujoco")
    if mujoco_extension is None:
        mujoco_extension = ET.SubElement(root, "mujoco")
    compiler = mujoco_extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_extension, "compiler")
    compiler.set("strippath", "false")
    compiler.set("discardvisual", "false")
    # FK/keypoint exporters need fixed URDF frames such as Link_Base,
    # *_hand_palm_link and *_tip_link to remain addressable by name.
    compiler.set("fusestatic", "false")
    package_root = urdf_path.parent.parent
    prefix = "package://marvin_wuji_d435_description/"
    unresolved: list[str] = []
    for link in root.findall("link"):
        link_name = link.get("name", "link")
        for kind in ("visual", "collision"):
            for index, geom in enumerate(link.findall(kind)):
                # The source URDF leaves these names empty.  MuJoCo accepts
                # that but emits one duplicate-name warning per geometry.
                if not geom.get("name"):
                    geom.set("name", f"{link_name}_{kind}_{index}")
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(prefix):
            resolved = package_root / filename[len(prefix) :]
            mesh.set("filename", str(resolved.resolve()))
        elif filename.startswith("package://"):
            unresolved.append(filename)
    if unresolved:
        raise ValueError(f"unresolved package URIs: {sorted(set(unresolved))}")
    return ET.tostring(root, encoding="unicode")


def load_model(urdf_path: Path) -> mujoco.MjModel:
    """Compile the combined URDF in memory; no source asset is modified."""
    xml = resolve_urdf_resources(urdf_path)
    try:
        return mujoco.MjModel.from_xml_string(xml)
    except ValueError as exc:
        raise RuntimeError(f"MuJoCo failed to import {urdf_path}: {exc}") from exc


def model_joint_map(
    model: mujoco.MjModel, names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_addresses = []
    lower = []
    upper = []
    missing = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            missing.append(name)
            continue
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name} is not a MuJoCo hinge joint")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        if model.jnt_limited[joint_id]:
            lower.append(float(model.jnt_range[joint_id, 0]))
            upper.append(float(model.jnt_range[joint_id, 1]))
        else:
            lower.append(-np.inf)
            upper.append(np.inf)
    if missing:
        raise ValueError(f"URDF is missing mapped joints: {missing}")
    return (
        np.asarray(qpos_addresses, dtype=np.int32),
        np.asarray(lower),
        np.asarray(upper),
    )


def replay(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    metadata: dict,
    *,
    start: int,
    stop: int,
    speed: float,
    loop: bool,
    headless: bool,
    realtime: bool,
) -> dict:
    names = dataset_joint_names(metadata)
    if trajectory.shape[1] != len(names):
        raise ValueError(
            f"qpos width is {trajectory.shape[1]}, mapping has {len(names)} joints"
        )
    addresses, lower, upper = model_joint_map(model, names)
    selected = trajectory[start:stop]
    clipped = np.clip(selected, lower, upper)
    violation = np.maximum(lower - selected, selected - upper)
    violation = np.maximum(violation, 0.0)
    violation_frames = int(np.count_nonzero(np.any(violation > 1e-7, axis=1)))
    max_violation = float(violation.max(initial=0.0))
    violation_by_joint = {
        name: {
            "frames": int(np.count_nonzero(violation[:, index] > 1e-7)),
            "max_rad": float(violation[:, index].max(initial=0.0)),
        }
        for index, name in enumerate(names)
        if np.any(violation[:, index] > 1e-7)
    }

    data = mujoco.MjData(model)
    frame_rate = float(metadata.get("frame_rate", 30.0))
    frame_period = 1.0 / (frame_rate * speed)
    viewer_context = nullcontext(None)
    if not headless:
        try:
            from mujoco import viewer as mj_viewer
        except ImportError as exc:
            raise RuntimeError("mujoco.viewer is unavailable") from exc
        viewer_context = mj_viewer.launch_passive(model, data)

    frames_shown = 0
    with viewer_context as viewer:
        while True:
            deadline = time.monotonic()
            for frame in clipped:
                if viewer is not None and not viewer.is_running():
                    return {
                        "frames_replayed": frames_shown,
                        "limit_violation_frames": violation_frames,
                        "max_limit_violation_rad": max_violation,
                        "limit_violations_by_joint": violation_by_joint,
                    }
                data.qpos[addresses] = frame
                data.qvel[:] = 0.0
                data.time = frames_shown / frame_rate
                mujoco.mj_forward(model, data)
                if viewer is not None:
                    viewer.sync()
                frames_shown += 1
                if realtime:
                    deadline += frame_period
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            if not loop:
                break
    return {
        "frames_replayed": frames_shown,
        "limit_violation_frames": violation_frames,
        "max_limit_violation_rad": max_violation,
        "limit_violations_by_joint": violation_by_joint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--start", type=int, default=0, help="first frame")
    parser.add_argument("--stop", type=int, help="exclusive last frame")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--headless", action="store_true", help="validate/replay without a window"
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="do not wait between frames (useful with --headless)",
    )
    parser.add_argument(
        "--export-mjcf",
        type=Path,
        help="save MuJoCo's compiled MJCF before replay",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="print episode/model summary and exit"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    trajectory, metadata = load_episode(args.episode_dir)
    model = load_model(args.urdf)
    names = dataset_joint_names(metadata)
    model_joint_map(model, names)
    summary = {
        "episode": str(args.episode_dir.resolve()),
        "frames": int(trajectory.shape[0]),
        "frame_rate": float(metadata.get("frame_rate", 30.0)),
        "qpos_shape": list(trajectory.shape),
        "camera_names": metadata.get("camera_names", []),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "mapped_joints": len(names),
        "active_hand_sides": metadata.get("active_hand_sides", []),
        "zero_filled_hand_sides": metadata.get("zero_filled_hand_sides", []),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.export_mjcf:
        output = args.export_mjcf.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(output), model)
        print(f"exported MJCF: {output}")
    if args.inspect:
        return 0

    stop = trajectory.shape[0] if args.stop is None else args.stop
    if not (0 <= args.start < stop <= trajectory.shape[0]):
        raise ValueError(
            f"require 0 <= start < stop <= {trajectory.shape[0]}, "
            f"got {args.start}, {stop}"
        )
    result = replay(
        model,
        trajectory,
        metadata,
        start=args.start,
        stop=stop,
        speed=args.speed,
        loop=args.loop,
        headless=args.headless,
        realtime=not args.no_realtime,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
