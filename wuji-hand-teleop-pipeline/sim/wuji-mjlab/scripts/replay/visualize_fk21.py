#!/usr/bin/env python3
"""Replay Tianji + Wuji and visualize canonical FK-21 trajectories.

All overlay coordinates are read from the exported FK-21 file and are
expressed in the Tianji ``Link_Base`` frame.  The imported URDF fixes
``Link_Base`` at MuJoCo world origin, so the overlay can be drawn directly.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

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
DEFAULT_FK21 = PROJECT_ROOT / "generated" / "tianji_wuji_fk21.npz"

FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
FINGER_COLORS = np.asarray(
    [
        [0.95, 0.30, 0.25, 1.0],
        [0.25, 0.80, 0.30, 1.0],
        [0.20, 0.55, 1.00, 1.0],
        [0.95, 0.75, 0.20, 1.0],
        [0.75, 0.30, 0.95, 1.0],
    ],
    dtype=np.float32,
)
TIP_INDICES = (4, 8, 12, 16, 20)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"model has no body named {name!r}")
    return body_id


def _add_sphere(scene, position, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0]),
        np.asarray(position, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_segment(scene, start, end, width, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    # mjv_connector only sets the connector's type, size and transform.  The
    # remaining mjvGeom fields must first receive valid defaults.
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
    scene.ngeom += 1


def draw_link_base_frame(
    scene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_body_id: int,
    *,
    axis_length: float = 0.65,
) -> None:
    """Draw Link_Base axes in MuJoCo world: X red, Y green, Z blue."""
    origin = data.xpos[base_body_id]
    rotation_world_base = data.xmat[base_body_id].reshape(3, 3)
    for axis, color in zip(
        rotation_world_base.T,
        (
            [1.0, 0.1, 0.1, 1.0],
            [0.1, 1.0, 0.1, 1.0],
            [0.1, 0.35, 1.0, 1.0],
        ),
    ):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.018,
            np.asarray(origin, dtype=np.float64),
            np.asarray(origin + axis_length * axis, dtype=np.float64),
        )
        geom.rgba[:] = np.asarray(color, dtype=np.float32)
        scene.ngeom += 1
    _add_sphere(scene, origin, 0.018, [1.0, 1.0, 1.0, 1.0])


def draw_fk21(
    scene,
    positions: np.ndarray,
    frame: int,
    observed_sides: np.ndarray,
    *,
    trail_frames: int,
    trail_stride: int,
    show_unobserved: bool,
    reset_scene: bool = True,
) -> None:
    if reset_scene:
        scene.ngeom = 0
    first = max(0, frame - trail_frames)
    for side_index in range(positions.shape[1]):
        if not observed_sides[side_index] and not show_unobserved:
            continue
        side_alpha = 1.0 if observed_sides[side_index] else 0.25
        current = positions[frame, side_index]

        # Current canonical 21-point skeleton.
        _add_sphere(
            scene, current[0], 0.009,
            [1.0, 1.0, 1.0, side_alpha],
        )
        for finger_index, chain in enumerate(FINGER_CHAINS):
            color = FINGER_COLORS[finger_index].copy()
            color[3] *= side_alpha
            for point_index in chain[1:]:
                radius = 0.008 if point_index in TIP_INDICES else 0.006
                _add_sphere(scene, current[point_index], radius, color)
            for start_index, end_index in zip(chain[:-1], chain[1:]):
                _add_segment(
                    scene,
                    current[start_index],
                    current[end_index],
                    0.0025,
                    color,
                )

        # Flow trails for all 21 points.  Consecutive history samples are
        # connected and fade with age, so motion direction remains readable.
        history_frames = list(range(first, frame, trail_stride))
        if history_frames and history_frames[-1] != frame:
            history_frames.append(frame)
        for start_frame, end_frame in zip(
            history_frames[:-1], history_frames[1:]
        ):
            age = (end_frame - first + 1) / max(frame - first + 1, 1)
            for point_index in range(21):
                if point_index == 0:
                    color = np.asarray(
                        [1.0, 1.0, 1.0, (0.04 + 0.48 * age) * side_alpha]
                    )
                else:
                    finger_index = min((point_index - 1) // 4, 4)
                    color = FINGER_COLORS[finger_index].copy()
                    color[3] = (0.03 + 0.42 * age) * side_alpha
                _add_segment(
                    scene,
                    positions[start_frame, side_index, point_index],
                    positions[end_frame, side_index, point_index],
                    0.0012 + 0.0008 * age,
                    color,
                )


def tcp_positions_in_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_body_id: int,
    tcp_body_ids: tuple[int, int],
) -> np.ndarray:
    base_position = data.xpos[base_body_id]
    base_rotation_inv = data.xmat[base_body_id].reshape(3, 3).T
    return np.stack(
        [
            base_rotation_inv @ (data.xpos[body_id] - base_position)
            for body_id in tcp_body_ids
        ]
    )


def draw_tcp_trails(
    scene,
    tcp_paths: np.ndarray,
    frame_index: int,
    *,
    first: int,
    trail_stride: int,
) -> None:
    for side_index, color in enumerate(
        ([0.1, 0.9, 1.0, 0.8], [1.0, 0.4, 0.1, 0.8])
    ):
        path_frames = list(range(first, frame_index, trail_stride))
        if path_frames and path_frames[-1] != frame_index:
            path_frames.append(frame_index)
        for path_start, path_end in zip(path_frames[:-1], path_frames[1:]):
            age = (path_end - first + 1) / max(
                frame_index - first + 1, 1
            )
            trail_color = np.asarray(color, dtype=np.float32)
            trail_color[3] = 0.08 + 0.72 * age
            _add_segment(
                scene,
                tcp_paths[path_start, side_index],
                tcp_paths[path_end, side_index],
                0.002 + 0.001 * age,
                trail_color,
            )


def record_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: np.ndarray,
    addresses: np.ndarray,
    positions: np.ndarray,
    observed_sides: np.ndarray,
    tcp_paths: np.ndarray,
    base_body_id: int,
    *,
    output: Path,
    start: int,
    stop: int,
    fps: float,
    width: int,
    height: int,
    trail_frames: int,
    trail_stride: int,
    show_unobserved: bool,
) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    camera.lookat[:] = np.asarray([0.68, 0.0, 0.90])
    camera.distance = 1.45
    camera.azimuth = 135.0
    camera.elevation = -18.0
    try:
        with mujoco.Renderer(
            model, height=height, width=width, max_geom=10000
        ) as renderer:
            for frame_index in range(start, stop):
                data.qpos[addresses] = trajectory[frame_index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                draw_fk21(
                    renderer.scene,
                    positions,
                    frame_index,
                    observed_sides,
                    trail_frames=trail_frames,
                    trail_stride=trail_stride,
                    show_unobserved=show_unobserved,
                    reset_scene=False,
                )
                draw_link_base_frame(
                    renderer.scene, model, data, base_body_id
                )
                first = max(start, frame_index - trail_frames)
                draw_tcp_trails(
                    renderer.scene,
                    tcp_paths,
                    frame_index,
                    first=first,
                    trail_stride=trail_stride,
                )
                frame = renderer.render()
                if encoder.stdin is None:
                    raise RuntimeError("ffmpeg stdin is unavailable")
                encoder.stdin.write(frame.tobytes())
                if (frame_index - start + 1) % 100 == 0:
                    print(
                        f"recorded {frame_index - start + 1}/{stop - start}",
                        flush=True,
                    )
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"saved video: {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--fk21", type=Path, default=DEFAULT_FK21)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--trail-frames", type=int, default=90)
    parser.add_argument("--trail-stride", type=int, default=5)
    parser.add_argument("--show-unobserved", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--video", type=Path, help="record visualization to MP4")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    args = parser.parse_args()
    if args.speed <= 0 or args.trail_frames < 0 or args.trail_stride <= 0:
        raise ValueError("speed/trajectory options must be positive")

    replay = _load_module(HERE.with_name("replay_teleop.py"), "replay_teleop")
    trajectory, metadata = replay.load_episode(args.episode_dir)
    model = replay.load_model(args.urdf)
    data = mujoco.MjData(model)
    names = replay.dataset_joint_names(metadata)
    addresses, _, _ = replay.model_joint_map(model, names)

    exported = np.load(args.fk21.expanduser().resolve(), allow_pickle=False)
    positions = np.asarray(exported["positions"], dtype=np.float64)
    observed_sides = np.asarray(exported["side_is_observed"], dtype=np.bool_)
    coordinate_frame = str(exported["coordinate_frame"])
    if positions.shape != (trajectory.shape[0], 2, 21, 3):
        raise ValueError(
            f"FK-21 shape {positions.shape} does not match "
            f"trajectory frames {trajectory.shape[0]}"
        )
    if coordinate_frame != "Link_Base":
        raise ValueError(f"expected Link_Base FK-21, got {coordinate_frame}")

    stop = trajectory.shape[0] if args.stop is None else args.stop
    if not (0 <= args.start < stop <= trajectory.shape[0]):
        raise ValueError(f"invalid frame interval [{args.start}, {stop})")

    base_body_id = _body_id(model, "Link_Base")
    tcp_body_ids = (
        _body_id(model, "TCP_Link_L"),
        _body_id(model, "TCP_Link_R"),
    )
    # Precompute Tianji TCP paths from the same measured arm trajectory.
    tcp_paths = np.empty((trajectory.shape[0], 2, 3), dtype=np.float64)
    for frame_index, qpos in enumerate(trajectory):
        data.qpos[addresses] = qpos
        mujoco.mj_forward(model, data)
        tcp_paths[frame_index] = tcp_positions_in_base(
            model, data, base_body_id, tcp_body_ids
        )

    summary = {
        "frames": int(trajectory.shape[0]),
        "fk21_shape": list(positions.shape),
        "coordinate_frame": coordinate_frame,
        "units": str(exported["units"]),
        "observed_sides": observed_sides.tolist(),
        "overlay": "current FK-21 + all-point history + Tianji TCP history",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.video:
        if args.video_width <= 0 or args.video_height <= 0:
            raise ValueError("video dimensions must be positive")
        record_video(
            model,
            data,
            trajectory,
            addresses,
            positions,
            observed_sides,
            tcp_paths,
            base_body_id,
            output=args.video,
            start=args.start,
            stop=stop,
            fps=float(metadata.get("frame_rate", 30.0)) * args.speed,
            width=args.video_width,
            height=args.video_height,
            trail_frames=args.trail_frames,
            trail_stride=args.trail_stride,
            show_unobserved=args.show_unobserved,
        )
        return 0

    viewer_context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer_context = mj_viewer.launch_passive(model, data)

    frame_period = 1.0 / (float(metadata.get("frame_rate", 30.0)) * args.speed)
    with viewer_context as viewer:
        while True:
            deadline = time.monotonic()
            for frame_index in range(args.start, stop):
                if viewer is not None and not viewer.is_running():
                    return 0
                if viewer is not None:
                    # launch_passive renders from a separate thread.  Hold
                    # its lock while changing MjData and the user scene so
                    # the render thread never observes a partially updated
                    # kinematic state or overlay.
                    with viewer.lock():
                        data.qpos[addresses] = trajectory[frame_index]
                        data.qvel[:] = 0.0
                        mujoco.mj_forward(model, data)
                        draw_fk21(
                            viewer.user_scn,
                            positions,
                            frame_index,
                            observed_sides,
                            trail_frames=args.trail_frames,
                            trail_stride=args.trail_stride,
                            show_unobserved=args.show_unobserved,
                        )
                        draw_link_base_frame(
                            viewer.user_scn, model, data, base_body_id
                        )
                        first = max(
                            args.start, frame_index - args.trail_frames
                        )
                        draw_tcp_trails(
                            viewer.user_scn,
                            tcp_paths,
                            frame_index,
                            first=first,
                            trail_stride=args.trail_stride,
                        )
                    viewer.sync()
                else:
                    data.qpos[addresses] = trajectory[frame_index]
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                if not args.no_realtime:
                    deadline += frame_period
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            if not args.loop:
                break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
