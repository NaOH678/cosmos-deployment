#!/usr/bin/env python3
"""Convert replay HDF5 into a strict single-task, three-view joint cache."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cosmos_framework.utils.bench2dex_contract import (
    CAMERAS,
    JOINT_NAMES,
    ROBOT,
    compose_rgb,
    permutation,
)


def text(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def convert(source, output, task, task_text, view_width=640):
    import cv2
    import h5py

    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    files = [source] if source.is_file() else sorted(source.rglob("*.hdf5"))
    if not files:
        raise ValueError("No HDF5 episodes found")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Duplicate episode names; provide one task/robot/replay directory")
    output.mkdir(parents=True)
    (output / "episodes").mkdir()
    (output / "video_frames").mkdir()
    rows, videos = [], []
    first_runtime = None
    for p in files:
        with h5py.File(p) as f:
            if text(f["meta/robot_key"][()]) != ROBOT or text(f["meta/scene_name"][()]) != task:
                raise ValueError(f"{p}: wrong robot or task")
            if float(f["meta/fps"][()]) != 20:
                raise ValueError(f"{p}: only native 20Hz is supported")
            if (
                text(f["action/control_mode"][()]) != "joint_position"
                or text(f["action/action_type"][()]) != "absolute"
            ):
                raise ValueError(f"{p}: expected absolute joint position commands")
            names = [text(x) for x in f["robot/joint_names"][:]]
            action_names = [text(x) for x in f["action/action_names"][:]]
            si, ai = permutation(names), permutation(action_names)
            first_runtime = first_runtime or names
            steps = f["time/sim_step"][:]
            if np.any(np.diff(steps) <= 0):
                raise ValueError(f"{p}: non-monotonic sim steps")
            marker = int(f["meta/homing_start_sim_step"][()])
            if marker < 0:
                raise ValueError(f"{p}: missing valid explicit homing marker")
            n = int(np.searchsorted(steps, marker))
            if n < 33:
                raise ValueError(f"{p}: fewer than 33 pre-homing observations")
            state = np.asarray(f["robot/qpos"][:n], np.float32)[:, si]
            action = np.asarray(f["action/commanded"][:n], np.float32)[:, ai]
            valid = np.asarray(f["action/action_valid"][:n], bool)
            valid &= np.isfinite(action).all(axis=1)
            # Initial capture can be off cadence. Reject crossing transitions,
            # without deleting frames or joining observations across a gap.
            valid[:-1] &= np.diff(steps[:n]) == int(f["meta/step_stride"][()])
            if state.shape != (n, 52) or action.shape != (n, 52) or not np.isfinite(state).all():
                raise ValueError(f"{p}: malformed state/action")
            # Keep invalid frames in place: the loader rejects crossing windows.
            action[~valid] = 0
            shape = (n, 3, view_width * 3 // 4 + view_width * 3 // 8, view_width)
            rel = f"video_frames/{p.stem}.npy"
            # Sequential NPY writing also works on filesystems without writable mmap.
            with (output / rel).open("wb") as stream:
                np.lib.format.write_array_header_1_0(stream, dict(descr="|u1", fortran_order=False, shape=shape))
                for t in range(n):
                    images = {}
                    for camera in CAMERAS:
                        raw = np.asarray(f[f"cameras/{camera}/rgb"][t])
                        if raw.ndim == 1:
                            bgr = cv2.imdecode(raw.astype(np.uint8), cv2.IMREAD_COLOR)
                            if bgr is None:
                                raise ValueError(f"{p}: invalid JPEG {camera}/{t}")
                            raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        images[camera] = raw
                    stream.write(compose_rgb(images, view_width).tobytes())
            np.savez(
                output / "episodes" / f"{p.stem}.npz",
                state=state,
                action=action,
                action_valid=valid,
                sim_step=steps[:n],
            )
            rows.append(
                dict(
                    name=p.stem,
                    trajectory_group=(
                        re.match(r"episode_\d+", p.stem).group(0) if re.match(r"episode_\d+", p.stem) else p.stem
                    ),
                    num_frames=n,
                    source_fps=20.0,
                    source=str(p.resolve()),
                    homing_start_sim_step=marker,
                    valid_actions=int(valid.sum()),
                )
            )
            videos.append(dict(name=p.stem, path=rel, shape=shape))
            print(f"{p.stem}: {n} frames before homing", flush=True)
    manifest = dict(
        schema="cosmos_bench2dex_joint_v1",
        schema_version=1,
        robot_key=ROBOT,
        task=task,
        task_text=task_text,
        arm_action_space="joint",
        state_dim=52,
        action_dim=52,
        units=dict(state="radian", action="radian"),
        joint_names=JOINT_NAMES,
        cameras=CAMERAS,
        view_width=view_width,
        action_alignment="obs_t_to_commanded_t",
        num_episodes=len(rows),
        episodes=rows,
    )
    (output / "video_manifest.json").write_text(
        json.dumps(
            dict(
                schema="cosmos_bench2dex_video_v1",
                schema_version=1,
                resolution="480",
                dtype="uint8",
                layout="TCHW",
                episodes=videos,
            ),
            indent=2,
        )
    )
    (output / "runtime_joint_names.json").write_text(json.dumps(first_runtime, indent=2))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", default="21_condiment_box_loading")
    p.add_argument("--task-text", required=True)
    p.add_argument("--view-width", type=int, default=640)
    a = p.parse_args()
    convert(a.source, a.output, a.task, a.task_text, a.view_width)


if __name__ == "__main__":
    main()
