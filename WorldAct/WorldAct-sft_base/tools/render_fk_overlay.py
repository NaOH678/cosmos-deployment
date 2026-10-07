"""Render the verified FK-21 overlay as a video and a stills contact sheet.

Uses the extrinsics verified against the tracked point cloud:
    p_cam = Rz(180 deg) @ R_urdf^T @ (p_base - t_urdf)

Usage:
    python tools/render_fk_overlay.py --episode-index 0 --out /tmp/fk_vis
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_fk_camera_projection import (  # noqa: E402
    EDGES,
    FINGER_COLORS,
    IMG_H,
    IMG_W,
    URDF,
    base_to_head_camera,
    finger_of,
    load_episode_map,
    load_fk,
    project,
    read_frame,
    roll_about_z,
)

RAW_ROOT = (
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/"
    "raw_data/singlerighthand_sandwich_100"
)


def project_episode(pts_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised base->camera->pixel for every frame. Returns (uv[T,21,2], in_front[T,21])."""
    r, t = base_to_head_camera(URDF)
    r = roll_about_z(180.0) @ r
    t = roll_about_z(180.0) @ t
    cam = pts_all @ r.T + t  # [T,21,3]
    z = cam[..., 2]
    in_front = z > 1e-6
    zs = np.where(in_front, z, 1.0)
    u = 605.5706176757812 * cam[..., 0] / zs + 324.4994812011719
    v = 604.4129638671875 * cam[..., 1] / zs + 238.25637817382812
    return np.stack([u, v], axis=-1), in_front


def draw(frame: np.ndarray, uv: np.ndarray, valid: np.ndarray, scale: int, label: str) -> np.ndarray:
    canvas = cv2.resize(frame, (IMG_W * scale, IMG_H * scale), interpolation=cv2.INTER_LINEAR)

    def pt(k: int) -> tuple[int, int]:
        return int(round(uv[k, 0] * scale)), int(round(uv[k, 1] * scale))

    for a, b in EDGES:
        if valid[a] and valid[b]:
            cv2.line(canvas, pt(a), pt(b), (0, 0, 0), 5 * scale, cv2.LINE_AA)
            cv2.line(canvas, pt(a), pt(b), FINGER_COLORS[finger_of(b)], 2 * scale, cv2.LINE_AA)
    for k in range(21):
        if valid[k]:
            cv2.circle(canvas, pt(k), 4 * scale, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt(k), 3 * scale, FINGER_COLORS[finger_of(k)], -1, cv2.LINE_AA)
        else:
            # off-image or behind camera: pin a magenta cross to the border
            x = int(np.clip(uv[k, 0] * scale, 6, IMG_W * scale - 6))
            y = int(np.clip(uv[k, 1] * scale, 6, IMG_H * scale - 6))
            cv2.drawMarker(canvas, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 7 * scale, 2 * scale)

    cv2.rectangle(canvas, (0, 0), (IMG_W * scale, 26 * scale), (0, 0, 0), -1)
    cv2.putText(canvas, label, (6 * scale, 19 * scale),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), 1 * scale, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--out", default="/tmp/fk_vis")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stills", default="200,500,800,1100,1400,1700")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ep_map = load_episode_map()
    name = ep_map[args.episode_index]
    video_path = os.path.join(RAW_ROOT, name, "videos", "head.mp4")
    pts_all = load_fk(args.episode_index)
    uv_all, front_all = project_episode(pts_all)
    print(f"episode {args.episode_index} = {name}   frames={pts_all.shape[0]}")

    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video frames = {n}")

    out_mp4 = os.path.join(args.out, f"fk_overlay_ep{args.episode_index:03d}.mp4")
    writer = cv2.VideoWriter(
        out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (IMG_W * args.scale, IMG_H * args.scale),
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    stills_wanted = {int(x) for x in args.stills.split(",")}
    stills: list[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(uv_all):
            break
        uv, front = uv_all[i], front_all[i]
        valid = front & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
        label = f"ep{args.episode_index}  frame {i:4d}   FK-21 (URDF extrinsics + 180deg roll)"
        canvas = draw(frame, uv, valid, args.scale, label)
        writer.write(canvas)
        if i in stills_wanted:
            stills.append(canvas)
        if i % 250 == 0:
            print(f"  ... {i}/{n}")
        i += 1
    cap.release()
    writer.release()
    print(f"wrote {out_mp4}  ({i} frames)")

    if stills:
        cols = 2
        rows = (len(stills) + cols - 1) // cols
        h, w = stills[0].shape[:2]
        sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
        for k, s in enumerate(stills):
            rr, cc = divmod(k, cols)
            sheet[rr * h:(rr + 1) * h, cc * w:(cc + 1) * w] = s
        sheet_path = os.path.join(args.out, f"fk_overlay_ep{args.episode_index:03d}_stills.png")
        cv2.imwrite(sheet_path, sheet)
        print(f"wrote {sheet_path}")


if __name__ == "__main__":
    main()
