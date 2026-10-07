"""Verify the URDF head-camera extrinsics by projecting FK-21 keypoints onto head.mp4.

The URDF's `head_d435_link_optical_frame` is a *zero-offset alias* of
`head_d435_link`, and the README states the camera's local +Y is kept upward.
That is NOT the image convention (y down), so the roll about the view axis is
undetermined by the URDF alone.  This script renders every roll candidate over
the real video frame so the correct one is obvious by eye.

Usage:
    python tools/verify_fk_camera_projection.py \
        --episode-index 0 --frame auto --out /tmp/fk_proj
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
URDF = "/data/shichaojian/wuji-mjlab/marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf"
SLIM_ROOT = "/data/shichaojian/datasets/singlerighthand-sandwich-100-lerobot-slim"
RAW_ROOT = "/data/shichaojian/raw_data/singlerighthand_sandwich_100"

# Head colour stream intrinsics, from
# raw_data/<episode>/auxiliary_camera/metadata.json
FX, FY = 605.5706176757812, 604.4129638671875
CX, CY = 324.4994812011719, 238.25637817382812
IMG_W, IMG_H = 640, 480

# Fixed chain from Link_Base to the head camera optical frame.
CHAIN = [
    "Joint_Stand",
    "head_camera_base_joint",
    "head_camera_bracket_joint",
    "head_d435_mount_joint",
    "head_d435_link_optical_joint",
]

KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

# Per-finger colours (BGR), wrist grey.
FINGER_COLORS = [
    (255, 255, 255),  # wrist
    (0, 0, 255),  # thumb   - red
    (0, 165, 255),  # index   - orange
    (0, 255, 255),  # middle  - yellow
    (0, 255, 0),  # ring    - green
    (255, 0, 0),  # pinky   - blue
]

# Kinematic edges: wrist -> each MCP, then along each finger.
EDGES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
for base in (1, 5, 9, 13, 17):
    EDGES += [(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)]


def finger_of(k: int) -> int:
    if k == 0:
        return 0
    return (k - 1) // 4 + 1


# --------------------------------------------------------------------------
# URDF -> fixed transform
# --------------------------------------------------------------------------
def _rot_rpy(rpy: np.ndarray) -> np.ndarray:
    """URDF uses fixed-axis rpy, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def base_to_head_camera(urdf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) such that p_cam = R @ p_base + t.

    A URDF joint origin is the transform from the *child* link into the
    *parent* link, so composing the chain from Link_Base downwards yields
    T_camera_link -> Link_Base, not the inverse.  Camera coordinates are
    therefore  p_cam_urdf = R_chain^T @ (p_base - t_chain).
    """
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}

    m = np.eye(4)  # T_<current link> -> Link_Base
    for name in CHAIN:
        j = joints[name]
        if j.get("type") != "fixed":
            raise ValueError(f"{name} is not a fixed joint")
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()])
        m_joint = np.eye(4)  # T_child -> parent
        m_joint[:3, :3] = _rot_rpy(rpy)
        m_joint[:3, 3] = xyz
        m = m @ m_joint

    r_chain = m[:3, :3].copy()
    t_chain = m[:3, 3].copy()  # camera origin expressed in Link_Base

    # invert: base -> camera
    r = r_chain.T
    t = -r_chain.T @ t_chain
    return r, t


def roll_about_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# Projection / drawing
# --------------------------------------------------------------------------
def project(pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection. Returns (uv[21,2], in_front[21])."""
    z = pts_cam[:, 2]
    in_front = z > 1e-6
    z_safe = np.where(in_front, z, 1.0)
    u = FX * pts_cam[:, 0] / z_safe + CX
    v = FY * pts_cam[:, 1] / z_safe + CY
    return np.stack([u, v], axis=-1), in_front


def in_image(uv: np.ndarray, margin: int = 0) -> np.ndarray:
    return (uv[:, 0] >= margin) & (uv[:, 0] < IMG_W - margin) & (uv[:, 1] >= margin) & (uv[:, 1] < IMG_H - margin)


def draw_overlay(
    frame: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    title: str,
    scale: int = 2,
) -> np.ndarray:
    canvas = cv2.resize(frame, (IMG_W * scale, IMG_H * scale), interpolation=cv2.INTER_NEAREST)

    def pt(k: int) -> tuple[int, int]:
        return int(round(uv[k, 0] * scale)), int(round(uv[k, 1] * scale))

    for a, b in EDGES:
        if valid[a] and valid[b]:
            cv2.line(canvas, pt(a), pt(b), (60, 60, 60), 4 * scale, cv2.LINE_AA)
            cv2.line(canvas, pt(a), pt(b), FINGER_COLORS[finger_of(b)], 2 * scale, cv2.LINE_AA)

    for k in range(21):
        color = FINGER_COLORS[finger_of(k)]
        if valid[k]:
            cv2.circle(canvas, pt(k), 4 * scale, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt(k), 3 * scale, color, -1, cv2.LINE_AA)
        else:
            # invalid: project into frame if possible, else clamp to the border
            x = int(np.clip(uv[k, 0] * scale, 8, IMG_W * scale - 8))
            y = int(np.clip(uv[k, 1] * scale, 8, IMG_H * scale - 8))
            cv2.drawMarker(canvas, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 8 * scale, 2 * scale)

    cv2.rectangle(canvas, (0, 0), (IMG_W * scale, 30 * scale), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (8 * scale, 21 * scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * scale,
        (255, 255, 255),
        1 * scale,
        cv2.LINE_AA,
    )
    return canvas


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_episode_map() -> dict[int, str]:
    import json

    out = {}
    with open(os.path.join(SLIM_ROOT, "meta", "source_episodes.jsonl")) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[int(rec["episode_index"])] = rec["source_episode"]
    return out


def load_fk(episode_index: int) -> np.ndarray:
    path = os.path.join(SLIM_ROOT, "fk21", "chunk-000", f"episode_{episode_index:06d}.npz")
    d = np.load(path, allow_pickle=True)
    return d["right_positions_abs"].astype(np.float64)


def read_frame(source_episode: str, frame_idx: int) -> np.ndarray:
    path = os.path.join(RAW_ROOT, source_episode, "videos", "head.mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} of {path}")
    if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
        frame = cv2.resize(frame, (IMG_W, IMG_H))
    return frame


def pick_frame(pts_base: np.ndarray, r: np.ndarray, t: np.ndarray) -> int:
    """Frame index maximising in-frame keypoints and centring, at roll=180."""
    best, best_score = 0, -1e9
    centre = np.array([IMG_W / 2.0, IMG_H / 2.0])
    for i in range(pts_base.shape[0]):
        cam = pts_base[i] @ r.T + t
        cam = cam @ roll_about_z(180.0).T
        uv, front = project(cam)
        vis = front & in_image(uv, margin=20)
        if vis.sum() < 21:
            continue
        d = np.linalg.norm(uv - centre, axis=-1).mean()
        score = vis.sum() * 100.0 - d
        if score > best_score:
            best, best_score = i, score
    return best


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--frame", default="auto", help="'auto' or an integer frame index")
    ap.add_argument("--rolls", default="0,90,180,270")
    ap.add_argument("--out", default="/tmp/fk_proj")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    r_urdf, t = base_to_head_camera(URDF)
    cam_pos_base = -r_urdf.T @ t
    print("=== Link_Base -> head camera (URDF), p_cam = R @ p_base + t ===")
    np.set_printoptions(precision=8, suppress=True)
    print("R =\n", r_urdf)
    print("t =", t, " m")
    print(
        "camera position in Link_Base:", np.round(cam_pos_base, 4), "  height =", round(float(cam_pos_base[2]), 4), "m"
    )
    print(
        "orthonormal:",
        float(np.abs(r_urdf @ r_urdf.T - np.eye(3)).max()) < 1e-9,
        " det =",
        float(np.linalg.det(r_urdf)),
    )
    for i, ax in enumerate("xyz"):
        print(f"  camera {ax} -> base {np.round(r_urdf.T[:, i], 4)}")

    ep_map = load_episode_map()
    source_episode = ep_map[args.episode_index]
    pts_base = load_fk(args.episode_index)
    print(f"\nepisode_index={args.episode_index} -> {source_episode}  frames={pts_base.shape[0]}")

    frame_idx = pick_frame(pts_base, r_urdf, t) if args.frame == "auto" else int(args.frame)
    print(f"frame = {frame_idx}")
    frame = read_frame(source_episode, frame_idx)
    pts = pts_base[frame_idx]
    print("hand keypoints (base frame) frame 0 wrist:", np.round(pts[0], 4))

    rolls = [float(x) for x in args.rolls.split(",")]
    tiles = []
    uv_by_roll = []
    cam_urdf = pts @ r_urdf.T + t  # base -> camera (URDF frame, +Y up)
    for deg in rolls:
        cam = cam_urdf @ roll_about_z(deg).T  # roll about the view axis
        uv, front = project(cam)
        uv_by_roll.append(uv)
        valid = front & in_image(uv)
        print(
            f"  roll {deg:6.1f} deg : in_front {int(front.sum()):2d}/21  in_image {int(valid.sum()):2d}/21"
            f"  uv_x [{uv[:, 0].min():7.1f},{uv[:, 0].max():7.1f}]"
            f"  uv_y [{uv[:, 1].min():7.1f},{uv[:, 1].max():7.1f}]"
        )
        tiles.append(draw_overlay(frame, uv, valid, f"roll {deg:.0f} deg", args.scale))

    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        rr, cc = divmod(i, cols)
        grid[rr * h : (rr + 1) * h, cc * w : (cc + 1) * w] = tile

    out_png = os.path.join(args.out, f"fk_projection_ep{args.episode_index}_frame{frame_idx}.png")
    cv2.imwrite(out_png, grid)
    print(f"\nwrote {out_png}")

    np.savez(
        os.path.join(args.out, f"uv_ep{args.episode_index}_frame{frame_idx}.npz"),
        uvs=np.stack(uv_by_roll),
        rolls=np.array(rolls),
        keypoint_names=np.array(KEYPOINT_NAMES),
    )


if __name__ == "__main__":
    main()
