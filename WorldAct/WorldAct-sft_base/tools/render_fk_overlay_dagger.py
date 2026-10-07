#!/usr/bin/env python3

# ---------------------------------------------------------------------------
# MIGRATED FROM THE mano WORKTREE -- PREMISE SUPERSEDED, READ BEFORE USING.
#
# This script assumes the dagger rig shares the sandwich camera extrinsic because the
# head unit is the same (serial 147122073219).  That assumption did not hold: rendering
# dagger with the sandwich extrinsic puts the skeleton ~200 px low, and the bracket is
# in fact different.  The supported route is tools/build_dagger_camera_urdf.py, which
# solves the mount by matching installation holes rather than fitting video pixels;
# regenerate the extrinsic from that URDF (tools/export_fk_camera_extrinsic.py) first.
# Kept because it is the only dagger overlay renderer in this repo, not because its
# camera model is right.
# ---------------------------------------------------------------------------
"""FK-21 overlay on the dagger episodes' real head video.

Same projection as tools/render_fk_overlay.py -- the camera is the same unit
(serial 147122073219, identical fx/fy/ppx/ppy), so the extrinsics and intrinsics
carry over unchanged.  What differs is only where the data lives and how the FK
file is laid out: dagger keeps one wuji_fk21.npz per episode under a SEPARATE
tree, holding both hands, where sandwich's slim pack had one npz per index
already reduced to the right hand.

    PYTHONPATH=. <venv>/bin/python tools/render_fk_overlay_dagger.py \
        --episode episode_0000_20260830_155455 --out /data/shichaojian/renders/fk_dagger
    PYTHONPATH=. <venv>/bin/python tools/render_fk_overlay_dagger.py --list
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
    roll_about_z,
)

RAW_ROOT = "/data/shichaojian/raw_data/dropper_dagger"
FK_ROOT = "/data/shichaojian/raw_data/dropper_dagger_mix_fk21"
RIGHT_SIDE = "right"  # side_names is ['left', 'right']


def episodes() -> list[str]:
    """The episodes that have BOTH the video and the annotations.

    Not every annotated episode has raw video: the fk21 tree is a superset
    (71 vs 51), and the extra ones are the 20260903 batch.  Intersecting here
    means --list shows exactly what can be rendered.
    """
    if not os.path.isdir(RAW_ROOT) or not os.path.isdir(FK_ROOT):
        raise SystemExit(f"missing data root: {RAW_ROOT} / {FK_ROOT}")
    raw = {d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))}
    fk = {d for d in os.listdir(FK_ROOT) if os.path.isdir(os.path.join(FK_ROOT, d))}
    both = sorted(n for n in raw & fk if os.path.isfile(os.path.join(RAW_ROOT, n, "videos", "head.mp4")))
    if not both:
        raise SystemExit("no episode has both videos/head.mp4 and an FK annotation")
    return both


def load_fk(name: str) -> np.ndarray:
    """``[T, 21, 3]`` base-frame metres for the RIGHT hand.

    ``positions`` is [T, 2, 21, 3] -- both sides in one array.  Indexing the
    wrong side yields the left hand, which projects to a plausible-looking
    skeleton somewhere it does not belong; that is the first thing to suspect
    if the overlay is badly off (see the doc's troubleshooting ladder).
    """
    path = os.path.join(FK_ROOT, name, "annotations", "wuji_fk21.npz")
    with np.load(path, allow_pickle=True) as d:
        positions = d["positions"]
        sides = [str(s) for s in d["side_names"]]
        if sides.count(RIGHT_SIDE) != 1:
            raise ValueError(f"{name}: side_names={sides} has no unique {RIGHT_SIDE!r}")
        return positions[:, sides.index(RIGHT_SIDE)].astype(np.float64)


def project_episode(pts_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised base -> camera -> pixel.  Returns (uv[T,21,2], in_front[T,21]).

    Identical to render_fk_overlay.project_episode; repeated rather than imported
    so this file stands alone and cannot drift from it silently.
    """
    r, t = base_to_head_camera(URDF)
    r = roll_about_z(180.0) @ r
    t = roll_about_z(180.0) @ t
    cam = pts_all @ r.T + t
    z = cam[..., 2]
    in_front = z > 1e-6
    zs = np.where(in_front, z, 1.0)
    u = 605.5706176757812 * cam[..., 0] / zs + 324.4994812011719
    v = 604.4129638671875 * cam[..., 1] / zs + 238.25637817382812
    return np.stack([u, v], axis=-1), in_front


def draw(frame, uv, valid, scale, label):
    canvas = cv2.resize(frame, (IMG_W * scale, IMG_H * scale), interpolation=cv2.INTER_LINEAR)

    def pt(k):
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
            x = int(np.clip(uv[k, 0] * scale, 6, IMG_W * scale - 6))
            y = int(np.clip(uv[k, 1] * scale, 6, IMG_H * scale - 6))
            cv2.drawMarker(canvas, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 7 * scale, 2 * scale)
    cv2.rectangle(canvas, (0, 0), (IMG_W * scale, 26 * scale), (0, 0, 0), -1)
    cv2.putText(canvas, label, (6 * scale, 19 * scale),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), 1 * scale, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episode", help="an episode directory name; default: the first")
    ap.add_argument("--list", action="store_true", help="print the renderable episodes and exit")
    ap.add_argument("--all", action="store_true", help="render every renderable episode")
    ap.add_argument("--out", default="/data/shichaojian/renders/fk_dagger")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=float, default=30.0, help="source is 30 Hz; 15 gives half-speed playback")
    ap.add_argument("--stills", default="100,500,1000,1500,2000",
                    help="frame indices for the contact sheet (default keeps them inside the shortest episode)")
    args = ap.parse_args()

    names = episodes()
    if args.list:
        print(f"{len(names)} renderable episodes (raw videos AND FK annotation):")
        for n in names:
            print(f"  {n}")
        return 0

    if args.all:
        todo = names
    elif args.episode:
        if args.episode not in names:
            raise SystemExit(f"{args.episode!r} is not renderable; --list to see what is")
        todo = [args.episode]
    else:
        todo = names[:1]

    os.makedirs(args.out, exist_ok=True)
    stills_wanted = {int(x) for x in args.stills.split(",") if x.strip()}

    for name in todo:
        video_path = os.path.join(RAW_ROOT, name, "videos", "head.mp4")
        pts_all = load_fk(name)
        uv_all, front_all = project_episode(pts_all)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  {name}: cannot open {video_path}; skipped")
            continue
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # The two are 1:1 in every episode checked (2251 == 2251).  A mismatch is
        # not fatal -- it is truncated to the shorter -- but it means the frame
        # correspondence assumption is wrong for this episode, and every skeleton
        # after the divergence point would sit on the wrong pose.  Say so.
        if n_video != len(pts_all):
            print(f"  {name}: WARNING video has {n_video} frames, FK has {len(pts_all)} "
                  f"-- assuming index i == frame i and truncating to the shorter")
        print(f"  {name}: {n_video} video frames, {len(pts_all)} FK frames")

        out_mp4 = os.path.join(args.out, f"fk_overlay_{name}.mp4")
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                 (IMG_W * args.scale, IMG_H * args.scale))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open {out_mp4}")

        stills, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(uv_all):
                break
            uv, front = uv_all[i], front_all[i]
            valid = (front & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W)
                     & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H))
            canvas = draw(frame, uv, valid, args.scale,
                          f"{name[:26]}  frame {i:4d}   FK-21 right (base->D435 + Rz180)")
            writer.write(canvas)
            if i in stills_wanted:
                stills.append(canvas)
            i += 1
        cap.release()
        writer.release()
        print(f"    wrote {out_mp4}  ({i} frames)")

        if stills:
            cols = 2
            rows = (len(stills) + cols - 1) // cols
            h, w = stills[0].shape[:2]
            sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
            for k, s in enumerate(stills):
                rr, cc = divmod(k, cols)
                sheet[rr * h:(rr + 1) * h, cc * w:(cc + 1) * w] = s
            sheet_path = os.path.join(args.out, f"fk_overlay_{name}_stills.png")
            cv2.imwrite(sheet_path, sheet)
            print(f"    wrote {sheet_path}  ({len(stills)} stills)")

    print("\nCHECK FIRST: is the skeleton on the HAND in the first still?")
    print("If it is not, the projection is wrong and nothing else in the frame means")
    print("anything. See the troubleshooting ladder in docs/fk_overlay_dagger_howto.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
