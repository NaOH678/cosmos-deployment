"""Decisive extrinsics check using the D435's OWN depth, not the depth-model point cloud.

The PointFlow `position.npy` is produced by a monocular depth model that assumed
its own camera intrinsics (fitted: f=568, cx=320, cy=224 on a 640x448 grid),
which differ from the real D435 colour intrinsics.  Comparing FK against it is
therefore comparing against a distorted frame.

The D435 depth stream is real sensor data with known intrinsics and a known
depth->colour transform.  Back-projecting it gives a colour-frame point cloud
that is independent of any learned model.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_fk_camera_projection import (  # noqa: E402
    IMG_H,
    IMG_W,
    URDF,
    base_to_head_camera,
    load_episode_map,
    load_fk,
    read_frame,
    roll_about_z,
)

RAW_ROOT = (
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/"
    "raw_data/singlerighthand_sandwich_100"
)


DEPTH_DUMP = "/tmp/depth_ep0_npy"


def load_depth_dumps() -> dict[int, np.ndarray]:
    """Read depth frames pre-extracted from the LMDB (see module docstring).

    The venv has no `lmdb` and GPFS has no mmap, so frames are dumped with the
    system interpreter first:
        /usr/bin/python3 -c "<see notes>"   -> /tmp/depth_ep0_npy/depth_XXXXXX.npy
    """
    if not os.path.isdir(DEPTH_DUMP):
        raise SystemExit(
            f"{DEPTH_DUMP} missing. Dump the depth frames first with the system python."
        )
    out = {}
    for f in sorted(os.listdir(DEPTH_DUMP)):
        if f.startswith("depth_") and f.endswith(".npy"):
            out[int(f[6:12])] = np.load(os.path.join(DEPTH_DUMP, f)).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--frames", default="500,1100,1400")
    ap.add_argument("--out", default="/tmp/fk_vis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ep_map = load_episode_map()
    episode = ep_map[args.episode_index]
    ep_dir = os.path.join(RAW_ROOT, episode, "auxiliary_camera")

    meta = json.load(open(os.path.join(ep_dir, "metadata.json")))
    head = meta["capture_metadata"]["cameras"]["head"]["streams"]
    dk = head["depth"]["intrinsics"]
    d_scale = head["depth"]["depth_scale_m"]
    e2c = head["depth"]["extrinsics_to_color"]
    r_d2c = np.array(e2c["rotation"]).reshape(3, 3)
    t_d2c = np.array(e2c["translation"])
    ck = head["color"]["intrinsics"]
    print("D435 depth intrinsics :", {k: round(dk[k], 3) for k in ("fx", "fy", "ppx", "ppy")},
          f"{dk['width']}x{dk['height']}")
    print("D435 colour intrinsics:", {k: round(ck[k], 3) for k in ("fx", "fy", "ppx", "ppy")},
          f"{ck['width']}x{ck['height']}")
    print("depth->colour t =", np.round(t_d2c, 5), "m")

    dumps = load_depth_dumps()
    keys = sorted(dumps)
    print(f"\ndepth frames available: {len(keys)}  (range {keys[0]}..{keys[-1]})")

    fk = load_fk(args.episode_index)
    r_urdf, t_urdf = base_to_head_camera(URDF)
    r_cam = roll_about_z(180.0) @ r_urdf
    t_cam = roll_about_z(180.0) @ t_urdf

    fx, fy = ck["fx"], ck["fy"]
    cx, cy = ck["ppx"], ck["ppy"]

    yy, xx = np.mgrid[0:dk["height"], 0:dk["width"]]
    tiles = []
    print(f"\n{'video frame':>11} {'depth key':>16} {'N valid':>9} {'FK->cloud 中位':>14} {'p90':>8}  (cm)")
    for vf in [int(x) for x in args.frames.split(",")]:
        dkey = min(keys, key=lambda k: abs(k - vf))
        depth = dumps[dkey] * d_scale
        m = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
        z = depth[m]
        pts_d = np.stack(
            [(xx[m] - dk["ppx"]) / dk["fx"] * z, (yy[m] - dk["ppy"]) / dk["fy"] * z, z], -1
        )
        cloud = pts_d @ r_d2c.T + t_d2c  # depth frame -> colour frame
        # keep only points inside the colour image
        zc = cloud[:, 2]
        u = fx * cloud[:, 0] / zc + cx
        v = fy * cloud[:, 1] / zc + cy
        vis = (zc > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        cloud, u, v = cloud[vis], u[vis], v[vis]

        cam = (fk[dkey] if dkey < len(fk) else fk[vf]) @ r_cam.T + t_cam
        d = np.linalg.norm(cloud[None] - cam[:, None], axis=-1).min(1)
        print(f"{vf:11d} {dkey:16d} {len(cloud):9d} "
              f"{np.median(d)*100:14.2f} {np.percentile(d,90)*100:8.2f}")

        # render: real depth cloud reprojected + FK skeleton
        fr = read_frame(episode, vf)
        S = 2
        canvas = cv2.resize(fr, (IMG_W * S, IMG_H * S), interpolation=cv2.INTER_LINEAR)
        step = max(1, len(cloud) // 25000)
        for uu, vv in zip(u[::step], v[::step]):
            cv2.circle(canvas, (int(uu * S), int(vv * S)), 1, (0, 255, 255), -1)
        from verify_fk_camera_projection import EDGES, FINGER_COLORS, finger_of

        zz = cam[:, 2]
        ok = zz > 0
        uvk = np.stack([fx * cam[:, 0] / np.where(ok, zz, 1) + cx,
                        fy * cam[:, 1] / np.where(ok, zz, 1) + cy], -1)
        good = ok & (uvk[:, 0] >= 0) & (uvk[:, 0] < IMG_W) & (uvk[:, 1] >= 0) & (uvk[:, 1] < IMG_H)
        for a, b in EDGES:
            if good[a] and good[b]:
                p1 = (int(uvk[a, 0] * S), int(uvk[a, 1] * S))
                p2 = (int(uvk[b, 0] * S), int(uvk[b, 1] * S))
                cv2.line(canvas, p1, p2, (0, 0, 0), 5 * S, cv2.LINE_AA)
                cv2.line(canvas, p1, p2, FINGER_COLORS[finger_of(b)], 2 * S, cv2.LINE_AA)
        for k in range(21):
            if good[k]:
                p = (int(uvk[k, 0] * S), int(uvk[k, 1] * S))
                cv2.circle(canvas, p, 4 * S, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, p, 3 * S, FINGER_COLORS[finger_of(k)], -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (IMG_W * S, 26 * S), (0, 0, 0), -1)
        cv2.putText(canvas, f"frame {vf}  yellow = REAL D435 depth  |  skeleton = FK-21 (URDF)",
                    (8, 19 * S), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * S, (255, 255, 255), 1 * S, cv2.LINE_AA)
        tiles.append(canvas)


    if tiles:
        out = os.path.join(args.out, f"fk_vs_REAL_depth_ep{args.episode_index:03d}.png")
        cv2.imwrite(out, np.vstack(tiles))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
