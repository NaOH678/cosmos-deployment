#!/usr/bin/env python3
"""Overlay the FK hand *surface* — as yellow points — on a head video.

Deliberately minimal: the frame is the original video with nothing added except
the yellow dots marking where FK thinks the hand's surface is.  No skeleton, no
keypoints, no extra decoration (matching the optcentre_*.png reference images).

The points are sampled on the hand-link meshes and posed by MuJoCo, so what you
see is the actual hand geometry, not the 21 joint centres.  Only the surface
facing the camera is drawn (back-facing points are culled), so the blob reads as
the hand's silhouette.

Works on both the single-arm (singlerighthand_sandwich_100) and dual-arm
(dualhand_micropipette) recordings — both record the same 54-dim qpos layout.

Usage:
    python tools/render_hand_surface.py \
        --dataset dualhand_micropipette --episode episode_0007_20260824_174548 \
        --sides both --out /path/to/dir --video /path/to/out.mp4
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian")
MJLAB = ROOT / "wuji-mjlab"
URDF = MJLAB / "marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf"
RAW = ROOT / "raw_data"

for _p in (ROOT / "mjlib", Path("/tmp/mjlib"), Path(__file__).resolve().parent):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# D435 head camera, factory calibration (metadata.json)
COLOR_FX, COLOR_FY = 605.5706176757812, 604.4129638671875
COLOR_CX, COLOR_CY = 324.4994812011719, 238.25637817382812

YELLOW = (0, 255, 255)  # BGR — OpenCV's yellow


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def kinematics_only_urdf(urdf_path: Path) -> str:
    """MuJoCo rejects the shipped URDF's degenerate meshes; FK needs no geometry."""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    ext = root.find("mujoco")
    if ext is None:
        ext = ET.SubElement(root, "mujoco")
    compiler = ext.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(ext, "compiler")
    compiler.set("strippath", "false")
    compiler.set("discardvisual", "true")
    compiler.set("fusestatic", "false")
    compiler.set("boundmass", "1e-6")
    compiler.set("boundinertia", "1e-6")
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for geom in list(link.findall(kind)):
                link.remove(geom)
    for mesh in list(root.findall(".//mesh")):
        mesh.getparent().remove(mesh)
    return ET.tostring(root, encoding="unicode")


def hand_links(side: str) -> list[str]:
    return [f"{side}_hand_palm_link"] + [
        f"{side}_hand_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5)
    ] + [f"{side}_hand_finger{f}_tip_link" for f in range(1, 6)]


def load_surface(side: str, n_per_link: int, seed: int = 0):
    """link -> (points, normals) in that link's frame, with the visual origin folded in."""
    import trimesh
    import xml.etree.ElementTree as ET

    def rpy_to_R(rpy):
        r, p, y = rpy
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))

    rng = np.random.default_rng(seed)
    wanted = set(hand_links(side))
    origins, meshes = {}, {}
    for link in ET.parse(URDF).getroot().findall("link"):
        name = link.get("name")
        if name not in wanted:
            continue
        vis = link.find("visual")
        if vis is None:
            continue
        o = vis.find("origin")
        if o is not None:
            origins[name] = (np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]),
                             np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]))
        m = vis.find("geometry/mesh")
        if m is not None:
            meshes[name] = m.get("filename")

    pts_out, nrm_out = {}, {}
    for name, uri in meshes.items():
        rel = uri.split("package://", 1)[-1].split("/", 1)[1]
        path = URDF.parent.parent / rel
        if not path.is_file():
            continue
        mesh = trimesh.load(path, force="mesh")
        p, fidx = trimesh.sample.sample_surface(mesh, n_per_link, seed=int(rng.integers(1 << 30)))
        xyz, rpy = origins.get(name, (np.zeros(3), np.zeros(3)))
        R = rpy_to_R(rpy)
        pts_out[name] = np.asarray(p, float) @ R.T + xyz
        nrm_out[name] = np.asarray(mesh.face_normals)[fidx] @ R.T
    return pts_out, nrm_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dualhand_micropipette",
                    help="directory under raw_data/ (e.g. dualhand_micropipette, "
                         "singlerighthand_sandwich_100)")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--sides", default="both", choices=["left", "right", "both"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=None, help="write an mp4 of the whole episode")
    ap.add_argument("--frames", default=None, help="comma list for stills, e.g. 300,600,900")
    ap.add_argument("--n-per-link", type=int, default=200)
    ap.add_argument("--dot", type=int, default=2, help="dot radius in px")
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/fk_surface"))
    ap.add_argument("--opt-centre", default=None, metavar="DX,DY,DZ",
                    help="D435 optical centre offset (mm) in the camera link frame; "
                         "subtracted from t so the projection origin moves from the "
                         "link origin to the optical centre")
    ap.add_argument("--limit", type=int, default=0, help="only render the first N frames (0 = all)")
    args = ap.parse_args()

    try:
        import mujoco
    except ModuleNotFoundError:
        print(f"FATAL: mujoco not importable. Looked in {ROOT / 'mjlib'} and /tmp/mjlib", flush=True)
        return 1
    import cv2

    sys.path.insert(0, str(MJLAB / "scripts" / "replay"))
    replay = _load_module("replay_teleop", MJLAB / "scripts/replay/replay_teleop.py")
    from verify_fk_camera_projection import base_to_head_camera, roll_about_z

    ep_src = RAW / args.dataset / args.episode
    if not ep_src.is_dir():
        print(f"FATAL: no such episode: {ep_src}", flush=True)
        return 1
    video_path = ep_src / "videos" / "head.mp4"
    if not video_path.is_file():
        print(f"FATAL: no head video at {video_path}", flush=True)
        return 1

    # GPFS cannot serve lmdb mmap — work from a local copy.
    ep = args.scratch / args.episode
    if not (ep / "lmdb").is_dir():
        ep.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ep_src / "lmdb", ep / "lmdb", dirs_exist_ok=True)
        for extra in ("meta_info.pkl", "sync_timestamps.json"):
            if (ep_src / extra).is_file():
                shutil.copy2(ep_src / extra, ep / extra)
        print(f"copied episode to {ep}", flush=True)

    traj, meta = replay.load_episode(ep)
    model = mujoco.MjModel.from_xml_string(kinematics_only_urdf(URDF))
    data = mujoco.MjData(model)
    addresses, _lo, _hi = replay.model_joint_map(model, replay.dataset_joint_names(meta))
    print(f"{args.episode}: {traj.shape[0]} frames, {len(addresses)} joints", flush=True)

    sides = ["left", "right"] if args.sides == "both" else [args.sides]
    surfs, norms = {}, {}
    for side in sides:
        surfs[side], norms[side] = load_surface(side, args.n_per_link)
        print(f"  {side}: {len(surfs[side])} links, "
              f"{sum(len(v) for v in surfs[side].values())} sample points", flush=True)

    R, t = base_to_head_camera(URDF)
    M = roll_about_z(180.0)
    R, t = M @ R, M @ t
    label = "nominal"
    if args.opt_centre:
        oc = np.array([float(x) for x in args.opt_centre.split(",")]) / 1000.0
        t = t - oc
        label = f"optcentre {args.opt_centre} mm"
    print(f"camera pose: {label}   position(Link_Base) = {np.round(-R.T @ t, 4)}", flush=True)

    def surface_uv(f):
        """Front-facing FK hand surface points for frame f, projected to pixels."""
        data.qpos[addresses] = traj[f]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        us, vs = [], []
        for side in sides:
            P, N = [], []
            for name in surfs[side]:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    continue
                Rl = data.xmat[bid].reshape(3, 3)
                tl = data.xpos[bid]
                P.append(surfs[side][name] @ Rl.T + tl)
                N.append(norms[side][name] @ Rl.T)
            if not P:
                continue
            P = np.concatenate(P) @ R.T + t
            N = np.concatenate(N) @ R.T
            view = P / np.linalg.norm(P, axis=1, keepdims=True)
            front = (N * view).sum(1) < 0            # keep only the camera-facing side
            P = P[front]
            ok = P[:, 2] > 1e-6
            P = P[ok]
            if not len(P):
                continue
            us.append(COLOR_FX * P[:, 0] / P[:, 2] + COLOR_CX)
            vs.append(COLOR_FY * P[:, 1] / P[:, 2] + COLOR_CY)
        if not us:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(us), np.concatenate(vs)

    def stamp(img, u, v, f):
        inside = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
        for uu, vv in zip(u[inside], v[inside]):
            cv2.circle(img, (int(uu), int(vv)), args.dot, YELLOW, -1)
        cv2.putText(img, f"f{f}  {args.episode}  [{label}]  {int(inside.sum())} pts",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(n_video, traj.shape[0])
    if args.limit:
        n_use = min(n_use, args.limit)

    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
        if not writer.isOpened():
            print(f"FATAL: cannot open writer for {args.video}", flush=True)
            return 1
        t0 = time.time()
        for f in range(n_use):
            ok, img = cap.read()
            if not ok:
                break
            stamp(img, *surface_uv(f), f)
            writer.write(img)
            if f % 300 == 0:
                el = time.time() - t0
                print(f"  frame {f:5d}/{n_use}  elapsed {el:.0f}s  "
                      f"eta {el / max(f, 1) * (n_use - f):.0f}s", flush=True)
        writer.release()
        print(f"wrote {args.video}", flush=True)

    if args.frames:
        tiles = []
        for f in [int(x) for x in args.frames.split(",") if x.strip()]:
            if f >= n_use:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            stamp(img, *surface_uv(f), f)
            path = args.out / f"hand_surface_f{f:04d}.png"
            cv2.imwrite(str(path), img)
            tiles.append(img)
            print(f"  still f{f} -> {path}", flush=True)
        if tiles:
            grid = np.vstack(tiles)
            cv2.imwrite(str(args.out / "hand_surface_grid.png"), grid)

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
