#!/usr/bin/env python3
"""Render FK-21 skeletons for BOTH hands of a dual-arm episode onto its head video.

The dualhand_micropipette dataset has no pre-computed FK annotation, so the
keypoints are computed here from the recorded qpos with the same MuJoCo model and
the same 21-point definition the single-hand exporter uses.

Camera pose: the URDF's head-camera chain plus roll_about_z(180).  Note this is
the *nominal* pose — it is known to be off by a few centimetres (see
docs/notes on the extrinsic), so treat the overlay as indicative, not exact.

Self-configures sys.path (mujoco on shared storage, falling back to /tmp) so it
does not depend on the caller's PYTHONPATH or on which worker node it lands on.

Usage:
    python tools/render_dualhand_fk.py --episode episode_0007_20260824_174548 \
        --out /path/to/dir --frames 100,400,700,1000
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
RAW_DUAL = ROOT / "raw_data/dualhand_micropipette"

for _p in (ROOT / "mjlib", Path("/tmp/mjlib"), Path(__file__).resolve().parent):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# D435 head camera, factory calibration (metadata.json)
COLOR_FX, COLOR_FY = 605.5706176757812, 604.4129638671875
COLOR_CX, COLOR_CY = 324.4994812011719, 238.25637817382812

# Left hand in cool tones, right hand in warm tones, wrist white/grey.
LEFT_COLORS = [(255, 255, 255), (255, 128, 0), (255, 200, 0), (200, 255, 0), (128, 255, 0), (0, 255, 128)]
RIGHT_COLORS = [(255, 255, 255), (0, 0, 255), (0, 128, 255), (0, 200, 255), (0, 255, 255), (128, 0, 255)]


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def kinematics_only_urdf(urdf_path: Path) -> str:
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


def hand_edges() -> list[tuple[int, int]]:
    """wrist -> each MCP, then along each finger (same topology as the single-hand tool)."""
    edges = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
    for base in (1, 5, 9, 13, 17):
        edges += [(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)]
    return edges


def finger_of(k: int) -> int:
    return 0 if k == 0 else (k - 1) // 4 + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="episode_0007_20260824_174548")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default="100,400,700,1000,1500,1900")
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/fk_dual"))
    ap.add_argument("--scale", type=int, default=1, help="upscale factor for the saved PNGs")
    ap.add_argument("--video", type=Path, default=None,
                    help="also write an mp4 of the whole episode here")
    ap.add_argument("--opt-centre", default=None, metavar="DX,DY,DZ",
                    help="D435 optical centre offset (mm) in the camera link frame, "
                         "e.g. '-17.5,0,-4.2'.  Subtracted from t, i.e. the projection "
                         "origin moves from the link origin to the optical centre.")
    ap.add_argument("--no-stills", action="store_true", help="skip the per-frame PNGs")
    args = ap.parse_args()

    try:
        import mujoco
    except ModuleNotFoundError:
        print("FATAL: mujoco not importable. Looked in "
              f"{ROOT / 'mjlib'} and /tmp/mjlib", flush=True)
        return 1
    import cv2

    import xml.etree.ElementTree as ET
    sys.path.insert(0, str(MJLAB / "scripts" / "replay"))
    replay = _load_module("replay_teleop", MJLAB / "scripts/replay/replay_teleop.py")
    exporter = _load_module("export_wuji_fk21", MJLAB / "scripts/replay/export_wuji_fk21.py")

    args.out.mkdir(parents=True, exist_ok=True)

    ep_src = RAW_DUAL / args.episode
    if not ep_src.is_dir():
        print(f"FATAL: no such episode: {ep_src}", flush=True)
        return 1

    # GPFS cannot serve lmdb mmap, so work from a local copy.
    ep = args.scratch / args.episode
    if not (ep / "lmdb").is_dir():
        ep.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ep_src / "lmdb", ep / "lmdb", dirs_exist_ok=True)
        for extra in ("meta_info.pkl", "sync_timestamps.json"):
            if (ep_src / extra).is_file():
                shutil.copy2(ep_src / extra, ep / extra)
        print(f"copied episode to {ep}", flush=True)

    video = ep_src / "videos" / "head.mp4"
    traj, meta = replay.load_episode(ep)
    print(f"episode {args.episode}: {traj.shape[0]} frames, qpos dim {traj.shape[1]}", flush=True)
    print(f"active sides: arms={meta.get('active_arm_sides')} hands={meta.get('active_hand_sides')}", flush=True)

    model = mujoco.MjModel.from_xml_string(kinematics_only_urdf(URDF))
    data = mujoco.MjData(model)
    names = replay.dataset_joint_names(meta)
    addresses, _lo, _hi = replay.model_joint_map(model, names)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_Base")
    print(f"mujoco: {len(names)} joints mapped, Link_Base id={base_body_id}", flush=True)

    # nominal head-camera pose: URDF chain + the view-axis roll
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_fk_camera_projection import base_to_head_camera, roll_about_z

    R, t = base_to_head_camera(URDF)
    M = roll_about_z(180.0)
    R, t = M @ R, M @ t
    print(f"nominal camera position (Link_Base) = {np.round(-R.T @ t, 4)}", flush=True)

    if args.opt_centre:
        # The URDF declares *_optical_frame as a zero-offset alias of the camera
        # body link, so the projection origin sits at the link origin.  Moving it
        # to the real optical centre subtracts that offset from the translation.
        oc = np.array([float(x) for x in args.opt_centre.split(",")]) / 1000.0
        t = t - oc
        print(f"optical centre applied: {np.round(oc * 1000, 2)} mm  "
              f"-> camera position now {np.round(-R.T @ t, 4)}", flush=True)

    edges = hand_edges()
    cap = cv2.VideoCapture(str(video))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(n_video, traj.shape[0])

    def draw_frame(img, f):
        """Pose the FK for frame f, draw both hands onto img, return the visible count."""
        data.qpos[addresses] = traj[f]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        n_drawn = 0
        for side, palette in (("left", LEFT_COLORS), ("right", RIGHT_COLORS)):
            sources = exporter.fk21_sources(model, side)
            pts = exporter.positions_in_base(model, data, base_body_id, sources)
            cam = pts @ R.T + t
            if cam[:, 2].min() <= 1e-6:
                continue
            u = COLOR_FX * cam[:, 0] / cam[:, 2] + COLOR_CX
            v = COLOR_FY * cam[:, 1] / cam[:, 2] + COLOR_CY
            vis = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
            n_drawn += int(vis.sum())
            for a, b in edges:
                if vis[a] and vis[b]:
                    cv2.line(img, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), palette[finger_of(a)], 2)
            for k in range(21):
                if vis[k]:
                    cv2.circle(img, (int(u[k]), int(v[k])), 4, palette[finger_of(k)], -1)
            if vis[0]:  # label the wrist so the two hands are told apart
                cv2.putText(img, side[0].upper(), (int(u[0]) + 8, int(v[0]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, palette[1], 2)
        return n_drawn

    # ---- full-episode video -------------------------------------------------
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
        if not writer.isOpened():
            print(f"FATAL: cannot open video writer for {args.video}", flush=True)
            return 1
        t0 = time.time()
        for f in range(n_use):
            ok, img = cap.read()
            if not ok:
                break
            n_drawn = draw_frame(img, f)
            cv2.putText(img, f"{args.episode}  frame {f}   left=cool  right=warm", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(img, f"{n_drawn}/42 keypoints in frame", (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            writer.write(img)
            if f % 200 == 0:
                el = time.time() - t0
                print(f"  frame {f:5d}/{n_use}  elapsed {el:.0f}s  "
                      f"eta {el / max(f, 1) * (n_use - f):.0f}s", flush=True)
        writer.release()
        print(f"wrote {args.video}", flush=True)

    # ---- selected stills ----------------------------------------------------
    if not args.no_stills:
        wanted = [int(x) for x in args.frames.split(",") if x.strip()]
        tiles = []
        for f in wanted:
            if f >= n_use:
                print(f"  frame {f}: beyond episode length, skipped", flush=True)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                print(f"  frame {f}: cannot read video", flush=True)
                continue
            n_drawn = draw_frame(img, f)
            cv2.putText(img, f"{args.episode}  frame {f}   left=cool  right=warm", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(img, f"{n_drawn}/42 keypoints inside frame", (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            print(f"  frame {f:5d}: {n_drawn}/42 keypoints in frame", flush=True)
            if args.scale > 1:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(args.out / f"dualhand_fk_f{f:04d}.png"), img)
            tiles.append(img)
        if tiles:
            h = min(t.shape[0] for t in tiles)
            w = min(t.shape[1] for t in tiles)
            grid = np.vstack([np.hstack([t[:h, :w] for t in tiles[i:i + 2]]) for i in range(0, len(tiles), 2)])
            cv2.imwrite(str(args.out / "dualhand_fk_grid.png"), grid)
            print(f"wrote {len(tiles)} stills + grid to {args.out}", flush=True)

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
