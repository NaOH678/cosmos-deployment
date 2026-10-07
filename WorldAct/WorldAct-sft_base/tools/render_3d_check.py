#!/usr/bin/env python3
"""Render the scene reconstruction and the FK hand together in 3D.

Why: every 2D overlay goes through "camera-frame xyz -> pixel", and that step is
one of the suspects.  Viewing both in 3D removes it — the virtual camera used
here is ours, not the D435's, so nothing about the projection can hide a 3D
mismatch, and nothing about the projection can fake an alignment either.

What is drawn, all in the D435 colour frame:
  * scene  — the D435 depth back-projected to 3D, coloured by the video pixel
  * hand   — the FK hand meshes posed by MuJoCo and pushed through the extrinsic
  * joints — the 21 FK keypoints

If the hand surface lands on the scene's hand, the 3D is right and the residual
lives in the 2D projection.  If it sits off the hand in 3D, the extrinsic is the
problem and no projection tweak will help.

Usage:
    python tools/render_3d_check.py --episode episode_0013_20260731_133649 \
        --out /path/to/dir --frames 300,600,900
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
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

# D435 head camera, factory calibration
COLOR_FX, COLOR_FY = 605.5706176757812, 604.4129638671875
COLOR_CX, COLOR_CY = 324.4994812011719, 238.25637817382812
DEPTH_FX = DEPTH_FY = 384.1614990234375
DEPTH_CX, DEPTH_CY = 320.2146301269531, 235.62696838378906
DEPTH_R = np.array([
    0.9999253749847412, -0.011212949641048908, -0.0048545487225055695,
    0.011222291737794876, 0.9999352097511292, 0.0019013523124158382,
    0.004832914564758539, -0.0019556896295398474, 0.9999864101409912,
]).reshape(3, 3)
DEPTH_T = np.array([0.014961255714297295, 5.274948853184469e-05, 2.5462672056164593e-05])


def _load_module(name, path):
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
    for k, v in (("strippath", "false"), ("discardvisual", "true"), ("fusestatic", "false"),
                 ("boundmass", "1e-6"), ("boundinertia", "1e-6")):
        compiler.set(k, v)
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for geom in list(link.findall(kind)):
                link.remove(geom)
    for mesh in list(root.findall(".//mesh")):
        mesh.getparent().remove(mesh)
    return ET.tostring(root, encoding="unicode")


# Virtual cameras, in the D435 colour frame: +X right, +Y down, +Z forward (away
# from the camera).  `cam` puts the eye back on the optical axis, so what it shows
# is the same left/right relationship the 2D overlay shows — if the yellow mesh sits
# left of the black glove here, it also sits left in the image.  `top` looks down and
# therefore shows that same left-right shift with nothing else mixed in.
# Each entry is (azimuth, elevation, up).  `up` matters: looking straight down with
# the default up=(0,-1,0) is degenerate, and the fallback picks +Z as screen-right —
# so the "top" view would silently show depth along the horizontal axis instead of the
# left-right shift it claims to show.  up=(0,0,-1) puts +X (image right) on screen-right.
VIEWS = {
    "cam": (0.0, 8.0, (0, -1, 0)),
    "left": (-78.0, 12.0, (0, -1, 0)),
    "right": (78.0, 12.0, (0, -1, 0)),
    "top": (0.0, 84.0, (0, 0, -1)),
}
VIEW_LABEL = {
    "cam": "along the D435 axis (same left/right as the image)",
    "left": "side view",
    "right": "side view",
    "top": "looking down - a left-right shift shows directly",
}

HAND_LINKS = ["right_hand_palm_link"] + [
    f"right_hand_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5)
] + [f"right_hand_finger{f}_tip_link" for f in range(1, 6)]


def load_hand_surface(n_per_link: int, seed: int = 0):
    import trimesh
    import xml.etree.ElementTree as ET

    def rpy_to_R(rpy):
        r, p, y = rpy
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))

    rng = np.random.default_rng(seed)
    origins, meshes = {}, {}
    for link in ET.parse(URDF).getroot().findall("link"):
        name = link.get("name")
        if name not in HAND_LINKS:
            continue
        vis = link.find("visual")
        if vis is None:
            continue
        o = vis.find("origin")
        origins[name] = (np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]),
                         np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()])) \
            if o is not None else (np.zeros(3), np.zeros(3))
        m = vis.find("geometry/mesh")
        if m is not None:
            meshes[name] = m.get("filename")

    pts_out = {}
    for name, uri in meshes.items():
        rel = uri.split("package://", 1)[-1].split("/", 1)[1]
        path = URDF.parent.parent / rel
        if not path.is_file():
            continue
        mesh = trimesh.load(path, force="mesh")
        p, _ = trimesh.sample.sample_surface(mesh, n_per_link, seed=int(rng.integers(1 << 30)))
        xyz, rpy = origins[name]
        pts_out[name] = np.asarray(p, float) @ rpy_to_R(rpy).T + xyz
    return pts_out


# --------------------------------------------------------------------------
# a tiny software point renderer
# --------------------------------------------------------------------------
def look_at(eye, target, up=(0, -1, 0)):
    """Camera-to-world (R, t) with the OpenCV convention: +x right, +y down, +z forward."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    r = np.cross(up, f)
    n = np.linalg.norm(r)
    if n < 1e-9:                       # looking straight along `up`
        r = np.cross((1.0, 0.0, 0.0), f)
        n = np.linalg.norm(r)
    r /= n
    d = np.cross(f, r)
    R = np.stack([r, d, f])            # world->camera rotation
    return R, -R @ eye


def render(layers, R, t, W, H, fx, background=(105, 105, 110)):
    """Splat several point layers through one shared z-buffer.

    background is mid-grey on purpose: the real hand is a *black* glove, so on a dark
    background its points vanish into the void and the picture silently loses the very
    thing being compared against.

    Each layer is (points, colours, radius, ztest).  With ztest the layer competes for
    depth; without it the layer is painted over the result, far points first.  The FK
    mesh is drawn *without* ztest and sparsely (see --mesh-step): the hand mesh is a
    closed surface, so depth-testing it would simply swallow the glove it is meant to
    be compared against and the picture would always look aligned.
    """
    img = np.full((H, W, 3), background, np.uint8)
    zbuf = np.full((H, W), np.inf, np.float32)

    def splat(points, colors, rad, ztest):
        points = np.asarray(points, float)
        cam = points @ R.T + t
        z = cam[:, 2]
        ok = z > 0.05
        pts, zs, cols = cam[ok], z[ok], np.asarray(colors, np.uint8)[ok]
        if not len(pts):
            return
        u = np.rint(fx * pts[:, 0] / zs + W * 0.5).astype(np.int32)
        v = np.rint(fx * pts[:, 1] / zs + H * 0.5).astype(np.int32)
        for i in np.argsort(-zs):       # far -> near
            x, y, zi = int(u[i]), int(v[i]), zs[i]
            x0, x1 = max(0, x - rad), min(W, x + rad + 1)
            y0, y1 = max(0, y - rad), min(H, y + rad + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            hit = (xx - x) ** 2 + (yy - y) ** 2 <= rad * rad
            if ztest:
                hit &= zi < zbuf[y0:y1, x0:x1]
                zbuf[y0:y1, x0:x1][hit] = zi
            img[y0:y1, x0:x1][hit] = cols[i]

    for ztest in (True, False):
        for points, colors, rad, want_z in layers:
            if want_z == ztest:
                splat(points, colors, rad, ztest)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="singlerighthand_sandwich_100")
    ap.add_argument("--episode", default="episode_0013_20260731_133649")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default="300,600,900")
    ap.add_argument("--n-per-link", type=int, default=120)
    ap.add_argument("--scene-step", type=int, default=2, help="depth pixel stride for the scene cloud")
    ap.add_argument("--sphere-m", type=float, default=0.35, help="scene points kept around the hand")
    ap.add_argument("--metric-sphere-m", type=float, default=0.12,
                    help="radius for the numeric comparison only.  Kept tight on purpose: "
                         "the black plate is dark like the glove, and a wide radius lets it "
                         "into the measurement")
    ap.add_argument("--radius", type=float, default=0.18, help="virtual camera distance from the hand (m)")
    ap.add_argument("--tile", type=int, default=760, help="pixels per view")
    ap.add_argument("--mesh-step", type=int, default=6,
                    help="draw every Nth FK mesh point; the mesh is a closed surface, so a "
                         "dense draw hides the very glove it is being compared against")
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/fk_3d"))
    ap.add_argument("--views", default="cam,left,right,top",
                    help="strip layout; each is one of cam (the D435's own view axis), "
                         "left/right (side-on, shows depth error), top (looks down, shows "
                         "a left-right shift most directly)")
    args = ap.parse_args()

    import cv2
    import lmdb
    import mujoco

    sys.path.insert(0, str(MJLAB / "scripts" / "replay"))
    replay = _load_module("replay_teleop", MJLAB / "scripts" / "replay" / "replay_teleop.py")
    exporter = _load_module("export_wuji_fk21", MJLAB / "scripts" / "replay" / "export_wuji_fk21.py")
    from verify_fk_camera_projection import base_to_head_camera, roll_about_z

    ep_src = RAW / args.dataset / args.episode
    if not ep_src.is_dir():
        print(f"FATAL: no such episode: {ep_src}", flush=True)
        return 1

    ep = args.scratch / args.episode
    if not (ep / "lmdb").is_dir():
        ep.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ep_src / "lmdb", ep / "lmdb", dirs_exist_ok=True)
        for extra in ("meta_info.pkl", "sync_timestamps.json"):
            if (ep_src / extra).is_file():
                shutil.copy2(ep_src / extra, ep / extra)
    depth_local = args.scratch / "depth.lmdb"
    if not (depth_local / "data.mdb").is_file():
        shutil.copytree(ep_src / "auxiliary_camera" / "depth.lmdb", depth_local, dirs_exist_ok=True)

    model = mujoco.MjModel.from_xml_string(kinematics_only_urdf(URDF))
    data = mujoco.MjData(model)
    traj, meta = replay.load_episode(ep)
    addresses, _lo, _hi = replay.model_joint_map(model, replay.dataset_joint_names(meta))
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_Base")
    surf = load_hand_surface(args.n_per_link)

    R, t = base_to_head_camera(URDF)
    M = roll_about_z(180.0)
    R, t = M @ R, M @ t
    print(f"camera position (Link_Base) = {np.round(-R.T @ t, 4)}", flush=True)

    env = lmdb.open(str(depth_local), readonly=True, lock=False, max_readers=4)
    with env.begin() as txn:
        dkeys = sorted(int(k.decode().split("/")[-1]) for k in txn.cursor().iternext(keys=True, values=False)
                       if k.decode().startswith("depth/head/"))
    cap = cv2.VideoCapture(str(ep_src / "videos" / "head.mp4"))
    args.out.mkdir(parents=True, exist_ok=True)
    view_names = [x.strip() for x in args.views.split(",") if x.strip()]
    bad = [v for v in view_names if v not in VIEWS]
    if bad:
        print(f"FATAL: unknown view(s) {bad}; pick from {sorted(VIEWS)}", flush=True)
        return 1

    for f in [int(x) for x in args.frames.split(",") if x.strip()]:
        dk = min(dkeys, key=lambda k: abs(k - f))
        with env.begin() as txn:
            raw = txn.get(f"depth/head/{dk:06d}".encode())
        D = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        okr, frame = cap.read()
        if not okr:
            continue

        # --- scene cloud in the colour frame, coloured by the video ---
        h, w = D.shape
        s = args.scene_step
        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        Xc = np.stack([(jj - DEPTH_CX) / DEPTH_FX * D, (ii - DEPTH_CY) / DEPTH_FY * D, D], -1) @ DEPTH_R.T + DEPTH_T
        # where each depth pixel lands in the colour image, for colour lookup
        zc = Xc[..., 2]
        uc = np.clip((COLOR_FX * Xc[..., 0] / np.where(zc == 0, 1, zc) + COLOR_CX), 0, w - 1)
        vc = np.clip((COLOR_FY * Xc[..., 1] / np.where(zc == 0, 1, zc) + COLOR_CY), 0, h - 1)
        m = (D > 0.05)
        ui, vi = uc.astype(int), vc.astype(int)
        scene = Xc[m][::s]
        scol = frame[vi[m], ui[m]][::s].copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        is_glove = ((hsv[..., 2] < 85) & (hsv[..., 1] < 70))[vi[m], ui[m]][::s]

        # Colour the cloud with the video pixel itself.  The real hand is a black
        # glove on a white table, so this alone makes it read as a black hand shape
        # — no hand-segmentation heuristic involved (an earlier attempt to pick the
        # glove out by darkness also grabbed the black plate and every shadow, which
        # is why the naive mask was dropped).  Gamma-lifted so the glove does not
        # crush to pure black against the dark background.
        scol = (255.0 * (scol.astype(np.float32) / 255.0) ** 0.65).astype(np.uint8)

        # --- FK hand ---
        data.qpos[addresses] = traj[f]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        P = []
        for name in surf:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            Rl = data.xmat[bid].reshape(3, 3)
            P.append(surf[name] @ Rl.T + data.xpos[bid])
        hand3 = np.concatenate(P) @ R.T + t
        sources = exporter.fk21_sources(model, "right")
        kp3 = exporter.positions_in_base(model, data, base_body_id, sources) @ R.T + t

        keep = np.linalg.norm(scene - kp3.mean(0), axis=1) < args.sphere_m
        scene, scol, is_glove = scene[keep], scol[keep], is_glove[keep]
        print(f"  frame {f:5d}: scene {len(scene):6d} pts, hand {len(hand3)} pts", flush=True)

        # --- the number the picture is there to back up ------------------------
        # Nearest-neighbour distance between the FK hand *surface* and the scene, in
        # 3D, with no projection involved.  The dark-pixel mask is only a rough way to
        # favour the glove over the table; it also catches the black plate, so the pair
        # of numbers (all scene points vs dark ones only) is reported rather than one.
        from scipy.spatial import cKDTree
        near = np.linalg.norm(scene - kp3.mean(0), axis=1) < args.metric_sphere_m
        d_fk_to_scene = cKDTree(scene).query(hand3)[0]
        print(f"    FK mesh -> nearest scene point : median {np.median(d_fk_to_scene)*100:.2f} cm  "
              f"p90 {np.percentile(d_fk_to_scene, 90)*100:.2f} cm", flush=True)
        g = scene[is_glove & near]
        h = hand3[np.linalg.norm(hand3 - kp3.mean(0), axis=1) < args.metric_sphere_m]
        if len(g) > 50 and len(h) > 50:
            off = g[cKDTree(g).query(h)[1]] - h
            dx, dy, dz = np.median(off, axis=0) * 100
            print(f"    nearest dark scene point to each FK mesh point sits "
                  f"{dx:+.2f} cm in X (image right +), {dy:+.2f} cm in Y (down +), "
                  f"{dz:+.2f} cm in Z (away from camera +)   [{len(g)} dark px, "
                  f"{len(h)} mesh pts in sphere]", flush=True)
            print("      nearest-neighbour matching shrinks |offset| somewhat; read the "
                  "direction, not the magnitude", flush=True)

        centre = kp3.mean(0)
        radius, W, H = args.radius, args.tile, args.tile
        fx = W * 0.9
        tiles = []
        for name in view_names:
            az_d, el_d, up = VIEWS[name]
            az, el = np.deg2rad(az_d), np.deg2rad(el_d)
            eye = centre + radius * np.array([np.sin(az) * np.cos(el), -np.sin(el), np.cos(az) * np.cos(el)])
            Rv, tv = look_at(eye, centre, up=up)
            img = render([
                (scene, scol, 2, True),                                   # real scene, depth-tested
                (hand3[::args.mesh_step],                                 # FK mesh, sparse so
                 np.tile([0, 200, 255], (len(hand3[::args.mesh_step]), 1)), 2, False),  # the glove shows through
                (kp3, np.tile([0, 0, 220], (len(kp3), 1)), 3, False),     # FK joints, always on top
            ], Rv, tv, W, H, fx)
            cv2.putText(img, VIEW_LABEL[name], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
            tiles.append(img)
        strip = np.hstack(tiles)
        cv2.putText(strip, f"{args.episode}  frame {f}    scene = D435 depth cloud in its own video colours "
                           f"(the black glove IS the real hand)",
                    (8, H + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(strip, "YELLOW = FK hand mesh   RED = FK keypoints   "
                           "all in the D435 colour frame - the 2D projection step is not involved",
                    (8, H + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)
        out = args.out / f"fk3d_f{f:04d}.png"
        cv2.imwrite(str(out), strip)
        print(f"    -> {out}", flush=True)

    cap.release()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
