#!/usr/bin/env python3
"""Turn the head D435 depth into a shaded surface reconstruction and orbit it.

This is the "make it look like a real 3D scene" version of render_3d_check.py.
The depth image is a regular grid, so every 2x2 block of valid pixels is a real
quad on a real surface — that is a reconstruction, not a scatter of guesses.
Three things make it read as solid rather than as dots:

  * full resolution (no subsampling), so the points tile the surface
  * normals differenced off the depth grid, then lit, so the surface has form
  * several frames of the same static head camera fused, which denoises the flat
    areas and fills the holes where a single frame dropped depth

What it cannot do, and no amount of code will change: the head camera is one
camera in one place, so this is a 2.5D shell.  There is no back side to orbit to.
The black glove is where the IR return is worst, so the hand is the one place the
depth is full of holes.

Usage:
    python tools/render_3d_recon.py --episode episode_0013_20260731_133649 \
        --out /path/to/dir --frames 300,600 --video /path/to/orbit.mp4
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
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

# Light directions in the colour frame (x right, y down, z away from the camera).
# Two of them, from the front-upper-left and front-upper-right, so a flat patch and
# a curved one do not end up equally bright.
LIGHTS = [np.array([-0.45, -0.75, -0.49]), np.array([0.55, -0.55, -0.63])]
AMBIENT = 0.34


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


HAND_LINKS = ["right_hand_palm_link"] + [
    f"right_hand_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5)
] + [f"right_hand_finger{f}_tip_link" for f in range(1, 6)]


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def load_hand_surface(n_per_link: int, seed: int = 0):
    """link -> (points, normals) in that link's frame, with the visual origin folded in."""
    import trimesh
    import xml.etree.ElementTree as ET

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

    pts_out, nrm_out = {}, {}
    for name, uri in meshes.items():
        rel = uri.split("package://", 1)[-1].split("/", 1)[1]
        path = URDF.parent.parent / rel
        if not path.is_file():
            continue
        mesh = trimesh.load(path, force="mesh")
        p, fidx = trimesh.sample.sample_surface(mesh, n_per_link, seed=int(rng.integers(1 << 30)))
        xyz, rpy = origins[name]
        R = rpy_to_R(rpy)
        pts_out[name] = np.asarray(p, float) @ R.T + xyz
        nrm_out[name] = np.asarray(mesh.face_normals)[fidx] @ R.T
    return pts_out, nrm_out


# --------------------------------------------------------------------------
# renderer
# --------------------------------------------------------------------------
def look_at(eye, target, up=(0, -1, 0)):
    """Camera-to-world (R, t), OpenCV convention: +x right, +y down, +z forward."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
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
    return np.stack([r, d, f]), -np.stack([r, d, f]) @ eye


# FK-21.  0 is the wrist, then five fingers of four joints each.  Drawn as the finger
# chains plus one closed palm outline, rather than a fan of wrist-to-knuckle spokes:
# the spokes all leave the same point and, at this zoom, read as a scribble.
KP_CHAINS = []
for _b in (1, 5, 9, 13, 17):
    KP_CHAINS += [(_b, _b + 1), (_b + 1, _b + 2), (_b + 2, _b + 3)]
KP_PALM = [(1, 5), (5, 9), (9, 13), (13, 17), (17, 0), (0, 1)]

LIFT_TARGET = 150.0


def shade(base_rgb, normals, lift=0.0, extra=1.0):
    """Lambert over the base colour.  `normals` must already face the viewer.

    `lift` blends the base colour toward mid-grey before lighting, and it is what makes
    the real hand readable at all.  The glove is black: multiply black by any lighting
    and every pixel is still black, so the surface has no shading, no form, and reads as
    a flat blob.  Lifting the base gives it albedo to modulate, which is the whole reason
    a black object is visible to a human eye in the first place.  The video colour is
    still the dominant term, so the glove stays the darkest thing in frame.
    """
    base = np.asarray(base_rgb, np.float32)
    if lift > 0:
        base = (1.0 - lift) * base + lift * LIFT_TARGET
    lit = np.full(len(normals), AMBIENT, np.float32)
    for L in LIGHTS:
        L = L / np.linalg.norm(L)
        lit += (np.clip(normals @ L, 0, None) ** 1.2).astype(np.float32) * 0.45
    return np.clip(base * np.clip(lit * extra, 0, 1.6)[:, None], 0, 255).astype(np.uint8)


def _disc(rad):
    """Integer offsets covering a disc of the given radius."""
    r = int(rad)
    if r <= 0:
        return [(0, 0)]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    m = xx * xx + yy * yy <= r * r
    return list(zip(xx[m].tolist(), yy[m].tolist()))


def draw(points, colors, rad, alpha, cam_R, cam_t, W, H, fx, zbuf, img, ztest, max_depth):
    """Splat one layer.  Fully vectorised: for repeated pixels the *last* write wins,
    and points are visited far-to-near, so the nearest survivor is the correct one and
    no per-point Python loop is needed.

    alpha < 1 blends instead of overwriting.  That is what lets the FK hand and the
    glove be seen at the same time: z-buffering alone hides the hand, because the glove
    really is ~1 cm in front of the bare-hand mesh, and hiding it is not useful when the
    question is how the two line up.
    """
    cam = np.asarray(points, float) @ cam_R.T + cam_t
    z = cam[:, 2]
    ok = (z > 0.05) & (z < max_depth)
    if not ok.any():
        return
    cam, z, colors = cam[ok], z[ok], np.asarray(colors, np.float32)[ok]
    u = np.rint(fx * cam[:, 0] / z + W * 0.5)
    v = np.rint(fx * cam[:, 1] / z + H * 0.5)
    order = np.argsort(-z)
    u, v, z, colors = u[order], v[order], z[order], colors[order]
    # Work on the image's own buffer: a separate float copy would have to be written back
    # once per disc offset, which for an 81-offset splat is 81 full-frame copies per layer.
    flat_img = img.reshape(-1, 3)

    for du, dv in _disc(rad):
        x = (u + du).astype(np.int32)
        y = (v + dv).astype(np.int32)
        keep = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not keep.any():
            continue
        flat = (y[keep] * W + x[keep])
        zz = z[keep]
        cc = colors[keep]
        if ztest:
            cur = zbuf[flat]
            win = zz <= cur
            flat, zz, cc = flat[win], zz[win], cc[win]
            zbuf[flat] = zz
        if alpha >= 1.0:
            flat_img[flat] = cc
        else:
            # rint, not a plain cast: truncation on every one of the ~50 overlapping
            # splats would darken the layer by a visible amount.
            flat_img[flat] = np.rint((1.0 - alpha) * flat_img[flat] + alpha * cc)


def render(layers, cam_R, cam_t, W, H, fx, background, max_depth=1.10):
    """layers: (points, colours, radius, ztest, alpha).  ztest layers first, then overlays."""
    img = np.full((H, W, 3), background, np.uint8)
    zbuf = np.full(H * W, np.inf, np.float32)
    for ztest in (True, False):
        for points, colors, rad, want, alpha in layers:
            if want == ztest:
                draw(points, colors, rad, alpha, cam_R, cam_t, W, H, fx, zbuf, img, ztest, max_depth)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="singlerighthand_sandwich_100")
    ap.add_argument("--episode", default="episode_0013_20260731_133649")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default="300,600")
    ap.add_argument("--video", type=Path, default=None, help="write an orbiting mp4")
    ap.add_argument("--video-frame", type=int, default=600, help="frame to orbit around")
    ap.add_argument("--video-frames", type=int, default=120)
    ap.add_argument("--fill-px", type=int, default=3,
                    help="fill remaining depth pinholes from the nearest valid sample within this many "
                         "pixels.  Kept small: a large radius bridges across silhouettes and invents "
                         "surface between the hand and the table")
    ap.add_argument("--fuse", type=int, default=8,
                    help="fuse +-N depth frames of the static head camera; 0 disables.  "
                         "Pixels whose depth is stable across the window take the median "
                         "(denoised, holes filled); pixels that move keep this frame's own "
                         "value, so the hand does not smear")
    ap.add_argument("--roi-m", type=float, default=0.30, help="keep points this close to the hand")
    ap.add_argument("--dist-m", type=float, default=None,
                    help="virtual camera distance; default 0.2 x roi.  Separate from roi so the "
                         "hand can fill the frame without also dragging in the far scene")
    ap.add_argument("--max-depth", type=float, default=1.10,
                    help="drop points beyond this (m).  The far room is noisy speckle and adds "
                         "nothing to a hand comparison")
    ap.add_argument("--lift", type=float, default=0.62, metavar="0..1",
                    help="blend the scene's video colour toward mid-grey before lighting.  "
                         "Without it the black glove has no albedo to shade, so it renders as a "
                         "flat blob and no amount of orbiting reveals its shape")
    ap.add_argument("--hand-alpha", type=float, default=0.12,
                    help="translucency of the FK hand, so the glove underneath stays visible")
    ap.add_argument("--background", default="112,112,118",
                    help="R,G,B.  Mid-grey on purpose: the real hand is a *black* glove, so on a "
                         "dark background its points are the same colour as the void and vanish")
    ap.add_argument("--tile", type=int, default=820)
    ap.add_argument("--ss", type=int, default=2, help="supersample factor")
    ap.add_argument("--n-per-link", type=int, default=3000,
                    help="surface samples per hand link.  At 400 the mesh is sparser than the "
                         "pixels it covers and reads as a cage instead of a surface")
    ap.add_argument("--mesh-step", type=int, default=1,
                    help="thin the FK hand surface; 1 = every point (it is lit now, so a "
                         "solid draw no longer hides the glove — colour tells them apart)")
    ap.add_argument("--scratch", type=Path, default=Path("/var/tmp/fk_recon"),
                    help="local scratch.  /tmp is tmpfs on these nodes, so the 120-380 MB depth "
                         "lmdb would be charged to RAM")
    args = ap.parse_args()
    args.background = [int(x) for x in args.background.split(",")]
    if args.dist_m is None:
        args.dist_m = args.roi_m * 0.85

    import cv2
    import lmdb
    import mujoco

    sys.path.insert(0, str(MJLAB / "scripts" / "replay"))
    replay = _load_module("replay_teleop", MJLAB / "scripts/replay/replay_teleop.py")
    exporter = _load_module("export_wuji_fk21", MJLAB / "scripts/replay/export_wuji_fk21.py")
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
    # Keyed by episode: a shared path here silently reused the first episode's depth for
    # every later one, which is invisible in the output and wrong in the scene.
    depth_local = ep / "depth.lmdb"
    if not (depth_local / "data.mdb").is_file():
        shutil.copytree(ep_src / "auxiliary_camera" / "depth.lmdb", depth_local, dirs_exist_ok=True)

    model = mujoco.MjModel.from_xml_string(kinematics_only_urdf(URDF))
    data = mujoco.MjData(model)
    traj, meta = replay.load_episode(ep)
    addresses, _lo, _hi = replay.model_joint_map(model, replay.dataset_joint_names(meta))
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_Base")
    surf, snorm = load_hand_surface(args.n_per_link)

    R, t = base_to_head_camera(URDF)
    M = roll_about_z(180.0)
    R, t = M @ R, M @ t

    env = lmdb.open(str(depth_local), readonly=True, lock=False, max_readers=4)
    with env.begin() as txn:
        dkeys = sorted(int(k.decode().split("/")[-1]) for k in txn.cursor().iternext(keys=True, values=False)
                       if k.decode().startswith("depth/head/"))

    def read_depth(f):
        dk = min(dkeys, key=lambda k: abs(k - f))
        with env.begin() as txn:
            raw = txn.get(f"depth/head/{dk:06d}".encode())
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001

    cap = cv2.VideoCapture(str(ep_src / "videos" / "head.mp4"))

    def read_frame(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        return img if ok else None

    # --- depth grid -> 3D + normals, once per frame ------------------------
    import scipy.ndimage as ndi

    def scene_cloud(f, radius, centre):
        """(points, normals, colours) for the scene around `centre`, colour frame."""
        D0 = read_depth(f)
        h, w = D0.shape
        if args.fuse > 0:
            stack = np.stack([read_depth(f + k) for k in range(-args.fuse, args.fuse + 1)])
            valid = stack > 0.05
            cnt = valid.sum(0)
            # Median over the frames that actually returned depth, NOT over all frames.
            # "No depth" is stored as 0, so a plain median drags every hole to 0 and the
            # pixel stays empty even when most of the window saw the surface.  On the
            # black glove that is most of the hand: the depth is there in the frames, a
            # naive median was discarding it.
            zs = np.where(valid, stack, np.nan)
            with np.errstate(invalid="ignore", all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN columns
                med = np.nanmedian(zs, axis=0)
                mad = np.nanmedian(np.abs(zs - med[None]), axis=0)
            # Consistent across the window -> trust the median (denoised, and it fills
            # holes the current frame happened to drop).  Inconsistent -> the surface
            # moved (the hand), so keep this frame's own measurement and do not smear.
            consistent = (cnt >= 3) & (mad < 0.012) & np.isfinite(med)
            D = np.where(consistent, med, D0)
        else:
            D = D0

        # D435 single-frame depth is speckled and grows "flying pixels" along every
        # silhouette.  A 5x5 median is the standard cure -- but only where the window is
        # mostly valid.  "No depth" is stored as 0, and a median happily averages zeros
        # in, so on the black glove (where most neighbours are invalid) a blind median
        # deletes the surface entirely.  That is why the hand disappeared from an earlier
        # render: the hole is real, the filter made it bigger.
        cnt = cv2.boxFilter((D > 0.05).astype(np.float32), -1, (5, 5), normalize=False)
        D = np.where(cnt >= 20, cv2.medianBlur(D, 5), D)

        # What the temporal pass could not recover: pinholes no frame ever filled.  Copy
        # in the nearest valid depth, but only from very close by, so the fill cannot
        # bridge a real silhouette (hand to table) and invent a surface that is not there.
        if args.fill_px > 0:
            hole = D < 0.05
            if hole.any():
                idx = ndi.distance_transform_edt(hole, return_distances=False, return_indices=True)
                near = ndi.distance_transform_edt(hole)
                D = np.where(hole & (near <= args.fill_px), D[tuple(idx)], D)

        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        Xc = np.stack([(jj - DEPTH_CX) / DEPTH_FX * D, (ii - DEPTH_CY) / DEPTH_FY * D, D], -1) @ DEPTH_R.T + DEPTH_T

        # Normals off the depth grid: the neighbours on either side are real samples of
        # the same surface, so the cross product of the two tangents is the surface
        # normal.  Edges and holes fall back to "facing the camera", which lights flat.
        P = Xc
        dz = np.zeros_like(P)
        dz[:, 1:-1] = P[:, 2:] - P[:, :-2]
        dy = np.zeros_like(P)
        dy[1:-1, :] = P[2:, :] - P[:-2, :]
        n = np.cross(dz, dy)
        ln = np.linalg.norm(n, axis=-1, keepdims=True)
        bad = (ln[..., 0] < 1e-9) | (D < 0.05)
        n = np.divide(n, np.where(ln < 1e-9, 1, ln))
        n[bad] = np.array([0.0, 0.0, -1.0])
        view = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-9)
        n[(n * view).sum(-1) > 0] *= -1.0            # point every normal at the camera

        frame = read_frame(f)
        zc = P[..., 2]
        uc = np.clip(COLOR_FX * P[..., 0] / np.where(zc == 0, 1, zc) + COLOR_CX, 0, w - 1).astype(int)
        vc = np.clip(COLOR_FY * P[..., 1] / np.where(zc == 0, 1, zc) + COLOR_CY, 0, h - 1).astype(int)
        scol = frame[vc, uc]

        m = (D > 0.05) & (P[..., 2] < args.max_depth) & (np.linalg.norm(P - centre, axis=-1) < radius)
        return P[m], n[m], scol[m]

    def hand_cloud(f):
        data.qpos[addresses] = traj[f]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        P, N = [], []
        for name in surf:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            Rl = data.xmat[bid].reshape(3, 3)
            P.append(surf[name] @ Rl.T + data.xpos[bid])
            N.append(snorm[name] @ Rl.T)
        P = np.concatenate(P)[::args.mesh_step]
        N = np.concatenate(N)[::args.mesh_step]
        return P @ R.T + t, N @ R.T

    def keypoints(f):
        data.qpos[addresses] = traj[f]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        srcs = exporter.fk21_sources(model, "right")
        return exporter.positions_in_base(model, data, base_body_id, srcs) @ R.T + t

    # --- views -------------------------------------------------------------
    def build(f):
        """Everything for one frame, posed and lit once, ready for any number of views.

        The fusion window reads ~2N+1 depth frames, so this must not be called per view.
        """
        kp = keypoints(f)
        pts, nrm, col = scene_cloud(f, args.roi_m, kp.mean(0))
        hpts, hnrm = hand_cloud(f)
        return {
            "centre": kp.mean(0),
            "scene": (pts, shade(col, nrm, lift=args.lift), args.ss, True, 1.0),
            "hand": (hpts, hnrm, shade(np.tile([70, 215, 255], (len(hpts), 1)), hnrm, lift=args.lift * 0.5),
                     args.ss, False, args.hand_alpha),
            "kp": (kp, np.tile([0, 0, 235], (len(kp), 1)), 3 * args.ss, False, 1.0),
            "handcol": np.tile([70, 215, 255], (len(hpts), 1)),
        }

    def hand_layer(layers, eye):
        """Only the half of the hand mesh facing this camera.

        A translucent layer still saturates to opaque where enough points overlap, and
        the far half of a closed mesh is behind the near half — so drawing both buries
        the glove.  Dropping the back half also halves the overdraw.
        """
        pts, nrm, col, rad, zt, alpha = layers["hand"]
        front = (nrm * (eye - pts)).sum(1) > 0
        return (pts[front], col[front], rad, zt, alpha)

    def scene_radius(layers, eye, W, H, fx):
        """Splat size that actually tiles the surface at this zoom.

        Scene points are one per depth pixel, so they carry the D435's angular pitch
        (1/DEPTH_FX rad).  Seen from a virtual camera standing much closer than the D435,
        the same physical gap subtends a wider angle, and a fixed 2 px splat leaves the
        surface as disconnected wisps.  Scale the splat to the projected gap.
        """
        pts = layers["scene"][0]
        z_orig = np.median(np.linalg.norm(pts, axis=1))          # distance from the D435
        z_virt = np.median(np.linalg.norm(pts - eye, axis=1))    # distance from our eye
        gap_px = fx * (1.0 / DEPTH_FX) * (z_orig / max(z_virt, 1e-6))
        return int(np.clip(round(gap_px * 0.85), 1, 48)), gap_px

    def draw_skeleton(img, kp, Rv, tv, W, H, fx, lw, dot):
        """FK-21 as bones + joints, in 2D, on top of everything.

        The FK hand *surface* is what hides the real hand: it is a closed shell sitting
        around the glove, so at any translucency it still stacks to opaque over exactly
        the region being compared.  Lines and dots mark the same joint positions without
        covering the surface underneath.
        """
        cam = kp @ Rv.T + tv
        z = cam[:, 2]
        u = fx * cam[:, 0] / z + W * 0.5
        v = fx * cam[:, 1] / z + H * 0.5
        zs = (W // args.ss, H // args.ss)
        su, sv, sz = u / args.ss, v / args.ss, z
        for edges, col, w in ((KP_PALM, (120, 120, 235), lw), (KP_CHAINS, (60, 60, 235), lw)):
            for a, b in edges:
                if sz[a] <= 0.05 or sz[b] <= 0.05:
                    continue
                cv2.line(img, (int(round(su[a])), int(round(sv[a]))),
                         (int(round(su[b])), int(round(sv[b]))), col, w, cv2.LINE_AA)
        for i in range(len(kp)):
            if sz[i] <= 0.05:
                continue
            col = (0, 0, 255) if i == 0 else (0, 255, 255)
            cv2.circle(img, (int(round(su[i])), int(round(sv[i]))), dot, col, -1, cv2.LINE_AA)

    def view(layers, centre, W, H, fx, az_deg, el_deg, up, dist):
        a, e = np.deg2rad(az_deg), np.deg2rad(el_deg)
        eye = centre + dist * np.array([np.sin(a) * np.cos(e), -np.sin(e), np.cos(a) * np.cos(e)])
        Rv, tv = look_at(eye, centre, up=up)
        rad, gap = scene_radius(layers, eye, W, H, fx)
        pts, col, _rad, zt, alpha = layers["scene"]
        img = render([(pts, col, rad, zt, alpha), hand_layer(layers, eye)],
                     Rv, tv, W, H, fx, background=args.background, max_depth=args.max_depth)
        img = cv2.resize(img, (W // args.ss, H // args.ss), interpolation=cv2.INTER_AREA)
        draw_skeleton(img, layers["kp"][0], Rv, tv, W, H, fx,
                      max(1, round(args.ss * 0.9)), max(2, round(args.ss * 1.6)))
        cv2.putText(img, f"scene splat {rad}px (gap {gap:.1f}px)", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        return img

    args.out.mkdir(parents=True, exist_ok=True)
    stills = [int(x) for x in args.frames.split(",") if x.strip()]

    for f in stills:
        layers = build(f)
        W = H = args.tile * args.ss
        fx = W * 0.9
        tiles = []
        for label, az, el, up in (("along the D435 axis (same left/right as the image)", 0, 8, (0, -1, 0)),
                                  ("looking down", 0, 84, (0, 0, -1)),
                                  ("side view", -78, 12, (0, -1, 0)),
                                  ("side view", 78, 12, (0, -1, 0))):
            img = view(layers, layers["centre"], W, H, fx, az, el, up, args.dist_m)
            cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1)
            tiles.append(img)
        strip = np.hstack(tiles)
        cv2.putText(strip, f"{args.episode}  frame {f}    surface = D435 depth, lit    "
                           f"yellow = FK hand (also lit)    red = FK keypoints",
                    (8, img.shape[0] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(strip, f"fused +-{args.fuse} depth frames (static head camera)   "
                           "2.5D: one viewpoint, so there is no back side to orbit to",
                    (8, img.shape[0] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)
        path = args.out / f"recon_f{f:04d}.png"
        cv2.imwrite(str(path), strip)
        print(f"  frame {f}: scene {len(layers['scene'][0])} pts, hand {len(layers['hand'][0])} pts -> {path}",
              flush=True)

    if args.video:
        f = args.video_frame
        layers = build(f)
        W = H = 720 * args.ss
        fx = W * 0.9
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                                 (W // args.ss, H // args.ss))
        if not writer.isOpened():
            print(f"FATAL: cannot open writer for {args.video}", flush=True)
            return 1
        centre = layers["centre"]
        print(f"  orbit around frame {f}: scene {len(layers['scene'][0])} pts, "
              f"hand {len(layers['hand'][0])} pts", flush=True)
        for i in range(args.video_frames):
            az = 360.0 * i / args.video_frames
            a = np.deg2rad(az)
            eye = centre + args.dist_m * 1.15 * np.array([np.sin(a), -0.42, np.cos(a)])
            Rv, tv = look_at(eye, centre)
            rad_s, _g = scene_radius(layers, eye, W, H, fx)
            pts, col, _r, zt, alpha = layers["scene"]
            img = render([(pts, col, rad_s, zt, alpha), hand_layer(layers, eye), layers["kp"]],
                         Rv, tv, W, H, fx, background=args.background, max_depth=args.max_depth)
            img = cv2.resize(img, (W // args.ss, H // args.ss), interpolation=cv2.INTER_AREA)
            draw_skeleton(img, layers["kp"][0], Rv, tv, W, H, fx,
                          max(1, round(args.ss * 0.9)), max(2, round(args.ss * 1.6)))
            writer.write(img)
            if i % 30 == 0:
                print(f"    orbit {i}/{args.video_frames}", flush=True)
        writer.release()
        print(f"wrote {args.video}", flush=True)

    cap.release()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
