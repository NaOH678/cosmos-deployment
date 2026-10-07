#!/usr/bin/env python3
"""Solve the head-camera extrinsic (Link_Base -> D435 colour) from data.

WHAT THIS SOLVES
    The URDF's head-camera mount pose is a nominal value (matched to CAD hole
    positions from photographs).  Measurement says it is ~5 cm away from the
    real camera.  This script finds that correction by fitting the FK hand to
    the hand the D435 actually sees.

THE TWO HANDS
    A = FK hand surface.  Posed by MuJoCo from the recorded joint angles, so it
        lives in Link_Base.  We use the hand's *meshes* (26 links, sampled
        surfaces), not the 21 keypoints: keypoints are joint centres sitting
        inside a ~2 cm-thick finger, and fitting those to a depth surface pulls
        the hand inward.
    B = the hand as the D435 sees it, in the D435 colour frame.  Produced by
        back-projecting depth with the factory intrinsics from metadata.json.

    If the camera pose is right, A transformed into the colour frame lands on B.

WHY THE LAST ATTEMPT FAILED (and what changed)
    The target set used to be "everything within a 22 cm sphere of the FK hand".
    FK was ~10 cm off, so that sphere swallowed a large slab of TABLE.  Trimmed
    ICP then slid the hand mesh onto the table — a flat plane fits anything — and
    reported a flattering 0.64 cm residual.  Two guards now: a 20 cm sphere AND
    a hard depth cut (the hand sits at 0.7-0.9 m, the table at 1.2 m).

RESOURCE BOUNDS (the previous version took the dev box down)
    7 cores / 15 GB machine.  Everything here is single-threaded; there is no
    `workers=-1` anywhere.  Peak RSS is ~500 MB.  Run under `ulimit -v 4000000`.

USAGE
    See the shell block this file was delivered with.  Progress goes to the log
    file with flush, so `tail -f` works.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian")
MJLAB = ROOT / "wuji-mjlab"

# Self-configure the import path.  Two traps this avoids:
#   * a run that silently lost the caller's PYTHONPATH fails with a bare
#     ModuleNotFoundError;
#   * /tmp is PER MACHINE, so a copy under /tmp on one node is invisible on
#     another worker.  The shared-storage copy is checked first.
MJLIB_CANDIDATES = [
    ROOT / "mjlib",          # shared storage — visible from every worker
    Path("/tmp/mjlib"),      # local scratch — may not exist on this node
]
MJLIB = next((p for p in MJLIB_CANDIDATES if (p / "mujoco").is_dir()), MJLIB_CANDIDATES[0])
for _p in (Path(__file__).resolve().parent, MJLIB):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

URDF = MJLAB / "marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf"
RAW = ROOT / "raw_data"

# D435 head-camera factory calibration, from auxiliary_camera/metadata.json.
# These are per-unit values and are trustworthy; the URDF pose is not.
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

# The user's independent physical measurement of the current offset, for the
# final sanity check.  Calibration should land near this.
EXPECTED_CORRECTION_CM = 4.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def npy_probe(path: Path):
    """(shape, dtype, data_offset) without loading the array (GPFS has no mmap)."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a .npy file: {path}")
        major, _minor = f.read(2)
        hlen = int.from_bytes(f.read(2), "little") if major == 1 else int.from_bytes(f.read(4), "little")
        import ast
        header = ast.literal_eval(f.read(hlen).decode("latin1"))
        return tuple(header["shape"]), np.dtype(header["descr"]), f.tell()


def read_npy_frame(path: Path, idx: int) -> np.ndarray:
    shape, dtype, offset = npy_probe(path)
    per = int(np.prod(shape[1:]))
    with open(path, "rb") as f:
        f.seek(offset + idx * per * dtype.itemsize)
        buf = f.read(per * dtype.itemsize)
    return np.frombuffer(buf, dtype=dtype).reshape(shape[1:])


def ensure_local(src: Path, dst: Path) -> Path:
    """GPFS cannot serve lmdb's mmap ('No such device'), so copy to local scratch."""
    if dst.exists() and any(dst.iterdir()):
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    log(f"copying {src} -> {dst}")
    for item in src.iterdir():
        shutil.copy2(item, dst / item.name)
    return dst


# --------------------------------------------------------------------------
# FK hand surface (MuJoCo)
# --------------------------------------------------------------------------
def _load_module(name: str, path: Path):
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


HAND_LINKS = ["right_hand_palm_link"] + [
    f"right_hand_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5)
] + [f"right_hand_finger{f}_tip_link" for f in range(1, 6)]


def load_hand_surface(n_per_link: int, seed: int = 0):
    """link -> (points, normals) in that link's frame, with visual origin folded in."""
    import trimesh
    import xml.etree.ElementTree as ET

    def rpy_to_R(rpy):
        r, p, y = rpy
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))

    rng = np.random.default_rng(seed)
    root = ET.parse(URDF).getroot()
    origins, meshes = {}, {}
    for link in root.findall("link"):
        name = link.get("name")
        if name not in HAND_LINKS:
            continue
        vis = link.find("visual")
        if vis is None:
            continue
        o = vis.find("origin")
        origins[name] = tuple(
            np.array([float(v) for v in (o.get(k) or "0 0 0").split()]) for k in ("xyz", "rpy")
        ) if o is not None else (np.zeros(3), np.zeros(3))
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


def pose_hand(model, data, addresses, qpos, surf, norms, R_cam, t_cam, front_only=True):
    """Pose every hand link for one frame; return points in the camera frame."""
    import mujoco

    data.qpos[addresses] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    P, N = [], []
    for name in surf:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        Rl = data.xmat[bid].reshape(3, 3)
        tl = data.xpos[bid]
        P.append(surf[name] @ Rl.T + tl)
        N.append(norms[name] @ Rl.T)
    P = np.concatenate(P)
    N = np.concatenate(N)
    Pc = P @ R_cam.T + t_cam
    if front_only:
        Nc = N @ R_cam.T
        view = Pc / np.linalg.norm(Pc, axis=1, keepdims=True)
        Pc = Pc[(Nc * view).sum(1) < 0]
    return Pc


# --------------------------------------------------------------------------
# D435 depth -> colour-frame point cloud
# --------------------------------------------------------------------------
def depth_cloud(depth_png: bytes) -> np.ndarray:
    import cv2

    d = cv2.imdecode(np.frombuffer(depth_png, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
    h, w = d.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    xyz = np.stack([(jj - DEPTH_CX) / DEPTH_FX * d, (ii - DEPTH_CY) / DEPTH_FY * d, d], -1)
    pts = xyz.reshape(-1, 3) @ DEPTH_R.T + DEPTH_T
    pts[d.reshape(-1) <= 0.05] = np.nan
    return pts


def depth_keys(depth_dir: Path) -> list[int]:
    import lmdb

    env = lmdb.open(str(depth_dir), readonly=True, lock=False, max_readers=4)
    with env.begin() as txn:
        out = sorted(int(k.decode().split("/")[-1]) for k in txn.cursor().iternext(keys=True, values=False)
                     if k.decode().startswith("depth/head/"))
    env.close()
    return out


def fetch_depth(depth_dir: Path, key: int) -> bytes | None:
    import lmdb

    env = lmdb.open(str(depth_dir), readonly=True, lock=False, max_readers=4)
    with env.begin() as txn:
        out = txn.get(f"depth/head/{key:06d}".encode())
    env.close()
    return out


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def evaluate(src_list, kdtrees, R, t, keep=0.5) -> float:
    """Trimmed mean distance (m) across frames for a candidate pose."""
    tot, n = 0.0, 0
    for src, tree in zip(src_list, kdtrees):
        d, _ = tree.query(src @ R.T + t, k=1, workers=1)
        thr = np.quantile(d, keep)
        sel = d <= thr
        tot += float((d[sel] ** 2).sum())
        n += int(sel.sum())
    return float(np.sqrt(tot / max(n, 1)))


def grid_search(src_list, kdtrees, R, t0, span, step, keep=0.5):
    """Translation-only search around t0. Returns (best_t, best_rms)."""
    base = [s @ R.T for s in src_list]
    offsets = np.arange(-span, span + 1e-9, step)
    best_t, best = t0.copy(), np.inf
    tried = 0
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                tt = t0 + np.array([dx, dy, dz])
                tot, n = 0.0, 0
                for b, tree in zip(base, kdtrees):
                    d, _ = tree.query(b + tt, k=1, workers=1)
                    thr = np.quantile(d, keep)
                    sel = d <= thr
                    tot += float((d[sel] ** 2).sum())
                    n += int(sel.sum())
                rms = float(np.sqrt(tot / max(n, 1)))
                tried += 1
                if rms < best:
                    best, best_t = rms, tt
    return best_t, best, tried


def icp(src_list, tgt_list, R, t, iters=40, keep=0.6, tol=1e-5):
    from scipy.spatial import cKDTree

    trees = [cKDTree(tg) for tg in tgt_list]
    rms = np.inf
    for it in range(iters):
        a, b = [], []
        for src, tg, tree in zip(src_list, tgt_list, trees):
            d, idx = tree.query(src @ R.T + t, k=1, workers=1)
            sel = d <= np.quantile(d, keep)
            a.append(src[sel])
            b.append(tg[idx[sel]])
        a = np.concatenate(a)
        b = np.concatenate(b)
        ca, cb = a.mean(0), b.mean(0)
        u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
        dd = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1, 1, dd]) @ u.T
        R, t = rot, cb - rot @ ca
        new = float(np.sqrt(((a @ R.T + t - b) ** 2).sum(1).mean()))
        if it % 5 == 0 or it == iters - 1:
            log(f"    ICP iter {it:2d}  RMS = {new * 100:.3f} cm")
        if abs(rms - new) < tol:
            rms = new
            break
        rms = new
    return R, t, rms


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="e.g. episode_0013_20260731_133649")
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/fk_calib"),
                    help="local dir for lmdb copies (GPFS cannot mmap lmdb)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--n-surface", type=int, default=2000, help="FK surface points per frame")
    ap.add_argument("--sphere-m", type=float, default=0.20)
    ap.add_argument("--depth-cut-m", type=float, default=1.05,
                    help="drop depth beyond this; the hand is 0.7-0.9 m, the table 1.2 m")
    ap.add_argument("--velocity-cm", type=float, default=1.0,
                    help="max hand displacement over 5 frames (1/6 s) to count as static")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log("=" * 68)
    log(f"camera extrinsic calibration — {args.episode}")
    log(f"out = {args.out}")
    log("=" * 68)

    # -- local copies (GPFS has no lmdb mmap) -------------------------------
    ep_raw = RAW / "singlerighthand_sandwich_100" / args.episode
    ep_local = ensure_local(ep_raw / "lmdb", args.scratch / args.episode / "lmdb")
    for extra in ("meta_info.pkl", "sync_timestamps.json"):
        if (ep_raw / extra).is_file() and not (ep_local.parent / extra).exists():
            shutil.copy2(ep_raw / extra, ep_local.parent / extra)
    depth_local = ensure_local(
        ep_raw / "auxiliary_camera" / "depth.lmdb", args.scratch / args.episode / "depth.lmdb")

    fk_npz = RAW / "sandwich_fk21" / args.episode / "annotations" / "wuji_fk21.npz"
    if not fk_npz.is_file():
        log(f"FATAL: no FK annotation at {fk_npz}")
        return 1

    # -- FK setup -----------------------------------------------------------
    try:
        import mujoco
    except ModuleNotFoundError:
        log(f"FATAL: cannot import mujoco. Looked on sys.path for {MJLIB}.")
        log(f"  /tmp/mjlib exists: {MJLIB.is_dir()}")
        if MJLIB.is_dir():
            log(f"  contents: {sorted(p.name for p in MJLIB.iterdir())[:12]}")
        log("  /tmp is cleared on reboot — mujoco must be reinstalled there.")
        return 1

    sys.path.insert(0, str(MJLAB / "scripts" / "replay"))
    replay = _load_module("replay_teleop", MJLAB / "scripts" / "replay" / "replay_teleop.py")
    model = mujoco.MjModel.from_xml_string(kinematics_only_urdf(URDF))
    data = mujoco.MjData(model)
    traj, meta = replay.load_episode(ep_local.parent)
    names = replay.dataset_joint_names(meta)
    addresses, _lo, _hi = replay.model_joint_map(model, names)
    log(f"episode: {len(traj)} frames, {len(names)} joints")

    surf, norms = load_hand_surface(max(1, args.n_surface // 26))
    log(f"hand mesh: {len(surf)} links, {sum(len(v) for v in surf.values())} sample points")

    # -- nominal extrinsic = URDF chain + the roll chosen by eye -------------
    from verify_fk_camera_projection import base_to_head_camera, roll_about_z

    R0, t0 = base_to_head_camera(URDF)
    M = roll_about_z(180.0)
    R0, t0 = M @ R0, M @ t0
    log(f"nominal camera position (Link_Base) = {np.round(-R0.T @ t0, 4)}")

    # -- pick static frames --------------------------------------------------
    fk = np.load(fk_npz, allow_pickle=True)["positions"][:, 1]
    c = fk.mean(1)
    disp = np.r_[np.zeros(5), np.linalg.norm(c[5:] - c[:-5], axis=1)] * 100
    static = [int(f) for f in np.nonzero(disp < args.velocity_cm)[0] if f < len(traj)]
    log(f"static frames (hand moves < {args.velocity_cm} cm per 1/6 s): {len(static)} / {len(traj)}")
    if not static:
        log("FATAL: no static frames")
        return 1
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(static, min(args.n_frames, len(static)), replace=False).tolist())
    log(f"using {len(pick)} frames: {pick}")

    dkeys = depth_keys(depth_local)

    # -- collect A (FK surface) and B (hand depth points) --------------------
    log("-" * 68)
    log("collecting correspondences")
    src_list, tgt_list = [], []
    for f in pick:
        dk = min(dkeys, key=lambda k: abs(k - f))
        raw = fetch_depth(depth_local, dk)
        if raw is None:
            continue
        cloud = depth_cloud(raw)
        cloud = cloud[np.isfinite(cloud).all(1)]
        pts = pose_hand(model, data, addresses, traj[f], surf, norms, R0, t0)
        centre = pts.mean(0)
        near = np.linalg.norm(cloud - centre, axis=1) < args.sphere_m
        tgt = cloud[near & (cloud[:, 2] < args.depth_cut_m)]
        if len(tgt) < 500:
            log(f"  frame {f:4d} depth#{dk}: only {len(tgt)} target points — skipped")
            continue
        src = pts[rng.choice(len(pts), min(args.n_surface, len(pts)), replace=False)]
        src_list.append(src)
        tgt_list.append(tgt)
        log(f"  frame {f:4d} depth#{dk}: A={len(src):5d} pts  B={len(tgt):6d} pts  "
            f"(sphere {args.sphere_m*100:.0f} cm, depth < {args.depth_cut_m} m)")
    if not src_list:
        log("FATAL: no usable frames")
        return 1
    log(f"pooled: {len(src_list)} frames")

    # -- coarse-to-fine search ----------------------------------------------
    log("-" * 68)
    coarse_src = [s[rng.choice(len(s), min(120, len(s)), replace=False)] for s in src_list]
    from scipy.spatial import cKDTree
    coarse_trees = [cKDTree(t) for t in tgt_list]

    log("stage 1: translation grid, ±10 cm @ 2 cm")
    t1, r1, n1 = grid_search(coarse_src, coarse_trees, R0, t0, 0.10, 0.02)
    log(f"  best: shift = {np.round((t1 - t0) * 100, 2)} cm   trimmed RMS = {r1 * 100:.2f} cm   "
        f"({n1} candidates)")

    log("stage 2: refine grid, ±1.5 cm @ 0.5 cm")
    t2, r2, n2 = grid_search(coarse_src, coarse_trees, R0, t1, 0.015, 0.005)
    log(f"  best: shift = {np.round((t2 - t0) * 100, 2)} cm   trimmed RMS = {r2 * 100:.2f} cm   "
        f"({n2} candidates)")

    log("stage 3: ICP, 6 DOF")
    R, t, rms = icp(src_list, tgt_list, R0, t2)

    # -- report --------------------------------------------------------------
    log("=" * 68)
    shift = t - t0
    R_rel = R @ R0.T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))))
    log("RESULT")
    log(f"  rotation correction : {ang:.2f} deg")
    log(f"  translation shift   : {np.round(shift * 100, 2)} cm   |  {np.linalg.norm(shift) * 100:.2f} cm")
    log(f"  camera position     : {np.round(-R.T @ t, 4)}   (was {np.round(-R0.T @ t0, 4)})")
    log(f"  residual RMS        : {rms * 100:.3f} cm")
    log("")
    log(f"  sanity check: your bench measurement was ~{EXPECTED_CORRECTION_CM} cm")
    got = np.linalg.norm(shift) * 100
    if abs(got - EXPECTED_CORRECTION_CM) < 2.0:
        log(f"  -> {got:.1f} cm agrees with the bench measurement. Calibration is credible.")
    else:
        log(f"  -> {got:.1f} cm does NOT match the bench measurement. Treat as suspect.")

    np.savez(args.out / "extrinsic.npz", R=R, t=t, R0=R0, t0=t0)
    (args.out / "report.json").write_text(json.dumps({
        "episode": args.episode,
        "R": R.tolist(), "t": t.tolist(), "R0": R0.tolist(), "t0": t0.tolist(),
        "rotation_deg": ang,
        "shift_m": shift.tolist(),
        "shift_cm": float(np.linalg.norm(shift) * 100),
        "residual_rms_m": rms,
        "frames": pick,
        "n_frames_used": len(src_list),
        "sphere_m": args.sphere_m,
        "depth_cut_m": args.depth_cut_m,
    }, indent=2))
    log("")
    log(f"wrote {args.out}/extrinsic.npz and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
