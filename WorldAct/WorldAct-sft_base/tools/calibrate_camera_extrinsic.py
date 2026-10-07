"""Solve the head-camera extrinsic (base -> camera) from FK hand surface + D435 depth.

Why this shape of solution:
  * The head camera is bolted to the robot stand, so ONE rigid transform holds for
    every frame and every episode.  Pooling frames is what makes the 6 DOF
    well-conditioned — a single hand position leaves directions unconstrained.
  * The FK side contributes the *hand mesh surface* (tools/fk_hand_mesh.py), not the
    21 joint centres.  Joint centres sit inside a ~2 cm-thick finger, so fitting them
    to a depth surface silently shrinks the hand and biases the answer.
  * The observation side is the D435 depth, the only real-camera-frame 3D data.
    Its factory calibration (metadata.json) is trustworthy; the URDF pose is not.

Coarse search runs first because the URDF pose is off by O(10 cm), which is outside
ICP's basin for a 15 cm hand.

Usage:
    PYTHONPATH=/tmp/mjlib python tools/calibrate_camera_extrinsic.py \
        --episodes ep13=/tmp/ep13_local \
        --out /tmp/calib
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
from pathlib import Path

import cv2
import lmdb
import numpy as np
from scipy.spatial import cKDTree

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fk_hand_mesh import (  # noqa: E402
    URDF,
    hand_surface_at,
    load_episode_qpos,
    load_hand_surface_points,
    load_mujoco_model,
    qpos_addresses,
)
from verify_fk_camera_projection import base_to_head_camera, roll_about_z  # noqa: E402

# D435 factory calibration, from raw_data/<ep>/auxiliary_camera/metadata.json
COLOR_FX, COLOR_FY = 605.5706176757812, 604.4129638671875
COLOR_CX, COLOR_CY = 324.4994812011719, 238.25637817382812
DEPTH_FX = DEPTH_FY = 384.1614990234375
DEPTH_CX, DEPTH_CY = 320.2146301269531, 235.62696838378906
DEPTH_R = np.array(
    [0.9999253749847412, -0.011212949641048908, -0.0048545487225055695,
     0.011222291737794876, 0.9999352097511292, 0.0019013523124158382,
     0.004832914564758539, -0.0019556896295398474, 0.9999864101409912]
).reshape(3, 3)
DEPTH_T = np.array([0.014961255714297295, 5.274948853184469e-05, 2.5462672056164593e-05])


def urdf_extrinsic() -> tuple[np.ndarray, np.ndarray]:
    """The nominal pose currently in use: URDF chain + the empirically chosen 180 roll."""
    r0, t0 = base_to_head_camera(URDF)
    m = roll_about_z(180.0)
    return m @ r0, m @ t0


# --------------------------------------------------------------------------
# D435 depth -> colour-frame point cloud
# --------------------------------------------------------------------------
def depth_to_cloud(depth_png: bytes) -> np.ndarray:
    """(H*W, 3) points in the D435 *colour* frame, NaN where depth is invalid."""
    d = cv2.imdecode(np.frombuffer(depth_png, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
    h, w = d.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    xd = (jj - DEPTH_CX) / DEPTH_FX * d
    yd = (ii - DEPTH_CY) / DEPTH_FY * d
    pts = np.stack([xd, yd, d], -1).reshape(-1, 3) @ DEPTH_R.T + DEPTH_T
    pts[d.reshape(-1) <= 0.05] = np.nan
    return pts


def load_depth_index(depth_lmdb: Path) -> dict[int, int]:
    """training_step -> value offset, so a frame's PNG can be fetched without mmap."""
    env = lmdb.open(str(depth_lmdb), readonly=True, lock=False, max_readers=8)
    with env.begin() as txn:
        keys = [k.decode() for k in txn.cursor().iternext(keys=True, values=False)
                if k.decode().startswith("depth/head/")]
    env.close()
    return {int(k.split("/")[-1]) for k in keys}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def trimmed_icp(
    src: np.ndarray,
    tgt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    keep: float = 0.6,
    iters: int = 40,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP with a trimmed objective (robust to table/arm points)."""
    tree = cKDTree(tgt)
    prev = np.inf
    for _ in range(iters):
        p = src @ R.T + t
        dist, idx = tree.query(p, k=1, workers=-1)
        thr = np.quantile(dist, keep)
        sel = dist <= thr
        if sel.sum() < 10:
            break
        a, b = src[sel], tgt[idx[sel]]
        ca, cb = a.mean(0), b.mean(0)
        u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1, 1, d]) @ u.T
        R, t = rot, cb - rot @ ca
        rms = float(np.sqrt((dist[sel] ** 2).mean()))
        if abs(prev - rms) < tol:
            break
        prev = rms
    return R, t, rms


def sphere_target(cloud: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """Depth points within `radius` of `centre` — keeps the hand, drops table/arm."""
    m = np.linalg.norm(cloud - centre, axis=1) < radius
    return cloud[m]


def coarse_search(
    src: np.ndarray, tgt: np.ndarray, R0: np.ndarray, t0: np.ndarray,
    span: float = 0.30, step: float = 0.03, keep: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Translation-only grid search around the current pose; returns (t, trimmed RMS)."""
    tree = cKDTree(tgt)
    base = src @ R0.T
    best_t, best = t0, np.inf
    grid = np.arange(-span, span + 1e-9, step)
    for dx in grid:
        for dy in grid:
            for dz in grid:
                tt = t0 + np.array([dx, dy, dz])
                d, _ = tree.query(base + tt, k=1, workers=-1)
                thr = np.quantile(d, keep)
                rms = float(np.sqrt((d[d <= thr] ** 2).mean()))
                if rms < best:
                    best, best_t = rms, tt
    return best_t, best


def search_pose(
    frames: list[tuple[np.ndarray, np.ndarray]],
    R_urdf: np.ndarray,
    t_urdf: np.ndarray,
    rolls: np.ndarray,
    sphere: float,
    span: float,
    step: float,
    icp_iters: int = 8,
    keep: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, float, list[tuple[float, float]]]:
    """Joint (roll, translation, ICP) search.

    The roll about the view axis is the one DOF the URDF leaves undefined, so it is
    the one most likely to be grossly wrong — a translation-only coarse stage cannot
    recover from it.  For each roll candidate: restrict the depth to a sphere around
    the predicted hand, grid-search translation, run a few ICP iterations, keep the
    best trimmed RMS.
    """
    rng = np.random.default_rng(0)
    log: list[tuple[float, float]] = []
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for roll in rolls:
        Rc = roll_about_z(float(roll)) @ R_urdf
        pairs = []
        for src, cloud in frames:
            centre = (src @ Rc.T + t_urdf).mean(0)
            tgt = sphere_target(cloud, centre, sphere)
            if len(tgt) < 500:
                continue
            if len(tgt) > 40000:
                tgt = tgt[rng.choice(len(tgt), 40000, replace=False)]
            pairs.append((src, tgt))
        if not pairs:
            continue
        sc = np.concatenate([s[rng.choice(len(s), min(400, len(s)), replace=False)] for s, _ in pairs])
        tc = np.concatenate([t[rng.choice(len(t), min(4000, len(t)), replace=False)] for _, t in pairs])
        t_coarse, _r = coarse_search(sc, tc, Rc, t_urdf, span=span, step=step)
        R, t = Rc, t_coarse
        for _ in range(icp_iters):
            a, b = [], []
            for s, tg in pairs:
                tree = cKDTree(tg)
                p = s @ R.T + t
                d, idx = tree.query(p, k=1, workers=-1)
                sel = d <= np.quantile(d, keep)
                a.append(s[sel]); b.append(tg[idx[sel]])
            a = np.concatenate(a); b = np.concatenate(b)
            ca, cb = a.mean(0), b.mean(0)
            u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
            dd = np.sign(np.linalg.det(vt.T @ u.T))
            rot = vt.T @ np.diag([1, 1, dd]) @ u.T
            R, t = rot, cb - rot @ ca
        rms = float(np.sqrt(((a @ R.T + t - b) ** 2).sum(1).mean()))
        log.append((float(roll), rms))
        if best is None or rms < best[2]:
            best = (R, t, rms)
    assert best is not None
    return best[0], best[1], best[2], log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True,
                    help="name=path pairs; path is a local episode dir with lmdb/")
    ap.add_argument("--depth-root", type=Path,
                    default=Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/"
                                 "raw_data/singlerighthand_sandwich_100"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/calib"))
    ap.add_argument("--n-frames", type=int, default=12, help="frames per episode")
    ap.add_argument("--n-surface", type=int, default=2500, help="FK surface points per frame")
    ap.add_argument("--n-target", type=int, default=40000, help="depth points per frame")
    ap.add_argument("--velocity-threshold", type=float, default=1.0,
                    help="max hand displacement (cm) over 5 frames for a frame to count as static")
    ap.add_argument("--sphere-m", type=float, default=0.22,
                    help="radius around the FK hand centroid in which depth points are kept")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import mujoco

    model = load_mujoco_model()
    data = mujoco.MjData(model)
    samples_full = load_hand_surface_points(n_per_link=max(1, args.n_surface // 26))
    R0, t0 = urdf_extrinsic()
    print(f"初始外参 (URDF 名义值 + roll180):\n  R=\n{np.round(R0,4)}\n  t={np.round(t0,4)}")
    print(f"  相机位置(基座系)={np.round(-R0.T @ t0, 4)}")

    rng = np.random.default_rng(0)
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    meta_rows = []
    for spec in args.episodes:
        name, _, rest = spec.partition("=")
        path, _, depth_override = rest.partition("|")
        ep_dir = Path(path)
        traj, meta = load_episode_qpos(ep_dir)
        addrs = qpos_addresses(model, meta)
        # GPFS does not support lmdb mmap, so the depth store must be a local copy.
        depth_lmdb = Path(depth_override) if depth_override else (
            args.depth_root / name / "auxiliary_camera" / "depth.lmdb")
        dkeys = sorted(load_depth_index(depth_lmdb))
        env = lmdb.open(str(depth_lmdb), readonly=True, lock=False, max_readers=8)

        # Static-frame selection: hand displacement over the depth interval (5 frames @30fps).
        fk = np.load(args.depth_root.parent / "sandwich_fk21" / name / "annotations" / "wuji_fk21.npz",
                     allow_pickle=True)["positions"][:, 1]
        c = fk.mean(1)
        disp = np.r_[np.zeros(5), np.linalg.norm(c[5:] - c[:-5], axis=1)] * 100
        static = np.nonzero(disp < args.velocity_threshold)[0]
        static = static[static < len(traj)]
        if len(static) > args.n_frames:
            static = rng.choice(static, args.n_frames, replace=False)
        print(f"\n{name}: {len(traj)} 帧, 静止帧阈值 {args.velocity_threshold}cm -> 取 {len(static)} 帧")

        for f in sorted(static):
            dk = min(dkeys, key=lambda k: abs(k - f))
            with env.begin() as txn:
                raw = txn.get(f"depth/head/{dk:06d}".encode())
            if raw is None:
                continue
            cloud = depth_to_cloud(raw)
            cloud = cloud[np.isfinite(cloud).all(1)]
            surf = hand_surface_at(model, data, samples_full, addrs, traj[f])
            if len(cloud) < 500:
                continue
            src = surf[rng.choice(len(surf), min(args.n_surface, len(surf)), replace=False)]
            frames.append((src, cloud))
            meta_rows.append(dict(episode=name, frame=int(f), depth_key=int(dk),
                                  n_src=len(src), n_cloud=len(cloud)))
        env.close()

    print(f"\n汇集 {len(frames)} 帧, 共 {sum(len(s) for s, _ in frames)} 源点")
    if not frames:
        print("没有可用帧"); return 1

    # ---- joint (roll, translation, ICP) search ----
    rolls = np.arange(0, 360, 30)
    print(f"\n[1/3] 联合搜索：滚转 {len(rolls)} 个候选 × 平移网格 × {8} 步 ICP")
    R, t, rms_coarse, log = search_pose(
        frames, R_urdf=R0, t_urdf=t0, rolls=rolls,
        sphere=args.sphere_m, span=0.30, step=0.04, icp_iters=8,
    )
    for roll, r in sorted(log, key=lambda x: x[1])[:6]:
        print(f"      roll={roll:5.0f}°  trimmed RMS={r * 100:6.2f} cm")
    best_roll = min(log, key=lambda x: x[1])[0]
    print(f"    → 最优滚转候选 = {best_roll:.0f}°   (脚本此前硬编码的是 180°)")

    # ---- fine ICP, pooled ----
    print("\n[2/3] ICP 精修（6 自由度）...")
    pairs = []
    for s, cloud in frames:
        centre = (s @ R.T + t).mean(0)
        tg = sphere_target(cloud, centre, args.sphere_m)
        if len(tg) > args.n_target:
            tg = tg[rng.choice(len(tg), args.n_target, replace=False)]
        pairs.append((s, tg))
    for it in range(40):
        a, b = [], []
        for s, tg in pairs:
            tree = cKDTree(tg)
            p = s @ R.T + t
            d, idx = tree.query(p, k=1, workers=-1)
            sel = d <= np.quantile(d, 0.6)
            a.append(s[sel]); b.append(tg[idx[sel]])
        a = np.concatenate(a); b = np.concatenate(b)
        ca, cb = a.mean(0), b.mean(0)
        u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
        dd = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1, 1, dd]) @ u.T
        R, t = rot, cb - rot @ ca
        rms = float(np.sqrt(((a @ R.T + t - b) ** 2).sum(1).mean()))
        if it % 5 == 0 or it == 39:
            print(f"      iter {it:2d}: RMS={rms * 100:.3f} cm")
        if rms < 0.002:
            break

    # ---- report ----
    print("\n[3/3] 结果")
    R_rel = R @ R0.T
    ang = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    print(f"  修正旋转 角度 = {ang:.2f}°")
    print(f"  修正平移 向量 = {np.round(t - t0, 4)} m   模长 {np.linalg.norm(t - t0) * 100:.2f} cm")
    print(f"  新相机位置(基座系) = {np.round(-R.T @ t, 4)}  (旧 {np.round(-R0.T @ t0, 4)})")
    print(f"  拟合残差 RMS = {rms * 100:.3f} cm")

    np.savez(args.out / "extrinsic.npz", R=R, t=t, R0=R0, t0=t0)
    (args.out / "report.json").write_text(json.dumps(dict(
        R=R.tolist(), t=t.tolist(), R0=R0.tolist(), t0=t0.tolist(),
        correction_deg=float(ang), correction_m=float(np.linalg.norm(t - t0)),
        rms_m=float(rms), frames=meta_rows,
        coarse_rms_m=float(rms_coarse),
    ), indent=2))
    print(f"\n写出 {args.out}/extrinsic.npz 和 report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
