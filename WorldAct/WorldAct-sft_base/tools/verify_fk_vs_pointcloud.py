"""Quantitative check of the URDF head-camera extrinsics against the tracked point cloud.

The PointFlow dense data stores, per frame, a 448x640 grid of tracked 3D points
in the head-camera frame.  The FK-21 keypoints describe the same physical hand in
Link_Base.  If the extrinsics are right, the transformed FK keypoints must sit on
(very close to) the observed hand points in the cloud.

GPFS does not support mmap, so frames are read with seek/read.
"""

from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_fk_camera_projection import (  # noqa: E402
    URDF,
    base_to_head_camera,
    roll_about_z,
)

SLIM = (
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/"
    "datasets/singlerighthand-sandwich-100-lerobot-slim"
)
PF_CANDIDATES = [
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/"
    "track4world_dense_pointflow_sandwich",
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/"
    "sandwich_dense_fullseq_10_0298_20260908/outputs",
]


def npy_probe(path: str) -> tuple[tuple[int, ...], np.dtype, int]:
    """Return (shape, dtype, data_offset) without loading the array."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError("not a .npy file")
        major, _minor = f.read(2)
        if major == 1:
            hlen = int.from_bytes(f.read(2), "little")
        else:
            hlen = int.from_bytes(f.read(4), "little")
        header = ast.literal_eval(f.read(hlen).decode("latin1"))
        offset = f.tell()
    return tuple(header["shape"]), np.dtype(header["descr"]), offset


def read_frame(path: str, idx: int) -> np.ndarray:
    shape, dtype, offset = npy_probe(path)
    per = int(np.prod(shape[1:]))
    with open(path, "rb") as f:
        f.seek(offset + idx * per * dtype.itemsize)
        buf = f.read(per * dtype.itemsize)
    return np.frombuffer(buf, dtype=dtype).reshape(shape[1:])


def load_fk(episode_index: int) -> np.ndarray:
    p = os.path.join(SLIM, "fk21", "chunk-000", f"episode_{episode_index:06d}.npz")
    return np.load(p, allow_pickle=True)["right_positions_abs"].astype(np.float64)


def episode_map() -> dict[str, int]:
    out = {}
    with open(os.path.join(SLIM, "meta", "source_episodes.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["source_episode"]] = int(r["episode_index"])
    return out


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform mapping a -> b (both [N,3])."""
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


def main() -> None:
    ep_map = episode_map()
    pf_root = next(p for p in PF_CANDIDATES if os.path.isdir(p))
    pf_eps = sorted(os.listdir(pf_root))
    shared = [e for e in pf_eps if e in ep_map]
    if not shared:
        raise SystemExit("no overlapping episodes")
    name = shared[0]
    ep_idx = ep_map[name]
    d = os.path.join(pf_root, name)
    print(f"episode: {name}  (lerobot index {ep_idx})")

    comp_path = os.path.join(d, "COMPLETE.json")
    summary_path = os.path.join(d, "dense_summary.json")
    if os.path.exists(comp_path):
        comp = json.load(open(comp_path))
    elif os.path.exists(summary_path):
        comp = json.load(open(summary_path))
    else:
        comp = {}
    for k in ("coordinate", "metric_scale", "fps", "start_frame", "frames", "pixel_queries"):
        if k in comp:
            print(f"  meta {k} = {comp[k]}")

    pos_path, val_path = os.path.join(d, "position.npy"), os.path.join(d, "valid.npy")
    fi_path = os.path.join(d, "frame_indices.npy")
    if os.path.exists(fi_path):
        fi = np.load(fi_path)
    else:
        start = int(comp.get("start_frame", 0))
        n = int(comp.get("frames", npy_probe(pos_path)[0][0]))
        fi = np.arange(start, start + n)
    fk = load_fk(ep_idx)
    print(f"  pointcloud frames={len(fi)}  fk frames={len(fk)}  frame_ids {fi[0]}..{fi[-1]}")

    r_urdf, t = base_to_head_camera(URDF)

    # sample several frames spread through the episode
    probe_raw = [int(fi[i]) for i in np.linspace(0, len(fi) - 1, 21).astype(int)]
    print(f"\n{'raw frame':>9} {'N_cloud':>8} " + " ".join(f"{'r'+str(int(d)):>8}" for d in (0, 90, 180, 270)))
    print("(单位: 厘米 —— FK 关键点到最近点云点的距离中位数)")
    per_roll = {d: [] for d in (0.0, 90.0, 180.0, 270.0)}
    results = {}
    for raw in probe_raw:
        j = int(np.where(fi == raw)[0][0])
        cloud = read_frame(pos_path, j).reshape(-1, 3).astype(np.float64)
        valid = read_frame(val_path, j).reshape(-1).astype(bool)
        cloud = cloud[valid & np.isfinite(cloud).all(1)]
        if len(cloud) < 100:
            continue
        pts = fk[raw]
        row = []
        for deg in (0.0, 90.0, 180.0, 270.0):
            cam = pts @ r_urdf.T + t
            cam = cam @ roll_about_z(deg).T
            dist = np.linalg.norm(cloud[None, :, :] - cam[:, None, :], axis=-1).min(axis=1)
            results[(raw, deg)] = (cam, dist, cloud)
            per_roll[deg].append(float(np.median(dist)) * 100)
            row.append(float(np.median(dist)) * 100)
        print(f"{raw:9d} {len(cloud):8d} " + " ".join(f"{v:8.2f}" for v in row))

    print("\n=== 汇总: 各滚转的中位距离 (cm) ===")
    for deg in (0.0, 90.0, 180.0, 270.0):
        v = np.array(per_roll[deg])
        if len(v):
            print(f"  roll {deg:5.0f} deg : 中位 {np.median(v):7.2f} cm   均值 {v.mean():7.2f} cm   "
                  f"p90帧 {np.percentile(v,90):7.2f} cm   (n={len(v)})")

    # residual correction via Kabsch between FK keypoints and their nearest cloud points
    mid = probe_raw[len(probe_raw) // 2]
    cam, dist, cloud = results[(mid, 180.0)]
    nn = np.linalg.norm(cloud[None] - cam[:, None], axis=-1).argmin(1)
    r_corr, t_corr = kabsch(cam, cloud[nn])
    ang = np.rad2deg(np.arccos(np.clip((np.trace(r_corr) - 1) / 2, -1, 1)))
    print(f"\n=== 用最近邻 + Kabsch 求出的残差修正 (raw frame {mid}, roll 180) ===")
    print(f"  残余旋转 = {ang:.3f} deg")
    print(f"  残余平移 = {np.round(t_corr, 4)} m  |t| = {np.linalg.norm(t_corr)*100:.2f} cm")
    print(f"  逐关键点最近距离 (cm) = {np.round(dist*100, 2)}")
    print("  顺序: wrist, thumb(cmc,mcp,ip,tip), index(...), middle, ring, pinky")


if __name__ == "__main__":
    main()
