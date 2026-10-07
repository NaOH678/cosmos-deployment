#!/usr/bin/env python3
"""Find the frames across a dataset where the hand is most open.

Why this exists: the sandwich recordings are grasping tasks — the hand is closed or
closing for the whole episode, so there is no frame where the fingers are straight and
separated, which makes it hard to see where FK puts each fingertip.  This scans a whole
dataset and reports the frames that come closest, so the pick is evidence rather than a
guess.

"Open" is measured from FK itself, which is fine: the question here is only *which
frames to look at*, not whether FK is right.  A second filter keeps the frames that are
actually usable — the whole hand inside the image, fingers apart enough to tell the tips
apart, and the hand not flying across the frame.

Loading an episode needs the LMDB, and LMDB cannot mmap on GPFS, so each episode is
copied to local scratch, read, and deleted again — one at a time, never the whole set.

Usage:
    python tools/scan_open_hand.py --dataset dualhand_micropipette --side right \
        --out /path/to/open_frames.json --top 40
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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

TIPS = [4, 8, 12, 16, 20]


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def joint_bend(P: np.ndarray) -> np.ndarray:
    """Per frame, the largest joint angle of the most-bent finger (degrees).

    P is (T, 21, 3).  Finger f occupies 1+4f .. 4+4f as mcp, pip, dip, tip, so the two
    bend angles are between consecutive bone directions.
    """
    out = []
    for f in range(5):
        b = 1 + 4 * f
        seg = [P[:, b + 1] - P[:, b], P[:, b + 2] - P[:, b + 1], P[:, b + 3] - P[:, b + 2]]
        seg = [s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9) for s in seg]
        out.append(np.maximum(
            np.degrees(np.arccos(np.clip((seg[0] * seg[1]).sum(1), -1, 1))),
            np.degrees(np.arccos(np.clip((seg[1] * seg[2]).sum(1), -1, 1))),
        ))
    return np.stack(out).max(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dualhand_micropipette")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--per-episode", type=int, default=3, help="at most this many frames from any one episode")
    ap.add_argument("--scratch", type=Path, default=Path("/var/tmp/fk_scan"))
    args = ap.parse_args()

    import mujoco

    sys.path.insert(0, str(MJLAB / "scripts" / "replay"))
    replay = _load_module("replay_teleop", MJLAB / "scripts/replay/replay_teleop.py")
    exporter = _load_module("export_wuji_fk21", MJLAB / "scripts/replay/export_wuji_fk21.py")
    from verify_fk_camera_projection import base_to_head_camera, roll_about_z

    Rc, tc = base_to_head_camera(URDF)
    M = roll_about_z(180.0)
    Rc, tc = M @ Rc, M @ tc

    model = mujoco.MjModel.from_xml_string(_load_module(
        "render_3d_recon", Path(__file__).resolve().parent / "render_3d_recon.py"
    ).kinematics_only_urdf(URDF))
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_Base")

    episodes = sorted(p for p in (RAW / args.dataset).iterdir()
                      if p.is_dir() and (p / "lmdb" / "data.mdb").is_file())
    print(f"{len(episodes)} episodes under {RAW / args.dataset}", flush=True)

    args.scratch.mkdir(parents=True, exist_ok=True)
    found = []
    for n, src in enumerate(episodes):
        ep = args.scratch / src.name
        try:
            if not (ep / "lmdb").is_dir():
                ep.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src / "lmdb", ep / "lmdb", dirs_exist_ok=True)
                for extra in ("meta_info.pkl", "sync_timestamps.json"):
                    if (src / extra).is_file():
                        shutil.copy2(src / extra, ep / extra)
            traj, meta = replay.load_episode(ep)
            addr, _lo, _hi = replay.model_joint_map(model, replay.dataset_joint_names(meta))
            srcs = exporter.fk21_sources(model, args.side)

            P = np.empty((traj.shape[0], 21, 3))
            for i in range(traj.shape[0]):
                data.qpos[addr] = traj[i]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                P[i] = exporter.positions_in_base(model, data, base_id, srcs)

            T = P.shape[0]
            wb = joint_bend(P)
            spread = np.array([np.linalg.norm(P[i, TIPS][:, None] - P[i, TIPS][None], axis=-1).max()
                               for i in range(T)])
            c = P.mean(1)
            mv = np.r_[np.zeros(5), np.linalg.norm(c[5:] - c[:-5], axis=1)] * 100
            kp = P @ Rc.T + tc
            with np.errstate(divide="ignore", invalid="ignore"):
                u = COLOR_FX * kp[..., 0] / kp[..., 2] + COLOR_CX
                v = COLOR_FY * kp[..., 1] / kp[..., 2] + COLOR_CY
            inside = ((u > 30) & (u < 610) & (v > 30) & (v < 450) & (kp[..., 2] > 0.2)).sum(1)

            ok = (inside >= 21) & (mv < 1.2) & (spread > 0.07)
            order = np.argsort(np.where(ok, wb, 1e9))[:args.per_episode]
            for i in order:
                if not ok[i]:
                    continue
                found.append({"episode": src.name, "frame": int(i), "worst_bend": float(wb[i]),
                              "spread_m": float(spread[i]), "move_cm": float(mv[i]), "inside": int(inside[i])})
            if n % 10 == 0:
                print(f"  [{n:3d}/{len(episodes)}] {src.name}  T={T}  best {wb.min():.1f} deg", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  SKIP {src.name}: {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(ep, ignore_errors=True)

    found.sort(key=lambda r: r["worst_bend"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, indent=2))
    print(f"\n{'bend':>6s} {'spread':>7s} {'move':>5s}  episode / frame")
    for r in found[:args.top]:
        print(f"{r['worst_bend']:6.1f} {r['spread_m']*100:6.1f}c {r['move_cm']:5.2f}  "
              f"{r['episode']} f{r['frame']}")
    print(f"\nwrote {args.out}  ({len(found)} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
