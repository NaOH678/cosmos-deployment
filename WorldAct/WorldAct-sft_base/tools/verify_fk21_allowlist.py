"""Recompute FK for every episode of the small-sample allowlist and compare.

Stage-1 gate for the FK modality.  The design (docs/fk_modality_design.md §1.1)
uses the shipped ``wuji_fk21.npz`` annotations rather than recomputing them at
training time, so this tool exists to prove that shortcut is safe: an independent
replay through the same MuJoCo pipeline must reproduce the stored numbers exactly.

Why a driver rather than ten invocations of ``replay_verify_fk.py``: the gate is
"all ten agree", and one table makes a single disagreement legible instead of
buried in scrollback.  The per-episode work is delegated to that tool's URDF
patch so there is only one copy of it.

Environment.  The replay stack needs mujoco==3.2.5, numpy and lmdb
(``wuji-mjlab/requirements-replay.txt``).  No single venv here has all three, so
the intended invocation is:

    PYTHONPATH=<deps-with-lmdb> <mujoco venv>/bin/python tools/verify_fk21_allowlist.py

Concretely:

    PYTHONPATH=/mnt/.../shichaojian/.fk-replay-deps \\
      /mnt/.../WorldAct-lingbot-va/lingbot-va/.venv/bin/python \\
      tools/verify_fk21_allowlist.py

GPFS serves lmdb without mmap, so each episode's ~16 MB store is copied to a
local scratch dir first; ``replay_verify_fk.py`` documents the same constraint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_verify_fk import (  # noqa: E402
    MJLAB,
    REPLAY_DIR,
    _load_module,
    kinematics_only_urdf,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ALLOWLIST = REPO / "examples" / "pointflow_sandwich_10_episodes.txt"
ROOT = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian")
DEFAULT_RAW_ROOT = ROOT / "raw_data" / "singlerighthand_sandwich_100"
DEFAULT_FK_ROOT = ROOT / "raw_data" / "sandwich_fk21"
# Matches the "exactly identical" bar replay_verify_fk.py uses: float32 storage
# round-trip plus mj_forward summation order, not a modelling tolerance.
EXACT_TOL_M = 1e-4


def load_allowlist(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines()]
    return [n for n in names if n and not n.startswith("#")]


def stage_episode(raw_root: Path, name: str, scratch: Path) -> Path:
    """Copy one episode's lmdb (and metadata) to local scratch; return the dir."""
    src = raw_root / name
    dst = scratch / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "lmdb", dst / "lmdb")
    meta = src / "meta_info.pkl"
    if meta.is_file():
        shutil.copy2(meta, dst / meta.name)
    return dst


def compare(recomputed: Path, stored: Path) -> dict:
    """Right-hand keypoint error in metres; both files hold [T, 2, 21, 3]."""
    new = np.load(recomputed, allow_pickle=True)
    old = np.load(stored, allow_pickle=True)
    if "positions" not in old.files:
        raise ValueError(f"{stored} has no 'positions' key (has {old.files})")
    a = np.asarray(new["positions"], dtype=np.float64)
    b = np.asarray(old["positions"], dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: recomputed {a.shape} vs stored {b.shape}")
    # Index 1 is the right hand, the only observed side in this dataset.
    d = np.linalg.norm(a[:, 1] - b[:, 1], axis=-1)  # [T, 21] metres
    worst = np.unravel_index(int(np.argmax(d)), d.shape)
    return {
        "frames": int(a.shape[0]),
        "max_mm": float(d.max() * 1000),
        "mean_mm": float(d.mean() * 1000),
        "worst_frame": int(worst[0]),
        "worst_keypoint": int(worst[1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--fk-root", type=Path, default=DEFAULT_FK_ROOT)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/fk21_verify"))
    ap.add_argument("--urdf", type=Path,
                    default=MJLAB / "marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    import mujoco

    names = load_allowlist(args.allowlist)
    print(f"allowlist: {args.allowlist}  ({len(names)} 条)")

    # Patch the replay stack once: MuJoCo refuses the shipped URDF's degenerate
    # meshes, and FK only needs the kinematic tree.
    replay = _load_module("replay_teleop", REPLAY_DIR / "replay_teleop.py")
    replay.load_model = lambda p: mujoco.MjModel.from_xml_string(  # type: ignore[assignment]
        kinematics_only_urdf(Path(p))
    )
    exporter = _load_module("export_wuji_fk21", REPLAY_DIR / "export_wuji_fk21.py")
    exporter._load_replay_module = lambda: replay  # type: ignore[assignment]

    args.scratch.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for i, name in enumerate(names, 1):
        stored = args.fk_root / name / "annotations" / "wuji_fk21.npz"
        if not (args.raw_root / name / "lmdb").is_dir():
            failures.append((name, "缺少 lmdb"))
            print(f"[{i:2d}/{len(names)}] {name}  ❌ 缺少 lmdb")
            continue
        if not stored.is_file():
            failures.append((name, "缺少现成标注"))
            print(f"[{i:2d}/{len(names)}] {name}  ❌ 缺少现成标注")
            continue
        try:
            local = stage_episode(args.raw_root, name, args.scratch)
            ns = argparse.Namespace(
                episode_dir=local, urdf=args.urdf,
                output=local / "recomputed_fk21.npz", clip_limits=False,
            )
            exporter.export(ns)
            row = compare(local / "recomputed_fk21.npz", stored)
            row["episode"] = name
            rows.append(row)
            ok = row["max_mm"] < EXACT_TOL_M * 1000
            print(f"[{i:2d}/{len(names)}] {name}  {'✅' if ok else '❌'}  "
                  f"{row['frames']:5d} 帧   max={row['max_mm']:.6f} mm   mean={row['mean_mm']:.9f} mm")
        except Exception as exc:  # noqa: BLE001 - report and continue, one bad episode must not hide the rest
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[{i:2d}/{len(names)}] {name}  ❌ {type(exc).__name__}: {exc}")
        finally:
            if not args.keep_scratch:
                shutil.rmtree(args.scratch / name, ignore_errors=True)

    print("\n=== 结论 ===")
    if not rows:
        print("  ❌ 没有任何 episode 完成比对")
        return 1
    worst = max(r["max_mm"] for r in rows)
    print(f"  完成 {len(rows)}/{len(names)} 条，失败 {len(failures)} 条")
    print(f"  全局最大误差 = {worst:.6f} mm  (判据 < {EXACT_TOL_M * 1000} mm)")
    for name, why in failures:
        print(f"    ❌ {name}: {why}")
    verdict = worst < EXACT_TOL_M * 1000 and not failures and len(rows) == len(names)
    print(f"  判定: {'✅ 全部逐点一致，可直接使用现成标注' if verdict else '❌ 存在差异，需排查'}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"verdict": verdict, "max_mm": worst, "episodes": rows,
             "failures": [{"episode": n, "why": w} for n, w in failures]}, indent=2))
        print(f"  报告写出: {args.json_out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
