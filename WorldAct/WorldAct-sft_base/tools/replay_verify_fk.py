"""Re-run the wuji-mjlab FK exporter and compare against the stored annotations.

Why a wrapper: the shipped `load_model()` hands the raw URDF to MuJoCo, which
now rejects it ("mesh volume is too small: TCP_Link_L").  FK only needs the
kinematic tree, so this wrapper strips every visual/collision geom before
compiling and leaves the rest of the exporter untouched — same joint mapping,
same `positions_in_base`, so the comparison stays apples to apples.

Usage:
    PYTHONPATH=/tmp/mjlib python tools/replay_verify_fk.py \
        --episode-dir /tmp/ep0_local --stored <stored fk21 npz> --out /tmp/fk_replay
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MJLAB = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/wuji-mjlab")
REPLAY_DIR = MJLAB / "scripts" / "replay"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def kinematics_only_urdf(urdf_path: Path) -> str:
    """Resolve mesh URIs, then drop every geom so MuJoCo skips mesh inertia."""
    root = ET.parse(urdf_path).getroot()
    ext = root.find("mujoco")
    if ext is None:
        ext = ET.SubElement(root, "mujoco")
    compiler = ext.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(ext, "compiler")
    compiler.set("strippath", "false")
    compiler.set("discardvisual", "true")
    compiler.set("fusestatic", "false")  # keep Link_Base / *_link frames addressable
    compiler.set("boundmass", "1e-6")
    compiler.set("boundinertia", "1e-6")

    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for geom in list(link.findall(kind)):
                link.remove(geom)
    for mesh in list(root.findall(".//mesh")):
        mesh.getparent().remove(mesh)
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", type=Path, required=True)
    ap.add_argument("--urdf", type=Path,
                    default=MJLAB / "marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf")
    ap.add_argument("--stored", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/fk_replay"))
    ap.add_argument("--stored-key", default="positions")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import mujoco

    replay = _load_module("replay_teleop", REPLAY_DIR / "replay_teleop.py")

    def patched_load_model(urdf_path: Path):
        xml = kinematics_only_urdf(Path(urdf_path))
        return mujoco.MjModel.from_xml_string(xml)

    replay.load_model = patched_load_model  # type: ignore[assignment]

    exporter = _load_module("export_wuji_fk21", REPLAY_DIR / "export_wuji_fk21.py")
    exporter._load_replay_module = lambda: replay  # type: ignore[assignment]

    ns = argparse.Namespace(
        episode_dir=args.episode_dir, urdf=args.urdf,
        output=args.out / "recomputed_fk21.npz", clip_limits=False,
    )
    summary = exporter.export(ns)
    print("=== 重算完成 ===")
    print("  shape:", summary["shape"], " frame:", summary["coordinate_frame"])
    print("  limit_violation_frames:", summary["limit_violation_frames"],
          " max_violation(rad):", round(summary["max_limit_violation_rad"], 5))

    new = np.load(args.out / "recomputed_fk21.npz", allow_pickle=True)
    old = np.load(args.stored, allow_pickle=True)
    key = args.stored_key if args.stored_key in old.files else old.files[0]
    a = np.asarray(new["positions"], dtype=np.float64)
    b = np.asarray(old[key], dtype=np.float64)
    print(f"\n=== 对比 ===\n  重算 {a.shape}  vs  已存 ({key}) {b.shape}")
    if a.shape != b.shape:
        print("  !! 形状不一致，无法逐帧比对")
        return
    d = np.linalg.norm(a - b, axis=-1)  # [T,2,21] 米
    right = d[:, 1]          # 右手（本数据只有右手有效）
    print(f"\n  右手 21 点误差 (毫米):")
    print(f"    max  = {right.max()*1000:.6f}")
    print(f"    mean = {right.mean()*1000:.8f}")
    print(f"    p99  = {np.percentile(right,99)*1000:.8f}")
    print(f"  逐帧最大误差的分布 (毫米): median={np.median(right.max(1))*1000:.8f}"
          f"  p99={np.percentile(right.max(1),99)*1000:.8f}")
    worst = np.unravel_index(np.argmax(right), right.shape)
    print(f"  最差: frame={worst[0]} keypoint={worst[1]}  {right[worst]*1000:.6f} mm")
    print(f"\n  判定: {'完全一致 (浮点误差级)' if right.max() < 1e-4 else '存在差异'}")
    (args.out / "compare.json").write_text(json.dumps({
        "shape": list(a.shape), "max_mm": float(right.max() * 1000),
        "mean_mm": float(right.mean() * 1000),
    }, indent=2))


if __name__ == "__main__":
    main()
