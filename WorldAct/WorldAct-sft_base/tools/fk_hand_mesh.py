"""Hand surface points from FK, for camera-extrinsic calibration.

FK gives the 21 keypoints, but those are *joint centres* — they sit inside a
~2 cm-thick finger.  Fitting them to an observed depth surface therefore pulls
the hand inward and biases the calibration.  This module instead poses the
hand's actual meshes with MuJoCo and samples their surfaces, so the target
objective is surface-to-surface with no anatomical offset.

The model is compiled kinematics-only (see replay_verify_fk.kinematics_only_urdf)
because MuJoCo rejects the shipped URDF's degenerate meshes ("mesh volume is too
small: TCP_Link_L").

Usage:
    PYTHONPATH=/tmp/mjlib python tools/fk_hand_mesh.py --episode-dir /tmp/ep13_local
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import trimesh

TOOLS = Path(__file__).resolve().parent
MJLAB = Path("/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/wuji-mjlab")
URDF = MJLAB / "marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf"
PKG = URDF.parent.parent  # .../marvin_wuji_d435_description

# Right-hand bodies that carry geometry, in the order we care about.
HAND_LINKS = ["right_hand_palm_link"] + [
    f"right_hand_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5)
] + [f"right_hand_finger{f}_tip_link" for f in range(1, 6)]


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def kinematics_only_urdf(urdf_path: Path) -> str:
    """Copy of the loader in replay_verify_fk.py — kept local so this module
    does not import that script's argparse main()."""
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


def load_mujoco_model(path: Path = URDF):
    import mujoco

    return mujoco.MjModel.from_xml_string(kinematics_only_urdf(path))


# --------------------------------------------------------------------------
# Mesh surface sampling
# --------------------------------------------------------------------------
def _visual_origins(urdf_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """link name -> (xyz, rpy) of its first <visual><origin>, identity if absent."""
    import xml.etree.ElementTree as ET

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for link in ET.parse(urdf_path).getroot().findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        o = vis.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
        rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
        out[link.get("name")] = (xyz, rpy)
    return out


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def load_hand_surface_points(
    n_per_link: int = 400, seed: int = 0, urdf_path: Path = URDF
) -> dict[str, np.ndarray]:
    """Sample each right-hand link's mesh surface, expressed in that link's frame.

    Returns link_name -> (n, 3) float64.  The URDF's <visual><origin> offset is
    folded in, so points are in the *link* frame, ready to be posed by FK.
    """
    import xml.etree.ElementTree as ET

    rng = np.random.default_rng(seed)
    origins = _visual_origins(urdf_path)
    meshes: dict[str, str] = {}
    for link in ET.parse(urdf_path).getroot().findall("link"):
        name = link.get("name")
        if name not in HAND_LINKS:
            continue
        m = link.find("visual/geometry/mesh")
        if m is not None:
            meshes[name] = m.get("filename")

    out: dict[str, np.ndarray] = {}
    for name, uri in meshes.items():
        rel = uri.split("package://", 1)[-1]
        rel = rel.split("/", 1)[1] if rel.startswith("marvin_wuji_d435_description/") else rel
        path = PKG / rel
        if not path.is_file():
            continue
        mesh = trimesh.load(path, force="mesh")
        pts, _ = trimesh.sample.sample_surface(mesh, n_per_link, seed=int(rng.integers(1 << 30)))
        xyz, rpy = origins.get(name, (np.zeros(3), np.zeros(3)))
        pts = np.asarray(pts, float) @ _rpy_to_R(rpy).T + xyz
        out[name] = pts
    return out


# --------------------------------------------------------------------------
# Posing
# --------------------------------------------------------------------------
def link_poses(model, data, link_names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return link_name -> (R_world, t_world) after data.qpos has been set."""
    import mujoco

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in link_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        out[name] = (data.xmat[bid].reshape(3, 3).copy(), data.xpos[bid].copy())
    return out


def hand_surface_in_base(
    model, data, samples: dict[str, np.ndarray], link_names: list[str] | None = None
) -> np.ndarray:
    """Concatenate every sampled link surface, posed into the world (base) frame."""
    poses = link_poses(model, data, link_names or list(samples))
    chunks = []
    for name, pts in samples.items():
        if name not in poses:
            continue
        R, t = poses[name]
        chunks.append(pts @ R.T + t)
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, 3))


def load_episode_qpos(episode_dir: Path):
    """(trajectory[T, nq], metadata) via wuji-mjlab's replay reader."""
    replay = _load_module("replay_teleop", MJLAB / "scripts/replay/replay_teleop.py")
    return replay.load_episode(episode_dir)


def qpos_addresses(model, meta) -> np.ndarray:
    """Dataset qpos columns -> MuJoCo qpos addresses, via the recorder's joint order."""
    replay = _load_module("replay_teleop", MJLAB / "scripts/replay/replay_teleop.py")
    names = replay.dataset_joint_names(meta)
    addresses, _lower, _upper = replay.model_joint_map(model, names)
    return addresses


def hand_surface_at(model, data, samples, addresses, frame_qpos) -> np.ndarray:
    """Pose the sampled hand meshes for one frame's qpos; returns base-frame points."""
    import mujoco

    data.qpos[addresses] = frame_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return hand_surface_in_base(model, data, samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=600)
    ap.add_argument("--n-per-link", type=int, default=400)
    ap.add_argument("--fk", type=Path, default=None)
    args = ap.parse_args()

    import mujoco

    model = load_mujoco_model()
    data = mujoco.MjData(model)
    traj, meta = load_episode_qpos(args.episode_dir)
    print(f"trajectory {traj.shape}, frames={len(traj)}")

    samples = load_hand_surface_points(args.n_per_link)
    print(f"sampled {len(samples)} hand links, {sum(len(v) for v in samples.values())} surface points")

    addresses = qpos_addresses(model, meta)
    surf = hand_surface_at(model, data, samples, addresses, traj[args.frame])
    print(f"\nframe {args.frame}: hand surface in base frame = {surf.shape}")
    print(f"   x[{surf[:,0].min():+.3f},{surf[:,0].max():+.3f}] "
          f"y[{surf[:,1].min():+.3f},{surf[:,1].max():+.3f}] "
          f"z[{surf[:,2].min():+.3f},{surf[:,2].max():+.3f}]")

    # Cross-check the wrist keypoint against the stored FK annotation when given.
    if args.fk and args.fk.is_file():
        fk = np.load(args.fk, allow_pickle=True)["positions"][args.frame, 1]
        print(f"   stored FK wrist  = {np.round(fk[0], 4)}")
        print(f"   surface centroid = {np.round(surf.mean(0), 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
