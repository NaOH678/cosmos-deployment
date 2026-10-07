"""Freeze the base -> head-camera transform into a reviewable Python module.

Why freeze it at all: the training pipeline converts FK keypoints from the robot
base frame into the camera frame on every window, and that transform is a single
constant.  Computing it at training time would make the Cosmos package depend on
a URDF living in a sibling repository, so instead it is resolved once here and
written out as literals.  Literals rather than an .npz so the numbers are
diffable in git -- a silent change to the optical-centre offset would otherwise
be invisible.

The transform itself and why it is not just the URDF chain (the roll about the
view axis is undetermined by the URDF; the optical joint's 32.5 mm offset was
added after the shipped zero) are documented in tools/verify_fk_camera_projection.py
and docs/fk_modality_design.md.

Usage:
    python tools/export_fk_camera_extrinsic.py            # write the module
    python tools/export_fk_camera_extrinsic.py --check    # only report drift
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_fk_camera_projection import (  # noqa: E402
    URDF as PRIMARY_URDF,
    base_to_head_camera,
    roll_about_z,
)

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "cosmos_framework" / "data" / "fk_camera_extrinsic.py"

# The same URDF is checked out in a second repository; requiring both to agree
# turns "someone edited one copy" into a hard error instead of a quiet skew.
MIRROR_URDF = Path(
    "/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/wuji-mjlab/"
    "marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf"
)

ROLL_DEG = 180.0


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def resolve_urdf() -> tuple[Path, str]:
    primary = Path(PRIMARY_URDF)
    if not primary.is_file():
        raise FileNotFoundError(f"primary URDF missing: {primary}")
    digest = md5(primary)
    if MIRROR_URDF.is_file():
        mirror = md5(MIRROR_URDF)
        if mirror != digest:
            raise ValueError(
                f"the two URDF copies disagree:\n  {primary}\n    {digest}\n"
                f"  {MIRROR_URDF}\n    {mirror}\n"
                "Fix the copies before freezing the transform."
            )
    return primary, digest


def compute(urdf: Path) -> tuple[np.ndarray, np.ndarray]:
    r_chain, t_chain = base_to_head_camera(str(urdf))
    roll = roll_about_z(ROLL_DEG)
    return roll @ r_chain, roll @ t_chain


def render(R: np.ndarray, t: np.ndarray, urdf: Path, digest: str) -> str:
    # Each row needs its own brackets: joining them flat produces a (9,) array that
    # only fails later, as a matmul dimension error pointing at the caller.
    rows = ",\n".join(
        "        [" + ", ".join(f"{v: .12f}" for v in row) + "]" for row in R
    )
    return f'''"""Base -> head-camera transform for FK keypoints. GENERATED -- do not edit.

Regenerate with ``python tools/export_fk_camera_extrinsic.py``; verify drift with
``--check``.  Edit the generator, never this file.

``p_cam = R @ p_base + t`` puts a keypoint from the robot base frame (``Link_Base``,
metres -- the frame ``wuji_fk21.npz`` stores) into the {int(ROLL_DEG)}-degree-rolled real D435
optical frame.  Two things make this more than a URDF chain walk:

* the URDF leaves the camera link's +Y upward, which is not the image convention,
  so the roll about the view axis is undetermined by the URDF alone;
* the ``head_d435_link_optical_joint`` origin was {0.0325 * 1000:.1f} mm / {0.0043 * 1000:.1f} mm rather than the
  shipped zero (see docs/fk_modality_design.md).

Source URDF: {urdf}
md5:         {digest}
"""

import numpy as np

# p_cam = R @ p_base + t
R = np.array(
    [
{rows},
    ],
    dtype=np.float64,
)
t = np.array([{", ".join(f"{v: .12f}" for v in t)}], dtype=np.float64)

URDF_MD5 = "{digest}"
ROLL_DEG = {ROLL_DEG}

# Checked at import: a regenerated module that lost a bracket would otherwise
# surface as a matmul dimension error inside a training step, far from the cause.
assert R.shape == (3, 3), f"R must be [3,3], got {{R.shape}}"
assert t.shape == (3,), f"t must be [3], got {{t.shape}}"
assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "R is not orthonormal"
assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9), "R is not a rotation"


def base_to_camera(points: np.ndarray) -> np.ndarray:
    """``[..., 3]`` base-frame metres -> camera-frame metres."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"expected [..., 3], got {{points.shape}}")
    return points @ R.T + t
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift without writing")
    args = ap.parse_args()

    urdf, digest = resolve_urdf()
    R, t = compute(urdf)
    if not (np.isfinite(R).all() and np.isfinite(t).all()):
        raise ValueError("non-finite extrinsic")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-9):
        raise ValueError("R is not orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-9):
        raise ValueError("R is not a rotation (det != 1)")

    text = render(R, t, urdf, digest)
    print(f"URDF: {urdf}\n  md5 {digest}")
    print(f"  camera position (base frame) = {np.round(-R.T @ t, 6)}")
    print(f"R orthonormal ✅  det = {np.linalg.det(R):.12f}")

    if args.check:
        if not TARGET.is_file():
            print(f"❌ {TARGET} 不存在，需要生成")
            return 1
        current = TARGET.read_text()
        if current == text:
            print("✅ 与已冻结的模块一致")
            return 0
        print(f"❌ {TARGET} 与重新计算的结果不一致 —— 需要重新生成")
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text)
    print(f"✅ 写出 {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
