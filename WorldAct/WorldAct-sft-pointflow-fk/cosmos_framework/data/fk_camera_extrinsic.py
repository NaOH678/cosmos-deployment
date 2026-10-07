"""Base -> head-camera transform for FK keypoints. GENERATED -- do not edit.

Regenerate with ``python tools/export_fk_camera_extrinsic.py``; verify drift with
``--check``.  Edit the generator, never this file.

``p_cam = R @ p_base + t`` puts a keypoint from the robot base frame (``Link_Base``,
metres -- the frame ``wuji_fk21.npz`` stores) into the 180-degree-rolled real D435
optical frame.  Two things make this more than a URDF chain walk:

* the URDF leaves the camera link's +Y upward, which is not the image convention,
  so the roll about the view axis is undetermined by the URDF alone;
* the ``head_d435_link_optical_joint`` origin was 32.5 mm / 4.3 mm rather than the
  shipped zero (see docs/fk_modality_design.md).

Source URDF: /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-lingbot-va-mano/lingbot-va/wan_va/fk/assets/marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf
md5:         d687e8f52df18839f4f99efe8c12a909
"""

import numpy as np

# p_cam = R @ p_base + t
R = np.array(
    [
        [0.000000000014, -1.000000000000, -0.000000000003],
        [-0.642787609683, -0.000000000006, -0.766044443122],
        [0.766044443122, 0.000000000012, -0.642787609683],
    ],
    dtype=np.float64,
)
t = np.array([0.032500000004, 1.094353449231, 0.830049026781], dtype=np.float64)

URDF_MD5 = "d687e8f52df18839f4f99efe8c12a909"
ROLL_DEG = 180.0

# Checked at import: a regenerated module that lost a bracket would otherwise
# surface as a matmul dimension error inside a training step, far from the cause.
assert R.shape == (3, 3), f"R must be [3,3], got {R.shape}"
assert t.shape == (3,), f"t must be [3], got {t.shape}"
assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "R is not orthonormal"
assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9), "R is not a rotation"


def base_to_camera(points: np.ndarray) -> np.ndarray:
    """``[..., 3]`` base-frame metres -> camera-frame metres."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"expected [..., 3], got {points.shape}")
    return points @ R.T + t
