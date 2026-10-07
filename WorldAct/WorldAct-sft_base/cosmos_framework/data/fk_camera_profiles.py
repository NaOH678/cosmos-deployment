"""Dataset-specific FK camera mounts; legacy sandwich/dropper constants stay intact.

The DAgger values are derived from assets/dagger_fk/marvin_wuji_d435_dagger.urdf
with the same optical offset and 180-degree roll as the legacy camera.
"""

import numpy as np

from cosmos_framework.data.fk_camera_extrinsic import R, t

DAGGER_R = np.array(
    [
        [1.2874727130868986e-11, -1.0, -3.5985199405405417e-12],
        [-0.7771258898393769, -7.740572610447075e-12, -0.6293451766251622],
        [0.6293451766251622, 1.089915043107482e-11, -0.7771258898393769],
    ],
    dtype=np.float64,
)
DAGGER_T = np.array([0.03150330353089736, 0.921574664215312, 1.0102225677177419], dtype=np.float64)


def camera_transform(profile="legacy"):
    """Explicit per-dataset selection, never inferred from paths or global state."""
    if profile == "legacy":
        return R, t
    if profile == "dagger":
        return DAGGER_R, DAGGER_T
    raise ValueError(f"unknown FK camera profile {profile!r}; expected legacy or dagger")
