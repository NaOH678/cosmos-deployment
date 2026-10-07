"""Shared, CPU-only Bench2Dex training/deployment contract (RGB, radians)."""

from __future__ import annotations

import numpy as np

ROBOT = "multi_ur5_wuji_with_flange"
CAMERAS = ("cam_overhead", "cam_wrist_left", "cam_wrist_right")
ARM = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
JOINT_NAMES = (
    tuple("L_arm_" + n for n in ARM)
    + tuple(f"left_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))
    + ARM
    + tuple(f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))
)
VIEW_DESCRIPTION = "The upper view is from the overhead camera. The lower-left view is from the left wrist camera, and the lower-right view is from the right wrist camera."


def permutation(source, target=JOINT_NAMES):
    source, target = list(source), list(target)
    if (
        len(source) != 52
        or len(set(source)) != 52
        or set(source) != set(JOINT_NAMES)
        or set(target) != set(JOINT_NAMES)
        or len(target) != 52
    ):
        raise ValueError("Expected exactly the 52 named UR5/Wuji joints; no positional fallback")
    return np.asarray([source.index(n) for n in target], dtype=np.int64)


def compose_rgb(images, width=640):
    """Return uint8 CHW: overhead above two wrists, aspect-preserving letterbox."""
    import cv2

    if width < 8 or width % 4:
        raise ValueError("view_width must be >=8 and divisible by 4")

    def fit(image, w, h):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Expected nonempty uint8 HWC RGB")
        scale = min(w / image.shape[1], h / image.shape[0])
        rw, rh = (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        )
        result = np.zeros((h, w, 3), np.uint8)
        result[(h - rh) // 2 : (h - rh) // 2 + rh, (w - rw) // 2 : (w - rw) // 2 + rw] = cv2.resize(
            image,
            (rw, rh),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        return result

    top = fit(images[CAMERAS[0]], width, width * 3 // 4)
    bottom = np.concatenate([fit(images[k], width // 2, width * 3 // 8) for k in CAMERAS[1:]], axis=1)
    return np.ascontiguousarray(np.concatenate([top, bottom], axis=0).transpose(2, 0, 1))
