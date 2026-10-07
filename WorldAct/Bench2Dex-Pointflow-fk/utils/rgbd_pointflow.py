"""RGB-only causal tracking and pinhole axial-depth lifting; no scene-state inputs."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackingConfig:
    grid_step: int = 20
    border: int = 12
    fb_threshold_px: float = 1.0
    photometric_threshold: float = 20.0
    min_eigenvalue: float = 1e-4
    window: int = 21
    pyramid_levels: int = 3
    texture_filter: bool = True


def image_queries(gray, config):
    """Initial pixel grid, optionally texture-filtered; never reads depth or GT."""
    h, w = gray.shape
    y, x = np.mgrid[config.border:h-config.border:config.grid_step,
                    config.border:w-config.border:config.grid_step]
    uv = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    if not config.texture_filter:
        return uv
    texture = cv2.cornerMinEigenVal(gray, blockSize=7)
    score = texture[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    return uv[score >= config.min_eigenvalue]


def track_pair(previous, current, uv, alive, config):
    """Forward/backward LK uses only t-1 and t; dead IDs are never resurrected.

    Error values are diagnostics, not calibrated confidence or true visibility.
    """
    indices = np.flatnonzero(alive)
    result = np.full(uv.shape, np.nan, np.float32)
    valid = np.zeros(len(uv), bool)
    fb = np.full(len(uv), np.nan, np.float32)
    photo = np.full(len(uv), np.nan, np.float32)
    if not len(indices):
        return result, valid, fb, photo
    p = uv[indices].reshape(-1, 1, 2).astype(np.float32)
    options = dict(winSize=(config.window, config.window), maxLevel=config.pyramid_levels,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
                   minEigThreshold=config.min_eigenvalue)
    q, status, err = cv2.calcOpticalFlowPyrLK(previous, current, p, None, **options)
    if q is None:
        return result, valid, fb, photo
    finite = np.isfinite(q).all(axis=(1, 2))
    safe_q = np.where(finite[:, None, None], q, p)
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, safe_q, None, **options)
    if back is None:
        return result, valid, fb, photo
    error = np.linalg.norm((back-p).reshape(-1, 2), axis=1)
    h, w = current.shape
    xy = q.reshape(-1, 2)
    inside = ((xy[:, 0] >= config.border) & (xy[:, 0] < w-config.border) &
              (xy[:, 1] >= config.border) & (xy[:, 1] < h-config.border))
    ok = (finite & inside & status.ravel().astype(bool) & back_status.ravel().astype(bool)
          & (error <= config.fb_threshold_px) & (err.ravel() <= config.photometric_threshold))
    valid[indices] = ok
    result[indices[ok]] = xy[ok]
    fb[indices] = error
    photo[indices] = err.ravel()
    return result, valid, fb, photo


def lift_depth(uv, depth, intrinsic, radius=1, jump_abs_m=.02, jump_relative=.02):
    """Nearest-pixel Z; reject invalid/discontinuous neighborhoods, never blend surfaces."""
    n = len(uv)
    h, w = depth.shape
    z = np.full(n, np.nan, np.float32)
    valid = np.zeros(n, bool)
    finite = np.isfinite(uv).all(axis=1)
    index = np.flatnonzero(finite)
    xy = np.rint(uv[index]).astype(np.int64)
    inside = ((xy[:, 0] >= radius) & (xy[:, 0] < w-radius) &
              (xy[:, 1] >= radius) & (xy[:, 1] < h-radius))
    index, xy = index[inside], xy[inside]
    if len(index):
        patches = np.stack([depth[xy[:, 1]+dy, xy[:, 0]+dx]
                            for dy in range(-radius, radius+1)
                            for dx in range(-radius, radius+1)], axis=1)
        center = depth[xy[:, 1], xy[:, 0]]
        good = (np.isfinite(patches).all(axis=1) & (patches > 0).all(axis=1)
                & ((patches.max(axis=1)-patches.min(axis=1)) <= jump_abs_m+jump_relative*center))
        z[index] = center  # raw sampled Z remains diagnostic even if rejected
        valid[index] = good
    homogeneous = np.column_stack([uv, np.ones(n)])
    xyz = (homogeneous @ np.linalg.inv(intrinsic).T) * z[:, None]
    xyz[~valid] = np.nan
    return z, valid, xyz.astype(np.float32)


def isaac_world_from_optical(extrinsic):
    """Existing CameraRig extrinsic maps Isaac (+X forward,+Z up) to world."""
    optical_to_isaac = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    result = np.array(extrinsic, copy=True)
    result[..., :3, :3] = extrinsic[..., :3, :3] @ optical_to_isaac
    return result
