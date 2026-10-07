"""Causal point budgets shared by preprocessing and selection previews."""

import numpy as np


def anchor_fps(xyz, count, initial=()):
    """Deterministic XYZ FPS; ties use candidate order, never future labels."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite anchors")
    initial = np.asarray(initial, dtype=np.int64)
    count = min(count, len(xyz) - len(initial))
    selected = np.empty(count, dtype=np.int64)
    if count == 0:
        return selected
    index = int(np.argmin(np.square(xyz - xyz.mean(0)).sum(1)))
    nearest = np.full(len(xyz), np.inf)
    used = np.zeros(len(xyz), dtype=bool)
    for prior in initial:
        nearest = np.minimum(nearest, np.square(xyz - xyz[prior]).sum(1))
        used[prior] = True
    if len(initial):
        nearest[used] = -np.inf
        index = int(nearest.argmax())
    for k in range(count):
        selected[k] = index
        used[index] = True
        nearest = np.minimum(nearest, np.square(xyz - xyz[index]).sum(1))
        nearest[used] = -np.inf
        index = int(nearest.argmax())
    return selected


def hand_guided_fps(xyz, fk, count, radius):
    from scipy.spatial import cKDTree

    global_count = count // 2
    per_hand = (count - global_count) // 2
    selected = anchor_fps(xyz, global_count).tolist()
    distances = np.stack([cKDTree(fk[h * 21 : (h + 1) * 21]).query(xyz)[0] for h in range(2)], axis=1)
    nearest_hand = distances.argmin(1)
    details = dict(global_points=len(selected), radius_metres=radius, local_points=[])
    for hand in range(2):
        eligible = (nearest_hand == hand) & (distances[:, hand] <= radius)
        eligible[selected] = False
        candidates = np.flatnonzero(eligible)
        extra = candidates[anchor_fps(xyz[candidates], per_hand)]
        selected.extend(extra.tolist())
        details["local_points"].append(len(extra))
    fill = anchor_fps(xyz, count - len(selected), initial=selected)
    details["global_fallback_points"] = len(fill)
    selected.extend(fill.tolist())
    return np.asarray(selected, dtype=np.int64), details
