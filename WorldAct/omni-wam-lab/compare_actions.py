"""Compare native raw 32x27 absolute actions; no success/safety threshold inferred."""

import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate):
    if reference.shape != (32, 27) or candidate.shape != (32, 27):
        raise ValueError("Both inputs must be raw 32x27 WAM actions")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Non-finite actions")
    pos = np.linalg.norm(candidate[:, :3] - reference[:, :3], axis=1)
    rq = reference[:, 3:7]
    cq = candidate[:, 3:7]
    rn = np.linalg.norm(rq, axis=1)
    cn = np.linalg.norm(cq, axis=1)
    if np.any(rn < 1e-8) or np.any(cn < 1e-8):
        raise ValueError("Invalid quaternion")
    cosine = np.abs(np.sum(rq * cq, axis=1) / (rn * cn))
    angle = np.degrees(2 * np.arccos(np.clip(cosine, 0, 1)))
    hand = np.abs(candidate[:, 7:] - reference[:, 7:])
    return {
        "position_m_mean": float(pos.mean()),
        "position_m_max": float(pos.max()),
        "orientation_deg_mean": float(angle.mean()),
        "orientation_deg_max": float(angle.max()),
        "hand_native_units_mae": float(hand.mean()),
        "hand_native_units_max": float(hand.max()),
        "max_abs_all": float(np.abs(candidate - reference).max()),
        "note": "Offline numerical differences; not a task-success or robot-safety assessment.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.reference.suffix == ".npz":
        with np.load(a.reference, allow_pickle=False) as z:
            reference = z["raw_actions"].copy()
    else:
        reference = np.load(a.reference, allow_pickle=False)
    candidate = np.load(a.candidate, allow_pickle=False)
    result = compare(reference.astype(np.float64), candidate.astype(np.float64))
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
