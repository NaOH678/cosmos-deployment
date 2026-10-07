"""Read-only FK-reference audit of the middle column of the three-way video.

Uses cached full-frame measurements. No model, cache or scale policy is changed.
Median centers and nearest-surface residuals are descriptive, not anatomical
correspondences or calibration targets.
"""

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/da3_depth_range_diagnosis/fk_reference")
OUT.mkdir(parents=True, exist_ok=True)
a = np.load(OUT.parent / "depth_series.npz")["full"]
fk, point = a[:, 1], a[:, 2]
error = point - fk
near = fk <= np.quantile(fk, 0.25)
far = fk >= np.quantile(fk, 0.75)
examples = [77, 128, 1010, 1023, 1024]
report = {
    "reference": "FK, same transform and frames as the requested video",
    "scheme": "B + legacy last-chunk scale, unchanged",
    "definitions": {
        "center": "median camera Z of all 21 FK joints / visible hand point cloud",
        "nearest_residual": "median Z component of nearest 3D hand point minus each FK joint",
        "limitation": "Neither proxy is an exact joint-to-surface correspondence; no fit is applied.",
    },
    "frames": len(a),
    "near_frames": int(near.sum()),
    "near_center_farther_frames": int((error[near] > 0).sum()),
    "near_nearest_surface_farther_frames": int((a[near, 4] > 0).sum()),
    "far_frames": int(far.sum()),
    "far_center_nearer_frames": int((error[far] < 0).sum()),
    "examples": [
        dict(frame=f, time_s=f / 30, fk_Z_cm=fk[f] * 100,
             point_Z_cm=point[f] * 100, center_error_cm=error[f] * 100,
             nearest_surface_Z_error_cm=a[f, 4] * 100)
        for f in examples
    ],
    "boundary_1024": {
        "FK_change_mm": (fk[1024] - fk[1023]) * 1000,
        "point_change_mm": (point[1024] - point[1023]) * 1000,
        "change_after_diagnostic_undo_normalization_only_mm": (a[1024, 3] - a[1023, 3]) * 1000,
        "note": "Undo-only is an arithmetic diagnostic, not a pipeline change. Residual jump can involve chunk inference/calibration/point support.",
    },
    "chunk_slopes": [],
}
for start in np.unique(a[:, 5]):
    sel = a[:, 5] == start
    slope, intercept = np.polyfit(fk[sel], point[sel], 1)
    report["chunk_slopes"].append(dict(start=int(start), slope=float(slope), intercept_m=float(intercept)))
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
axes[0].plot(a[:, 0], fk * 100, label="FK joint median Z", color="tab:blue")
axes[0].plot(a[:, 0], point * 100, label="B hand surface median Z", color="tab:orange")
axes[0].set(ylabel="Camera Z (cm)", title="FK reference | B + legacy scale | unchanged data")
axes[1].plot(a[:, 0], error * 100, label="Difference of median Z")
axes[1].plot(a[:, 0], a[:, 4] * 100, label="Median nearest-surface Z residual", alpha=0.7)
axes[1].axhline(0, color="black", ls="--")
axes[1].set(xlabel="Frame", ylabel="Point minus FK (cm)", title="Positive = Point farther; negative = Point nearer")
for ax in axes:
    for boundary in np.unique(a[:, 5])[1:]:
        ax.axvline(boundary, color="grey", alpha=0.2)
    ax.legend()
    ax.grid(alpha=0.2)
fig.savefig(OUT / "fk_depth_timeline.png", dpi=160)
plt.close(fig)
video = Path("outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_three_continuous.mp4")
cap = cv2.VideoCapture(str(video))
images = []
for frame in [77, 1010, 1023, 1024]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    assert ok, frame
    images.append(im[:, 800:1600])
cap.release()
assert cv2.imwrite(str(OUT / "middle_column_examples.jpg"), np.concatenate(images, axis=1))
print(json.dumps(report, indent=2))
