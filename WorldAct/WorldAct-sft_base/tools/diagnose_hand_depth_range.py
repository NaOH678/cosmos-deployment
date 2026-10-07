"""Read-only depth-range diagnostic of the retained B+legacy scheme.

Compare full-episode FK proxies and same-frame, nearby-RGB-ray D435 surface depth.
Fits are descriptive only, never applied as calibration or written to datasets.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import render_fk_vs_pointflow as core
from scipy.spatial import cKDTree

EP = "episode_0013_20260731_133649"
OUT = Path("outputs/da3_depth_range_diagnosis")
OUT.mkdir(parents=True, exist_ok=True)
r = json.loads(Path("outputs/fk_vs_pointflow_B/scale_report.json").read_text())
ratio = r["ratio_B_over_original"]
chunks = json.loads(Path("outputs/fk_vs_pointflow_scale_fix/chunks_all.json").read_text())["chunks"]
last = max(chunks.values(), key=lambda c: c["start"])["norms"]["estimated"]
R, t = core.base_to_camera()
fk, _ = core.load_fk(EP, R, t)
obs = core.load_pointflow(EP)
K = core.colour_K(core.colour_intrinsics(EP))
nearest = np.load("outputs/fk_vs_pointflow_scale_fix/episode_0013_20260731_133649_three_continuous.npz")[
    "residual_xyz_mm"
]
depthdir = Path("/tmp/fk_vs_pointflow_depth") / EP / "depth.lmdb"
keys = set(core.depth_keys(depthdir))
meta = core.depth_metadata(EP)
full = []
sensor = []
details = []
for f in range(len(fk)):
    p, uv = core.hand_observations(obs, f)
    z = p[:, 2] * ratio
    c = next(c for c in chunks.values() if c["start"] <= f < c["end"])
    undo = (c["norms"]["estimated"] + 1e-6) / last
    full.append(
        [
            f,
            np.median(fk[f, :, 2]),
            np.median(z),
            np.median(z) * undo,
            np.median(nearest[f, 1, :, 2]) / 1000,
            c["start"],
        ]
    )
    if f in keys:
        u, v = core.canvas_uv_to_colour(uv)
        pf_uv = np.column_stack([u, v])
        # Both gates are independent of FK; report sensitivity to foreground selection.
        for gate in ["front", "none"]:
            cloud, info = core.d435_hand_points(EP, f, uv, K, depthdir, meta, gate=gate)
            if not len(cloud):
                continue
            h = cloud @ K.T
            cu = h[:, :2] / h[:, 2:]
            dist, idx = cKDTree(cu).query(pf_uv)
            keep = dist <= 1.5
            if keep.sum() < 100:
                continue
            dz = cloud[idx[keep], 2]
            pz = z[keep]
            sensor.append(
                [
                    f,
                    0 if gate == "front" else 1,
                    len(pz),
                    len(z),
                    np.median(dz),
                    np.median(pz),
                    np.median(pz - dz),
                    np.median(abs(pz - dz)),
                    np.median(pz) * undo,
                    c["start"],
                ]
            )
            details.append(dict(frame=f, **info, matched_pixels=int(keep.sum())))
    if f % 100 == 0:
        print(f"Processed {f}/{len(fk)}", flush=True)
full = np.array(full)
sensor = np.array(sensor)
np.savez_compressed(OUT / "depth_series.npz", full=full, sensor=sensor)


def describe(x, y, chunk):
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    chunk = chunk[valid]
    a, b = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    centered_x = x.copy()
    centered_y = y.copy()
    for c in np.unique(chunk):
        m = chunk == c
        centered_x[m] -= x[m].mean()
        centered_y[m] -= y[m].mean()
    within = float(np.dot(centered_x, centered_y) / np.dot(centered_x, centered_x))
    q = np.quantile(x, [0.25, 0.75])
    bins = {}
    for label, m in [("near_quartile", x <= q[0]), ("far_quartile", x >= q[1])]:
        bins[label] = dict(
            n=int(m.sum()),
            reference_mean_m=float(x[m].mean()),
            prediction_mean_m=float(y[m].mean()),
            median_bias_mm=float(np.median(y[m] - x[m]) * 1000),
            mean_bias_mm=float(np.mean(y[m] - x[m]) * 1000),
        )
    return dict(
        n=len(x),
        slope=float(a),
        intercept_m=float(b),
        correlation=float(corr),
        within_chunk_slope=within,
        reference_p10_p90_m=np.percentile(x, [10, 90]).tolist(),
        prediction_p10_p90_m=np.percentile(y, [10, 90]).tolist(),
        bins=bins,
    )


report = dict(
    episode=EP,
    retained_scheme="B with legacy last-chunk scale",
    fit_note="Descriptive OLS fit, not a calibration. FK center != visible surface. D435 is an independent sensor reference, not assumed ground truth; nearby RGB rays <=1.5px, same frame keys, >=100 matched pixels; both foreground and ungated sensitivity shown.",
    full_columns=[
        "frame",
        "fk_joint_median_Z_m",
        "B_hand_median_Z_m",
        "diagnostic_undo_chunk_normalization_only_Z_m",
        "median_3D_nearest_signed_Z_residual_m",
        "chunk_start",
    ],
    sensor_columns=[
        "frame",
        "gate_0_front_1_none",
        "matched_pixels",
        "total_pf_pixels",
        "sensor_median_Z_m",
        "B_matched_median_Z_m",
        "median_paired_Z_bias_m",
        "median_paired_abs_Z_error_m",
        "diagnostic_unwrap_only_m",
        "chunk_start",
    ],
)
report["fk_centers_vs_hand_surface"] = describe(full[:, 1], full[:, 2], full[:, 5])
report["fk_diagnostic_undo_only"] = describe(full[:, 1], full[:, 3], full[:, 5])
for gate in [0, 1]:
    s = sensor[sensor[:, 1] == gate]
    label = "front" if gate == 0 else "none"
    report["sensor_" + label] = describe(s[:, 4], s[:, 5], s[:, 9])
    report["sensor_" + label]["median_pair_bias_mm"] = float(np.median(s[:, 6]) * 1000)
    report["sensor_" + label]["median_frame_pixel_abs_error_mm"] = float(np.median(s[:, 7]) * 1000)
    report["sensor_" + label]["median_pixel_coverage"] = float(np.median(s[:, 2] / s[:, 3]))
    report["sensor_" + label + "_undo_only"] = describe(s[:, 4], s[:, 8], s[:, 9])
report["sensor_selection_details"] = details
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs[0, 0].plot(full[:, 0] / 30, full[:, 1], label="FK joint median Z")
axs[0, 0].plot(full[:, 0] / 30, full[:, 2], label="B hand surface median Z")
s = sensor[sensor[:, 1] == 0]
axs[0, 0].scatter(s[:, 0] / 30, s[:, 4], s=6, label="D435 matched visible surface")
axs[0, 0].set(xlabel="Time (s)", ylabel="Camera Z (m)")
axs[0, 0].legend()
for ax, x, y, title in [
    (axs[0, 1], full[:, 1], full[:, 2], "FK centers vs hand surface (different anatomy)"),
    (axs[1, 0], s[:, 4], s[:, 5], "Same-frame nearby rays: D435 vs B (foreground)"),
]:
    ax.scatter(x, y, s=8, alpha=0.5)
    a, b = np.polyfit(x, y, 1)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "k--", label="identity")
    ax.plot(lims, a * np.array(lims) + b, label=f"OLS slope {a:.3f}")
    ax.set(xlabel="Reference Z (m)", ylabel="B Z (m)", title=title)
    ax.legend()
axs[1, 1].scatter(s[:, 4], s[:, 6] * 1000, s=12)
axs[1, 1].axhline(0, color="black", ls="--")
axs[1, 1].set(
    xlabel="D435 matched surface Z (m)",
    ylabel="Median paired B - sensor Z (mm)",
    title="Positive = predicted farther; negative = predicted nearer",
)
fig.tight_layout()
fig.savefig(OUT / "depth_range.png", dpi=150)
plt.close(fig)
print(
    json.dumps({k: v for k, v in report.items() if isinstance(v, dict) and k != "sensor_selection_details"}, indent=2),
    flush=True,
)
