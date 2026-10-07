"""Descriptive paired full-episode FK-to-visible-surface distance evaluation."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np


def evaluate(distances, xyz, chunks, output):
    names = ["original", "B_legacy", "B_corrected"]
    medians = np.nanmedian(distances, axis=2)
    summary = {
        "definition": "Per joint: nearest visible hand surface distance in camera metres, reported in mm. Not anatomical correspondences, not FK prediction ADE. Same FK and full original selected point support for all methods. Means over frames/joints are descriptive, not independent statistical samples.",
        "frames": len(distances),
        "variants": {},
    }
    for i, name in enumerate(names):
        d = distances[:, i]
        m = medians[:, i]
        summary["variants"][name] = dict(
            valid_frames=int(np.isfinite(m).sum()),
            frame_median_mean_mm=float(np.nanmean(m)),
            frame_median_median_mm=float(np.nanmedian(m)),
            frame_median_p90_mm=float(np.nanpercentile(m, 90)),
            all_joint_mean_mm=float(np.nanmean(d)),
            all_joint_p95_mm=float(np.nanpercentile(d, 95)),
            mean_abs_xyz_mm=np.nanmean(abs(xyz[:, i]), axis=(0, 1)).tolist(),
            fraction_joints_within_20mm=float(np.mean(d < 20)),
            fraction_frames_median_below_20mm=float(np.mean(m < 20)),
        )
    valid = np.isfinite(medians[:, 1:]).all(1)
    delta = medians[valid, 2] - medians[valid, 1]
    summary["corrected_minus_legacy"] = dict(
        valid_paired_frames=int(valid.sum()),
        mean_frame_median_delta_mm=float(delta.mean()),
        median_frame_median_delta_mm=float(np.median(delta)),
        comparison_tolerance_mm=1.0,
        improved_frames=int(np.sum(delta < -1.0)),
        worsened_frames=int(np.sum(delta > 1.0)),
        tied_frames=int(np.sum(abs(delta) <= 1.0)),
        improved_fraction=float(np.mean(delta < -1.0)),
        pooled_joint_improved_fraction=float(np.nanmean(distances[:, 2] < distances[:, 1])),
    )
    summary["chunks"] = []
    for c in chunks:
        a, b = c["start"], c["end"]
        values = np.nanmean(medians[a:b], axis=0)
        summary["chunks"].append(
            dict(
                start=a,
                end=b,
                mean_frame_median_mm=dict(zip(names, values.tolist())),
                corrected_minus_legacy_mm=float(values[2] - values[1]),
                corrected_improved_fraction=float(np.mean(medians[a:b, 2] < medians[a:b, 1] - 1.0)),
            )
        )
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_seconds"] + [n + "_median_mm" for n in names] + ["corrected_minus_legacy_mm"])
        for f, row in enumerate(medians):
            writer.writerow([f, f / 30, *row, row[2] - row[1]])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ts = np.arange(len(medians)) / 30
    for i, name in enumerate(names):
        axes[0].plot(ts, medians[:, i], label=name, lw=1)
    axes[0].set(
        ylabel="Frame median joint-to-surface (mm)", title="Whole episode; same FK and point support; lower is closer"
    )
    axes[0].legend()
    diff = medians[:, 2] - medians[:, 1]
    axes[1].plot(ts, diff, color="black", lw=0.8)
    axes[1].fill_between(ts, 0, diff, where=diff < 0, color="green", alpha=0.3, label="Corrected closer")
    axes[1].fill_between(ts, 0, diff, where=diff > 0, color="red", alpha=0.3, label="Corrected farther")
    axes[1].set(xlabel="Time (s)", ylabel="Corrected - legacy (mm)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
        for c in chunks[1:]:
            ax.axvline(c["start"] / 30, color="gray", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".metrics.png"), dpi=150)
    plt.close(fig)
    print(json.dumps(summary["corrected_minus_legacy"]), flush=True)
