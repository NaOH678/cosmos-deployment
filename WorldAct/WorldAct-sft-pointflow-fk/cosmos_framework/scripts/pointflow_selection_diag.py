"""Render what each PointFlow point-selection configuration actually picks.

The GT-ranked selection (``select_motion_fraction``) is a diagnostic: it reads the
future, so nothing it produces can be reproduced at inference.  Before spending a
training run on a new selection we want to *see* which points it keeps and how far
they move -- the numbers alone hide two failure modes that a scatter on the anchor
frame makes obvious.

``voxel_size`` and ``min_voxel_members`` interact destructively.  The guard ranks a
point last when its voxel holds fewer than N members, so it only means something
while the voxel grid still merges points.  Once ``voxel_size`` drops below the point
spacing every voxel holds exactly one point, every point is ranked last, the
stable sort falls back to index order, and the "top 5%" silently becomes the first
410 raster-ordered pixels.  That configuration is included here on purpose so the
failure is visible rather than inferred.

CPU only.  Writes one PNG per episode plus an aggregate.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cosmos_framework.data.pointflow_window import PointFlowTiming, prepare_window  # noqa: E402

# (label, kwargs) -- each is a complete selection configuration, not a delta.
CONFIGS = (
    ("0.02 m + guard>=3\n(current guard3_full)", dict(voxel_size=0.02, min_voxel_members=3)),
    ("0.5 mm + no guard\n(proposed 1 point : 1 token)", dict(voxel_size=0.0005, min_voxel_members=0)),
    ("0.5 mm + guard>=3\n(the trap: raster-order fallback)", dict(voxel_size=0.0005, min_voxel_members=3)),
)
FRACTION = 0.05
MAX_POINTS = 8192


def load_manifest(path):
    document = json.loads(Path(path).resolve().read_text())
    root = Path(path).resolve().parent
    return {
        row["name"]: (root / row["pointflow_source"]["path"]).resolve()
        for row in document["episodes"]
        if row.get("pointflow_source")
    }


def anchor_frame(episode, frame_id):
    """The tracker's own input canvas: head video resized to the inference size."""
    metadata = json.loads((episode / "COMPLETE.json").read_text())
    capture = cv2.VideoCapture(metadata["video"])
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, bgr = capture.read()
        if not ok:
            raise ValueError(f"{episode.name}: cannot decode frame {frame_id}")
    finally:
        capture.release()
    size = (metadata["inference_width"], metadata["inference_height"])
    return cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def select(episode, frame, timing, voxel_size, min_voxel_members):
    window = prepare_window(
        episode,
        frame,
        MAX_POINTS,
        voxel_size,
        0,
        timing=timing,
        allow_empty=True,
        select_motion_fraction=FRACTION,
        min_voxel_members=min_voxel_members,
    )
    target = np.asarray(window["target_displacement"], dtype=np.float64)
    valid = np.asarray(window["target_valid"])
    magnitude = np.linalg.norm(target, axis=-1) * 1000.0  # mm
    counts = valid.sum(0)
    mean = np.where(counts > 0, (magnitude * valid).sum(0) / np.maximum(counts, 1), np.nan)
    return {
        "uv": np.asarray(window["anchor_uv"], dtype=np.float64),
        "mean_mm": mean,
        "curve_mm": np.where(valid, magnitude, np.nan),
        "valid_frames": counts,
        "raw_frame_ids": np.asarray(window["raw_frame_ids"]),
        "point_ids": np.asarray(window["point_ids"]),
    }


def panel_image(ax, image, result, vmax, title):
    ax.imshow(image)
    uv, mean = result["uv"], result["mean_mm"]
    order = np.argsort(np.nan_to_num(mean))
    scatter = ax.scatter(uv[order, 0], uv[order, 1], c=mean[order], s=4, cmap="turbo", vmin=0, vmax=vmax, linewidths=0)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])
    ax.set_title(
        f"{title}\nn={len(uv)}  mean |disp| median={np.nanmedian(mean):.0f} mm  "
        f"n<20mm={(np.nan_to_num(mean) < 20).mean():.0%}  in-frame={inside.mean():.0%}",
        fontsize=8,
    )
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


def render(episode_name, episode, frames, timing, out_dir):
    per_config = []
    image = None
    for label, kwargs in CONFIGS:
        per_config.append((label, [select(episode, f, timing, **kwargs) for f in frames]))

    vmax = np.nanpercentile(
        np.concatenate([np.concatenate([r["mean_mm"] for r in rows]) for _, rows in per_config]), 95
    )
    n_rows = len(frames)
    figure = plt.figure(figsize=(4.6 * len(CONFIGS), 3.5 * n_rows + 6.0))
    grid = figure.add_gridspec(n_rows + 2, 3, height_ratios=[2.6] * n_rows + [1.4, 1.4], hspace=0.28)
    figure.suptitle(f"{episode_name}   fraction={FRACTION}  max_points={MAX_POINTS}", fontsize=11)

    last_scatter = None
    for row, frame in enumerate(frames):
        image = anchor_frame(episode, per_config[0][1][row]["raw_frame_ids"][0])
        for column, (label, rows) in enumerate(per_config):
            last_scatter = panel_image(
                figure.add_subplot(grid[row, column]),
                image,
                rows[row],
                vmax,
                label if row == 0 else "",
            )
    if last_scatter is not None:
        bar = figure.colorbar(last_scatter, ax=figure.axes[: n_rows * len(CONFIGS)], fraction=0.012, pad=0.008)
        bar.set_label("mean |displacement| over the window (mm)", fontsize=8)

    axis = figure.add_subplot(grid[n_rows, 0])
    bins = np.linspace(0, vmax * 1.6, 45)
    for label, rows in per_config:
        values = np.concatenate([r["mean_mm"] for r in rows])
        axis.hist(values, bins=bins, histtype="step", linewidth=1.6, label=label.split("\n")[0])
    axis.set_xlabel("mean |displacement| (mm)")
    axis.set_ylabel("points")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)

    axis = figure.add_subplot(grid[n_rows, 1:])
    seconds = np.arange(per_config[0][1][0]["curve_mm"].shape[0]) / timing.fps
    for label, rows in per_config:
        curves = np.concatenate([r["curve_mm"] for r in rows], axis=1)
        median = np.nanmedian(curves, axis=1)
        low, high = np.nanpercentile(curves, [25, 75], axis=1)
        (line,) = axis.plot(seconds, median, linewidth=1.6, label=label.split("\n")[0])
        axis.fill_between(seconds, low, high, alpha=0.18, color=line.get_color())
    axis.set_xlabel(f"time from anchor (s, {timing.fps} Hz)")
    axis.set_ylabel("|displacement| (mm)")
    axis.set_title("median with the p25-p75 band across selected points", fontsize=8)
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)

    path = out_dir / f"{episode_name}.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=[222])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--steps-per-token", type=int, default=4)
    arguments = parser.parse_args()

    timing = PointFlowTiming(fps=arguments.fps, steps=arguments.steps, steps_per_token=arguments.steps_per_token)
    entries = load_manifest(arguments.manifest)
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in arguments.episodes:
        if name not in entries:
            raise SystemExit(f"{name}: not in manifest, or has no pointflow_source")
        print(render(name, entries[name], arguments.frames, timing, out_dir), flush=True)


if __name__ == "__main__":
    main()
