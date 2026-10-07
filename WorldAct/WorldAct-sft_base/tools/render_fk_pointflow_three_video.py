"""Continuous full-episode original/B comparison, using original renderer data helpers."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
from track4world_scale_fix import correct_cached_depth

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import render_fk_vs_pointflow as core
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-report", type=Path, required=True)
    parser.add_argument("--chunk-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.scale_report.read_text())
    episode = report["episode"]
    ratio = report["ratio_B_over_original"]
    chunk_report = json.loads(args.chunk_report.read_text())
    assert chunk_report["episode"] == episode
    chunks = sorted(chunk_report["chunks"].values(), key=lambda c: c["start"])
    last_scale = chunks[-1]["norms"]["estimated"]
    R, t = core.base_to_camera()
    skeleton, _ = core.load_fk(episode, R, t)
    observations = core.load_pointflow(episode)
    K = core.colour_K(core.colour_intrinsics(episode))
    T = len(skeleton)
    assert len(observations["frame_offsets"]) == T + 1
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == T
    assert all(a["end"] == b["start"] for a, b in zip(chunks, chunks[1:]))
    frame_chunks = [c for c in chunks for _ in range(c["start"], c["end"])]
    all_distances = np.full((T, 3, 21), np.nan)
    all_xyz = np.full((T, 3, 21, 3), np.nan)
    fps = 30
    data, metrics = [], []
    lo, hi = skeleton.min(axis=(0, 1)), skeleton.max(axis=(0, 1))
    for frame in range(T):
        original, uv = core.hand_observations(observations, frame)
        u, v = core.canvas_uv_to_colour(uv)
        z = original[:, 2] * ratio
        calibrated = np.column_stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z))
        chunk = frame_chunks[frame]
        z_fixed = correct_cached_depth(original[:, 2], chunk["norms"]["estimated"], last_scale, chunk["metric_ratio"])
        corrected = np.column_stack(((u - K[0, 2]) / K[0, 0] * z_fixed, (v - K[1, 2]) / K[1, 1] * z_fixed, z_fixed))
        pair, stats = [], []
        indices = np.linspace(0, len(original) - 1, min(2500, len(original))).astype(int)
        for variant, points in enumerate((original, calibrated, corrected)):
            if len(points):
                lo = np.minimum(lo, points.min(0))
                hi = np.maximum(hi, points.max(0))
                distance, nearest = cKDTree(points).query(skeleton[frame])
                distance *= 1000
                all_distances[frame, variant] = distance
                all_xyz[frame, variant] = (points[nearest] - skeleton[frame]) * 1000
                stats.append(float(np.median(distance)))
            else:
                stats.append(None)
            pair.append(points[indices])
        data.append(pair)
        metrics.append(
            dict(
                frame=frame,
                time_seconds=frame / fps,
                points=len(original),
                original_median_mm=stats[0],
                B_median_mm=stats[1],
                corrected_median_mm=stats[2],
            )
        )
        if frame % 100 == 0:
            print(f"Loaded {frame + 1}/{T}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), distances_mm=all_distances, residual_xyz_mm=all_xyz, fps=fps)
    from evaluate_fk_scale_video import evaluate

    evaluate(all_distances, all_xyz, chunks, args.output)
    lo -= 0.03
    hi += 0.03
    print(f"Fixed bounds: {lo} -> {hi}", flush=True)
    width, height = 2400, 850
    fig = plt.figure(figsize=(24, 8.5), dpi=100)
    heading = fig.suptitle("", fontsize=17)
    fig.text(
        0.5,
        0.035,
        "DA3 depth | fixed original observations, rerun-derived scale reconstruction | no FK fitting | learned motion not corrected",
        ha="center",
        fontsize=11,
    )
    artists = []
    for col, label in enumerate(("Original PointFlow + FK", "B: real K + legacy scale", "B: real K + own-chunk scale")):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.set(
            xlim=(lo[0], hi[0]),
            ylim=(lo[1], hi[1]),
            zlim=(lo[2], hi[2]),
            xlabel="Camera X (m)",
            ylabel="Camera Y (m)",
            zlabel="Camera Z (m)",
        )
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=22, azim=-62)
        points = ax.scatter([], [], [], s=3, c=core.PF_ORANGE, alpha=0.4, depthshade=False, label="PointFlow hand")
        joints = ax.scatter(
            [], [], [], s=28, c=core.FK_BLUE, edgecolors="white", linewidths=0.5, depthshade=False, label="FK-21"
        )
        bones = Line3DCollection([skeleton[0, [a, b]] for a, b in core.EDGES], colors=core.FK_BLUE, linewidths=2)
        ax.add_collection3d(bones)
        title = ax.set_title(label, fontsize=13)
        ax.legend(loc="upper right", fontsize=9)
        artists.append((points, joints, bones, title, label))
    fig.subplots_adjust(left=0.035, right=0.96, bottom=0.13, top=0.86, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for frame in range(T):
                heading.set_text(f"{episode} | frame {frame}/{T - 1} | {frame / fps:.2f} s")
                sk = skeleton[frame]
                for col, (points, joints, bones, title, label) in enumerate(artists):
                    points._offsets3d = tuple(data[frame][col].T)
                    joints._offsets3d = tuple(sk.T)
                    bones.set_segments([sk[[a, b]] for a, b in core.EDGES])
                    median = metrics[frame][("original_median_mm", "B_median_mm", "corrected_median_mm")[col]]
                    metric_text = "N/A" if median is None else f"{median:.1f} mm"
                    title.set_text(
                        f"{label}\nJoint-to-surface median: {metric_text} | {metrics[frame]['points']:,} points"
                    )
                fig.canvas.draw()
                encoder.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().tobytes())
                if frame % 100 == 0:
                    print(f"Rendered {frame + 1}/{T}", flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                fps=fps,
                frames=T,
                duration_seconds=T / fps,
                scale_report=report,
                chunk_report=chunk_report,
                bounds=[lo.tolist(), hi.tolist()],
                metrics=metrics,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output}: {T} frames at {fps} fps, {T / fps:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
