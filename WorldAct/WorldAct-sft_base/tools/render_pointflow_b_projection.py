"""Project the fixed-support original/B hand clouds onto native head RGB.

Uses the same observations as render_fk_pointflow_b_video, not training top-N.
All statistics use all valid, unique hand observations. Display uses track ID
modulo --display-stride, consistently across panels, with five-frame trails.
"""

import argparse
import json
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import render_fk_vs_pointflow as core


def project(points, k):
    homogeneous = points @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display-stride", type=int, default=8)
    args = parser.parse_args()
    if args.display_stride < 1:
        parser.error("display-stride must be positive")
    report = json.loads(args.scale_report.read_text())
    episode = report["episode"]
    ratio = float(report["ratio_B_over_original"])
    root = core.PF_ROOT / episode
    obs = core.load_pointflow(episode)
    tracks = np.load(root / "obs_track.npy", mmap_mode="r")
    frame_ids = np.load(root / "frame_indices.npy")
    timestamps = np.load(root / "timestamps_sec.npy")
    original_ks = np.load(root / "intrinsics.npy")
    k = core.colour_K(core.colour_intrinsics(episode))
    cap = cv2.VideoCapture(str(core.RAW_ROOT / episode / "videos/head.mp4"))
    if not cap.isOpened():
        raise RuntimeError("Cannot open source video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    assert (width, height) == (640, 480)
    assert np.allclose(np.diff(timestamps), 1 / fps, atol=1e-5)
    assert np.array_equal(frame_ids, np.arange(len(frame_ids)))
    # Keep compact display data only; metric computations include every point.
    frames, metrics, motion = [], [], {}
    previous = {}
    for row, raw_id in enumerate(frame_ids):
        start, end = map(int, obs["frame_offsets"][row : row + 2])
        keep = (
            (obs["obs_label"][start:end] == core.HAND_LABEL)
            & obs["obs_valid"][start:end]
            & obs["obs_unique"][start:end]
        )
        xyz = np.asarray(obs["obs_pos"][start:end][keep], dtype=np.float64)
        uv = np.asarray(obs["obs_uv"][start:end][keep], dtype=np.float64)
        ids = np.asarray(tracks[start:end][keep])
        good = np.isfinite(xyz).all(1) & np.isfinite(uv).all(1) & (xyz[:, 2] > 0)
        xyz, uv, ids = xyz[good], uv[good], ids[good]
        reference = np.column_stack(core.canvas_uv_to_colour(uv))
        z = xyz[:, 2] * ratio
        b = np.column_stack(((reference[:, 0] - k[0, 2]) / k[0, 0] * z, (reference[:, 1] - k[1, 2]) / k[1, 1] * z, z))
        original_pixels, b_pixels = project(xyz, k), project(b, k)
        own_pixels = project(xyz, original_ks[row]) * [640, 448] - 0.5
        errors = [np.linalg.norm(p - reference, axis=1) for p in (original_pixels, b_pixels)]
        own_error = float(np.max(np.abs(own_pixels - uv))) if len(ids) else 0.0
        if own_error > 0.5:
            raise ValueError(f"Original camera-coordinate check failed at {raw_id}: {own_error}")
        metrics.append(
            dict(
                frame=int(raw_id),
                points=len(ids),
                original_median_px=float(np.median(errors[0])) if len(ids) else None,
                original_p95_px=float(np.percentile(errors[0], 95)) if len(ids) else None,
                B_max_px=float(errors[1].max()) if len(ids) else None,
                original_own_K_max_axis_error_px=own_error,
            )
        )
        show = ids % args.display_stride == 0
        ids, xyz = ids[show], xyz[show]
        current = {}
        for track, point in zip(ids, xyz):
            track = int(track)
            current[track] = point
            if track in previous:
                total, count = motion.get(track, (0.0, 0))
                motion[track] = (total + float(np.linalg.norm(point - previous[track])) * 1000, count + 1)
        previous = current
        frames.append((ids, [reference[show], original_pixels[show], b_pixels[show]]))
        if row % 200 == 0:
            print(f"Geometry {row}/{len(frame_ids)}", flush=True)
    means = {track: total / count for track, (total, count) in motion.items()}
    color_scale = max(float(np.percentile(list(means.values()), 95)), 1.0)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    size = (width * 2, (height + 56) * 2)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{size[0]}x{size[1]}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    history = deque(maxlen=5)
    snapshots = {0, 300, 650, 1000}
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for row, (ids, pixels) in enumerate(frames):
                ok, rgb = cap.read()
                if not ok:
                    raise RuntimeError(f"Video decode failed at {row}")
                panels = [rgb.copy() for _ in range(4)]
                current = {int(track): np.stack([p[i] for p in pixels]) for i, track in enumerate(ids)}
                visible = []
                for mode in range(3):
                    count = 0
                    for track, positions in current.items():
                        p = positions[mode]
                        if not (np.isfinite(p).all() and 0 <= p[0] < width and 0 <= p[1] < height):
                            continue
                        count += 1
                        color = tuple(map(int, lut[min(255, int(means.get(track, 0) / color_scale * 255))]))
                        last = tuple(np.rint(p).astype(int))
                        for past in reversed(history):
                            if track not in past:
                                break
                            old = past[track][mode]
                            if not (0 <= old[0] < width and 0 <= old[1] < height):
                                break
                            old = tuple(np.rint(old).astype(int))
                            cv2.line(panels[mode + 1], old, last, color, 1, cv2.LINE_AA)
                            last = old
                        cv2.circle(panels[mode + 1], tuple(np.rint(p).astype(int)), 1, color, -1)
                    visible.append(count)
                history.append(current)
                titles = [
                    "Raw RGB",
                    "Cached 2D tracks (reference)",
                    "Original XYZ -> real RGB K",
                    "B XYZ -> real RGB K (round-trip)",
                ]
                subtitles = [
                    f"frame {row} | {row / fps:.2f}s | display track ID % {args.display_stride} = 0",
                    f"visible {visible[0]} | color: mean step motion, p95 {color_scale:.1f}mm",
                    f"visible {visible[1]} | median UV error {metrics[row]['original_median_px']:.2f}px",
                    f"visible {visible[2]} | pixel agreement does NOT validate depth",
                ]
                canvas = np.zeros((size[1], size[0], 3), np.uint8)
                for panel in range(4):
                    x, y = (panel % 2) * width, (panel // 2) * (height + 56)
                    canvas[y + 56 : y + 56 + height, x : x + width] = panels[panel]
                    for line, label in enumerate((titles[panel], subtitles[panel])):
                        cv2.putText(
                            canvas,
                            label,
                            (x + 8, y + 22 + line * 23),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.53,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                encoder.stdin.write(canvas.tobytes())
                if row in snapshots:
                    cv2.imwrite(str(args.output.with_name(args.output.stem + f"_frame_{row:04d}.jpg")), canvas)
                if row % 200 == 0:
                    print(f"Rendered {row}/{len(frames)}", flush=True)
        finally:
            encoder.stdin.close()
            cap.release()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                episode=episode,
                frames=len(frames),
                fps=fps,
                source=str(root),
                scale_report=report,
                support="Original obs_label=2 & obs_valid & obs_unique; not training top-N",
                display_stride=args.display_stride,
                trail_steps=5,
                color="Mean adjacent-valid-frame original 3D displacement per track; shared across all panels",
                caveat="B constructed from cached UV: round-trip agreement cannot validate depth.",
                metrics=metrics,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
