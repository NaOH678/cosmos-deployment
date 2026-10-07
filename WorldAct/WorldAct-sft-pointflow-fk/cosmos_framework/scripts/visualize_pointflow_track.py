"""Export one fixed Track4World point ID across an entire episode, without GPU."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cosmos_framework.data.pointflow_window import read_frame


def read_track(path, point_id, channels):
    """Seek to one query slot per frame instead of reading the dense episode."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f"Unsupported NPY version: {version}")
        shape, fortran, dtype = readers[version](stream)
        if fortran or dtype.hasobject:
            raise ValueError("Expected numeric C-order array")
        slots = int(np.prod(shape[1:])) // channels
        if not 0 <= point_id < slots:
            raise ValueError("point-id outside query grid")
        start = stream.tell()
        size = channels * dtype.itemsize
        result = np.empty((shape[0], channels), dtype=dtype)
        for row in range(shape[0]):
            stream.seek(start + (row * slots + point_id) * size)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError(f"Truncated array: {path}")
            result[row] = np.frombuffer(data, dtype=dtype)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--select-frame", type=int, default=596)
    parser.add_argument("--focus-uv", nargs=2, type=float, default=[450, 380])
    parser.add_argument("--point-id", type=int)
    parser.add_argument(
        "--trail-steps", type=int, default=30, help="0 shows all history; otherwise recent source frames"
    )
    args = parser.parse_args()
    if args.trail_steps < 0:
        parser.error("trail-steps must be nonnegative")
    meta = json.loads((args.episode / "COMPLETE.json").read_text())
    frame_ids = np.load(args.episode / "frame_indices.npy", allow_pickle=False)
    timestamps = np.load(args.episode / "timestamps_sec.npy", allow_pickle=False)
    width, height = meta["inference_width"], meta["inference_height"]
    point_id = args.point_id
    distance = None
    if point_id is None:
        rows = np.flatnonzero(frame_ids == args.select_frame)
        if len(rows) != 1:
            raise ValueError("select-frame not uniquely present")
        row = int(rows[0])
        uv = read_frame(args.episode / "uv_px.npy", row).reshape(-1, 2)
        xyz = read_frame(args.episode / "position.npy", row).reshape(-1, 3)
        valid = read_frame(args.episode / "valid.npy", row).reshape(-1)
        valid &= np.isfinite(uv).all(1) & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            raise ValueError("No valid visible points in selection frame")
        distances = np.linalg.norm(uv[candidates] - np.array(args.focus_uv), axis=1)
        index = int(np.argmin(distances))
        point_id, distance = int(candidates[index]), float(distances[index])
    uv = read_track(args.episode / "uv_px.npy", point_id, 2)
    xyz = read_track(args.episode / "position.npy", point_id, 3)
    source_valid = read_track(args.episode / "valid.npy", point_id, 1)[:, 0].astype(bool)
    if len(uv) != len(frame_ids) or len(timestamps) != len(frame_ids):
        raise ValueError("Trajectory/time lengths differ")
    valid = source_valid & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0) & np.isfinite(uv).all(1)
    visible = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    xy = np.rint(np.where(visible[:, None], uv, 0)).astype(int)
    delta = np.diff(timestamps)
    if (
        not len(delta)
        or np.any(np.diff(frame_ids) <= 0)
        or not np.allclose(delta, delta[0], atol=1e-4)
        or delta[0] <= 0
    ):
        raise ValueError("Expected increasing frame IDs and uniform timestamps")
    fps = float(1 / delta[0])
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"track_{point_id}"
    np.savez_compressed(
        args.output / f"{stem}.npz",
        point_id=point_id,
        frame_ids=frame_ids,
        timestamps_sec=timestamps,
        xyz=xyz,
        uv=uv,
        source_valid=source_valid,
        valid=valid,
        visible=visible,
    )
    capture = cv2.VideoCapture(meta["video"])
    writer = cv2.VideoWriter(
        str(args.output / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height + 40)
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("Cannot open input/output video")
    next_frame = 0
    try:
        for row, raw_id in enumerate(frame_ids):
            while next_frame < raw_id:
                if not capture.grab():
                    raise RuntimeError("Video ends before requested frame")
                next_frame += 1
            ok, frame = capture.read()
            next_frame += 1
            if not ok:
                raise RuntimeError(f"Cannot decode frame {raw_id}")
            canvas = np.zeros((height + 40, width, 3), dtype=np.uint8)
            image = cv2.resize(frame, (width, height))
            start = 1 if args.trail_steps == 0 else max(1, row - args.trail_steps + 1)
            for j in range(start, row + 1):
                if visible[j - 1] and visible[j] and frame_ids[j] == frame_ids[j - 1] + 1:
                    cv2.line(image, tuple(xy[j - 1]), tuple(xy[j]), (0, 180, 255), 1)
            if visible[row]:
                cv2.circle(image, tuple(xy[row]), 6, (0, 255, 255), 2)
            status = "VISIBLE" if visible[row] else "INVALID/OUTSIDE"
            canvas[40:] = image
            cv2.putText(
                canvas,
                f"GT ID {point_id} | frame {raw_id} | {status}",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            if raw_id == args.select_frame:
                cv2.imwrite(str(args.output / f"{stem}_selection.jpg"), canvas)
    finally:
        capture.release()
        writer.release()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe = np.where(valid[:, None], xyz, np.nan)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*safe.T, linewidth=0.8)
    ax.set(xlabel="Camera X (m)", ylabel="Camera Y (m)", zlabel="Camera Z (m)", title=f"Track4World ID {point_id}")
    ax2 = fig.add_subplot(122)
    for axis, name in enumerate(["X", "Y", "Z"]):
        ax2.plot(timestamps - timestamps[0], safe[:, axis], label=name)
    ax2.set(xlabel="Time (s)", ylabel="Camera position (m)", title="Invalid intervals are gaps")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(args.output / f"{stem}_xyz.png", dpi=160)
    plt.close(fig)
    report = {
        "source": "offline Track4World labels, not model predictions",
        "point_id": point_id,
        "selection_frame": args.select_frame,
        "selection_uv": args.focus_uv,
        "selection_distance_px": distance,
        "explicit_point_id": args.point_id is not None,
        "frames": len(frame_ids),
        "first_frame": int(frame_ids[0]),
        "last_frame": int(frame_ids[-1]),
        "fps": fps,
        "visible_frames": int(visible.sum()),
        "valid_frames": int(valid.sum()),
        "output": str(args.output),
    }
    (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
