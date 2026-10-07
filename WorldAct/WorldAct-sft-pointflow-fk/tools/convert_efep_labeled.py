#!/usr/bin/env python3
"""Convert a 3d_efep labeled episode into the flat labeled delivery schema.

The training pipeline (pointflow_window.prepare_window) reads the flat labeled
schema -- dense [T,N,*] arrays keyed by a persistent point identity.  The
3d_efep export stores ragged per-frame observation lists instead, but its
obs_track links observations into persistent tracks, so the per-point-token
paradigm carries over unchanged: candidates are the tracks alive at the window
anchor.  The efep pool is re-seeded every frame, which is exactly what recovers
surfaces the fixed-first-frame-query delivery can never see (hand flips, late
entries).

Tracks are filtered by whole-episode valid-observation count: a track that
cannot reach `select_min_valid_steps` in ANY window only bloats storage.
Frame-invalid slots carry the last known position/uv forward (valid=0), the
same parked-value semantics as the fixed-first-frame delivery, so the phantom
guard keeps working.

Memory-light: ragged inputs are read per frame via seek; dense outputs are
streamed row by row.  No mmap anywhere (GPFS rejects it).

    .venv/bin/python tools/convert_efep_labeled.py \
        --data-dir /path/to/pf_out/<export>/efep_labeled/<episode> \
        --output-dir <out>/<episode> [--min-valid-obs 32]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REQUIRED_INPUTS = (
    "frame_offsets.npy",
    "obs_track.npy",
    "obs_pos.npy",
    "obs_uv.npy",
    "obs_valid.npy",
    "track_label.npy",
    "frame_indices.npy",
    "timestamps_sec.npy",
    "report.json",
)
# obs_unique.npy: v6.1 (9.24) marks the first track per pixel per frame; training
# points are obs_valid & obs_unique.  Older efep_labeled exports lack it.


def _npy_header(path: Path):
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f"Unsupported NPY version: {path}")
        shape, fortran, dtype = readers[version](stream)
        if fortran or dtype.hasobject:
            raise ValueError(f"Expected numeric C-order array: {path}")
        return shape, dtype, stream.tell()


class RaggedReader:
    """Per-frame row slices of a ragged [total_obs, ...] NPY via seek."""

    def __init__(self, path: Path, offsets: np.ndarray):
        self.shape, self.dtype, self.data_start = _npy_header(path)
        self.offsets = offsets
        self.row_bytes = (
            int(np.prod(self.shape[1:])) * self.dtype.itemsize if len(self.shape) > 1 else self.dtype.itemsize
        )
        self.stream = path.open("rb")

    def frame(self, t: int) -> np.ndarray:
        start, end = int(self.offsets[t]), int(self.offsets[t + 1])
        count = end - start
        self.stream.seek(self.data_start + start * self.row_bytes)
        data = self.stream.read(count * self.row_bytes)
        if len(data) != count * self.row_bytes:
            raise ValueError(f"Truncated read at frame {t}")
        row = np.frombuffer(data, dtype=self.dtype, count=count * int(np.prod(self.shape[1:])))
        return row.reshape((count,) + self.shape[1:]).copy()


class DenseWriter:
    """Stream dense [T, N, ...] rows into a fresh NPY (header first, then rows)."""

    def __init__(self, path: Path, shape, dtype):
        self.stream = path.open("wb")
        # write_array_header_1_0 already emits the magic prefix itself.
        np.lib.format.write_array_header_1_0(
            self.stream,
            {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": tuple(shape)},
        )

    def write_row(self, row: np.ndarray) -> None:
        self.stream.write(row.tobytes(order="C"))

    def close(self) -> None:
        self.stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="efep_labeled/<episode> directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="flat-schema output directory")
    parser.add_argument("--min-valid-obs", type=int, default=32, help="whole-episode valid-observation floor per track")
    parser.add_argument("--video", default=None, help="head video path (older efep reports do not record one)")
    args = parser.parse_args()
    if args.min_valid_obs < 1:
        parser.error("--min-valid-obs must be positive")

    missing = [f for f in REQUIRED_INPUTS if not (args.data_dir / f).is_file()]
    if missing:
        raise ValueError(f"{args.data_dir}: incomplete efep_labeled delivery: {missing}")

    off = np.load(args.data_dir / "frame_offsets.npy", allow_pickle=False)
    tlabel = np.load(args.data_dir / "track_label.npy", allow_pickle=False)
    frame_indices = np.load(args.data_dir / "frame_indices.npy", allow_pickle=False)
    timestamps = np.load(args.data_dir / "timestamps_sec.npy", allow_pickle=False)
    report = json.loads((args.data_dir / "report.json").read_text())
    T = len(off) - 1
    if len(frame_indices) != T or len(timestamps) != T:
        raise ValueError("frame_indices/timestamps_sec do not match frame_offsets")

    # Pass 1: count valid observations per track (only track id + validity streams).
    tracks = RaggedReader(args.data_dir / "obs_track.npy", off)
    valids = RaggedReader(args.data_dir / "obs_valid.npy", off)
    uniques = (
        RaggedReader(args.data_dir / "obs_unique.npy", off) if (args.data_dir / "obs_unique.npy").is_file() else None
    )

    def frame_ok(t: int) -> np.ndarray:
        ok = valids.frame(t).astype(bool)
        if uniques is not None:
            ok &= uniques.frame(t).astype(bool)
        return ok

    counts = np.zeros(len(tlabel), dtype=np.int64)
    for t in range(T):
        tid = tracks.frame(t)
        np.add.at(counts, tid[frame_ok(t)], 1)
    keep = np.flatnonzero(counts >= args.min_valid_obs)
    slot = np.full(len(tlabel), -1, np.int32)
    slot[keep] = np.arange(len(keep))
    n_kept = len(keep)
    if n_kept < 3:
        raise ValueError(f"only {n_kept} tracks pass min_valid_obs={args.min_valid_obs}")

    positions = RaggedReader(args.data_dir / "obs_pos.npy", off)
    uvs = RaggedReader(args.data_dir / "obs_uv.npy", off)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pos_out = DenseWriter(args.output_dir / "position.npy", (T, n_kept, 3), np.float32)
    uv_out = DenseWriter(args.output_dir / "uv_px.npy", (T, n_kept, 2), np.float32)
    valid_out = DenseWriter(args.output_dir / "valid.npy", (T, n_kept), np.bool_)
    # Parked-value semantics: invalid slots carry the last observation forward.
    last_pos = np.zeros((n_kept, 3), dtype=np.float32)
    last_uv = np.zeros((n_kept, 2), dtype=np.float32)
    query_ids = np.full(n_kept, -1, dtype=np.int64)
    try:
        for t in range(T):
            tid = tracks.frame(t)
            valid = frame_ok(t)
            col = slot[tid]
            observed = col >= 0
            cols, seen = np.unique(col[observed], return_index=True)
            if len(cols) != observed.sum():
                # A track observed twice in one frame: keep the first sighting.
                first = np.zeros(len(col), bool)
                first[np.flatnonzero(observed)[seen]] = True
                observed &= first
            cols = col[observed]
            frame_valid = valid[observed]
            pos = positions.frame(t).astype(np.float32)[observed]
            uv = uvs.frame(t).astype(np.float32)[observed]
            last_pos[cols[frame_valid]] = pos[frame_valid]
            last_uv[cols[frame_valid]] = uv[frame_valid]
            fresh = frame_valid & (query_ids[cols] < 0)
            fresh_uv = uv[frame_valid][fresh[frame_valid]].astype(np.int64)
            query_ids[cols[fresh]] = fresh_uv[:, 1] * 640 + fresh_uv[:, 0]
            pos_out.write_row(last_pos)
            uv_out.write_row(last_uv)
            row_valid = np.zeros(n_kept, dtype=np.bool_)
            row_valid[cols[frame_valid]] = True
            valid_out.write_row(row_valid)
            if t % 200 == 0:
                print(f"frame {t}/{T}", flush=True)
    finally:
        pos_out.close()
        uv_out.close()
        valid_out.close()

    np.save(args.output_dir / "region_labels.npy", tlabel[keep])
    np.save(args.output_dir / "query_ids.npy", query_ids)
    np.save(args.output_dir / "frame_indices.npy", frame_indices)
    np.save(args.output_dir / "timestamps_sec.npy", timestamps)
    if (args.data_dir / "regions.json").is_file():
        shutil.copy2(args.data_dir / "regions.json", args.output_dir / "regions.json")

    kept_labels = tlabel[keep]
    per_label = {int(k): int(v) for k, v in zip(*np.unique(kept_labels, return_counts=True))}
    out_report = {
        "episode": report.get("episode", args.data_dir.name),
        "video": args.video or report.get("video"),
        "processing": f"tools/convert_efep_labeled.py from {args.data_dir} (min_valid_obs={args.min_valid_obs})",
        "native_pixel_queries": 640 * 448,
        "frames": T,
        "fps": report.get("fps", 30.0),
        "tracks_total": int(len(tlabel)),
        "tracks_kept": n_kept,
        "tracks_kept_per_label": per_label,
        "full_sequence": True,
        "semantics": report.get("semantics", "manual first-frame candidate regions; not contact labels"),
    }
    (args.output_dir / "report.json").write_text(json.dumps(out_report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"kept {n_kept}/{len(tlabel)} tracks (>= {args.min_valid_obs} valid obs), per label {per_label}\n"
        f"wrote {args.output_dir}"
    )


if __name__ == "__main__":
    main()
