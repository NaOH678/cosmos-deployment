# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-window PointFlow selection cache.

``prepare_window`` re-reads ~200 MB of full-track frames and re-runs the whole
selection pipeline for every occurrence of every window; on a network
filesystem that dominates the dataloader.  The selection is deterministic for a
fixed (episode, start_frame, config), so this module stores the final per-window
result once (a few hundred KB) and the training read path replays it.

Layout::

    <cache_root>/
      cache_manifest.json      # schema_version + enumeration + selection config
      <episode>.npz            # keys "<start_frame>/<field>"

The manifest records the enumeration (fps/chunk_length/sample_stride) and every
selection knob; the reader refuses a cache built with a different configuration
rather than silently returning the wrong points.  A missing episode file or
window key falls back to online ``prepare_window`` (identical result, slower).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from cosmos_framework.data.pointflow_window import prepare_window, read_frame, window_rows

SCHEMA_VERSION = 1

# Everything ``pointflow_sample`` reads from a prepared window, stored verbatim.
# ``target_seconds`` / ``point_block_seconds`` are recomputed from the timing at
# load; ``anchor_rgb`` and ``projection_error_px`` stay online-only (visualization
# aids, never consumed by training).
STORED_KEYS = (
    "point_ids",
    "anchor_xyz",
    "anchor_uv",
    "normal",
    "color",
    "coord",
    "feat",
    "grid_coord",
    "original_to_voxel",
    "voxel_representatives",
    "coord_shift",
    "intrinsics_normalized",
    "image_size_wh",
    "target_displacement",
    "target_valid",
    "raw_frame_ids",
    "timestamps_sec",
)

# Enumeration knobs, validated at the dataset level (it owns the window lattice),
# and selection knobs, validated at the PointFlowSource level (it owns selection).
ENUMERATION_KEYS = ("fps", "chunk_length", "sample_stride")
SELECTION_KEYS = (
    "max_points",
    "voxel_size",
    "seed",
    "select_motion_fraction",
    "select_top_n",
    "min_voxel_members",
    "supervise_cluster_n",
    "select_regions",
    "select_region_quotas",
    "select_min_valid_steps",
    "select_phantom_guard",
    "phantom_guard_disp_mm",
    "phantom_guard_uv_px",
)
CONFIG_KEYS = ENUMERATION_KEYS + SELECTION_KEYS


def write_episode_archive(path: Path, windows: dict[int, dict]) -> None:
    """Write one episode's windows atomically (tmp + rename, GPFS-safe)."""
    flat = {}
    for start, window in sorted(windows.items()):
        for key in STORED_KEYS:
            flat[f"{start}/{key}"] = window[key]
    tmp = path.with_name(path.name + ".tmp")
    # np.savez appends ".npz" to bare paths; a file handle writes exactly `tmp`.
    with tmp.open("wb") as stream:
        np.savez(stream, **flat)
    tmp.rename(path)


def write_manifest(cache_root: Path, config: dict) -> None:
    document = {"schema_version": SCHEMA_VERSION, "config": config}
    (cache_root / "cache_manifest.json").write_text(json.dumps(document, indent=2) + "\n")


def build_episode_archive(
    ep_path: Path,
    out_path: Path,
    *,
    fps: float,
    chunk_length: int,
    sample_stride: int,
    seed_fn,
    overwrite: bool = False,
    **prepare_kwargs,
):
    """Compute every window of one episode in a single streaming pass.

    Adjacent windows share 64 of 65 frames, so track arrays are read exactly
    once (rolling buffer) and the anchor video is decoded in one forward pass
    instead of a seek per window — ~15x less I/O than per-window preparation.
    ``seed_fn(start)`` must reproduce the training-side per-window seed
    (``pointflow_source.window_seed``).  Returns (n_written, failures).
    """
    if out_path.is_file() and not overwrite:
        return -1, []
    frame_ids = np.load(ep_path / "frame_indices.npy", allow_pickle=False)
    timestamps = np.load(ep_path / "timestamps_sec.npy", allow_pickle=False)
    source_stride = int(round(30.0 / fps))
    last_start = len(frame_ids) - 1 - chunk_length * source_stride

    buffers = {}  # labeled row -> (position, uv_px, valid)
    slot = {"position": 0, "uv_px": 1, "valid": 2}

    def provider(name, row):
        return buffers[row][slot[name]]

    video = json.loads((ep_path / "report.json").read_text())["video"]
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise ValueError(f"Cannot open head video: {video}")
    video_pos = 0
    anchor_bgr = None

    windows, failures = {}, []
    try:
        for start in range(0, last_start + 1, sample_stride):
            try:
                rows = window_rows(frame_ids, timestamps, start, fps, chunk_length)
                lo, hi = int(rows[0]), int(rows[-1])
                for row in list(buffers):
                    if row < lo:
                        del buffers[row]
                for row in range(lo, hi + 1):
                    if row not in buffers:
                        buffers[row] = (
                            read_frame(ep_path / "position.npy", row),
                            read_frame(ep_path / "uv_px.npy", row),
                            read_frame(ep_path / "valid.npy", row),
                        )
                target = int(frame_ids[rows[0]])
                if target < video_pos - 1:
                    raise ValueError(f"anchor video frames are not monotonic at start={start}")
                while video_pos <= target:
                    ok, anchor_bgr = capture.read()
                    if not ok:
                        raise ValueError(f"head video ended before frame {target}")
                    video_pos += 1
                windows[start] = prepare_window(
                    ep_path,
                    start,
                    seed=seed_fn(start),
                    frame_provider=provider,
                    anchor_frame_bgr=anchor_bgr,
                    **prepare_kwargs,
                )
            except Exception as error:  # degenerate windows fall back to online prep
                failures.append((start, str(error)[:200]))
    finally:
        capture.release()
    write_episode_archive(out_path, windows)
    return len(windows), failures


def _normalize_config_value(value):
    """JSON round-trips tuples to lists; compare sequence config by value."""
    if isinstance(value, (tuple, list)):
        return [_normalize_config_value(item) for item in value]
    return value


def validate_manifest(cache_root: Path, config: dict, keys=CONFIG_KEYS) -> None:
    """Refuse a cache whose enumeration/selection config differs from ours."""
    path = cache_root / "cache_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run tools/build_pointflow_window_cache.py first")
    document = json.loads(path.read_text())
    if int(document.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported pointflow window cache schema in {path}")
    stored = document.get("config", {})
    for key in keys:
        want = config.get(key)
        got = stored.get(key)
        if key == "voxel_size" or key.startswith("phantom_guard_") or key == "select_motion_fraction":
            if not np.isclose(float(want), float(got), atol=1e-9):
                raise ValueError(f"pointflow window cache config mismatch on {key}: cache={got} vs run={want}")
        elif _normalize_config_value(want) != _normalize_config_value(got):
            raise ValueError(f"pointflow window cache config mismatch on {key}: cache={got} vs run={want}")


def read_cached_window(archive, start: int, timing) -> dict | None:
    """Rebuild the prepare_window output dict from an open npz archive.

    Returns None on a missing key (caller falls back to online preparation).
    """
    prefix = f"{start}/"
    try:
        window = {key: archive[f"{prefix}{key}"] for key in STORED_KEYS}
    except KeyError:
        return None
    timestamps = window["timestamps_sec"]
    window["target_seconds"] = timestamps - timestamps[0]
    window["point_block_seconds"] = (
        np.arange(timing.steps_per_token, timing.steps + 1, timing.steps_per_token) / timing.fps
    )
    return window
