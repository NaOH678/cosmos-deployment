# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Round-trip tests for the per-window PointFlow selection cache."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from cosmos_framework.data.generator.action.pointflow_source import PointFlowSource, window_seed
from cosmos_framework.data.pointflow_window import PointFlowTiming, prepare_window
from cosmos_framework.data.pointflow_window_cache import (
    CONFIG_KEYS,
    STORED_KEYS,
    build_episode_archive,
    read_cached_window,
    validate_manifest,
    write_episode_archive,
    write_manifest,
)


def _write_episode(root: Path, frames=69, points=64):
    """Minimal labeled delivery: 64 tracks on an 8x8 grid, half hand half object."""
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    anchor = np.stack((x.ravel() * 0.03 + 0.3, y.ravel() * 0.03 + 0.3, np.ones(points)), axis=-1).astype(np.float32)
    speed = np.where(np.arange(points) < 32, 0.002, 0.0002).astype(np.float32)
    positions = anchor[None] + np.arange(frames, dtype=np.float32)[:, None, None] * speed[None, :, None] * np.array(
        [[[1.0, 0, 0]]], dtype=np.float32
    )
    uv = np.stack((x.ravel() + 320.5, y.ravel() + 224.5), axis=-1).astype(np.float32)
    labels = np.where(np.arange(points) < 32, 2, 3).astype(np.uint8)
    for key, value in {
        "position": positions,
        "uv_px": np.repeat(uv[None], frames, axis=0),
        "valid": np.ones((frames, points), dtype=bool),
        "frame_indices": np.arange(frames),
        "timestamps_sec": np.arange(frames) / 30,
        "region_labels": labels,
    }.items():
        np.save(root / f"{key}.npy", value)
    (root / "report.json").write_text(json.dumps({"video": "mock.mp4", "native_pixel_queries": 640 * 448}))
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((448, 640, 3), dtype=np.uint8))
    return capture


def _write_manifest(root: Path, episode_dir: Path, name="dense"):
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [
                    {
                        "name": name,
                        "pointflow_source": {
                            "path": episode_dir.name,
                            "uv_to_video": [[1, 0, 0], [0, 1.0714, 362.04]],
                            "video_size_wh": [640, 842],
                        },
                    }
                ],
            }
        )
    )
    return root / "manifest.json"


class WindowCacheTests(unittest.TestCase):
    def test_roundtrip_matches_online_and_miss_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_dir = root / "dense"
            episode_dir.mkdir()
            capture = _write_episode(episode_dir)
            manifest = _write_manifest(root, episode_dir)
            cache_root = root / "window_cache"
            cache_root.mkdir()

            kwargs = dict(
                timing=PointFlowTiming(),
                max_points=64,
                seed=42,
                select_top_n=16,
                select_region_quotas=((2, 0.5), (3, 0.5)),
            )
            source = PointFlowSource(manifest, **kwargs)

            with patch("cv2.VideoCapture", return_value=capture):
                online = prepare_window(
                    episode_dir,
                    0,
                    max_points=64,
                    seed=window_seed(42, "dense", 0),
                    timing=PointFlowTiming(),
                    allow_empty=True,
                    select_top_n=16,
                    select_region_quotas=((2, 0.5), (3, 0.5)),
                )
            write_episode_archive(cache_root / "dense.npz", {0: online})
            config = dict(source.selection_config())
            config.update({"fps": 15.0, "chunk_length": 32, "sample_stride": 1})
            write_manifest(cache_root, config)

            cached_source = PointFlowSource(manifest, window_cache_root=cache_root, **kwargs)
            frame_ids = np.arange(0, 65, 2)
            hit = cached_source.load("dense", frame_ids, 30.0, (640, 842))
            with patch("cv2.VideoCapture", return_value=capture):
                want = source.load("dense", frame_ids, 30.0, (640, 842))
            for key in want["inputs"]:
                np.testing.assert_array_equal(hit["inputs"][key], want["inputs"][key], err_msg=key)
            for key in want["targets"]:
                np.testing.assert_array_equal(hit["targets"][key], want["targets"][key], err_msg=key)
            self.assertEqual(cached_source.cache_misses, 0)

            # A start not in the archive replays the online path identically.
            late_ids = np.arange(2, 67, 2)
            with patch("cv2.VideoCapture", return_value=capture):
                want_late = source.load("dense", late_ids, 30.0, (640, 842))
                miss = cached_source.load("dense", late_ids, 30.0, (640, 842))
            self.assertEqual(cached_source.cache_misses, 1)
            np.testing.assert_array_equal(miss["inputs"]["point_ids"], want_late["inputs"]["point_ids"])
            np.testing.assert_array_equal(miss["targets"]["displacement"], want_late["targets"]["displacement"])
            archive = np.load(cache_root / "dense.npz")
            self.assertIsNone(read_cached_window(archive, 2, PointFlowTiming()))
            archive.close()

    def test_streaming_builder_matches_online(self):
        """build_episode_archive (rolling buffer + one video pass) must be bit-identical
        to per-window online preparation, including the derived window_seed."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_dir = root / "dense"
            episode_dir.mkdir()
            capture = _write_episode(episode_dir)  # 69 frames -> starts 0..4
            out = root / "cache"
            out.mkdir()
            kwargs = dict(
                max_points=64,
                timing=PointFlowTiming(),
                allow_empty=True,
                select_top_n=16,
                select_region_quotas=((2, 0.5), (3, 0.5)),
            )
            with patch("cv2.VideoCapture", return_value=capture):
                n, failures = build_episode_archive(
                    episode_dir,
                    out / "dense.npz",
                    fps=15.0,
                    chunk_length=32,
                    sample_stride=1,
                    seed_fn=lambda start: window_seed(42, "dense", start),
                    overwrite=True,
                    **kwargs,
                )
                self.assertEqual(failures, [])
                self.assertEqual(n, 5)
                archive = np.load(out / "dense.npz")
                for start in range(5):
                    online = prepare_window(episode_dir, start, seed=window_seed(42, "dense", start), **kwargs)
                    cached = read_cached_window(archive, start, PointFlowTiming())
                    self.assertIsNotNone(cached)
                    for key in STORED_KEYS:
                        np.testing.assert_array_equal(cached[key], online[key], err_msg=f"{start}/{key}")
                archive.close()

    def test_manifest_config_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_dir = root / "dense"
            episode_dir.mkdir()
            _write_episode(episode_dir)
            manifest = _write_manifest(root, episode_dir)
            cache_root = root / "window_cache"
            cache_root.mkdir()
            config = {
                key: value
                for key, value in {
                    "fps": 15.0,
                    "chunk_length": 32,
                    "sample_stride": 1,
                    "max_points": 64,
                    "voxel_size": 0.02,
                    "seed": 42,
                    "select_motion_fraction": 0.0,
                    "select_top_n": 16,
                    "min_voxel_members": 0,
                    "supervise_cluster_n": 0,
                    "select_regions": (),
                    "select_region_quotas": (),
                    "select_min_valid_steps": 0,
                    "select_phantom_guard": False,
                    "phantom_guard_disp_mm": 30.0,
                    "phantom_guard_uv_px": 2.0,
                }.items()
            }
            write_manifest(cache_root, config)
            validate_manifest(cache_root, config, CONFIG_KEYS)
            bad = dict(config, select_top_n=500)
            with self.assertRaises(ValueError):
                validate_manifest(cache_root, bad, CONFIG_KEYS)
            with self.assertRaises(ValueError):
                PointFlowSource(
                    manifest, timing=PointFlowTiming(), max_points=64, select_top_n=500, window_cache_root=cache_root
                )


if __name__ == "__main__":
    unittest.main()
