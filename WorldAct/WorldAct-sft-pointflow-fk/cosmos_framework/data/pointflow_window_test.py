"""Timeline, identity and future-label isolation tests for the dense reader."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from cosmos_framework.data.pointflow_window import PointFlowTiming, prepare_window, read_frame, window_rows


class WindowTests(unittest.TestCase):
    def test_timing_from_cosmos_and_malformed_timeline(self):
        timing = PointFlowTiming.from_cosmos(
            {"fps": 10, "chunk_length": 8},
            {"temporal_compression_factor": 4, "encode_exact_durations": [9]},
        )
        self.assertEqual(timing, PointFlowTiming(10, 8, 4))
        with self.assertRaises(ValueError):
            PointFlowTiming.from_cosmos(
                {"fps": 15, "chunk_length": 32},
                {"temporal_compression_factor": 4, "encode_exact_durations": [25]},
            )
        for stamps in (np.array([]), np.array([0, np.nan]), np.array([0, 0])):
            with self.assertRaises(ValueError):
                window_rows(np.arange(len(stamps)), stamps, 0)

    def test_timeline_and_short_window(self):
        ids = np.arange(100, 200)
        stamps = np.arange(100) / 30
        rows = window_rows(ids, stamps, 110)
        np.testing.assert_array_equal(ids[rows], np.arange(110, 175, 2))
        with self.assertRaises(ValueError):
            window_rows(ids, stamps, 150)

    def test_frame_seek_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.npy"
            array = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
            np.save(path, array)
            np.testing.assert_array_equal(read_frame(path, 1), array[1])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                read_frame(path, 2)

    def test_top_n_goes_below_the_fraction_floor_to_one_point(self):
        """The fraction floors at 3; select_top_n exists precisely to go lower."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, y = np.meshgrid(np.arange(8), np.arange(8))
            anchor = np.stack((x * 0.03, y * 0.03, np.ones_like(x)), axis=-1).astype(np.float32)
            positions = np.repeat(anchor[None], 65, axis=0)
            # Exactly one point moves; every other point is static.
            positions[:, 0, 0, 0] += np.arange(65, dtype=np.float32) * 0.01
            uv = np.stack((x + 5, y + 5), axis=-1).astype(np.float32)
            arrays = {
                "position": positions,
                "uv_px": np.repeat(uv[None], 65, axis=0),
                "valid": np.ones((65, 8, 8), dtype=bool),
                "frame_indices": np.arange(65),
                "timestamps_sec": np.arange(65) / 30,
                "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 65, axis=0),
            }
            for key, value in arrays.items():
                np.save(root / f"{key}.npy", value)
            (root / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "coordinate": "camera_depthanythingv3",
                        "metric_scale": True,
                        "inference_width": 32,
                        "inference_height": 32,
                        "video": "mock.mp4",
                    }
                )
            )
            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.read.return_value = (True, np.zeros((32, 32, 3), dtype=np.uint8))
            with patch("cv2.VideoCapture", return_value=capture):
                one = prepare_window(root, max_points=64, select_top_n=1)
                self.assertEqual(list(one["point_ids"]), [0], "the only moving point must be picked")
                self.assertEqual(one["target_displacement"].shape, (32, 1, 3))
                # The fraction path keeps its floor of 3 even when asked for fewer.
                floored = prepare_window(root, max_points=64, select_motion_fraction=1e-6)
                self.assertEqual(len(floored["point_ids"]), 3)
                # select_top_n takes precedence when both are given.
                both = prepare_window(root, max_points=64, select_motion_fraction=0.5, select_top_n=1)
                self.assertEqual(list(both["point_ids"]), [0])

    def _grid_episode(self, root):
        """65 anchors: 64 on a dense 1 cm grid, plus one isolated point 0.5 m away.

        Anchor 10 (inside the dense grid) and anchor 64 (the isolated one) both move;
        the isolated one moves more, so an unguarded ranking always picks it.
        """
        x, y = np.meshgrid(np.arange(8), np.arange(8))
        anchor = np.zeros((65, 3), dtype=np.float32)
        anchor[:64] = np.stack((x.ravel() * 0.01, y.ravel() * 0.01, np.ones(64)), axis=-1)
        anchor[64] = (0.5, 0.5, 1.0)
        positions = np.repeat(anchor[None], 65, axis=0)
        positions[:, 10, 0] += np.arange(65, dtype=np.float32) * 0.005
        positions[:, 64, 0] += np.arange(65, dtype=np.float32) * 0.010
        uv = np.zeros((65, 2), dtype=np.float32)
        for key, value in {
            "position": positions,
            "uv_px": np.repeat(uv[None], 65, axis=0),
            "valid": np.ones((65, 65), dtype=bool),
            "frame_indices": np.arange(65),
            "timestamps_sec": np.arange(65) / 30,
            "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 65, axis=0),
        }.items():
            np.save(root / f"{key}.npy", value)
        (root / "COMPLETE.json").write_text(
            json.dumps(
                {
                    "coordinate": "camera_depthanythingv3",
                    "metric_scale": True,
                    "inference_width": 32,
                    "inference_height": 32,
                    "video": "mock.mp4",
                }
            )
        )
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((32, 32, 3), dtype=np.uint8))
        return patch("cv2.VideoCapture", return_value=capture)

    def test_voxel_member_guard_demotes_the_isolated_mover(self):
        """A lost track is alone in its voxel; the guard is what keeps it out of the top."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._grid_episode(root):
                unguarded = prepare_window(root, max_points=65, select_top_n=1)
                guarded = prepare_window(root, max_points=65, select_top_n=1, min_voxel_members=2)
            self.assertEqual(list(unguarded["point_ids"]), [64], "the isolated mover ranks first")
            self.assertEqual(list(guarded["point_ids"]), [10], "the guard must demote it")

    def test_phantom_guard_demotes_uv_locked_drifters(self):
        """uv glued while 3D drifts = depth failure; the guard ranks it last."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._grid_episode(root):
                uv = np.load(root / "uv_px.npy")
                # Point 10's motion is real: its uv tracks it across the window.
                uv[:, 10, 0] += np.arange(65, dtype=np.float32) * 3.0
                np.save(root / "uv_px.npy", uv)
                unguarded = prepare_window(root, max_points=65, select_top_n=1)
                guarded = prepare_window(root, max_points=65, select_top_n=1, select_phantom_guard=True)
                # Without a selection budget the guard drops the drifter outright.
                full_guarded = prepare_window(root, max_points=65, select_phantom_guard=True)
            self.assertEqual(list(unguarded["point_ids"]), [64], "the isolated mover ranks first")
            self.assertEqual(list(guarded["point_ids"]), [10], "the uv-locked drifter must be demoted")
            self.assertNotIn(64, list(full_guarded["point_ids"]), "no budget: the drifter is dropped")
            self.assertIn(10, list(full_guarded["point_ids"]), "real motion survives")

    def test_supervise_cluster_masks_the_loss_without_touching_the_cloud(self):
        """The geometry encoder must still see every point; only the loss mask shrinks."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._grid_episode(root):
                window = prepare_window(root, max_points=65, min_voxel_members=2, supervise_cluster_n=1)
            self.assertEqual(len(window["point_ids"]), 65, "the cloud must be untouched")
            self.assertEqual(window["anchor_xyz"].shape, (65, 3))
            self.assertEqual(window["target_displacement"].shape, (32, 65, 3))
            self.assertEqual(int((window["target_valid"].sum(0) > 0).sum()), 1)

    def test_supervise_cluster_is_capped_by_its_voxel(self):
        """Points come from one voxel, so asking for more than it holds yields fewer."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._grid_episode(root):
                window = prepare_window(root, max_points=65, min_voxel_members=2, supervise_cluster_n=99)
            supervised = int((window["target_valid"].sum(0) > 0).sum())
            self.assertGreaterEqual(supervised, 1)
            self.assertLessEqual(supervised, 64, "never more than the voxel holds")

    def test_labeled_delivery_without_intrinsics(self):
        """report.json + flat (T,N,*) arrays + region_labels, no COMPLETE.json/intrinsics."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, y = np.meshgrid(np.arange(8), np.arange(8))
            anchor = np.stack((x.ravel() * 0.03 + 0.3, y.ravel() * 0.03 + 0.3, np.ones(64)), axis=-1).astype(np.float32)
            positions = np.repeat(anchor[None], 65, axis=0)
            uv = np.stack((x.ravel() + 320.5, y.ravel() + 224.5), axis=-1).astype(np.float32)
            labels = np.where(np.arange(64) < 32, 2, 3).astype(np.uint8)
            for key, value in {
                "position": positions,
                "uv_px": np.repeat(uv[None], 65, axis=0),
                "valid": np.ones((65, 64), dtype=bool),
                "frame_indices": np.arange(65),
                "timestamps_sec": np.arange(65) / 30,
                "region_labels": labels,
            }.items():
                np.save(root / f"{key}.npy", value)
            (root / "report.json").write_text(json.dumps({"video": "mock.mp4", "native_pixel_queries": 640 * 448}))
            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.read.return_value = (True, np.zeros((448, 640, 3), dtype=np.uint8))
            with patch("cv2.VideoCapture", return_value=capture):
                window = prepare_window(root, max_points=64)
                self.assertEqual(len(window["point_ids"]), 64)
                self.assertTrue(np.isnan(window["intrinsics_normalized"]).all(), "no intrinsics -> NaN marker")
                self.assertTrue(np.isnan(window["projection_error_px"]).all(), "the projection guard is skipped")
                objects = prepare_window(root, max_points=64, select_regions=(3,))
                self.assertEqual(len(objects["point_ids"]), 32)
                self.assertTrue((objects["point_ids"] >= 32).all(), "region 3 is the second half")
                hands = prepare_window(root, max_points=64, select_regions="2")
                self.assertTrue((hands["point_ids"] < 32).all())
                (root / "region_labels.npy").unlink()
                with self.assertRaises(ValueError):
                    prepare_window(root, max_points=64, select_regions=(3,))

    def test_labeled_delivery_without_native_pixel_queries(self):
        """Older deliveries lack the field; the uv range of the data guards the grid instead."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, y = np.meshgrid(np.arange(8), np.arange(8))
            anchor = np.stack((x.ravel() * 0.03 + 0.3, y.ravel() * 0.03 + 0.3, np.ones(64)), axis=-1).astype(np.float32)
            positions = np.repeat(anchor[None], 65, axis=0)
            uv = np.stack((x.ravel() + 320.5, y.ravel() + 224.5), axis=-1).astype(np.float32)
            for key, value in {
                "position": positions,
                "uv_px": np.repeat(uv[None], 65, axis=0),
                "valid": np.ones((65, 64), dtype=bool),
                "frame_indices": np.arange(65),
                "timestamps_sec": np.arange(65) / 30,
                "region_labels": np.full(64, 3, dtype=np.uint8),
            }.items():
                np.save(root / f"{key}.npy", value)
            (root / "report.json").write_text(json.dumps({"video": "mock.mp4"}))
            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.read.return_value = (True, np.zeros((448, 640, 3), dtype=np.uint8))
            with patch("cv2.VideoCapture", return_value=capture):
                window = prepare_window(root, max_points=64)
                self.assertEqual(len(window["point_ids"]), 64)
                bad = np.repeat(uv[None], 65, axis=0)
                bad[0, 0, 1] = 448.5
                np.save(root / "uv_px.npy", bad)
                with self.assertRaises(ValueError):
                    prepare_window(root, max_points=64)

    def _write_quota_episode(self, root):
        """64 tracks: 16 hand (fast), 32 object (medium), 16 surface (slow)."""
        x, y = np.meshgrid(np.arange(8), np.arange(8))
        anchor = np.stack((x.ravel() * 0.03 + 0.3, y.ravel() * 0.03 + 0.3, np.ones(64)), axis=-1).astype(np.float32)
        speed = np.where(np.arange(64) < 16, 0.01, np.where(np.arange(64) < 48, 0.001, 0.0001)).astype(np.float32)
        positions = anchor[None] + (
            np.arange(65, dtype=np.float32)[:, None, None]
            * speed[None, :, None]
            * np.array([[[1.0, 0, 0]]], dtype=np.float32)
        )
        uv = np.stack((x.ravel() + 320.5, y.ravel() + 224.5), axis=-1).astype(np.float32)
        labels = np.where(np.arange(64) < 16, 2, np.where(np.arange(64) < 48, 3, 4)).astype(np.uint8)
        for key, value in {
            "position": positions,
            "uv_px": np.repeat(uv[None], 65, axis=0),
            "valid": np.ones((65, 64), dtype=bool),
            "frame_indices": np.arange(65),
            "timestamps_sec": np.arange(65) / 30,
            "region_labels": labels,
        }.items():
            np.save(root / f"{key}.npy", value)
        (root / "report.json").write_text(json.dumps({"video": "mock.mp4", "native_pixel_queries": 640 * 448}))
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((448, 640, 3), dtype=np.uint8))
        return capture

    def test_region_quotas_stratify_the_top_n_budget(self):
        """Flat top-N would take only hand points; quotas guarantee region coverage."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self._write_quota_episode(root)
            with patch("cv2.VideoCapture", return_value=capture):
                flat = prepare_window(root, max_points=64, select_top_n=16)
                self.assertTrue((flat["point_ids"] < 16).all(), "flat top-16 is all hand (fastest)")
                window = prepare_window(
                    root,
                    max_points=64,
                    select_top_n=16,
                    select_region_quotas=((2, 0.25), (3, 0.5), (4, 0.25)),
                )
                ids = window["point_ids"]
                self.assertEqual(len(ids), 16)
                self.assertEqual((ids < 16).sum(), 4, "hand quota 0.25x16")
                self.assertEqual(((ids >= 16) & (ids < 48)).sum(), 8, "object quota 0.5x16")
                self.assertEqual((ids >= 48).sum(), 4, "surface quota 0.25x16")
                # float32 displacement magnitudes carry last-ulp noise, so ties are
                # not guaranteed; the contract is deterministic under a fixed seed.
                again = prepare_window(
                    root,
                    max_points=64,
                    select_top_n=16,
                    select_region_quotas=((2, 0.25), (3, 0.5), (4, 0.25)),
                )
                self.assertTrue(np.array_equal(ids, again["point_ids"]), "same seed, same selection")

    def test_region_quotas_refill_shortfall_globally(self):
        """A region with fewer points than its quota yields the budget to the others."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self._write_quota_episode(root)
            labels = np.load(root / "region_labels.npy")
            labels[2:16] = 3  # only tracks 0,1 stay hand-labeled
            np.save(root / "region_labels.npy", labels)
            # keep speeds consistent with the new labels: only tracks 0,1 move fast
            positions = np.load(root / "position.npy")
            speed = np.where(np.arange(64) < 2, 0.01, np.where(np.arange(64) < 48, 0.001, 0.0001)).astype(np.float32)
            positions = positions[0:1] + (
                np.arange(65, dtype=np.float32)[:, None, None]
                * speed[None, :, None]
                * np.array([[[1.0, 0, 0]]], dtype=np.float32)
            )
            np.save(root / "position.npy", positions)
            with patch("cv2.VideoCapture", return_value=capture):
                window = prepare_window(
                    root,
                    max_points=64,
                    select_top_n=16,
                    select_region_quotas=((2, 0.5), (3, 0.25), (4, 0.25)),
                )
                ids = window["point_ids"]
                self.assertEqual(len(ids), 16, "shortfall is redistributed, total stays 16")
                hand_labels = np.load(root / "region_labels.npy")[ids]
                self.assertEqual((hand_labels == 2).sum(), 2, "only two hands exist")
                self.assertTrue((ids[:2] == np.array([0, 1])).all(), "the two hands are the fastest overall")

    def test_region_quotas_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self._write_quota_episode(root)
            with patch("cv2.VideoCapture", return_value=capture):
                with self.assertRaises(ValueError, msg="quotas need the top_n budget"):
                    prepare_window(root, max_points=64, select_region_quotas=((2, 0.5),))
                with self.assertRaises(ValueError, msg="fractions must sum to <= 1"):
                    prepare_window(root, max_points=64, select_top_n=16, select_region_quotas=((2, 0.7), (3, 0.7)))
                with self.assertRaises(ValueError, msg="duplicate labels"):
                    prepare_window(root, max_points=64, select_top_n=16, select_region_quotas=((2, 0.3), (2, 0.3)))

    def test_select_min_valid_steps_demotes_unreliable_tracks(self):
        """A track that dies mid-window must lose the motion ranking to a slower survivor."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self._grid_episode(root):
                # The isolated mover sprints early (0.1 m per target step), then its
                # track dies after 3 target steps: the mean-over-valid ranking still
                # puts it first, only the retention guard can demote it.
                positions = np.load(root / "position.npy")
                positions[:, 64, 0] += np.arange(65, dtype=np.float32) * 0.05
                np.save(root / "position.npy", positions)
                valid = np.ones((65, 65), dtype=bool)
                valid[8:, 64] = False
                np.save(root / "valid.npy", valid)
                unguarded = prepare_window(root, max_points=65, select_top_n=1)
                guarded = prepare_window(root, max_points=65, select_top_n=1, select_min_valid_steps=30)
            self.assertEqual(list(unguarded["point_ids"]), [64], "unguarded ranking picks the fast mover")
            self.assertEqual(list(guarded["point_ids"]), [10], "the guard must demote the dead track")

    def test_future_labels_do_not_select_input_and_validity_can_return(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, y = np.meshgrid(np.arange(8), np.arange(8))
            anchor = np.stack((x * 0.03, y * 0.03, np.ones_like(x)), axis=-1).astype(np.float32)
            positions = np.repeat(anchor[None], 65, axis=0)
            uv = np.stack((x + 5, y + 5), axis=-1).astype(np.float32)
            arrays = {
                "position": positions,
                "uv_px": np.repeat(uv[None], 65, axis=0),
                "valid": np.ones((65, 8, 8), dtype=bool),
                "frame_indices": np.arange(65),
                "timestamps_sec": np.arange(65) / 30,
                "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 65, axis=0),
            }
            for key, value in arrays.items():
                np.save(root / f"{key}.npy", value)
            (root / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "coordinate": "camera_depthanythingv3",
                        "metric_scale": True,
                        "inference_width": 32,
                        "inference_height": 32,
                        "video": "mock.mp4",
                    }
                )
            )
            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.read.return_value = (True, np.zeros((32, 32, 3), dtype=np.uint8))
            with patch("cv2.VideoCapture", return_value=capture):
                first = prepare_window(root, max_points=32)
                positions[2] = np.nan
                positions[4] += np.array([0.1, 0, 0], dtype=np.float32)
                np.save(root / "position.npy", positions)
                second = prepare_window(root, max_points=32)
                from cosmos_framework.data.pointflow_dataset import PointFlowWindowDataset, collate_pointflow_windows

                dataset = PointFlowWindowDataset.from_cosmos(
                    root.parent,
                    [root.name],
                    {"fps": 10, "chunk_length": 8, "sample_stride": 4},
                    {"temporal_compression_factor": 4, "encode_exact_durations": [9]},
                    max_points=32,
                )
                self.assertEqual(len(dataset), 11)
                a, b = dataset[0], dataset[-1]
                self.assertEqual(b["metadata"]["start_frame"], 40)
                np.testing.assert_array_equal(a["metadata"]["raw_frame_ids"], np.arange(0, 25, 3))
                np.testing.assert_allclose(a["metadata"]["point_block_seconds"], [0.4, 0.8])
                np.testing.assert_array_equal(a["inputs"]["point_ids"], dataset[0]["inputs"]["point_ids"])
                self.assertNotIn("target_displacement", a["inputs"])
                valid = arrays["valid"].copy()
                valid[0] = False
                np.save(root / "valid.npy", valid)
                empty = dataset[0]
                self.assertTrue(empty["metadata"]["empty_anchor"])
                self.assertEqual(empty["targets"]["displacement"].shape, (8, 0, 3))
                # Different point budgets produce genuinely ragged batches.
                small = PointFlowWindowDataset(root.parent, [root.name], timing=dataset.timing, max_points=16)[4]
                packed = collate_pointflow_windows([empty, a, small])
                inp = packed["inputs"]
                self.assertEqual(packed["targets"]["displacement"].shape, (8, 48, 3))
                np.testing.assert_array_equal(inp["point_offsets"].numpy(), [0, 32, 48])
                self.assertTrue((inp["voxel_batch"][inp["original_to_voxel"]] == inp["point_batch"]).all())
                self.assertTrue((inp["point_batch"][inp["voxel_representatives"]] == inp["voxel_batch"]).all())
                all_empty = collate_pointflow_windows([empty, empty])
                self.assertEqual(all_empty["targets"]["valid"].shape, (8, 0))
                self.assertFalse(all_empty["inputs"]["has_geometry"].any())
            for key in ("point_ids", "feat", "grid_coord", "original_to_voxel", "anchor_uv"):
                np.testing.assert_array_equal(first[key], second[key])
            self.assertFalse(second["target_valid"][0].any())
            self.assertTrue(second["target_valid"][1].all())
            self.assertTrue(np.isfinite(second["target_displacement"]).all())
            np.testing.assert_allclose(second["target_displacement"][1, :, 0], 0.1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
