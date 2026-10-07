import json

import numpy as np
import pytest

from cosmos_framework.data.generator.action.pointflow_source import PointFlowSource, resize_pointflow_metadata
from cosmos_framework.data.pointflow_window import PointFlowTiming


def test_manifest_missing_is_not_unlabeled(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "episodes": [{"name": "plain", "pointflow_source": None}]}))
    source = PointFlowSource(path, timing=PointFlowTiming())
    assert source.load("plain", np.arange(33), 15, (8, 8)) is None
    with pytest.raises(KeyError):
        source.load("missing", np.arange(33), 15, (8, 8))


def test_resize_composes_head_offset_and_pixel_centers():
    point = {"metadata": {"uv_to_video": np.array([[2, 0, 0.5], [0, 2, 100.5]]), "video_size_wh": [1280, 1000]}}
    resize_pointflow_metadata(point, (1280, 1000), (640, 500), (640, 512))
    np.testing.assert_allclose(point["metadata"]["uv_to_video"], [[1, 0, 0], [0, 1, 50]])
    np.testing.assert_array_equal(point["metadata"]["video_size_wh"], [640, 512])
    with pytest.raises(ValueError, match="stale"):
        resize_pointflow_metadata(point, (1280, 1000), (640, 500), (640, 512))


def test_duplicate_episode_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    row = {"name": "plain", "pointflow_source": None}
    path.write_text(json.dumps({"schema_version": 1, "episodes": [row, row]}))
    with pytest.raises(ValueError, match="Duplicate"):
        PointFlowSource(path, timing=PointFlowTiming())


def test_source_rejects_misaligned_frames_and_time(tmp_path, monkeypatch):
    import cosmos_framework.data.generator.action.pointflow_source as module

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [
                    {
                        "name": "dense",
                        "pointflow_source": {
                            "path": "relative_dense",
                            "uv_to_video": [[1, 0, 0], [0, 1, 0]],
                            "video_size_wh": [8, 8],
                        },
                    }
                ],
            }
        )
    )
    source = PointFlowSource(path, timing=PointFlowTiming())
    assert source.entries["dense"]["path"] == tmp_path / "relative_dense"
    ids = np.arange(0, 65, 2)
    window = {
        "raw_frame_ids": ids + 1,
        "timestamps_sec": ids / 30,
        **{key: None for key in ("point_ids", "anchor_xyz", "anchor_uv", "normal", "color", "coord", "feat")},
        **{key: None for key in ("grid_coord", "original_to_voxel", "voxel_representatives", "coord_shift")},
        **{key: None for key in ("intrinsics_normalized", "image_size_wh", "target_displacement", "target_valid")},
        "target_seconds": ids / 30,
        "point_block_seconds": np.arange(4, 33, 4) / 15,
        "point_ids": np.empty(0, dtype=np.int64),  # len() is read to flag an empty anchor
    }
    monkeypatch.setattr(module, "prepare_window", lambda *args, **kwargs: window)
    with pytest.raises(ValueError, match="frame IDs"):
        source.load("dense", ids, 30, (8, 8))
    window["raw_frame_ids"] = ids
    window["timestamps_sec"] = ids / 30 + 0.1
    with pytest.raises(ValueError, match="timestamps"):
        source.load("dense", ids, 30, (8, 8))
    # ``video_size_wh`` names the canvas the stored affine maps into, and
    # ``resize_pointflow_metadata`` rebases it onto whatever tensor it is handed.
    # It must therefore stay the manifest's value: overwriting it with the canvas
    # the caller happens to return makes the align step see "recorded == actual",
    # skip the very rescale it exists to perform, and place every point wrongly.
    window["timestamps_sec"] = ids / 30
    loaded = source.load("dense", ids, 30, (16, 16))
    assert loaded["metadata"]["video_size_wh"].tolist() == [8, 8]
    assert loaded["metadata"]["returned_video_size_wh"].tolist() == [16, 16]
    np.testing.assert_allclose(loaded["metadata"]["uv_to_video"], [[1, 0, 0], [0, 1, 0]])

    # Simulation capture can start off cadence; its declared constant offset
    # must be respected without accepting an actual one-frame misalignment.
    document = json.loads(path.read_text())
    document["episodes"][0]["pointflow_source"]["timestamp_offset_sec"] = -1 / 60
    path.write_text(json.dumps(document))
    source = PointFlowSource(path, timing=PointFlowTiming())
    window["timestamps_sec"] = ids / 30 - 1 / 60
    source.load("dense", ids, 30, (8, 8))
    window["timestamps_sec"] += 1 / 30
    with pytest.raises(ValueError, match="timestamps"):
        source.load("dense", ids, 30, (8, 8))
    document["episodes"][0]["pointflow_source"]["timestamp_offset_sec"] = float("nan")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="timestamp_offset_sec"):
        PointFlowSource(path, timing=PointFlowTiming())
