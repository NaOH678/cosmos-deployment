import json
import pickle

import lmdb
import numpy as np
import pytest

from wuji_data_pipeline.auxiliary_camera import (
    AuxiliaryCameraWriter,
    load_auxiliary_depth,
)


def test_depth_streams_share_one_namespaced_lmdb_and_preserve_uint16(
    tmp_path,
):
    episode = tmp_path / "episode.inprogress"
    episode.mkdir()
    capture_metadata = tmp_path / "camera_metadata.json"
    capture_metadata.write_text(
        json.dumps({
            "cameras": {
                "head": {
                    "streams": {
                        "depth": {"depth_scale_m": 0.001}
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    writer = AuxiliaryCameraWriter(
        episode,
        depth_stream_names=("head_depth", "right_wrist_depth"),
        infrared_stream_names=(),
        frame_rate=30.0,
        map_size=1 << 26,
        video_fourcc="mp4v",
        queue_capacity=4,
        capture_metadata_path=capture_metadata,
    )
    head = np.arange(48, dtype=np.uint16).reshape(6, 8)
    wrist = np.full((4, 6), 1234, dtype=np.uint16)

    assert writer.enqueue(
        0,
        {
            "head_depth": head,
            "right_wrist_depth": wrist,
        },
        {
            "head_depth": 10.0,
            "right_wrist_depth": 10.01,
        },
        {
            "head_depth": 100,
            "right_wrist_depth": 200,
        },
    )
    assert writer.enqueue(
        1,
        {"head_depth": head + 1},
        {"head_depth": 10.033},
        {"head_depth": 101},
    )

    metadata = writer.finalize(num_steps=2)

    assert np.array_equal(load_auxiliary_depth(episode, "head", 0), head)
    assert np.array_equal(
        load_auxiliary_depth(episode, "right_wrist", 0), wrist
    )
    with pytest.raises(KeyError):
        load_auxiliary_depth(episode, "right_wrist", 1)

    env = lmdb.open(
        str(episode / "auxiliary_camera" / "depth.lmdb"),
        readonly=True,
        lock=False,
    )
    try:
        with env.begin() as txn:
            head_available = pickle.loads(
                txn.get(b"index/depth/head/available")
            )
            wrist_available = pickle.loads(
                txn.get(b"index/depth/right_wrist/available")
            )
            head_sequences = pickle.loads(
                txn.get(b"index/depth/head/sequences")
            )
    finally:
        env.close()

    assert head_available.tolist() == [1, 1]
    assert wrist_available.tolist() == [1, 0]
    assert head_sequences.tolist() == [100, 101]
    assert metadata["capture_metadata"]["cameras"]["head"]["streams"][
        "depth"
    ]["depth_scale_m"] == 0.001


def test_invalid_or_missing_auxiliary_frames_do_not_raise(tmp_path):
    episode = tmp_path / "episode.inprogress"
    episode.mkdir()
    writer = AuxiliaryCameraWriter(
        episode,
        depth_stream_names=("head_depth",),
        infrared_stream_names=("head_ir_left",),
        frame_rate=30.0,
        map_size=1 << 24,
        video_fourcc="mp4v",
        queue_capacity=2,
    )

    assert writer.enqueue(
        0,
        {
            "head_depth": np.zeros((3, 4), dtype=np.uint8),
            "head_ir_left": np.zeros((3, 4), dtype=np.uint16),
        },
        {"head_depth": 1.0, "head_ir_left": 1.0},
    )
    metadata = writer.finalize(num_steps=1)

    assert metadata["writer"]["invalid_frames"] == 2
    assert metadata["depth_streams"] == {}
    assert metadata["infrared_videos"] == {}


def test_infrared_video_keeps_training_step_index_and_deduplicates_source(
    tmp_path,
):
    episode = tmp_path / "episode.inprogress"
    episode.mkdir()
    writer = AuxiliaryCameraWriter(
        episode,
        depth_stream_names=(),
        infrared_stream_names=("head_ir_left", "head_ir_right"),
        frame_rate=6.0,
        map_size=1 << 24,
        video_fourcc="mp4v",
        queue_capacity=4,
    )
    infrared = np.full((48, 64), 37, dtype=np.uint8)
    for step in (0, 1):
        writer.enqueue(
            step,
            {
                "head_ir_left": infrared,
                "head_ir_right": infrared + 1,
            },
            {
                "head_ir_left": 10.0,
                "head_ir_right": 10.0,
            },
            {
                "head_ir_left": 5,
                "head_ir_right": 8,
            },
        )

    metadata = writer.finalize(num_steps=2)

    assert (
        episode / "auxiliary_camera" / "head_ir_left.mp4"
    ).stat().st_size > 0
    assert (
        episode / "auxiliary_camera" / "head_ir_right.mp4"
    ).stat().st_size > 0
    assert metadata["infrared_videos"]["head_ir_left"][
        "saved_frames"
    ] == 1
    assert metadata["infrared_videos"]["head_ir_left"][
        "frame_rate"
    ] == 6.0
    assert metadata["writer"]["duplicate_frames"] == 2

    env = lmdb.open(
        str(episode / "auxiliary_camera" / "depth.lmdb"),
        readonly=True,
        lock=False,
    )
    try:
        with env.begin() as txn:
            steps = pickle.loads(
                txn.get(
                    b"index/video/head_ir_left/training_steps"
                )
            )
    finally:
        env.close()
    assert steps.tolist() == [0]


def test_close_incomplete_discards_queued_work_without_hanging(tmp_path):
    episode = tmp_path / "episode.inprogress"
    episode.mkdir()
    writer = AuxiliaryCameraWriter(
        episode,
        depth_stream_names=("head_depth",),
        infrared_stream_names=(),
        frame_rate=30.0,
        map_size=1 << 24,
        video_fourcc="mp4v",
        queue_capacity=2,
    )
    writer.enqueue(
        0,
        {"head_depth": np.zeros((16, 16), dtype=np.uint16)},
        {"head_depth": 1.0},
    )

    writer.close_incomplete()

    assert writer._closed is True
