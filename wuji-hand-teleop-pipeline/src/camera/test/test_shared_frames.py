from pathlib import Path

import numpy as np

from stereocamera.camera_manager import enabled_camera_configs
from stereocamera.shared_frames import SharedFrameReader, SharedFrameWriter


def _image(value: int, width: int = 4, height: int = 3) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_shared_ring_delivers_timestamped_bgr_frames(tmp_path: Path):
    writer = SharedFrameWriter(
        "head", width=4, height=3, capacity=4, directory=tmp_path
    )
    reader = SharedFrameReader("head", directory=tmp_path)
    try:
        writer.write(_image(7), monotonic_ns=100, system_ns=200)
        batch = reader.read_since(0)
        assert batch.overwritten == 0
        assert len(batch.frames) == 1
        assert batch.frames[0].sequence == 1
        assert batch.frames[0].monotonic_ns == 100
        assert batch.frames[0].system_ns == 200
        assert np.array_equal(batch.frames[0].image, _image(7))
    finally:
        reader.close()
        writer.close()


def test_shared_ring_delivers_uint16_depth_without_changing_rgb_contract(
    tmp_path: Path,
):
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
    writer = SharedFrameWriter(
        "head_depth",
        width=4,
        height=3,
        channels=1,
        dtype=np.uint16,
        capacity=4,
        directory=tmp_path,
    )
    reader = SharedFrameReader("head_depth", directory=tmp_path)
    try:
        writer.write(depth, monotonic_ns=300, system_ns=400)
        frame = reader.latest()

        assert frame is not None
        assert frame.image.dtype == np.uint16
        assert frame.image.shape == (3, 4)
        assert np.array_equal(frame.image, depth)
    finally:
        reader.close()
        writer.close()


def test_shared_ring_delivers_single_channel_uint8_infrared(tmp_path: Path):
    infrared = np.full((3, 4), 19, dtype=np.uint8)
    writer = SharedFrameWriter(
        "head_ir_left",
        width=4,
        height=3,
        channels=1,
        capacity=2,
        directory=tmp_path,
    )
    reader = SharedFrameReader("head_ir_left", directory=tmp_path)
    try:
        writer.write(infrared, monotonic_ns=10, system_ns=20)

        assert np.array_equal(reader.latest().image, infrared)
    finally:
        reader.close()
        writer.close()


def test_slow_reader_reports_overwritten_frames_without_blocking_writer(tmp_path: Path):
    writer = SharedFrameWriter(
        "left_wrist", width=4, height=3, capacity=2, directory=tmp_path
    )
    reader = SharedFrameReader("left_wrist", directory=tmp_path)
    try:
        for sequence in range(1, 6):
            writer.write(
                _image(sequence),
                monotonic_ns=sequence,
                system_ns=sequence,
            )
        batch = reader.read_since(0)
        assert batch.overwritten == 3
        assert [frame.sequence for frame in batch.frames] == [4, 5]
        assert np.all(batch.frames[-1].image == 5)
    finally:
        reader.close()
        writer.close()


def test_reader_reopens_after_producer_replaces_ring(tmp_path: Path):
    first = SharedFrameWriter(
        "right_wrist", width=4, height=3, capacity=2, directory=tmp_path
    )
    reader = SharedFrameReader("right_wrist", directory=tmp_path)
    first.write(_image(1), monotonic_ns=1, system_ns=1)
    first_batch = reader.read_since(0)
    first_generation = first_batch.producer_generation

    second = SharedFrameWriter(
        "right_wrist", width=4, height=3, capacity=2, directory=tmp_path
    )
    try:
        second.write(_image(9), monotonic_ns=9, system_ns=9)
        second_batch = reader.read_since(first_batch.frames[-1].sequence)
        assert second_batch.producer_generation != first_generation
        assert [frame.sequence for frame in second_batch.frames] == [1]
        assert np.all(second_batch.frames[0].image == 9)
    finally:
        reader.close()
        first.close()
        second.close()


def test_placeholder_camera_serial_is_not_started():
    config = {
        "cameras": {
            "head": {"enabled": True, "type": "usb"},
            "left_wrist": {
                "enabled": True,
                "serial_number": "YOUR_LEFT_WRIST_CAM_SERIAL",
            },
            "right_wrist": {"enabled": False},
        }
    }
    assert list(enabled_camera_configs(config)) == ["head"]


def test_right_arm_starts_only_head_and_right_wrist_cameras():
    config = {
        "cameras": {
            name: {"enabled": True, "serial_number": f"serial-{name}"}
            for name in ("head", "left_wrist", "right_wrist")
        }
    }

    assert list(enabled_camera_configs(config, active_arm="right")) == [
        "head",
        "right_wrist",
    ]
