"""Bounded, latest-frame shared memory transport for camera images.

The transport deliberately has no ROS dependency.  One camera producer owns a
fixed-size ring in ``/dev/shm``; any number of GUI/recorder consumers may read
it without blocking the producer.  A slow consumer skips overwritten frames
instead of applying backpressure to camera capture or robot control.
"""

from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Optional

import numpy as np


MAGIC = b"WUJICAM1"
VERSION = 1
HEADER_BYTES = 128
SLOT_HEADER_BYTES = 64
DEFAULT_DIRECTORY = Path("/dev/shm/wuji_camera_v1")

# magic, version, header bytes, slot-header bytes, capacity, slot bytes,
# reserved, latest sequence, producer pid, producer start monotonic ns,
# producer-side capture failures
_HEADER = struct.Struct("<8sIIIIIIQQQQ")
_SLOT = struct.Struct("<QQQIIIIQ")
_WRITE_SEQUENCE_OFFSET = struct.calcsize("<8sIIIIII")
_CAPTURE_FAILURES_OFFSET = struct.calcsize("<8sIIIIIIQQQ")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_DTYPE_TO_CODE = {
    np.dtype(np.uint8): 0,
    np.dtype(np.uint16): 1,
}
_CODE_TO_DTYPE = {
    code: dtype for dtype, code in _DTYPE_TO_CODE.items()
}


@dataclass(frozen=True)
class SharedFrame:
    camera_name: str
    sequence: int
    monotonic_ns: int
    system_ns: int
    image: np.ndarray

    @property
    def timestamp(self) -> float:
        """System-clock timestamp in seconds, compatible with ROS wall time."""
        return self.system_ns * 1e-9


@dataclass(frozen=True)
class FrameBatch:
    frames: tuple[SharedFrame, ...]
    overwritten: int
    producer_generation: int
    capture_failures: int


def _validate_name(camera_name: str) -> str:
    normalized = str(camera_name).strip()
    if not _SAFE_NAME.fullmatch(normalized):
        raise ValueError(f"invalid camera name: {camera_name!r}")
    return normalized


def ring_path(
    camera_name: str, directory: str | os.PathLike[str] = DEFAULT_DIRECTORY
) -> Path:
    return Path(directory) / f"{_validate_name(camera_name)}.ring"


class SharedFrameWriter:
    """Single-producer fixed-capacity image ring.

    Existing RGB producers use the default ``uint8``/three-channel contract.
    Auxiliary RealSense streams may opt into ``uint16`` single-channel depth
    or ``uint8`` single-channel infrared without changing the RGB ring names
    or consumers.
    """

    def __init__(
        self,
        camera_name: str,
        *,
        width: int,
        height: int,
        channels: int = 3,
        dtype: np.dtype | type = np.uint8,
        capacity: int = 64,
        directory: str | os.PathLike[str] = DEFAULT_DIRECTORY,
    ) -> None:
        if min(width, height, channels, capacity) <= 0:
            raise ValueError("frame dimensions and ring capacity must be positive")
        normalized_dtype = np.dtype(dtype)
        if normalized_dtype not in _DTYPE_TO_CODE:
            raise ValueError(
                f"unsupported shared camera dtype: {normalized_dtype}"
            )
        self.camera_name = _validate_name(camera_name)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.dtype = normalized_dtype
        self.capacity = int(capacity)
        self.slot_bytes = (
            self.width
            * self.height
            * self.channels
            * self.dtype.itemsize
        )
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = ring_path(self.camera_name, self.directory)
        total_bytes = (
            HEADER_BYTES
            + self.capacity * (SLOT_HEADER_BYTES + self.slot_bytes)
        )

        # Publish the fully sized inode atomically.  Existing readers keep a
        # valid old mapping until they notice the inode change and reopen.
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.camera_name}.", dir=str(self.directory)
        )
        try:
            os.ftruncate(fd, total_bytes)
            os.fchmod(fd, 0o644)
            mapping = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_WRITE)
            os.replace(temporary, self.path)
        except Exception:
            os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        self._fd = fd
        self._mapping = mapping
        self._sequence = 0
        self._capture_failures = 0
        self._generation = __import__("time").monotonic_ns()
        _HEADER.pack_into(
            self._mapping,
            0,
            MAGIC,
            VERSION,
            HEADER_BYTES,
            SLOT_HEADER_BYTES,
            self.capacity,
            self.slot_bytes,
            _DTYPE_TO_CODE[self.dtype],
            0,
            os.getpid(),
            self._generation,
            0,
        )

    @property
    def sequence(self) -> int:
        return self._sequence

    def note_capture_failure(self) -> None:
        self._capture_failures += 1
        struct.pack_into(
            "<Q",
            self._mapping,
            _CAPTURE_FAILURES_OFFSET,
            self._capture_failures,
        )

    def write(
        self,
        image: np.ndarray,
        *,
        monotonic_ns: int,
        system_ns: int,
    ) -> int:
        frame = np.asarray(image)
        if frame.dtype != self.dtype:
            raise ValueError(
                f"shared camera image must be {self.dtype}, got {frame.dtype}"
            )
        if frame.ndim == 2:
            height, width = frame.shape
            channels = 1
        elif frame.ndim == 3:
            height, width, channels = frame.shape
        else:
            raise ValueError(
                "shared camera image must be HxW or HxWxC, "
                f"got shape {frame.shape}"
            )
        if channels != self.channels:
            raise ValueError(
                f"camera channels changed: expected {self.channels}, got {channels}"
            )
        byte_count = int(frame.nbytes)
        if byte_count > self.slot_bytes:
            raise ValueError(
                f"camera frame exceeds slot: {width}x{height}x{channels} "
                f"> {self.width}x{self.height}x{self.channels}"
            )
        contiguous = np.ascontiguousarray(frame)
        sequence = self._sequence + 1
        slot_index = (sequence - 1) % self.capacity
        slot_offset = HEADER_BYTES + slot_index * (
            SLOT_HEADER_BYTES + self.slot_bytes
        )
        data_offset = slot_offset + SLOT_HEADER_BYTES

        # An incomplete slot always has mismatched begin/end markers.
        struct.pack_into("<Q", self._mapping, slot_offset, 0)
        self._mapping[data_offset : data_offset + byte_count] = contiguous.tobytes()
        _SLOT.pack_into(
            self._mapping,
            slot_offset,
            sequence,
            int(monotonic_ns),
            int(system_ns),
            int(width),
            int(height),
            int(channels),
            byte_count,
            sequence,
        )
        # Publish the sequence only after the complete slot is visible.
        struct.pack_into(
            "<Q", self._mapping, _WRITE_SEQUENCE_OFFSET, sequence
        )
        self._sequence = sequence
        return sequence

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            try:
                mapping.flush()
            except Exception:
                pass
            mapping.close()
            self._mapping = None
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None

    def __enter__(self) -> "SharedFrameWriter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class SharedFrameReader:
    """Non-blocking consumer for a camera ring.

    ``read_since`` returns all still-resident frames after a sequence number.
    If the consumer fell behind, ``overwritten`` reports exactly how many
    frames were lost.
    """

    def __init__(
        self,
        camera_name: str,
        *,
        directory: str | os.PathLike[str] = DEFAULT_DIRECTORY,
    ) -> None:
        self.camera_name = _validate_name(camera_name)
        self.directory = Path(directory)
        self.path = ring_path(self.camera_name, self.directory)
        self._fd: Optional[int] = None
        self._mapping: Optional[mmap.mmap] = None
        self._inode: Optional[tuple[int, int]] = None
        self._size = 0
        self._capacity = 0
        self._slot_bytes = 0
        self._generation = 0
        self._dtype = np.dtype(np.uint8)

    @property
    def producer_generation(self) -> int:
        return self._generation

    def _disconnect(self) -> None:
        mapping, self._mapping = self._mapping, None
        fd, self._fd = self._fd, None
        if mapping is not None:
            mapping.close()
        if fd is not None:
            os.close(fd)
        self._inode = None
        self._size = 0
        self._capacity = 0
        self._slot_bytes = 0
        self._generation = 0
        self._dtype = np.dtype(np.uint8)

    def _connect(self) -> bool:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._disconnect()
            return False
        inode = (stat.st_dev, stat.st_ino)
        if (
            self._mapping is not None
            and inode == self._inode
            and stat.st_size == self._size
        ):
            return True
        self._disconnect()
        fd = os.open(self.path, os.O_RDONLY)
        try:
            mapping = mmap.mmap(fd, stat.st_size, access=mmap.ACCESS_READ)
            header = _HEADER.unpack_from(mapping, 0)
            (
                magic,
                version,
                header_bytes,
                slot_header_bytes,
                capacity,
                slot_bytes,
                dtype_code,
                _latest,
                _producer_pid,
                generation,
                _capture_failures,
            ) = header
            expected = header_bytes + capacity * (slot_header_bytes + slot_bytes)
            if (
                magic != MAGIC
                or version != VERSION
                or header_bytes != HEADER_BYTES
                or slot_header_bytes != SLOT_HEADER_BYTES
                or expected != stat.st_size
                or int(dtype_code) not in _CODE_TO_DTYPE
            ):
                raise ValueError(f"invalid shared camera ring: {self.path}")
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._mapping = mapping
        self._inode = inode
        self._size = stat.st_size
        self._capacity = int(capacity)
        self._slot_bytes = int(slot_bytes)
        self._generation = int(generation)
        self._dtype = _CODE_TO_DTYPE[int(dtype_code)]
        return True

    def read_since(self, last_sequence: int) -> FrameBatch:
        previous_generation = self._generation
        if not self._connect() or self._mapping is None:
            return FrameBatch((), 0, 0, 0)
        header = _HEADER.unpack_from(self._mapping, 0)
        latest = int(header[7])
        generation = int(header[9])
        capture_failures = int(header[10])
        if generation != previous_generation and previous_generation != 0:
            last_sequence = 0
        if latest <= 0:
            return FrameBatch((), 0, generation, capture_failures)
        oldest = max(1, latest - self._capacity + 1)
        requested = max(1, int(last_sequence) + 1)
        overwritten = max(0, oldest - requested)
        first = max(requested, oldest)
        frames = []
        for sequence in range(first, latest + 1):
            frame = self._read_sequence(sequence)
            if frame is not None:
                frames.append(frame)
        return FrameBatch(
            tuple(frames), overwritten, generation, capture_failures
        )

    def latest(self) -> Optional[SharedFrame]:
        if not self._connect() or self._mapping is None:
            return None
        latest = int(_HEADER.unpack_from(self._mapping, 0)[7])
        return self._read_sequence(latest) if latest > 0 else None

    def _read_sequence(self, sequence: int) -> Optional[SharedFrame]:
        if self._mapping is None or sequence <= 0:
            return None
        slot_index = (sequence - 1) % self._capacity
        slot_offset = HEADER_BYTES + slot_index * (
            SLOT_HEADER_BYTES + self._slot_bytes
        )
        first = _SLOT.unpack_from(self._mapping, slot_offset)
        (
            begin,
            monotonic_ns,
            system_ns,
            width,
            height,
            channels,
            byte_count,
            end,
        ) = first
        expected_bytes = (
            int(width)
            * int(height)
            * int(channels)
            * self._dtype.itemsize
        )
        if (
            begin != sequence
            or end != sequence
            or expected_bytes != byte_count
            or byte_count <= 0
            or byte_count > self._slot_bytes
        ):
            return None
        data_offset = slot_offset + SLOT_HEADER_BYTES
        payload = bytes(self._mapping[data_offset : data_offset + byte_count])
        second_begin = struct.unpack_from("<Q", self._mapping, slot_offset)[0]
        second_end = struct.unpack_from(
            "<Q", self._mapping, slot_offset + _SLOT.size - 8
        )[0]
        if second_begin != sequence or second_end != sequence:
            return None
        shape = (
            (int(height), int(width))
            if int(channels) == 1
            else (int(height), int(width), int(channels))
        )
        image = np.frombuffer(payload, dtype=self._dtype).reshape(shape)
        return SharedFrame(
            camera_name=self.camera_name,
            sequence=int(sequence),
            monotonic_ns=int(monotonic_ns),
            system_ns=int(system_ns),
            image=image,
        )

    def close(self) -> None:
        self._disconnect()

    def __enter__(self) -> "SharedFrameReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
