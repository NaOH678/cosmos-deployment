"""Best-effort persistence for analysis-only RealSense streams.

The original dexmanip-compatible LMDB and RGB videos remain owned by
``EpisodeWriter``.  This module writes optional depth/infrared data below an
``auxiliary_camera`` subdirectory and deliberately never raises from the
capture-time enqueue path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import queue
import threading
from typing import Any, Mapping, Optional

import numpy as np


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for auxiliary camera persistence"
        ) from exc
    return cv2


def _require_lmdb():
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError(
            "python lmdb is required for auxiliary depth persistence"
        ) from exc
    return lmdb


def _depth_camera_name(stream_name: str) -> str:
    suffix = "_depth"
    if not stream_name.endswith(suffix):
        raise ValueError(f"not a depth stream name: {stream_name!r}")
    return stream_name[: -len(suffix)]


@dataclass(frozen=True)
class _AuxiliaryJob:
    step: int
    frames: dict[str, np.ndarray]
    timestamps: dict[str, float]
    sequences: dict[str, int]


class AuxiliaryCameraWriter:
    """One bounded background writer for every optional camera stream."""

    def __init__(
        self,
        episode_dir: str | os.PathLike[str],
        *,
        depth_stream_names: tuple[str, ...],
        infrared_stream_names: tuple[str, ...],
        frame_rate: float,
        map_size: int,
        video_fourcc: str,
        infrared_frame_rate: Optional[float] = None,
        queue_capacity: int = 64,
        png_compression: int = 1,
        capture_metadata_path: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("auxiliary queue_capacity must be positive")
        if not 0 <= int(png_compression) <= 9:
            raise ValueError("depth PNG compression must be between 0 and 9")
        self.depth_stream_names = tuple(dict.fromkeys(depth_stream_names))
        self.infrared_stream_names = tuple(
            dict.fromkeys(infrared_stream_names)
        )
        self.stream_names = (
            self.depth_stream_names + self.infrared_stream_names
        )
        if len(set(self.stream_names)) != len(self.stream_names):
            raise ValueError("auxiliary stream names must be unique")
        for name in self.depth_stream_names:
            _depth_camera_name(name)

        self.root = Path(episode_dir) / "auxiliary_camera"
        self.root.mkdir(parents=True, exist_ok=True)
        self.depth_lmdb_dir = self.root / "depth.lmdb"
        self.frame_rate = float(frame_rate)
        self.infrared_frame_rate = float(
            infrared_frame_rate
            if infrared_frame_rate is not None
            else frame_rate
        )
        if self.infrared_frame_rate <= 0.0:
            raise ValueError("infrared_frame_rate must be positive")
        self.video_fourcc = str(video_fourcc)
        self.png_compression = int(png_compression)
        self.capture_metadata_path = (
            Path(capture_metadata_path)
            if capture_metadata_path is not None
            else None
        )

        lmdb = _require_lmdb()
        self._env = lmdb.open(
            str(self.depth_lmdb_dir),
            map_size=int(map_size),
        )
        self._queue: queue.Queue[Optional[_AuxiliaryJob]] = queue.Queue(
            maxsize=int(queue_capacity)
        )
        self._abort_event = threading.Event()
        self._closed = False
        self._worker_stopped = False
        self._video_writers: dict[str, Any] = {}
        self._video_info: dict[str, dict[str, Any]] = {}
        self._depth_info: dict[str, dict[str, Any]] = {}
        self._depth_records: dict[str, dict[int, tuple[float, int]]] = {
            name: {} for name in self.depth_stream_names
        }
        self._infrared_records: dict[
            str, list[tuple[int, float, int]]
        ] = {
            name: [] for name in self.infrared_stream_names
        }
        self._last_saved_sequences = {
            name: -1 for name in self.stream_names
        }
        self.submitted_jobs = 0
        self.queue_drops = 0
        self.invalid_frames = 0
        self.duplicate_frames = 0
        self.write_errors = 0
        self.last_error: Optional[str] = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="wuji-auxiliary-camera-writer",
        )
        self._thread.start()

    def enqueue(
        self,
        step: int,
        frames: Mapping[str, np.ndarray],
        timestamps: Mapping[str, float],
        sequences: Optional[Mapping[str, int]] = None,
    ) -> bool:
        """Try to enqueue one aligned auxiliary sample without blocking."""
        if self._closed or self._abort_event.is_set():
            return False
        prepared: dict[str, np.ndarray] = {}
        prepared_timestamps: dict[str, float] = {}
        prepared_sequences: dict[str, int] = {}
        source_sequences = sequences or {}
        for name, value in frames.items():
            if name not in self.stream_names:
                continue
            image = np.asarray(value)
            expected_dtype = (
                np.dtype(np.uint16)
                if name in self.depth_stream_names
                else np.dtype(np.uint8)
            )
            if image.ndim != 2 or image.dtype != expected_dtype:
                self.invalid_frames += 1
                continue
            timestamp = timestamps.get(name)
            if timestamp is None or not np.isfinite(float(timestamp)):
                self.invalid_frames += 1
                continue
            prepared[name] = np.ascontiguousarray(image)
            prepared_timestamps[name] = float(timestamp)
            prepared_sequences[name] = int(source_sequences.get(name, -1))
        if not prepared:
            return True
        job = _AuxiliaryJob(
            step=int(step),
            frames=prepared,
            timestamps=prepared_timestamps,
            sequences=prepared_sequences,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.queue_drops += 1
            return False
        self.submitted_jobs += 1
        return True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "queue_size": self._queue.qsize(),
            "submitted_jobs": self.submitted_jobs,
            "queue_drops": self.queue_drops,
            "invalid_frames": self.invalid_frames,
            "duplicate_frames": self.duplicate_frames,
            "write_errors": self.write_errors,
            "last_error": self.last_error,
        }

    def finalize(self, num_steps: int) -> dict[str, Any]:
        if self._closed:
            return {}
        self._stop_worker(drain=True)
        metadata = self._build_metadata(int(num_steps))
        try:
            self._write_indexes_and_metadata(metadata, int(num_steps))
            self._env.sync()
        finally:
            self._env.close()
            self._closed = True
        with (self.root / "metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return metadata

    def close_incomplete(self) -> None:
        if self._closed:
            return
        self._abort_event.set()
        self._stop_worker(drain=False)
        try:
            self._env.sync()
        except Exception:
            pass
        try:
            self._env.close()
        except Exception:
            pass
        self._closed = True

    def _stop_worker(self, *, drain: bool) -> None:
        if self._worker_stopped:
            return
        if not drain:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("auxiliary camera writer did not stop")
        self._worker_stopped = True

    def _run(self) -> None:
        try:
            while True:
                job = self._queue.get()
                try:
                    if job is None:
                        return
                    if not self._abort_event.is_set():
                        self._write_job(job)
                except Exception as exc:
                    self.write_errors += 1
                    self.last_error = str(exc)
                finally:
                    self._queue.task_done()
        finally:
            for writer in self._video_writers.values():
                writer.release()
            self._video_writers.clear()

    def _write_job(self, job: _AuxiliaryJob) -> None:
        cv2 = _require_cv2()
        depth_payloads: list[
            tuple[str, bytes, float, int, tuple[int, int]]
        ] = []
        for stream_name in self.depth_stream_names:
            image = job.frames.get(stream_name)
            if image is None:
                continue
            sequence = job.sequences[stream_name]
            if (
                sequence >= 0
                and sequence == self._last_saved_sequences[stream_name]
            ):
                self.duplicate_frames += 1
                continue
            shape = tuple(int(value) for value in image.shape)
            existing = self._depth_info.get(stream_name)
            if (
                existing is not None
                and existing["shape"] != list(shape)
            ):
                raise ValueError(
                    f"{stream_name} depth shape changed from "
                    f"{existing['shape']} to {list(shape)}"
                )
            ok, encoded = cv2.imencode(
                ".png",
                image,
                [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression],
            )
            if not ok:
                raise RuntimeError(
                    f"failed to encode depth frame: {stream_name}"
                )
            depth_payloads.append(
                (
                    stream_name,
                    bytes(encoded),
                    job.timestamps[stream_name],
                    sequence,
                    shape,
                )
            )
        if depth_payloads:
            txn = self._env.begin(write=True)
            try:
                for stream_name, payload, _timestamp, _sequence, _shape in (
                    depth_payloads
                ):
                    camera_name = _depth_camera_name(stream_name)
                    txn.put(
                        f"depth/{camera_name}/{job.step:06d}".encode(),
                        payload,
                    )
                txn.commit()
            except Exception:
                txn.abort()
                raise
            for stream_name, _payload, timestamp, sequence, shape in (
                depth_payloads
            ):
                info = self._depth_info.setdefault(
                    stream_name,
                    {
                        "camera_name": _depth_camera_name(stream_name),
                        "encoding": "png",
                        "dtype": "uint16",
                        "shape": list(shape),
                        "saved_frames": 0,
                    },
                )
                info["saved_frames"] += 1
                self._depth_records[stream_name][job.step] = (
                    timestamp,
                    sequence,
                )
                if sequence >= 0:
                    self._last_saved_sequences[stream_name] = sequence

        for stream_name in self.infrared_stream_names:
            image = job.frames.get(stream_name)
            if image is None:
                continue
            sequence = job.sequences[stream_name]
            if (
                sequence >= 0
                and sequence == self._last_saved_sequences[stream_name]
            ):
                self.duplicate_frames += 1
                continue
            self._write_infrared(
                stream_name,
                image,
                step=job.step,
                timestamp=job.timestamps[stream_name],
                sequence=sequence,
            )
            if sequence >= 0:
                self._last_saved_sequences[stream_name] = sequence

    def _write_infrared(
        self,
        stream_name: str,
        image: np.ndarray,
        *,
        step: int,
        timestamp: float,
        sequence: int,
    ) -> None:
        cv2 = _require_cv2()
        height, width = image.shape
        writer = self._video_writers.get(stream_name)
        if writer is None:
            path = self.root / f"{stream_name}.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*self.video_fourcc),
                self.infrared_frame_rate,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(
                    f"failed to open infrared video writer: {path}"
                )
            self._video_writers[stream_name] = writer
            self._video_info[stream_name] = {
                "path": f"auxiliary_camera/{stream_name}.mp4",
                "encoding": "mono8_converted_to_bgr_for_mp4",
                "size": [width, height],
                "fourcc": self.video_fourcc,
                "frame_rate": self.infrared_frame_rate,
                "saved_frames": 0,
            }
        info = self._video_info[stream_name]
        if info["size"] != [width, height]:
            raise ValueError(
                f"{stream_name} infrared shape changed from "
                f"{info['size']} to {[width, height]}"
            )
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        info["saved_frames"] += 1
        self._infrared_records[stream_name].append(
            (int(step), float(timestamp), int(sequence))
        )

    def _capture_metadata(self) -> dict[str, Any]:
        path = self.capture_metadata_path
        if path is None or not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _build_metadata(self, num_steps: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "best_effort": True,
            "num_training_steps": int(num_steps),
            "depth_lmdb": "auxiliary_camera/depth.lmdb",
            "depth_key_format": "depth/{camera}/{training_step:06d}",
            "depth_streams": self._depth_info,
            "infrared_videos": self._video_info,
            "capture_metadata": self._capture_metadata(),
            "writer": {
                "queue_capacity": self._queue.maxsize,
                "submitted_jobs": self.submitted_jobs,
                "queue_drops": self.queue_drops,
                "invalid_frames": self.invalid_frames,
                "duplicate_frames": self.duplicate_frames,
                "write_errors": self.write_errors,
                "last_error": self.last_error,
                "depth_png_compression": self.png_compression,
            },
        }

    def _write_indexes_and_metadata(
        self,
        metadata: Mapping[str, Any],
        num_steps: int,
    ) -> None:
        txn = self._env.begin(write=True)
        try:
            for stream_name in self.depth_stream_names:
                camera_name = _depth_camera_name(stream_name)
                available = np.zeros(num_steps, dtype=np.uint8)
                timestamps = np.full(num_steps, np.nan, dtype=np.float64)
                sequences = np.full(num_steps, -1, dtype=np.int64)
                for step, (timestamp, sequence) in self._depth_records[
                    stream_name
                ].items():
                    if 0 <= step < num_steps:
                        available[step] = 1
                        timestamps[step] = timestamp
                        sequences[step] = sequence
                prefix = f"index/depth/{camera_name}"
                txn.put(
                    f"{prefix}/available".encode(),
                    pickle.dumps(available),
                )
                txn.put(
                    f"{prefix}/timestamps".encode(),
                    pickle.dumps(timestamps),
                )
                txn.put(
                    f"{prefix}/sequences".encode(),
                    pickle.dumps(sequences),
                )
            for stream_name, records in self._infrared_records.items():
                steps = np.asarray(
                    [record[0] for record in records], dtype=np.int64
                )
                timestamps = np.asarray(
                    [record[1] for record in records], dtype=np.float64
                )
                sequences = np.asarray(
                    [record[2] for record in records], dtype=np.int64
                )
                prefix = f"index/video/{stream_name}"
                txn.put(
                    f"{prefix}/training_steps".encode(),
                    pickle.dumps(steps),
                )
                txn.put(
                    f"{prefix}/timestamps".encode(),
                    pickle.dumps(timestamps),
                )
                txn.put(
                    f"{prefix}/sequences".encode(),
                    pickle.dumps(sequences),
                )
            serialized = pickle.dumps(dict(metadata))
            txn.put(b"__metadata__", serialized)
            txn.put(b"meta_info", serialized)
            txn.commit()
        except Exception:
            txn.abort()
            raise


def load_auxiliary_depth(
    episode_dir: str | os.PathLike[str],
    camera_name: str,
    training_step: int,
) -> np.ndarray:
    """Load one losslessly stored ``uint16`` depth image."""
    cv2 = _require_cv2()
    lmdb = _require_lmdb()
    path = Path(episode_dir) / "auxiliary_camera" / "depth.lmdb"
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
    )
    try:
        with env.begin() as txn:
            payload = txn.get(
                f"depth/{camera_name}/{int(training_step):06d}".encode()
            )
    finally:
        env.close()
    if payload is None:
        raise KeyError(
            f"no auxiliary depth for {camera_name} at step {training_step}"
        )
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if image is None or image.dtype != np.uint16 or image.ndim != 2:
        raise ValueError(
            f"invalid stored depth for {camera_name} at step {training_step}"
        )
    return image
