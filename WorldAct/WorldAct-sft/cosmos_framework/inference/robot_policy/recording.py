"""Optional bounded, asynchronous recording of actual policy requests.

COSMOS_RECORDING_DIR enables recording; unset means no filesystem/thread work.
Only explicitly selected metadata is recorded, never request headers or auth.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class AsyncRequestRecorder:
    """Own CPU snapshots. Never perform disk I/O on the inference thread."""

    def __init__(self, root: str | Path, manifest: dict, capacity: int = 16):
        if capacity < 1:
            raise ValueError("recording capacity must be positive")
        self.directory = Path(root).expanduser() / f"run_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        self.manifest = manifest
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._counts = dict(enqueued=0, written=0, dropped=0, errors=0, stats_errors=0)
        self._drained = False
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, name="cosmos-request-recorder", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    @classmethod
    def from_environment(cls, config: Any) -> AsyncRequestRecorder | None:
        root = os.environ.get("COSMOS_RECORDING_DIR")
        if not root:
            return None
        try:
            capacity = int(os.environ.get("COSMOS_RECORDING_QUEUE_SIZE", "16"))
            raw = config.model_dump(mode="json")
            raw["model"].pop("credential_path", None)
            sections = (
                "deployment",
                "model",
                "image_preprocessing",
                "coordinate_frames",
                "timestamps",
                "timestamp",
                "safety",
            )
            manifest = {
                "format_version": 1,
                "recording_run_id": os.environ.get("COSMOS_RECORDING_RUN_ID"),
                "enabled": True,
                "created_wall_ns": time.time_ns(),
                "source_path": str(Path(__file__).resolve()),
                "config": {key: raw[key] for key in sections if key in raw},
                "units": {
                    "state": "native 27D: EEF xyz metres, quaternion xyzw, hand radians; joint profile: arm and hand radians",
                    "raw_actions": "native model output, condition row removed, before smoothing/normalization",
                    "images": "decoded RGB uint8, original camera dimensions",
                },
                "queue_capacity": capacity,
            }
            torch = sys.modules.get("torch")
            if torch is not None:
                manifest["runtime"] = {
                    "torch": torch.__version__,
                    "cuda_build": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_initialized() else None,
                }
            manifest["timing_note"] = (
                "Recording adds no CUDA synchronization; model_ms uses existing adapter synchronization. Record enqueue/snapshot time is excluded from adapter_total_ms."
            )
            return cls(root, manifest, capacity)
        except Exception:
            _LOG.exception("Invalid Cosmos recording configuration; recording disabled")
            return None

    def stats(self) -> dict:
        with self._lock:
            return dict(self._counts, pending=self._queue.qsize(), closed=self._stop.is_set(), drained=self._drained)

    def submit(self, images: dict, state: np.ndarray | None, raw_actions: np.ndarray | None, metadata: dict) -> bool:
        with self._lock:
            if self._stop.is_set() or self._queue.full():
                self._counts["dropped"] += 1
                return False
            self._sequence += 1
            sequence = self._sequence
        # Snapshot CPU arrays only; never transfer tensors or synchronize CUDA.
        try:
            arrays = {name: np.array(image, copy=True) for name, image in images.items()}
            if state is not None:
                arrays["state"] = np.array(state, copy=True)
            if raw_actions is not None:
                arrays["raw_actions"] = np.array(raw_actions, copy=True)
            info = dict(metadata)
            info["sequence"] = sequence
            info["recording_run_id"] = self.manifest.get("recording_run_id")
            info["finite"] = {key: bool(np.isfinite(array).all()) for key, array in arrays.items()}
            info["shapes"] = {key: list(array.shape) for key, array in arrays.items()}
            # Round-trip produces an immutable snapshot of small wire metadata.
            info = json.loads(json.dumps(_json_safe(info), allow_nan=False))
            with self._lock:
                if self._stop.is_set():
                    self._counts["dropped"] += 1
                    return False
                self._queue.put_nowait((sequence, arrays, info))
                self._counts["enqueued"] += 1
            return True
        except queue.Full:
            with self._lock:
                self._counts["dropped"] += 1
            return False
        except Exception:
            with self._lock:
                self._counts["errors"] += 1
            _LOG.exception("Could not snapshot policy request for recording")
            return False

    def _write_json(self, name: str, data: dict) -> None:
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def _write_record(self, sequence: int, arrays: dict, info: dict) -> None:
        stem = f"{sequence:08d}"
        temporary = self.directory / f"{stem}.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(self.directory / f"{stem}.npz")
        self._write_json(f"{stem}.json", info)  # JSON is the commit marker.

    def _run(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            source_files = [Path(__file__).resolve(), Path(__file__).resolve().with_name("adapters.py")]
            self.manifest["source_sha256"] = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
            }
            self._write_json("manifest.json", self.manifest)
            self._write_json("stats.json", self.stats())
        except Exception:
            with self._lock:
                self._counts["errors"] += 1
            _LOG.exception("Could not initialize Cosmos request recording")
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._write_record(*item)
                with self._lock:
                    self._counts["written"] += 1
            except Exception:
                with self._lock:
                    self._counts["errors"] += 1
                _LOG.exception("Could not write Cosmos request recording")
            finally:
                self._queue.task_done()
            try:
                self._write_json("stats.json", self.stats())
            except Exception:
                with self._lock:
                    self._counts["stats_errors"] += 1
        with self._lock:
            self._drained = True
        try:
            self._write_json("stats.json", self.stats())
        except Exception:
            with self._lock:
                self._counts["stats_errors"] += 1

    def close(self, timeout: float = 3.0) -> bool:
        with self._lock:
            self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            _LOG.warning("Cosmos recorder drain incomplete after %.1fs: %s", timeout, self.stats())
        return not self._thread.is_alive()
