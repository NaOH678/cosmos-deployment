"""Non-blocking JSONL diagnostics for the robot deployment client."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class DeploymentTraceWriter:
    """Write diagnostic events on a dedicated thread.

    Control and ROS callbacks only enqueue JSON-compatible dictionaries.  A
    full queue drops diagnostics instead of delaying robot commands.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        session_id: str,
        recording_run_id: str = "",
        queue_size: int = 4096,
        flush_interval_s: float = 0.5,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("deployment trace queue size must be positive")
        if flush_interval_s <= 0.0:
            raise ValueError("deployment trace flush interval must be positive")
        trace_directory = Path(directory).expanduser().resolve()
        trace_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = trace_directory / (
            f"deployment_trace_{timestamp}_{session_id[:8]}.jsonl"
        )
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=int(queue_size)
        )
        self._flush_interval_s = float(flush_interval_s)
        self._stop_event = threading.Event()
        self._dropped_events = 0
        self._writer_error = ""
        self._written_events = 0
        self._closed_cleanly = False
        self._inflight_event = False
        self._session_id = session_id
        self._recording_run_id = recording_run_id
        self._dropped_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="wuji-deployment-trace",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_events(self) -> int:
        with self._dropped_lock:
            return self._dropped_events

    @property
    def writer_error(self) -> str:
        with self._dropped_lock:
            return self._writer_error

    @property
    def written_events(self) -> int:
        with self._dropped_lock:
            return self._written_events

    @property
    def closed_cleanly(self) -> bool:
        with self._dropped_lock:
            return self._closed_cleanly

    @property
    def queued_events(self) -> int:
        return self._queue.qsize()

    def record(self, event: str, **fields: Any) -> None:
        if self._stop_event.is_set():
            if self.writer_error:
                with self._dropped_lock:
                    self._dropped_events += 1
            return
        item = {
            "wall_time": time.time(),
            "monotonic_time": time.monotonic(),
            "event": str(event),
            **fields,
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._dropped_lock:
                self._dropped_events += 1

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(0.0, float(timeout_s)))

    def _run(self) -> None:
        try:
            self._write_events()
        except Exception as exc:
            # Diagnostics must fail visibly without raising in a control
            # callback. Count the failed/in-flight record and queued records;
            # a flush failure may also invalidate previous buffered writes.
            with self._dropped_lock:
                self._writer_error = f"{type(exc).__name__}: {exc}"
                self._closed_cleanly = False
                self._dropped_events += int(self._inflight_event) + self._queue.qsize()
            self._stop_event.set()
            print(f"Deployment trace writer failed: {self._writer_error}", file=sys.stderr)
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def _write_events(self) -> None:
        last_flush = time.monotonic()
        with self.path.open("w", encoding="utf-8") as stream:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=self._flush_interval_s)
                except queue.Empty:
                    item = None
                if item is not None:
                    self._inflight_event = True
                    stream.write(
                        json.dumps(
                            item,
                            ensure_ascii=True,
                            separators=(",", ":"),
                            default=_json_default,
                        )
                    )
                    stream.write("\n")
                    with self._dropped_lock:
                        self._written_events += 1
                    self._inflight_event = False
                now = time.monotonic()
                if now - last_flush >= self._flush_interval_s:
                    stream.flush()
                    last_flush = now
            stream.write(json.dumps({
                "event": "trace_summary", "wall_time": time.time(),
                "monotonic_time": time.monotonic(),
                "session_id": self._session_id,
                "recording_run_id": self._recording_run_id,
                "written_events": self.written_events,
                "dropped_events": self.dropped_events,
                "queued_events": self.queued_events,
                "writer_error": "", "closed_cleanly": True,
                "complete": self.dropped_events == 0,
            }, separators=(",", ":")) + "\n")
            stream.flush()
            with self._dropped_lock:
                self._closed_cleanly = True
