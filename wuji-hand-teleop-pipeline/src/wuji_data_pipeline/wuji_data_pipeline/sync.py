"""Small timestamped ring buffers used by recorder and deployment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class TimedSample:
    timestamp: float
    value: Any


class TimedRingBuffer:
    def __init__(self, maxlen: int = 4096, max_age_s: float = 5.0):
        if maxlen <= 0 or max_age_s <= 0.0:
            raise ValueError("maxlen and max_age_s must be positive")
        self._samples: deque[TimedSample] = deque(maxlen=maxlen)
        self._max_age_s = float(max_age_s)
        self._lock = threading.Lock()

    def append(self, timestamp: float, value: Any) -> None:
        sample = TimedSample(float(timestamp), value)
        with self._lock:
            if self._samples and sample.timestamp < self._samples[-1].timestamp:
                # ROS clocks can jump when simulation time is enabled.  Keep
                # nearest lookup deterministic by restarting this source.
                self._samples.clear()
            self._samples.append(sample)
            cutoff = sample.timestamp - self._max_age_s
            while self._samples and self._samples[0].timestamp < cutoff:
                self._samples.popleft()

    def latest(self) -> Optional[TimedSample]:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def latest_after(self, timestamp: float) -> Optional[TimedSample]:
        with self._lock:
            for sample in reversed(self._samples):
                if sample.timestamp > timestamp:
                    return sample
        return None

    def nearest(self, timestamp: float, tolerance_s: float) -> Optional[TimedSample]:
        target = float(timestamp)
        tolerance = float(tolerance_s)
        with self._lock:
            if not self._samples:
                return None
            best = min(self._samples, key=lambda item: abs(item.timestamp - target))
            if abs(best.timestamp - target) <= tolerance:
                return best
        return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)


class FiniteDifferenceVelocity:
    """Fill missing JointState velocities from synchronized positions.

    Tianji publishes native joint positions in degrees while the Wuji driver
    publishes radians.  This helper deliberately preserves the caller's unit;
    the dataset schema performs the Tianji degree-to-radian conversion later.
    """

    def __init__(self, max_dt_s: float = 1.0) -> None:
        if max_dt_s <= 0.0:
            raise ValueError("max_dt_s must be positive")
        self._max_dt_s = float(max_dt_s)
        self._previous: dict[str, tuple[float, np.ndarray]] = {}

    def reset(self) -> None:
        self._previous.clear()

    def measure(
        self,
        key: str,
        timestamp: float,
        position: Any,
        reported_velocity: Any = None,
    ) -> np.ndarray:
        current = np.asarray(position, dtype=np.float32).reshape(-1)
        if current.size == 0 or not np.all(np.isfinite(current)):
            raise ValueError(f"{key} position is empty or non-finite")
        now = float(timestamp)
        previous = self._previous.get(key)
        self._previous[key] = (now, current.copy())

        if reported_velocity is not None:
            reported = np.asarray(reported_velocity, dtype=np.float32).reshape(-1)
            if reported.size:
                if reported.shape != current.shape or not np.all(np.isfinite(reported)):
                    raise ValueError(f"{key} reported velocity does not match position")
                return reported

        if previous is None:
            return np.zeros_like(current)
        previous_time, previous_position = previous
        dt = now - previous_time
        if dt <= 1e-6 or dt > self._max_dt_s or previous_position.shape != current.shape:
            return np.zeros_like(current)
        return ((current - previous_position) / dt).astype(np.float32)
