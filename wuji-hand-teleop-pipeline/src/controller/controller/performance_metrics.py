"""
Low-overhead rolling metrics for the Tianji realtime path.

This module deliberately has no ROS dependency so the accounting logic can be
unit-tested without hardware.  Samples are kept only until the next snapshot;
the ROS node publishes one compact JSON snapshot per second.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
import time
from typing import Any


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


class PerformanceWindow:
    """Collect callback/stage timings and emit reset-on-read windows."""

    def __init__(self, max_samples_per_series: int = 8192) -> None:
        if max_samples_per_series <= 0:
            raise ValueError("max_samples_per_series must be positive")
        self._max_samples = int(max_samples_per_series)
        self._series: dict[str, deque[float]] = {}
        self._window_counters: dict[str, int] = defaultdict(int)
        self._total_counters: dict[str, int] = defaultdict(int)
        self._last_loop_start_ns: dict[str, int] = {}
        self._window_started_ns = time.monotonic_ns()

    def observe(self, name: str, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            return
        samples = self._series.get(name)
        if samples is None:
            samples = deque(maxlen=self._max_samples)
            self._series[name] = samples
        samples.append(numeric)

    def increment(self, name: str, amount: int = 1) -> None:
        value = int(amount)
        self._window_counters[name] += value
        self._total_counters[name] += value

    def record_loop(
        self,
        name: str,
        started_ns: int,
        finished_ns: int,
        target_rate_hz: float,
        *,
        active: bool,
    ) -> None:
        target_rate = float(target_rate_hz)
        if target_rate <= 0.0:
            return
        target_period_ms = 1000.0 / target_rate
        duration_ms = max(0.0, (finished_ns - started_ns) / 1e6)
        self.observe(f"{name}.duration_ms", duration_ms)
        self.increment(f"{name}.callbacks")
        if active:
            self.increment(f"{name}.active_callbacks")

        previous_ns = self._last_loop_start_ns.get(name)
        self._last_loop_start_ns[name] = int(started_ns)
        if previous_ns is not None and started_ns >= previous_ns:
            period_ms = (started_ns - previous_ns) / 1e6
            self.observe(f"{name}.period_ms", period_ms)
            if period_ms > target_period_ms * 1.5:
                self.increment(f"{name}.scheduling_gaps")
                missed = max(1, int(period_ms / target_period_ms) - 1)
                self.increment(f"{name}.missed_periods_estimate", missed)
        if duration_ms > target_period_ms:
            self.increment(f"{name}.execution_overruns")

    @staticmethod
    def _series_summary(values: deque[float]) -> dict[str, Any]:
        ordered = sorted(values)
        count = len(ordered)
        total = float(sum(ordered))
        return {
            "count": count,
            "sum": total,
            "min": float(ordered[0]),
            "mean": total / count,
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
            "max": float(ordered[-1]),
        }

    def snapshot(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        duration_sec = max(
            (now_ns - self._window_started_ns) / 1e9,
            1e-9,
        )
        series = {
            name: self._series_summary(values)
            for name, values in sorted(self._series.items())
            if values
        }
        window_counters = dict(sorted(self._window_counters.items()))
        callback_rates_hz = {
            name.removesuffix(".callbacks"): count / duration_sec
            for name, count in window_counters.items()
            if name.endswith(".callbacks")
        }
        result = {
            "schema_version": 1,
            "wall_time_unix": time.time(),
            "monotonic_ns": now_ns,
            "window_duration_sec": duration_sec,
            "context": dict(context or {}),
            "callback_rates_hz": callback_rates_hz,
            "series": series,
            "window_counters": window_counters,
            "total_counters": dict(sorted(self._total_counters.items())),
        }
        self._series.clear()
        self._window_counters.clear()
        self._window_started_ns = now_ns
        return result


class MetricsSummary:
    """Merge published windows into one comparable baseline summary."""

    def __init__(self) -> None:
        self.window_count = 0
        self.duration_sec = 0.0
        self.counters: dict[str, int] = defaultdict(int)
        self.series: dict[str, dict[str, float]] = {}
        self.first_context: dict[str, Any] = {}
        self.last_context: dict[str, Any] = {}
        self.first_wall_time: float | None = None
        self.last_wall_time: float | None = None

    def add(self, snapshot: dict[str, Any]) -> None:
        duration = max(0.0, float(snapshot.get("window_duration_sec", 0.0)))
        self.window_count += 1
        self.duration_sec += duration
        context = snapshot.get("context")
        if isinstance(context, dict):
            if not self.first_context:
                self.first_context = dict(context)
            self.last_context = dict(context)
        wall_time = snapshot.get("wall_time_unix")
        if isinstance(wall_time, (int, float)):
            if self.first_wall_time is None:
                self.first_wall_time = float(wall_time)
            self.last_wall_time = float(wall_time)

        for name, value in snapshot.get("window_counters", {}).items():
            self.counters[str(name)] += int(value)

        for name, stats in snapshot.get("series", {}).items():
            if not isinstance(stats, dict):
                continue
            count = int(stats.get("count", 0))
            if count <= 0:
                continue
            target = self.series.setdefault(
                str(name),
                {
                    "count": 0,
                    "sum": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                    "worst_window_p95": 0.0,
                    "worst_window_p99": 0.0,
                },
            )
            target["count"] += count
            target["sum"] += float(stats.get("sum", 0.0))
            target["min"] = min(target["min"], float(stats.get("min", 0.0)))
            target["max"] = max(target["max"], float(stats.get("max", 0.0)))
            target["worst_window_p95"] = max(
                target["worst_window_p95"],
                float(stats.get("p95", 0.0)),
            )
            target["worst_window_p99"] = max(
                target["worst_window_p99"],
                float(stats.get("p99", 0.0)),
            )

    def result(self, label: str = "") -> dict[str, Any]:
        duration = max(self.duration_sec, 1e-9)
        rates = {
            name.removesuffix(".callbacks"): value / duration
            for name, value in sorted(self.counters.items())
            if name.endswith(".callbacks")
        }
        active_rates = {
            name.removesuffix(".active_callbacks"): value / duration
            for name, value in sorted(self.counters.items())
            if name.endswith(".active_callbacks")
        }
        series = {}
        for name, values in sorted(self.series.items()):
            count = int(values["count"])
            series[name] = {
                "count": count,
                "mean": values["sum"] / max(count, 1),
                "min": values["min"],
                "max": values["max"],
                "worst_window_p95": values["worst_window_p95"],
                "worst_window_p99": values["worst_window_p99"],
            }
        return {
            "schema_version": 1,
            "label": str(label),
            "window_count": self.window_count,
            "duration_sec": self.duration_sec,
            "first_wall_time_unix": self.first_wall_time,
            "last_wall_time_unix": self.last_wall_time,
            "first_context": self.first_context,
            "last_context": self.last_context,
            "callback_rates_hz": rates,
            "active_callback_rates_hz": active_rates,
            "counters": dict(sorted(self.counters.items())),
            "series": series,
        }
