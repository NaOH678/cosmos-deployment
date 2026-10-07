"""Opt-in per-phase wall-clock timing for the PointFlow training path.

The plain MSE and the MoT FLOPs counters say nothing about where a training
step actually spends its time, and the PointFlow branch adds several small,
launch-bound phases (sparse Sonata encoding, ragged token packing, per-block
decoding) that never show up as transformer FLOPs.  This module measures them.

Usage::

    from cosmos_framework.model.generator.pointflow_profiling import phase, report

    with phase("pf_sonata"):
        features = self.geometry(inputs)

Timing is a ``cuda.synchronize()`` pair around each phase, so it costs a few
tens of microseconds per call and perturbs the measurement slightly.  It is
therefore off by default: set ``POINTFLOW_PROFILE=1`` to enable it (the context
manager then costs one boolean test per phase).
"""

import os
import resource
import time
from collections import defaultdict
from contextlib import contextmanager

_ENABLED = os.environ.get("POINTFLOW_PROFILE", "0").lower() in {"1", "true", "yes", "on"}

_TOTAL_SECONDS: dict[str, float] = defaultdict(float)
_CPU_SECONDS: dict[str, float] = defaultdict(float)
_MINOR_FAULTS: dict[str, int] = defaultdict(int)
_MAJOR_FAULTS: dict[str, int] = defaultdict(int)
_CALLS: dict[str, int] = defaultdict(int)
_MARKS: list[tuple[str, float]] = []


def _page_faults() -> tuple[int, int]:
    """(minor, major) page faults of this process.

    A phase that is slow *and* faults heavily is paying the kernel to get
    memory back; a phase that is slow without faults is doing actual work.
    Major faults mean the data had to come from disk or swap.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_minflt, usage.ru_majflt


def system_memory() -> str:
    """One-line host memory watermark, to tell reclaim from ordinary allocation."""
    fields = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                fields[key] = rest.split()[0] if rest.split() else "?"
    except OSError:  # pragma: no cover - /proc is always there on Linux
        return ""
    if not fields:
        return ""
    return (
        f"MemAvailable {int(fields.get('MemAvailable', 0)) // 1024}MB"
        f" | SwapFree {int(fields.get('SwapFree', 0)) // 1024}MB"
        f" | Dirty {int(fields.get('Dirty', 0)) // 1024}MB"
    )


def profile_enabled() -> bool:
    return _ENABLED


@contextmanager
def phase(name: str):
    """Accumulate synchronized wall time under ``name``.

    The entry synchronization happens before the clock starts, so a phase never
    pays for work queued earlier -- but that wait is itself recorded under
    ``<name>_wait``.  A phase whose wait dominates its own time is overlapping
    with the GPU rather than doing anything expensive, which is the difference
    between "this code is slow" and "the step started before the GPU caught up".
    """
    if not _ENABLED:
        yield
        return
    import torch

    cuda = torch.cuda.is_available()
    if cuda:
        wait_start = time.perf_counter()
        torch.cuda.synchronize()
        wait = time.perf_counter() - wait_start
        _TOTAL_SECONDS[name + "_wait"] += wait
        _CALLS[name + "_wait"] += 1
    start = time.perf_counter()
    cpu_start = time.process_time()
    minor_start, major_start = _page_faults()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize()
        minor_end, major_end = _page_faults()
        _TOTAL_SECONDS[name] += time.perf_counter() - start
        _CPU_SECONDS[name] += time.process_time() - cpu_start
        _MINOR_FAULTS[name] += minor_end - minor_start
        _MAJOR_FAULTS[name] += major_end - major_start
        _CALLS[name] += 1


def averages() -> dict[str, float]:
    """Mean seconds per call for every phase recorded since the last reset."""
    return {name: _TOTAL_SECONDS[name] / _CALLS[name] for name in _TOTAL_SECONDS if _CALLS[name]}


def cpu_averages() -> dict[str, float]:
    """Mean CPU seconds per call, i.e. time this process actually spent running."""
    return {name: _CPU_SECONDS[name] / _CALLS[name] for name in _CPU_SECONDS if _CALLS[name]}


def reset() -> None:
    _TOTAL_SECONDS.clear()
    _CPU_SECONDS.clear()
    _CALLS.clear()
    _MARKS.clear()


def mark(name: str) -> None:
    """Timestamp one boundary of the backward pass, e.g. a subgraph's first gradient.

    Unlike :func:`phase` there is no exit side: the segments are the differences
    between consecutive marks, flushed as ``bwd_<name>`` phases by
    :func:`flush_marks`.
    """
    if not _ENABLED:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _MARKS.append((name, time.perf_counter()))


def flush_marks(prefix: str = "bwd_") -> dict[str, float]:
    """Fold the marks recorded since the last flush into the phase accumulators.

    Segment ``i`` is named after its closing mark, so marks ``begin, head, end``
    produce ``bwd_head`` and ``bwd_end``.
    """
    marks, _MARKS[:] = list(_MARKS), []
    segments: dict[str, float] = {}
    for (_, start), (name, end) in zip(marks, marks[1:]):
        key = f"{prefix}{name}"
        delta = end - start
        _TOTAL_SECONDS[key] += delta
        _CALLS[key] += 1
        segments[key] = segments.get(key, 0.0) + delta
    return segments


def format_report(prefix: str = "PointFlow phases") -> str:
    """One-line ``name=ms`` summary ordered by descending cost.

    Each entry carries the CPU time this process consumed and the page faults it
    took, so three very different situations can be told apart:

    - ``cpu`` far below the wall time: the trainer was descheduled.
    - ``cpu`` close to the wall time with few faults: it ran, and the work is
      genuinely expensive.
    - ``cpu`` close to the wall time with many faults -- especially major ones:
      the kernel spent the time getting memory back, so the fix is to reduce
      the footprint, not to rewrite the code.
    """
    wall_seconds = averages()
    cpu_seconds = cpu_averages()
    entries = [(name, seconds) for name, seconds in wall_seconds.items() if seconds >= 5e-5]
    if not entries:
        return f"{prefix}: no samples"
    ordered = sorted(entries, key=lambda item: -item[1])
    rendered = []
    for name, seconds in ordered:
        text = f"{name} {seconds * 1000.0:.1f}ms"
        if name in cpu_seconds:
            text += f"(cpu {cpu_seconds[name] * 1000.0:.1f}ms"
            calls = _CALLS.get(name, 0)
            if calls:
                text += f" flt {_MINOR_FAULTS[name] // calls}/{_MAJOR_FAULTS[name] // calls}"
            text += ")"
        rendered.append(text)
    memory = system_memory()
    return prefix + ": " + " | ".join(rendered) + (f" || {memory}" if memory else "")
