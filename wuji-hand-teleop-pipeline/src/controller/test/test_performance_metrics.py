from controller.performance_metrics import MetricsSummary, PerformanceWindow


def test_snapshot_reports_rates_percentiles_and_resets_window(monkeypatch):
    times = iter([1_000_000_000, 2_000_000_000, 3_000_000_000])
    monkeypatch.setattr(
        "controller.performance_metrics.time.monotonic_ns",
        lambda: next(times),
    )
    metrics = PerformanceWindow()
    metrics.observe("control.duration_ms", 1.0)
    metrics.observe("control.duration_ms", 3.0)
    metrics.increment("target_hold.frames", 2)
    metrics.record_loop(
        "control",
        1_900_000_000,
        1_901_000_000,
        100.0,
        active=True,
    )

    first = metrics.snapshot({"enabled": True})

    assert first["window_duration_sec"] == 1.0
    assert first["context"]["enabled"] is True
    assert first["callback_rates_hz"]["control"] == 1.0
    assert first["series"]["control.duration_ms"]["count"] == 3
    assert first["series"]["control.duration_ms"]["mean"] == 5.0 / 3.0
    assert first["window_counters"]["target_hold.frames"] == 2
    assert first["total_counters"]["target_hold.frames"] == 2

    second = metrics.snapshot()

    assert second["series"] == {}
    assert second["window_counters"] == {}
    assert second["total_counters"]["target_hold.frames"] == 2


def test_loop_accounting_detects_schedule_gaps_and_execution_overruns(
    monkeypatch,
):
    monkeypatch.setattr(
        "controller.performance_metrics.time.monotonic_ns",
        lambda: 1_000_000_000,
    )
    metrics = PerformanceWindow()
    metrics.record_loop(
        "state",
        100_000_000,
        103_000_000,
        500.0,
        active=True,
    )
    metrics.record_loop(
        "state",
        110_000_000,
        111_000_000,
        500.0,
        active=True,
    )

    snapshot = metrics.snapshot()

    assert snapshot["window_counters"]["state.callbacks"] == 2
    assert snapshot["window_counters"]["state.active_callbacks"] == 2
    assert snapshot["window_counters"]["state.execution_overruns"] == 1
    assert snapshot["window_counters"]["state.scheduling_gaps"] == 1
    assert (
        snapshot["window_counters"]["state.missed_periods_estimate"]
        == 4
    )
    assert snapshot["series"]["state.period_ms"]["mean"] == 10.0


def test_non_finite_samples_are_ignored(monkeypatch):
    times = iter([1_000_000_000, 2_000_000_000])
    monkeypatch.setattr(
        "controller.performance_metrics.time.monotonic_ns",
        lambda: next(times),
    )
    metrics = PerformanceWindow()
    metrics.observe("bad", float("nan"))
    metrics.observe("bad", float("inf"))

    assert metrics.snapshot()["series"] == {}


def test_summary_merges_window_rates_and_stage_means():
    summary = MetricsSummary()
    summary.add(
        {
            "window_duration_sec": 1.0,
            "wall_time_unix": 10.0,
            "context": {"arm_enabled": True},
            "window_counters": {"control.callbacks": 120},
            "series": {
                "control.duration_ms": {
                    "count": 2,
                    "sum": 6.0,
                    "min": 2.0,
                    "max": 4.0,
                    "p95": 3.9,
                    "p99": 3.98,
                }
            },
        }
    )
    summary.add(
        {
            "window_duration_sec": 1.0,
            "wall_time_unix": 11.0,
            "context": {"arm_enabled": True},
            "window_counters": {"control.callbacks": 118},
            "series": {
                "control.duration_ms": {
                    "count": 1,
                    "sum": 9.0,
                    "min": 9.0,
                    "max": 9.0,
                    "p95": 9.0,
                    "p99": 9.0,
                }
            },
        }
    )

    result = summary.result("gui_recording")

    assert result["callback_rates_hz"]["control"] == 119.0
    assert result["active_callback_rates_hz"] == {}
    assert result["series"]["control.duration_ms"]["mean"] == 5.0
    assert (
        result["series"]["control.duration_ms"]["worst_window_p99"]
        == 9.0
    )
    assert result["label"] == "gui_recording"
