import importlib
import time

import pytest

from cosmos_framework.model.generator import pointflow_profiling
from cosmos_framework.model.generator.pointflow_profiling import (
    averages,
    flush_marks,
    format_report,
    mark,
    phase,
    reset,
)


@pytest.fixture(autouse=True)
def record_phases(monkeypatch):
    """Recording tests opt in; production defaults to disabled."""
    monkeypatch.setattr(pointflow_profiling, "_ENABLED", True)
    reset()
    yield
    reset()


def test_phase_accumulates_mean_seconds_per_call():
    for _ in range(2):
        with phase("alpha"):
            time.sleep(0.02)
    with phase("beta"):
        time.sleep(0.01)
    result = averages()
    assert result["alpha"] == pytest.approx(0.02, abs=0.015)
    assert result["beta"] == pytest.approx(0.01, abs=0.015)
    assert format_report().startswith("PointFlow phases: alpha")
    assert "beta" in format_report()


def test_reset_clears_the_window():
    with phase("alpha"):
        pass
    assert averages()
    reset()
    assert averages() == {}
    assert format_report() == "PointFlow phases: no samples"


def test_phase_records_even_when_the_body_raises():
    with pytest.raises(RuntimeError):
        with phase("alpha"):
            raise RuntimeError("boom")
    assert averages()["alpha"] >= 0


def test_marks_become_segments_named_after_their_closing_mark():
    mark("begin")
    time.sleep(0.02)
    mark("head")
    time.sleep(0.01)
    mark("end")
    segments = flush_marks()
    assert set(segments) == {"bwd_head", "bwd_end"}
    assert averages()["bwd_head"] == pytest.approx(0.02, abs=0.015)
    assert averages()["bwd_end"] == pytest.approx(0.01, abs=0.015)


def test_flush_without_marks_records_nothing():
    assert flush_marks() == {}
    assert averages() == {}


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("POINTFLOW_PROFILE", raising=False)
    importlib.reload(pointflow_profiling)
    try:
        assert pointflow_profiling.profile_enabled() is False
        with pointflow_profiling.phase("alpha"):
            time.sleep(0.01)
        assert pointflow_profiling.averages() == {}
    finally:
        importlib.reload(pointflow_profiling)
