from pathlib import Path

import pytest

from wuji_teleop_monitor.ui.record_gui_core import (
    RecorderState,
    SessionOptions,
    TaskCatalog,
    format_arm_snapshot,
    validate_task_name,
)


@pytest.mark.parametrize("name", ["pick_red_block", "task-02", "A1"])
def test_task_name_accepts_portable_ascii(name):
    assert validate_task_name(name) == name


@pytest.mark.parametrize(
    "name",
    ["", "中文任务", "has space", "../escape", "_leading"],
)
def test_task_name_rejects_unsafe_or_non_ascii_names(name):
    with pytest.raises(ValueError):
        validate_task_name(name)


def test_task_catalog_lists_only_valid_task_directories(tmp_path):
    catalog = TaskCatalog(tmp_path)
    catalog.create_task("pick_box")
    (tmp_path / "episode_0000_legacy").mkdir()
    (tmp_path / "bad name").mkdir()
    (tmp_path / "pick_box.inprogress").mkdir()

    assert catalog.list_tasks() == ["pick_box"]
    assert catalog.task_path("pick_box") == tmp_path / "pick_box"


def test_default_gui_session_enables_both_hands_and_cameras(tmp_path):
    options = SessionOptions(task_name="pick_box")

    command = options.command(repo_root=tmp_path)

    assert command == [
        str(tmp_path / "src" / "scripts" / "start_record_session.sh"),
        "both",
        "--task",
        "pick_box",
        "--handoff-ramp-sec",
        "6.0",
        "--camera-transport",
        "direct",
        "--with-camera",
    ]


def test_recorder_state_encodes_pedal_guards():
    idle = RecorderState.from_status(
        {"state": "idle", "recording": False}
    )
    recording = RecorderState.from_status(
        {"state": "recording", "recording": True}
    )
    pending = RecorderState.from_status(
        {"state": "pending_save", "recording": False}
    )

    assert idle.pedal3_allowed is False
    assert recording.pedal1_action == "finish"
    assert recording.pedal2_allowed is False
    assert recording.pedal3_allowed is False
    assert pending.pedal1_action == "start"
    assert pending.pedal2_allowed is True
    assert pending.pedal3_allowed is True


def test_lightweight_arm_snapshot_is_formatted_without_detailed_diagnostics():
    text = format_arm_snapshot({
        "feedback_age_s": 0.0123,
        "sdk_fault_latched": False,
        "arms": {
            "left": {"state": 3, "err_code": 0},
            "right": {"state": 0, "err_code": 0},
        },
    })

    assert "left: state=3, err=0" in text
    assert "right: state=0, err=0" in text
    assert "反馈年龄=0.012s" in text


def test_record_gui_subscribes_to_cache_and_never_polls_full_arm_status():
    source = (
        Path(__file__).resolve().parents[1]
        / "wuji_teleop_monitor"
        / "ui"
        / "run_record.py"
    ).read_text(encoding="utf-8")

    assert '"/tianji_arm/status_snapshot"' in source
    assert '"/tianji_arm_controller/arm_status"' not in source
    assert "SharedFrameReader" in source
    assert 'if camera_transport == "ros":' in source
    assert "CameraMonitorWindow" in source
    assert 'for name in ("head",)' in source
    assert '"/cam_left_wrist/color/image_raw"' not in source
    assert '"/cam_right_wrist/color/image_raw"' not in source
    assert 'QGroupBox("在线相机' not in source
