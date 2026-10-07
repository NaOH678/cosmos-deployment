import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from wuji_teleop_monitor.ui.record_camera_window import CameraMonitorWindow


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def monitor(app):
    window = CameraMonitorWindow("camera_test")
    yield window
    window.shutdown()
    app.processEvents()


def _image() -> QImage:
    image = QImage(32, 24, QImage.Format_RGB888)
    image.fill(0x336699)
    return image


def _grid_position(window, name):
    tile = window._camera_tiles[name]
    index = window._camera_grid.indexOf(tile)
    return window._camera_grid.getItemPosition(index)


def test_frames_are_ignored_until_a_camera_session_starts(monitor):
    monitor.set_frame("head", _image())

    assert monitor.active_camera_names == ()


def test_operator_monitor_only_displays_the_head_camera(monitor):
    monitor.begin_session(True)

    monitor.set_frame("head", _image())
    assert monitor.active_camera_names == ("head",)
    assert _grid_position(monitor, "head") == (0, 0, 2, 2)

    monitor.set_frame("right_wrist", _image())
    monitor.set_frame("left_wrist", _image())
    assert monitor.active_camera_names == ("head",)
    assert _grid_position(monitor, "head") == (0, 0, 2, 2)


def test_session_end_clears_and_hides_the_monitor(monitor, app):
    monitor.begin_session(True)
    monitor.set_frame("head", _image())

    monitor.end_session()
    app.processEvents()

    assert monitor.active_camera_names == ()
    assert monitor.isHidden()


def test_recording_banner_shows_live_elapsed_time_and_steps(monitor):
    monitor.begin_session(True)

    monitor.set_recorder_status({
        "state": "recording",
        "recording": True,
        "elapsed_s": 12.34,
        "steps": 371,
    })

    assert "正在采集" in monitor.record_banner_text
    assert "12.3 秒" in monitor.record_banner_text
    assert "371" in monitor.record_banner_text
    assert "#b71c1c" in monitor._record_banner.styleSheet()


def test_pending_banner_prompts_operator_to_press_f8(monitor):
    monitor.begin_session(True)

    monitor.set_recorder_status({
        "state": "pending_save",
        "recording": False,
        "elapsed_s": 8.5,
        "pending_steps": 255,
    })

    assert "采集已结束" in monitor.record_banner_text
    assert "F8 保存" in monitor.record_banner_text
    assert "255" in monitor.record_banner_text
    assert "#e65100" in monitor._record_banner.styleSheet()


def test_camera_health_uses_recorder_online_status_without_rendering_wrists(
    monitor,
):
    monitor.begin_session(True)
    monitor.set_recorder_status({
        "state": "idle",
        "configured_cameras": ["head", "left_wrist", "right_wrist"],
        "online_cameras": ["head", "right_wrist"],
    })

    assert monitor._camera_health_labels["head"].text() == "主视角：在线"
    assert monitor._camera_health_labels["left_wrist"].text() == "左腕：离线"
    assert monitor._camera_health_labels["right_wrist"].text() == "右腕：在线"
    assert "#1b5e20" in monitor._camera_health_labels["head"].styleSheet()
    assert "#b71c1c" in monitor._camera_health_labels["left_wrist"].styleSheet()
    assert monitor.active_camera_names == ()


def test_save_success_stays_visible_until_next_capture(monitor):
    monitor.begin_session(True)
    monitor.set_recorder_status({
        "state": "idle",
        "pending_steps": 0,
    })

    monitor.show_save_success(
        "/tmp/episode_0001 (123 steps)"
    )
    monitor.set_recorder_status({"state": "idle"})

    assert "保存成功" in monitor.record_banner_text
    assert "123 帧" in monitor.record_banner_text
    assert "#1b5e20" in monitor._record_banner.styleSheet()

    monitor.set_recorder_status({
        "state": "recording",
        "steps": 1,
        "elapsed_s": 0.1,
    })
    assert "正在采集" in monitor.record_banner_text
