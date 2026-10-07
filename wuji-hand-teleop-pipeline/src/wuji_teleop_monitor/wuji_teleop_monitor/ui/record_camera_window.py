"""Independent camera-monitor window for the recording cockpit."""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .theme import DARK_THEME_CSS


CAMERA_TITLES = {
    "head": "主视角",
}
CAMERA_ORDER = tuple(CAMERA_TITLES)
CAMERA_HEALTH_TITLES = {
    "head": "主视角",
    "left_wrist": "左腕",
    "right_wrist": "右腕",
}

RECORD_BANNER_STYLES = {
    "idle": (
        "background:#263238; color:#eceff1; border:3px solid #607d8b; "
        "border-radius:8px; padding:10px;"
    ),
    "recording": (
        "background:#b71c1c; color:#ffffff; border:4px solid #ff5252; "
        "border-radius:8px; padding:10px;"
    ),
    "pending": (
        "background:#e65100; color:#ffffff; border:4px solid #ffb74d; "
        "border-radius:8px; padding:10px;"
    ),
    "saved": (
        "background:#1b5e20; color:#ffffff; border:4px solid #69f0ae; "
        "border-radius:8px; padding:10px;"
    ),
    "discarded": (
        "background:#5d4037; color:#fff3e0; border:4px solid #ffb74d; "
        "border-radius:8px; padding:10px;"
    ),
    "error": (
        "background:#7f0000; color:#ffffff; border:4px solid #ff1744; "
        "border-radius:8px; padding:10px;"
    ),
}


class CameraTile(QLabel):
    """One latest-frame display with a visible stale-feed state."""

    STALE_AFTER_S = 2.0

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.last_frame_at = 0.0
        self.stale = False
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 220)
        self.reset()

    def set_image(self, image: QImage) -> None:
        self.last_frame_at = time.monotonic()
        self.stale = False
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.setText("")
        self.setPixmap(pixmap)
        self.setStyleSheet(
            "background:#111; color:#aaa; border:2px solid #2e7d32;"
        )

    def refresh_stale(self) -> None:
        if (
            self.last_frame_at
            and time.monotonic() - self.last_frame_at > self.STALE_AFTER_S
        ):
            self.stale = True
            self.setPixmap(QPixmap())
            self.setText(f"{self.title}\n图像超时")
            self.setStyleSheet(
                "background:#111; color:#ef5350; "
                "border:2px solid #c62828;"
            )

    def reset(self) -> None:
        self.last_frame_at = 0.0
        self.stale = False
        self.setPixmap(QPixmap())
        self.setText(f"{self.title}\n等待图像…")
        self.setStyleSheet(
            "background:#111; color:#888; border:1px solid #555;"
        )
        self.hide()


class CameraMonitorWindow(QMainWindow):
    """Top-level latest-frame monitor designed for a second display."""

    visible_cameras_changed = pyqtSignal(object)

    def __init__(self, task_name: str, parent=None):
        super().__init__(parent)
        self.task_name = task_name
        self._session_enabled = False
        self._shutdown_requested = False
        self._active_names: set[str] = set()
        self._recorder_status: dict = {}
        self._result_notification_active = False
        self._build_ui()
        self._stale_timer = QTimer(self)
        self._stale_timer.timeout.connect(self._refresh_staleness)
        self._stale_timer.start(1000)

    @property
    def active_camera_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in CAMERA_ORDER if name in self._active_names
        )

    def _build_ui(self) -> None:
        self.setWindowTitle(f"Wuji 主视角监看 — {self.task_name}")
        self.setMinimumSize(900, 600)
        self.resize(1400, 850)
        self.setStyleSheet(DARK_THEME_CSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(f"任务：{self.task_name}　主视角监看")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        header.addWidget(title)
        self._status_label = QLabel("等待进入准备")
        self._status_label.setAlignment(Qt.AlignCenter)
        header.addWidget(self._status_label, 1)

        self._second_screen_btn = QPushButton("移到另一块屏幕")
        self._second_screen_btn.clicked.connect(
            self.move_to_secondary_screen
        )
        header.addWidget(self._second_screen_btn)

        self._fullscreen_btn = QPushButton("全屏")
        self._fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        header.addWidget(self._fullscreen_btn)

        hide_button = QPushButton("隐藏监看")
        hide_button.clicked.connect(self.hide)
        header.addWidget(hide_button)
        root.addLayout(header)

        self._record_banner = QLabel()
        self._record_banner.setAlignment(Qt.AlignCenter)
        self._record_banner.setWordWrap(True)
        self._record_banner.setMinimumHeight(76)
        self._record_banner.setFont(QFont("Arial", 20, QFont.Bold))
        root.addWidget(self._record_banner)
        self._render_recorder_state()

        camera_health = QHBoxLayout()
        camera_health.addWidget(QLabel("采集相机状态"))
        self._camera_health_labels = {}
        for name, title_text in CAMERA_HEALTH_TITLES.items():
            label = QLabel(f"{title_text}：检测中")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(36)
            label.setFont(QFont("Arial", 12, QFont.Bold))
            camera_health.addWidget(label, 1)
            self._camera_health_labels[name] = label
        root.addLayout(camera_health)
        self._render_camera_health()

        self._camera_container = QWidget()
        self._camera_grid = QGridLayout(self._camera_container)
        self._camera_grid.setContentsMargins(0, 0, 0, 0)
        self._camera_grid.setSpacing(8)
        self._waiting_label = QLabel(
            "等待主视角相机图像…"
        )
        self._waiting_label.setAlignment(Qt.AlignCenter)
        self._waiting_label.setFont(QFont("Arial", 16))
        self._camera_grid.addWidget(self._waiting_label, 0, 0)
        root.addWidget(self._camera_container, 1)

        self._camera_tiles = {
            name: CameraTile(title, self._camera_container)
            for name, title in CAMERA_TITLES.items()
        }

    def begin_session(self, camera_enabled: bool) -> None:
        self._session_enabled = bool(camera_enabled)
        self._recorder_status = {}
        self._result_notification_active = False
        self._render_recorder_state()
        self._render_camera_health()
        self._reset_frames()
        if self._session_enabled:
            self._status_label.setText("等待在线相机")
            self.show_monitor()
        else:
            self._status_label.setText("本次未启动或记录相机")

    def end_session(self) -> None:
        self._session_enabled = False
        self._recorder_status = {}
        self._result_notification_active = False
        self._set_record_banner(
            "idle",
            "数采 Session 已结束",
        )
        self._reset_frames()
        self._render_camera_health()
        self._status_label.setText("Session 已结束")
        self.hide()

    def set_frame(self, name: str, image: QImage) -> None:
        if not self._session_enabled or name not in self._camera_tiles:
            return
        if name not in self._active_names:
            self._active_names.add(name)
            self._update_layout()
        self._camera_tiles[name].set_image(image)
        self._update_status()

    @property
    def record_banner_text(self) -> str:
        return self._record_banner.text()

    def set_recorder_status(self, status: dict) -> None:
        """Update the persistent F7 capture state from the recorder cache."""
        self._recorder_status = dict(status)
        self._render_camera_health()
        state = str(status.get("state", "idle"))
        if state in ("recording", "pending_save", "saving"):
            self._result_notification_active = False
            self._render_recorder_state()
        elif not self._result_notification_active:
            self._render_recorder_state()

    def show_capture_result(self, detail: str) -> None:
        """Provide immediate F7 feedback while the status topic catches up."""
        self._result_notification_active = False
        if "awaiting save" in detail:
            steps = self._steps_from_status(pending=True)
            suffix = f"　帧数 {steps}" if steps else ""
            self._set_record_banner(
                "pending",
                "■ 本次采集已结束　请踩 F8 保存" + suffix,
            )
            return
        self._set_record_banner(
            "recording",
            "● 正在采集　再次踩 F7 结束本次采集",
        )

    def show_save_success(self, detail: str) -> None:
        self._result_notification_active = True
        steps = self._steps_from_status(pending=True)
        if not steps:
            steps = self._extract_step_count(detail)
        suffix = f"　共 {steps} 帧" if steps else ""
        self._set_record_banner(
            "saved",
            "✓ 保存成功" + suffix + "　踩 F7 开始下一条",
        )

    def show_discarded(self) -> None:
        self._result_notification_active = True
        self._set_record_banner(
            "discarded",
            "上一条未保存轨迹已丢弃　正在准备新一条轨迹",
        )

    def show_record_error(self, detail: str) -> None:
        self._result_notification_active = True
        summary = detail.removeprefix("ERROR:").strip()
        self._set_record_banner(
            "error",
            "✕ 数采操作失败\n" + summary,
        )

    def show_monitor(self) -> None:
        if self.isFullScreen():
            self.showFullScreen()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self._fullscreen_btn.setText("全屏")
        else:
            self.showFullScreen()
            self._fullscreen_btn.setText("退出全屏")

    def move_to_secondary_screen(self) -> None:
        screens = QApplication.screens()
        if len(screens) < 2:
            QMessageBox.information(
                self,
                "没有第二块屏幕",
                "系统当前只检测到一块屏幕。",
            )
            return
        current_name = self.screen().name() if self.screen() is not None else ""
        target = next(
            (screen for screen in screens if screen.name() != current_name),
            screens[1],
        )
        if self.isFullScreen():
            self.showNormal()
        self.setGeometry(target.availableGeometry())
        self.showMaximized()
        self._fullscreen_btn.setText("全屏")
        self.raise_()
        self.activateWindow()

    def shutdown(self) -> None:
        self._shutdown_requested = True
        self._stale_timer.stop()
        self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def _reset_frames(self) -> None:
        self._active_names.clear()
        for tile in self._camera_tiles.values():
            self._camera_grid.removeWidget(tile)
            tile.reset()
        self._update_layout()

    def _update_layout(self) -> None:
        for tile in self._camera_tiles.values():
            self._camera_grid.removeWidget(tile)
            tile.hide()
        self._camera_grid.removeWidget(self._waiting_label)
        for row in range(2):
            self._camera_grid.setRowStretch(row, 0)
        for column in range(2):
            self._camera_grid.setColumnStretch(column, 0)

        names = self.active_camera_names
        if not names:
            self._waiting_label.show()
            self._camera_grid.addWidget(self._waiting_label, 0, 0)
            self.visible_cameras_changed.emit(names)
            return

        self._waiting_label.hide()
        self._camera_grid.addWidget(
            self._camera_tiles["head"], 0, 0, 2, 2
        )
        self._camera_grid.setRowStretch(0, 1)
        self._camera_grid.setColumnStretch(0, 1)

        for name in names:
            self._camera_tiles[name].show()
        self.visible_cameras_changed.emit(names)

    def _refresh_staleness(self) -> None:
        for name in self._active_names:
            self._camera_tiles[name].refresh_stale()
        self._update_status()

    def _update_status(self) -> None:
        if not self._session_enabled:
            return
        names = self.active_camera_names
        if not names:
            self._status_label.setText("等待在线相机")
            return
        parts = []
        for name in names:
            tile = self._camera_tiles[name]
            state = "超时" if tile.stale else "在线"
            parts.append(f"{tile.title}:{state}")
        self._status_label.setText("　".join(parts))

    def _render_recorder_state(self) -> None:
        state = str(self._recorder_status.get("state", "idle"))
        elapsed = float(self._recorder_status.get("elapsed_s", 0.0))
        if state == "recording":
            steps = self._steps_from_status()
            self._set_record_banner(
                "recording",
                "● 正在采集"
                f"　时长 {elapsed:.1f} 秒　帧数 {steps}"
                "　再次踩 F7 结束",
            )
        elif state == "pending_save":
            steps = self._steps_from_status(pending=True)
            self._set_record_banner(
                "pending",
                "■ 本次采集已结束"
                f"　时长 {elapsed:.1f} 秒　帧数 {steps}"
                "　请踩 F8 保存",
            )
        elif state == "saving":
            self._set_record_banner(
                "pending",
                "正在保存轨迹，请稍候…",
            )
        else:
            self._set_record_banner(
                "idle",
                "等待开始采集　踩 F7 开始",
            )

    def _render_camera_health(self) -> None:
        configured = {
            str(name)
            for name in self._recorder_status.get("configured_cameras", [])
        }
        online = {
            str(name)
            for name in self._recorder_status.get("online_cameras", [])
        }
        status_received = "configured_cameras" in self._recorder_status
        for name, title in CAMERA_HEALTH_TITLES.items():
            label = self._camera_health_labels[name]
            if not self._session_enabled:
                state = "未启用"
                style = "background:#37474f; color:#cfd8dc;"
            elif not status_received:
                state = "检测中"
                style = "background:#37474f; color:#cfd8dc;"
            elif name not in configured:
                state = "未配置"
                style = "background:#37474f; color:#cfd8dc;"
            elif name in online:
                state = "在线"
                style = "background:#1b5e20; color:#ffffff;"
            else:
                state = "离线"
                style = "background:#b71c1c; color:#ffffff;"
            label.setText(f"{title}：{state}")
            label.setStyleSheet(
                style + " border-radius:6px; padding:6px;"
            )

    def _set_record_banner(self, level: str, text: str) -> None:
        self._record_banner.setText(text)
        self._record_banner.setStyleSheet(RECORD_BANNER_STYLES[level])

    def _steps_from_status(self, pending: bool = False) -> int:
        key = "pending_steps" if pending else "steps"
        try:
            return int(self._recorder_status.get(key, 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _extract_step_count(detail: str) -> int:
        marker = " steps)"
        before, separator, _ = detail.partition(marker)
        if not separator:
            return 0
        token = before.rsplit("(", 1)[-1]
        try:
            return int(token)
        except ValueError:
            return 0

    def closeEvent(self, event) -> None:
        if self._shutdown_requested:
            event.accept()
            return
        self.hide()
        event.ignore()
