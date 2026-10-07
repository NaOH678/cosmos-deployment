#!/usr/bin/env python3
"""Standalone Tianji + Wuji data-collection GUI."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import pty
import re
import subprocess
import sys
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int8, String

from stereocamera.shared_frames import (
    DEFAULT_DIRECTORY,
    SharedFrameReader,
)

from .pedal_input import KeyboardShortcutPedalInput, PedalInputAdapter
from .qt_setup import setup_qt_plugins
from .record_gui_core import (
    GUI_HANDOFF_RAMP_SEC,
    RecorderState,
    SessionOptions,
    TaskCatalog,
    format_arm_snapshot,
    repository_root,
    validate_task_name,
)
from .theme import DARK_THEME_CSS, LOG_TEXTEDIT_CSS

setup_qt_plugins()

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QFont, QImage, QTextCursor  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .record_camera_window import CameraMonitorWindow  # noqa: E402


_LOCK_FILE = "/tmp/wuji_record_gui.lock"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_LIFECYCLE_NAMES = {
    0: "INITIALIZING",
    1: "ENABLING",
    2: "READY",
    3: "DISABLED",
    4: "ENABLE_FAILED",
    5: "SDK_ERROR",
    6: "RECOVERING",
    7: "RECOVERY_READY",
    8: "RECOVERY_FAILED",
    9: "RECOVERY_PARTIAL",
    10: "TARGET_HOLD",
    11: "CLUTCH_DISCONNECTED",
}
_KEY_NODES = {
    "Tracker": "/openvr_input",
    "Tianji": "/tianji_arm_controller",
    "Recorder": "/wuji_teleop_recorder",
    "MANUS": "/manus_data_publisher",
    "Left hand": "/wujihand_controller_left",
    "Right hand": "/wujihand_controller_right",
}
_CAMERA_SUBSCRIPTIONS = (
    ("head", "compressed", "/cam_head/color/image_raw/compressed"),
    ("head", "raw", "/cam_head/color/image_raw"),
    ("head", "raw", "/cam_head/color/image_rect_raw"),
    # Retain compatibility with the optional USB stereo head camera.
    ("head", "compressed", "/stereo/left/compressed"),
)


class _Signals(QObject):
    lifecycle = pyqtSignal(int)
    hand_recovery = pyqtSignal(str, int)
    teleop_status = pyqtSignal(str)
    recorder_status = pyqtSignal(object)
    status_snapshot = pyqtSignal(object)
    node_names = pyqtSignal(object)
    camera_frame = pyqtSignal(str, QImage)


def _compressed_to_qimage(message: CompressedImage) -> Optional[QImage]:
    image = QImage()
    if not image.loadFromData(bytes(message.data)):
        return None
    return image


def _raw_to_qimage(message: Image) -> Optional[QImage]:
    encoding = str(message.encoding).lower()
    formats = {
        "rgb8": QImage.Format_RGB888,
        "bgr8": QImage.Format_BGR888,
        "rgba8": QImage.Format_RGBA8888,
        "bgra8": QImage.Format_ARGB32,
        "mono8": QImage.Format_Grayscale8,
    }
    image_format = formats.get(encoding)
    if image_format is None:
        return None
    expected = int(message.height) * int(message.step)
    data = bytes(message.data)
    if len(data) < expected:
        return None
    return QImage(
        data,
        int(message.width),
        int(message.height),
        int(message.step),
        image_format,
    ).copy()


class RecordGuiRosNode(Node):
    def __init__(self, signals: _Signals, camera_transport: str = "direct"):
        super().__init__("wuji_record_gui")
        self._signals = signals
        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Int8,
            "/tianji_arm/lifecycle_state",
            lambda msg: signals.lifecycle.emit(int(msg.data)),
            latched,
        )
        for side in ("left", "right"):
            self.create_subscription(
                Int8,
                f"/{side}_hand/recovery_state",
                lambda msg, s=side: signals.hand_recovery.emit(
                    s, int(msg.data)
                ),
                latched,
            )
        self.create_subscription(
            String,
            "/tianji_arm/teleop_status",
            lambda msg: signals.teleop_status.emit(str(msg.data)),
            latched,
        )
        self.create_subscription(
            String,
            "/wuji_teleop_recorder/status_text",
            self._recorder_status_callback,
            10,
        )
        self.create_subscription(
            String,
            "/tianji_arm/status_snapshot",
            self._status_snapshot_callback,
            latched,
        )
        if camera_transport == "ros":
            for name, message_type, topic in _CAMERA_SUBSCRIPTIONS:
                if message_type == "compressed":
                    self.create_subscription(
                        CompressedImage,
                        topic,
                        lambda msg, n=name: self._emit_compressed(n, msg),
                        qos_profile_sensor_data,
                    )
                else:
                    self.create_subscription(
                        Image,
                        topic,
                        lambda msg, n=name: self._emit_raw(n, msg),
                        qos_profile_sensor_data,
                    )
        self.create_timer(1.0, self._poll_system)

    def _recorder_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            status = {"state": "invalid", "error": message.data}
        self._signals.recorder_status.emit(status)

    def _status_snapshot_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            status = {"error": "invalid cached status: " + message.data}
        self._signals.status_snapshot.emit(status)

    def _emit_compressed(self, name: str, message: CompressedImage) -> None:
        image = _compressed_to_qimage(message)
        if image is not None:
            self._signals.camera_frame.emit(name, image)

    def _emit_raw(self, name: str, message: Image) -> None:
        image = _raw_to_qimage(message)
        if image is not None:
            self._signals.camera_frame.emit(name, image)

    def _poll_system(self) -> None:
        names = {
            f"{namespace.rstrip('/')}/{name}" if namespace else f"/{name}"
            for name, namespace in self.get_node_names_and_namespaces()
        }
        self._signals.node_names.emit(names)


class DirectCameraPreview:
    """Poll only the latest shared frame; never queue GUI history."""

    def __init__(
        self,
        signals: _Signals,
        directory: str = str(DEFAULT_DIRECTORY),
    ) -> None:
        self._signals = signals
        self._readers = {
            name: SharedFrameReader(name, directory=directory)
            for name in ("head",)
        }
        self._last_identity = {
            name: (0, 0) for name in self._readers
        }
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="wuji-direct-camera-preview",
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        for reader in self._readers.values():
            reader.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for name, reader in self._readers.items():
                try:
                    frame = reader.latest()
                except (OSError, ValueError):
                    continue
                if frame is None:
                    continue
                identity = (reader.producer_generation, frame.sequence)
                if identity == self._last_identity[name]:
                    continue
                self._last_identity[name] = identity
                if time.monotonic_ns() - frame.monotonic_ns > 2_000_000_000:
                    continue
                height, width, channels = frame.image.shape
                if channels != 3:
                    continue
                image = QImage(
                    frame.image.data,
                    width,
                    height,
                    int(frame.image.strides[0]),
                    QImage.Format_BGR888,
                ).copy()
                self._signals.camera_frame.emit(name, image)
            self._stop_event.wait(1.0 / 30.0)


class PtyRecordSession(QObject):
    output = pyqtSignal(str)
    exited = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process: Optional[subprocess.Popen] = None
        self._master_fd: Optional[int] = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, command: list[str], cwd: Path) -> None:
        if self.running:
            raise RuntimeError("record session is already running")
        master_fd, slave_fd = pty.openpty()
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)
        self._master_fd = master_fd
        threading.Thread(
            target=self._reader,
            daemon=True,
            name="wuji-record-session-pty",
        ).start()

    def send_key(self, key: str) -> None:
        if not self.running or self._master_fd is None:
            raise RuntimeError("record session is not running")
        os.write(self._master_fd, key.encode("ascii"))

    def _reader(self) -> None:
        master_fd = self._master_fd
        process = self.process
        if master_fd is None or process is None:
            return
        try:
            while True:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                text = chunk.decode(
                    "utf-8", errors="replace"
                ).replace("\r", "")
                text = _ANSI_RE.sub("", text)
                if text:
                    self.output.emit(text)
        finally:
            code = process.wait()
            try:
                os.close(master_fd)
            except OSError:
                pass
            self._master_fd = None
            self.process = None
            self.exited.emit(int(code))


class TaskSelectionDialog(QDialog):
    def __init__(self, catalog: TaskCatalog, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.selected_task: Optional[str] = None
        self.setWindowTitle("选择数采任务")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "一个任务对应一个轨迹目录。请选择旧任务，或新建英文任务名。"
        ))
        form = QFormLayout()
        self._existing = QComboBox()
        self._existing.addItem("新建任务…", None)
        for task in catalog.list_tasks():
            self._existing.addItem(task, task)
        self._new_name = QLineEdit()
        self._new_name.setPlaceholderText("例如: pick_red_block")
        self._existing.currentIndexChanged.connect(self._selection_changed)
        form.addRow("旧任务", self._existing)
        form.addRow("新任务名称", self._new_name)
        layout.addLayout(form)
        self._path_label = QLabel(str(catalog.root))
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._selection_changed()

    def _selection_changed(self) -> None:
        self._new_name.setEnabled(self._existing.currentData() is None)

    def _accept(self) -> None:
        existing = self._existing.currentData()
        if existing is not None:
            self.selected_task = str(existing)
            self.accept()
            return
        try:
            name = validate_task_name(self._new_name.text())
            self.catalog.create_task(name)
        except FileExistsError:
            QMessageBox.warning(
                self, "任务已存在", "该任务已经存在，请从旧任务列表选择。"
            )
            return
        except ValueError as exc:
            QMessageBox.warning(self, "任务名称无效", str(exc))
            return
        self.selected_task = name
        self.accept()


class RecordWindow(QMainWindow):
    pedal_event = pyqtSignal(int)

    def __init__(
        self,
        task_name: str,
        catalog: TaskCatalog,
        signals: _Signals,
        pedal_input: Optional[PedalInputAdapter] = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.catalog = catalog
        self.signals = signals
        self.session = PtyRecordSession(self)
        self.lifecycle: Optional[int] = None
        self.hand_recovery_states = {"left": None, "right": None}
        self.recorder_status: dict = {}
        self.node_names: set[str] = set()
        self._close_requested = False
        self._log_buffer = ""
        self._handoff_duration_s = GUI_HANDOFF_RAMP_SEC
        self._handoff_until = 0.0
        self._camera_window = CameraMonitorWindow(
            task_name,
            parent=self,
        )

        self.session.output.connect(self._append_output)
        self.session.exited.connect(self._session_exited)
        signals.lifecycle.connect(self._set_lifecycle)
        signals.hand_recovery.connect(self._set_hand_recovery)
        signals.teleop_status.connect(self._set_teleop_status)
        signals.recorder_status.connect(self._set_recorder_status)
        signals.recorder_status.connect(
            self._camera_window.set_recorder_status
        )
        signals.status_snapshot.connect(self._set_status_snapshot)
        signals.node_names.connect(self._set_node_names)
        signals.camera_frame.connect(self._camera_window.set_frame)
        self.pedal_event.connect(self._handle_pedal)

        self._build_ui()
        self.pedal_input = (
            pedal_input or KeyboardShortcutPedalInput(self)
        )
        self.pedal_input.start(self.pedal_event.emit)
        self._refresh_controls()
        self._control_timer = QTimer(self)
        self._control_timer.timeout.connect(self._refresh_controls)
        self._control_timer.start(200)

    def _build_ui(self) -> None:
        self.setWindowTitle(f"Wuji 数采 GUI — {self.task_name}")
        self.resize(1250, 850)
        self.setStyleSheet(DARK_THEME_CSS)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        config_group = QGroupBox("任务与系统准备")
        config = QHBoxLayout(config_group)
        self._task_label = QLabel(
            f"任务：{self.task_name}\n目录：{self.catalog.task_path(self.task_name)}"
        )
        self._task_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        config.addWidget(self._task_label, 1)
        self._hand_combo = QComboBox()
        self._hand_combo.addItem("双臂 + 双手", "both")
        self._hand_combo.addItem("左臂 + 左手", "left")
        self._hand_combo.addItem("右臂 + 右手", "right")
        config.addWidget(QLabel("数采模式"))
        config.addWidget(self._hand_combo)
        self._camera_check = QCheckBox("启动并记录在线相机")
        self._camera_check.setChecked(True)
        config.addWidget(self._camera_check)
        self._camera_monitor_btn = QPushButton("打开主视角监看窗口")
        self._camera_monitor_btn.clicked.connect(
            self._camera_window.show_monitor
        )
        config.addWidget(self._camera_monitor_btn)
        self._prepare_btn = QPushButton("进入准备")
        self._prepare_btn.setMinimumHeight(44)
        self._prepare_btn.clicked.connect(self._start_session)
        config.addWidget(self._prepare_btn)
        layout.addWidget(config_group)

        status_group = QGroupBox("系统状态")
        status_grid = QGridLayout(status_group)
        self._lifecycle_label = QLabel("Lifecycle：OFFLINE")
        self._hand_recovery_label = QLabel("Hand Recovery：OFFLINE")
        self._teleop_label = QLabel("Teleop：等待启动")
        self._recorder_label = QLabel("Recorder：OFFLINE")
        self._result_label = QLabel("最近轨迹结果：无")
        self._result_label.setWordWrap(True)
        self._result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._arm_label = QLabel("硬件缓存：等待状态")
        self._nodes_label = QLabel("节点：等待启动")
        self._nodes_label.setWordWrap(True)
        status_grid.addWidget(self._lifecycle_label, 0, 0)
        status_grid.addWidget(self._recorder_label, 0, 1)
        status_grid.addWidget(self._hand_recovery_label, 1, 0, 1, 2)
        status_grid.addWidget(self._teleop_label, 2, 0, 1, 2)
        status_grid.addWidget(self._result_label, 3, 0, 1, 2)
        status_grid.addWidget(self._arm_label, 4, 0, 1, 2)
        status_grid.addWidget(self._nodes_label, 5, 0, 1, 2)
        layout.addWidget(status_group)

        controls = QGroupBox("操作控制")
        control_grid = QGridLayout(controls)
        self._recovery_btn = QPushButton("Recovery（r）")
        self._enable_btn = QPushButton(
            "Enable 机械臂 + 手（a，机械臂6秒 / 手5秒）"
        )
        self._exit_btn = QPushButton("退出系统（q，不保存未决轨迹）")
        self._pedal1_btn = QPushButton("踏板1（F7）：开始采集")
        self._pedal2_btn = QPushButton("踏板2（F8）：保存轨迹")
        self._pedal3_btn = QPushButton("踏板3模拟：断开人体控制")
        self._recovery_btn.clicked.connect(lambda: self._send_key("r"))
        self._enable_btn.clicked.connect(self._enable)
        self._exit_btn.clicked.connect(self.close)
        self._pedal1_btn.clicked.connect(lambda: self.pedal_event.emit(1))
        self._pedal2_btn.clicked.connect(lambda: self.pedal_event.emit(2))
        self._pedal3_btn.clicked.connect(lambda: self.pedal_event.emit(3))
        control_grid.addWidget(self._recovery_btn, 0, 0)
        control_grid.addWidget(self._enable_btn, 0, 1)
        control_grid.addWidget(self._exit_btn, 0, 2)
        control_grid.addWidget(self._pedal1_btn, 1, 0)
        control_grid.addWidget(self._pedal2_btn, 1, 1)
        control_grid.addWidget(self._pedal3_btn, 1, 2)
        layout.addWidget(controls)

        log_group = QGroupBox("当前 Session 日志")
        log_layout = QVBoxLayout(log_group)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setStyleSheet(LOG_TEXTEDIT_CSS)
        self._log.document().setMaximumBlockCount(5000)
        log_layout.addWidget(self._log)
        layout.addWidget(log_group, 1)

    def _start_session(self) -> None:
        options = SessionOptions(
            task_name=self.task_name,
            active_hand=str(self._hand_combo.currentData()),
            camera_enabled=self._camera_check.isChecked(),
            camera_transport=os.environ.get(
                "WUJI_CAMERA_TRANSPORT", "direct"
            ),
        )
        try:
            command = options.command()
            self.session.start(command, cwd=repository_root())
        except Exception as exc:
            QMessageBox.critical(self, "启动失败", str(exc))
            return
        self._append_output(">>> " + " ".join(command) + "\n")
        self._camera_window.begin_session(options.camera_enabled)
        self._prepare_btn.setEnabled(False)
        self._hand_combo.setEnabled(False)
        self._camera_check.setEnabled(False)
        self._refresh_controls()

    def _send_key(self, key: str) -> None:
        try:
            self.session.send_key(key)
        except Exception as exc:
            QMessageBox.warning(self, "操作失败", str(exc))

    def _enable(self) -> None:
        self._teleop_label.setText(
            "Teleop：正在 Enable 机械臂；READY 后手用5秒渐进接管"
        )
        self._send_key("a")

    def _handle_pedal(self, pedal_id: int) -> None:
        state = RecorderState.from_status(self.recorder_status)
        if not self.session.running:
            QMessageBox.warning(self, "踏板无效", "请先点击“进入准备”。")
            return
        if pedal_id == 1:
            if self.lifecycle != 2 and not state.recording:
                QMessageBox.warning(
                    self, "踏板1无效", "Tianji 尚未进入 READY。"
                )
                return
            remaining = self._handoff_until - time.monotonic()
            if remaining > 0.0 and not state.recording:
                QMessageBox.warning(
                    self,
                    "接管尚未完成",
                    f"请等待约 {remaining:.1f} 秒后再开始采集。",
                )
                return
            self._send_key("s")
        elif pedal_id == 2:
            if not state.pedal2_allowed:
                QMessageBox.warning(
                    self, "踏板2无效", "当前没有等待保存的轨迹。"
                )
                return
            self._send_key("z")
        elif pedal_id == 3:
            reconnecting = self.lifecycle == 11
            if not reconnecting and not state.pedal3_allowed:
                QMessageBox.critical(
                    self,
                    "禁止断联",
                    "必须先踩踏板1结束一次采集；采集中禁止断联。",
                )
                return
            self._send_key("c")

    def _set_lifecycle(self, value: int) -> None:
        previous = self.lifecycle
        self.lifecycle = value
        if value == 2 and previous in (1, 11):
            self._handoff_until = (
                time.monotonic() + self._handoff_duration_s
            )
        elif value not in (2, 10):
            self._handoff_until = 0.0
        self._lifecycle_label.setText(
            f"Lifecycle：{value}/{_LIFECYCLE_NAMES.get(value, 'UNKNOWN')}"
        )
        self._refresh_controls()

    def _set_hand_recovery(self, side: str, value: int) -> None:
        self.hand_recovery_states[side] = value
        names = {0: "IDLE", 1: "RECOVERING", 2: "READY", 3: "FAILED"}
        active_hand = str(self._hand_combo.currentData())
        sides = ("left", "right") if active_hand == "both" else (active_hand,)
        detail = "  ".join(
            f"{item}={names.get(self.hand_recovery_states[item], 'WAITING')}"
            for item in sides
        )
        self._hand_recovery_label.setText("Hand Recovery：" + detail)
        self._refresh_controls()

    def _selected_hands_recovery_ready(self) -> bool:
        active_hand = str(self._hand_combo.currentData())
        sides = ("left", "right") if active_hand == "both" else (active_hand,)
        return all(self.hand_recovery_states[side] == 2 for side in sides)

    def _set_teleop_status(self, value: str) -> None:
        self._teleop_label.setText("Teleop：" + value)

    def _set_recorder_status(self, status: dict) -> None:
        self.recorder_status = dict(status)
        state = str(status.get("state", "unknown"))
        steps = int(
            status.get("steps", 0)
            if state == "recording"
            else status.get("pending_steps", 0)
        )
        elapsed = float(status.get("elapsed_s", 0.0))
        configured = {
            str(name) for name in status.get("configured_cameras", [])
        }
        online = {str(name) for name in status.get("online_cameras", [])}
        camera_parts = []
        for name, title in (
            ("head", "主视角"),
            ("left_wrist", "左腕"),
            ("right_wrist", "右腕"),
        ):
            camera_state = (
                "在线"
                if name in online
                else "离线"
                if name in configured
                else "未配置"
            )
            camera_parts.append(f"{title}={camera_state}")
        self._recorder_label.setText(
            f"Recorder：{state}  帧数={steps}  时长={elapsed:.1f}s\n"
            "相机：" + "  ".join(camera_parts)
        )
        self._refresh_controls()

    def _set_status_snapshot(self, status: dict) -> None:
        self._arm_label.setText(format_arm_snapshot(status))

    def _set_node_names(self, names: set[str]) -> None:
        self.node_names = set(names)
        active_hand = str(self._hand_combo.currentData())
        parts = []
        for label, name in _KEY_NODES.items():
            if label == "Left hand" and active_hand == "right":
                continue
            if label == "Right hand" and active_hand == "left":
                continue
            parts.append(f"{label}={'在线' if name in names else '离线'}")
        self._nodes_label.setText("节点：" + "  ".join(parts))

    def _refresh_controls(self) -> None:
        running = self.session.running
        state = RecorderState.from_status(self.recorder_status)
        handoff_remaining = max(
            0.0, self._handoff_until - time.monotonic()
        )
        self._recovery_btn.setEnabled(
            running
            and not state.recording
            and self.lifecycle in (0, 3)
        )
        self._enable_btn.setEnabled(
            running
            and self.lifecycle == 7
            and self._selected_hands_recovery_ready()
        )
        self._exit_btn.setEnabled(running)
        self._pedal1_btn.setEnabled(
            running
            and (self.lifecycle == 2 or state.recording)
            and (handoff_remaining <= 0.0 or state.recording)
        )
        self._pedal2_btn.setEnabled(running and state.pedal2_allowed)
        self._pedal3_btn.setEnabled(
            running
            and self.lifecycle in (2, 11)
            and (self.lifecycle == 11 or state.pedal3_allowed)
        )
        if (
            handoff_remaining > 0.0
            and self.lifecycle == 2
            and not state.recording
        ):
            self._pedal1_btn.setText(
                f"踏板1（F7）：等待接管 {handoff_remaining:.1f}s"
            )
        elif state.recording:
            self._pedal1_btn.setText("踏板1（F7）：结束本次采集")
        elif state.pending:
            self._pedal1_btn.setText(
                "踏板1（F7）：丢弃旧轨迹并开始新采集"
            )
        else:
            self._pedal1_btn.setText("踏板1（F7）：开始采集")
        self._pedal3_btn.setText(
            "踏板3模拟：恢复人体控制"
            if self.lifecycle == 11
            else "踏板3模拟：断开人体控制"
        )

    def _append_output(self, text: str) -> None:
        self._log.moveCursor(QTextCursor.End)
        self._log.insertPlainText(text)
        self._log.ensureCursorVisible()
        self._log_buffer += text
        lines = self._log_buffer.split("\n")
        self._log_buffer = lines.pop()
        for line in lines:
            normalized = line.strip()
            if normalized.startswith("SAVED:"):
                detail = normalized[6:].strip()
                self._result_label.setText(
                    "最近轨迹结果：已保存 — "
                    + detail
                )
                self._result_label.setStyleSheet("color:#66bb6a;")
                self._camera_window.show_save_success(detail)
            elif normalized.startswith("DISCARDED:"):
                detail = normalized[10:].strip()
                if detail != "no unsaved episode":
                    self._result_label.setText(
                        "最近轨迹结果：已丢弃未保存轨迹 — " + detail
                    )
                    self._result_label.setStyleSheet("color:#ffb74d;")
                    self._camera_window.show_discarded()
            elif normalized.startswith("CAPTURE:"):
                detail = normalized[8:].strip()
                self._camera_window.show_capture_result(detail)
                if "discarded unsaved" in normalized:
                    self._result_label.setText(
                        "最近轨迹结果：上一条未保存轨迹已丢弃"
                    )
                    self._result_label.setStyleSheet("color:#ffb74d;")
            elif (
                normalized.startswith("ERROR:") and (
                    "episode" in normalized.lower()
                    or "capture" in normalized.lower()
                    or "record" in normalized.lower()
                )
            ):
                self._result_label.setText("最近轨迹结果：失败 — " + normalized)
                self._result_label.setStyleSheet("color:#ef5350;")
                self._camera_window.show_record_error(normalized)

    def _session_exited(self, code: int) -> None:
        self._append_output(f"\n>>> Session exited with code {code}\n")
        self.lifecycle = None
        self.hand_recovery_states = {"left": None, "right": None}
        self._handoff_until = 0.0
        self.recorder_status = {}
        self._arm_label.setText("硬件缓存：等待状态")
        self._hand_recovery_label.setText("Hand Recovery：OFFLINE")
        self._prepare_btn.setEnabled(True)
        self._hand_combo.setEnabled(True)
        self._camera_check.setEnabled(True)
        self._camera_window.end_session()
        self._refresh_controls()
        if self._close_requested:
            self._close_requested = False
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event) -> None:
        if self.session.running:
            if self._close_requested:
                event.ignore()
                return
            answer = QMessageBox.question(
                self,
                "退出数采系统",
                "退出会丢弃当前正在采集或尚未保存的轨迹，并安全关闭机械臂。继续吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._close_requested = True
            self._send_key("q")
            event.ignore()
            return
        self.pedal_input.stop()
        self._camera_window.shutdown()
        event.accept()


def _acquire_lock():
    handle = open(_LOCK_FILE, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("数采 GUI 已经在运行")
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> None:
    try:
        lock = _acquire_lock()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    catalog = TaskCatalog()
    catalog.root.mkdir(parents=True, exist_ok=True)
    selector = TaskSelectionDialog(catalog)
    if selector.exec_() != QDialog.Accepted or selector.selected_task is None:
        lock.close()
        return

    rclpy.init(args=None)
    signals = _Signals()
    camera_transport = os.environ.get(
        "WUJI_CAMERA_TRANSPORT", "direct"
    ).strip().lower()
    if camera_transport not in ("direct", "ros"):
        print(
            "WUJI_CAMERA_TRANSPORT must be direct or ros",
            file=sys.stderr,
        )
        lock.close()
        raise SystemExit(2)
    node = RecordGuiRosNode(signals, camera_transport=camera_transport)
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
        name="wuji-record-gui-ros",
    )
    spin_thread.start()
    window = RecordWindow(selector.selected_task, catalog, signals)
    direct_preview = None
    if camera_transport == "direct":
        direct_preview = DirectCameraPreview(
            signals,
            directory=os.environ.get(
                "WUJI_CAMERA_SHARED_MEMORY_DIR",
                str(DEFAULT_DIRECTORY),
            ),
        )
        direct_preview.start()
    window.show()
    exit_code = app.exec_()
    if direct_preview is not None:
        direct_preview.stop()
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    spin_thread.join(timeout=2.0)
    lock.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
