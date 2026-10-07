import os

import pytest

from wuji_teleop_monitor.ui.pedal_input import (
    DEFAULT_KEYBOARD_BINDINGS,
    KeyboardShortcutPedalInput,
)


def test_default_keyboard_bindings_only_enable_first_two_pedals():
    assert DEFAULT_KEYBOARD_BINDINGS == ((1, "F7"), (2, "F8"))


def test_keyboard_shortcuts_deliver_logical_pedal_events():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5")
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    events = []
    adapter = KeyboardShortcutPedalInput(parent)
    adapter.start(events.append)
    parent.show()
    parent.activateWindow()
    app.processEvents()

    QTest.keyClick(parent, Qt.Key_F7)
    QTest.keyClick(parent, Qt.Key_F8)
    app.processEvents()

    assert events == [1, 2]
    adapter.stop()
    parent.close()
