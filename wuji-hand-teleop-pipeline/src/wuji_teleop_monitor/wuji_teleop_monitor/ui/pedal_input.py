"""Foot-pedal adapters for the data-collection GUI.

Adapters call the callback with logical pedal IDs.  The GUI's physical inputs
and simulation buttons therefore share the same state guards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional


PedalCallback = Callable[[int], None]
DEFAULT_KEYBOARD_BINDINGS = ((1, "F7"), (2, "F8"))


class PedalInputAdapter(ABC):
    @abstractmethod
    def start(self, callback: PedalCallback) -> None:
        """Start delivering pedal IDs to ``callback``."""

    @abstractmethod
    def stop(self) -> None:
        """Stop delivery and release hardware resources."""


class NullPedalInput(PedalInputAdapter):
    """Adapter for tests or deployments without physical pedals."""

    def __init__(self) -> None:
        self.callback: Optional[PedalCallback] = None

    def start(self, callback: PedalCallback) -> None:
        self.callback = callback

    def stop(self) -> None:
        self.callback = None


class KeyboardShortcutPedalInput(PedalInputAdapter):
    """Map keyboard-mode USB pedals to logical pedal events.

    LinTx pedals enumerate as ordinary HID keyboards.  Qt shortcuts avoid a
    direct ``/dev/input`` dependency and therefore do not need elevated input
    device permissions.  Auto-repeat is disabled so holding a pedal produces
    one logical event.
    """

    def __init__(
        self,
        parent,
        bindings=DEFAULT_KEYBOARD_BINDINGS,
    ) -> None:
        self._parent = parent
        self._bindings = tuple(bindings)
        self._callback: Optional[PedalCallback] = None
        self._shortcuts = []

    @property
    def bindings(self):
        return self._bindings

    def start(self, callback: PedalCallback) -> None:
        if self._shortcuts:
            self._callback = callback
            return

        # Imported lazily because run_record configures Qt's plugin path before
        # constructing this adapter.
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtWidgets import QShortcut

        self._callback = callback
        for pedal_id, key_name in self._bindings:
            shortcut = QShortcut(QKeySequence(key_name), self._parent)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(
                partial(self._deliver, pedal_id)
            )
            self._shortcuts.append(shortcut)

    def stop(self) -> None:
        self._callback = None
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._shortcuts.clear()

    def _deliver(self, pedal_id: int) -> None:
        callback = self._callback
        if callback is not None:
            callback(pedal_id)
