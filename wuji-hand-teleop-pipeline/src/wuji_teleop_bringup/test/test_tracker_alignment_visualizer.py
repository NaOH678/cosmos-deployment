import math
from types import SimpleNamespace

import numpy as np

from wuji_teleop_bringup.tracker_alignment_visualizer import (
    TrackerAlignmentVisualizer,
    compute_chest_anchor,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_error_deg,
    rpy_to_matrix,
)


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


CHEST_CONFIG = {
    "shared": {"x": 0.0, "y": -0.2, "z": 0.12},
    "left": {
        "roll": math.pi / 2, "pitch": 0.0, "yaw": -math.pi / 2,
        "local_x": 0.0, "local_y": 0.0, "local_z": 0.037,
    },
    "right": {
        "roll": math.pi / 2, "pitch": 0.0, "yaw": math.pi / 2,
        "local_x": 0.0, "local_y": 0.0, "local_z": 0.037,
    },
}


def test_quaternion_roundtrip():
    rotation = rpy_to_matrix(0.4, -0.2, 1.1)
    recovered = quaternion_to_matrix(matrix_to_quaternion(rotation))
    np.testing.assert_allclose(recovered, rotation, atol=1e-9)


def test_rotation_error_uses_shortest_angle():
    assert rotation_error_deg(np.eye(3), rpy_to_matrix(0, 0, math.pi / 2)) == 90.0


def test_chest_anchor_places_robot_shoulder_at_virtual_shoulder():
    chest_position = np.array([0.4, 1.2, -0.7])
    chest_rotation = rpy_to_matrix(0.1, -0.3, 0.2)
    base_position, base_rotation = compute_chest_anchor(
        chest_rotation, chest_position, CHEST_CONFIG, np.array([0.0, 1.0, 0.0]))
    expected_shoulder = chest_position + chest_rotation @ np.array([0.0, -0.2, 0.12])
    actual_shoulder = base_position + base_rotation @ np.array([0.0, 0.0, 1.121])
    np.testing.assert_allclose(actual_shoulder, expected_shoulder, atol=1e-9)
    np.testing.assert_allclose(base_rotation.T @ base_rotation, np.eye(3), atol=1e-9)


def test_controller_detection_pauses_and_resumes_offline_preview():
    logger = RecordingLogger()
    visualizer = SimpleNamespace(
        _controller_online=False,
        _teleop_status="OFFLINE PREVIEW: no hardware commands",
        count_publishers=lambda topic: 1,
        get_logger=lambda: logger,
    )

    TrackerAlignmentVisualizer._update_controller_source(visualizer)

    assert visualizer._controller_online is True
    assert "handed over" in logger.messages[-1]

    visualizer.count_publishers = lambda topic: 0
    TrackerAlignmentVisualizer._update_controller_source(visualizer)

    assert visualizer._controller_online is False
    assert visualizer._teleop_status == "OFFLINE PREVIEW: no hardware commands"
    assert "resumed" in logger.messages[-1]
