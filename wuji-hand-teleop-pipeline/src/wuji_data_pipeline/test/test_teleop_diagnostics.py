from types import SimpleNamespace

import numpy as np
import pytest

from wuji_data_pipeline.teleop_diagnostics import (
    MEDIAPIPE_MANUS_JOINTS,
    MANUS_ERGONOMICS_TYPE_CODES,
    TRACKER_DIAGNOSTIC_COLUMNS,
    decode_manus_message,
    decode_tracker_diagnostic,
)


def _point(values):
    return SimpleNamespace(x=values[0], y=values[1], z=values[2])


def _quaternion(values=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(x=values[0], y=values[1], z=values[2], w=values[3])


def _pose(position, orientation=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        position=_point(position),
        orientation=_quaternion(orientation),
    )


def test_tracker_diagnostic_decoder_preserves_all_five_hardware_rows():
    rows = np.zeros((5, TRACKER_DIAGNOSTIC_COLUMNS), dtype=np.float64)
    rows[:, 0:7] = np.arange(35).reshape(5, 7)
    rows[:, 7:14] = np.arange(35, 70).reshape(5, 7)
    rows[:, 20:23] = 1
    rows[:, 23] = [200, 201, 202, 203, 204]
    rows[:, 24] = [3, 4, 5, 6, 7]
    rows[:, 25] = 1

    decoded = decode_tracker_diagnostic(rows.reshape(-1))

    assert decoded["teleop_tracker_raw_pose"].shape == (5, 7)
    assert decoded["teleop_tracker_corrected_pose"].shape == (5, 7)
    assert decoded["teleop_tracker_valid"].tolist() == [1, 1, 1, 1, 1]
    assert decoded["teleop_tracker_corrected_valid"].tolist() == [1, 1, 1, 1, 1]
    assert decoded["teleop_tracker_device_index"].tolist() == [3, 4, 5, 6, 7]


def test_tracker_diagnostic_decoder_rejects_partial_messages():
    with pytest.raises(ValueError, match="must contain"):
        decode_tracker_diagnostic([0.0] * 10)


def test_manus_decoder_records_raw_nodes_and_exact_controller_keypoints():
    # Include the four extra non-thumb Metacarpal nodes so the raw skeleton is
    # 25 nodes while the semantic control representation remains 21x3.
    labels = list(MEDIAPIPE_MANUS_JOINTS)
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        labels.append((finger, "MCP"))
    nodes = []
    semantic_positions = {}
    for index, (chain, joint) in enumerate(labels):
        position = (index + 0.1, index + 0.2, index + 0.3)
        semantic_positions[(chain.lower(), joint.lower())] = position
        nodes.append(
            SimpleNamespace(
                node_id=100 - index,
                parent_node_id=99 - index,
                chain_type=chain,
                joint_type=joint,
                pose=_pose(position),
            )
        )
    message = SimpleNamespace(
        glove_id=77,
        side="Right",
        raw_node_count=25,
        raw_nodes=list(reversed(nodes)),
        ergonomics_count=1,
        ergonomics=[SimpleNamespace(type="ThumbMCPStretch", value=12.5)],
        raw_sensor_orientation=_quaternion((0.1, 0.2, 0.3, 0.9)),
        raw_sensor_count=1,
        raw_sensor=[_pose((1.0, 2.0, 3.0))],
    )

    decoded = decode_manus_message(message, "right")

    assert decoded["teleop_manus_right_raw_node_pose"].shape == (25, 7)
    assert decoded["teleop_manus_right_node_valid"].sum() == 25
    assert decoded["teleop_manus_right_keypoints_valid"].tolist() == [1]
    expected = np.asarray(
        [
            semantic_positions[(chain.lower(), joint.lower())]
            for chain, joint in MEDIAPIPE_MANUS_JOINTS
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        decoded["teleop_manus_right_keypoints_21"], expected
    )
    assert decoded["teleop_manus_right_glove_id"].tolist() == [77]
    assert decoded["teleop_manus_right_ergonomics_type_code"][0] == (
        MANUS_ERGONOMICS_TYPE_CODES.index("ThumbMCPStretch")
    )
    assert decoded["teleop_manus_right_raw_sensor_valid"].tolist() == [1, 0, 0, 0, 0]
