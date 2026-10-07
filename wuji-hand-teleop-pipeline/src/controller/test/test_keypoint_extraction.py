"""Unit tests for glove skeleton -> MediaPipe (21,3) extraction."""
from types import SimpleNamespace

import numpy as np

from controller.wujihand_node import (
    _convert_to_mediapipe,
    _extract_wuji_glove_keypoints,
)


def _make_skeleton(positions, frame_id="r_wrist"):
    """Build a minimal skeleton stub: header + joints[*].pose.position."""
    joints = [
        SimpleNamespace(pose=SimpleNamespace(position=list(p)))
        for p in positions
    ]
    header = SimpleNamespace(frame_id=frame_id)
    return SimpleNamespace(header=header, joints=joints)


def test_extracts_21_keypoints_with_correct_shape():
    positions = [(i * 0.01, i * 0.02, i * 0.03) for i in range(21)]
    skel = _make_skeleton(positions)
    kp = _extract_wuji_glove_keypoints(skel)
    assert kp.shape == (21, 3)
    assert kp.dtype == np.float32
    np.testing.assert_allclose(kp[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(kp[20], [0.20, 0.40, 0.60], atol=1e-6)


def test_returns_none_on_wrong_joint_count():
    positions = [(0.0, 0.0, 0.0)] * 15  # too few
    skel = _make_skeleton(positions)
    assert _extract_wuji_glove_keypoints(skel) is None


def test_returns_none_on_too_many_joints():
    positions = [(0.0, 0.0, 0.0)] * 25
    skel = _make_skeleton(positions)
    assert _extract_wuji_glove_keypoints(skel) is None


def _make_manus_node(chain, joint, position, node_id):
    point = SimpleNamespace(x=position[0], y=position[1], z=position[2])
    pose = SimpleNamespace(position=point)
    return SimpleNamespace(
        node_id=node_id,
        chain_type=chain,
        joint_type=joint,
        pose=pose,
    )


def _make_manus_message():
    labels = [
        ("Hand", "Invalid"),
        ("Thumb", "MCP"), ("Thumb", "PIP"),
        ("Thumb", "DIP"), ("Thumb", "TIP"),
        ("Index", "MCP"), ("Index", "PIP"), ("Index", "IP"),
        ("Index", "DIP"), ("Index", "TIP"),
        ("Middle", "MCP"), ("Middle", "PIP"), ("Middle", "IP"),
        ("Middle", "DIP"), ("Middle", "TIP"),
        ("Ring", "MCP"), ("Ring", "PIP"), ("Ring", "IP"),
        ("Ring", "DIP"), ("Ring", "TIP"),
        ("Pinky", "MCP"), ("Pinky", "PIP"), ("Pinky", "IP"),
        ("Pinky", "DIP"), ("Pinky", "TIP"),
    ]
    nodes = [
        _make_manus_node(chain, joint, (i, i + 0.25, i + 0.5), 100 - i)
        for i, (chain, joint) in enumerate(labels)
    ]
    return SimpleNamespace(raw_nodes=list(reversed(nodes)))


def test_manus_mapping_uses_semantic_labels_not_node_ids():
    keypoints = _convert_to_mediapipe(_make_manus_message())
    assert keypoints.shape == (21, 3)
    assert keypoints.dtype == np.float32
    np.testing.assert_allclose(keypoints[0], [0.0, 0.25, 0.5])
    # Non-thumb Metacarpal nodes are skipped. Proximal (labelled PIP by the
    # ROS wrapper) maps to MediaPipe MCP, and Intermediate/IP maps to PIP.
    np.testing.assert_allclose(keypoints[5], [6.0, 6.25, 6.5])
    np.testing.assert_allclose(keypoints[6], [7.0, 7.25, 7.5])
    np.testing.assert_allclose(keypoints[20], [24.0, 24.25, 24.5])


def test_manus_mapping_rejects_incomplete_skeleton():
    msg = _make_manus_message()
    msg.raw_nodes = [
        node for node in msg.raw_nodes
        if not (node.chain_type == "Index" and node.joint_type == "TIP")
    ]
    assert _convert_to_mediapipe(msg) is None
