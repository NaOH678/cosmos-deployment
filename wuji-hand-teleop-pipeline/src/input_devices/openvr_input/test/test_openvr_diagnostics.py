from types import SimpleNamespace

import numpy as np

from openvr_input.openvr_input_node import (
    OpenVRInputNode,
    TRACKER_DIAGNOSTIC_COLUMNS,
    TRACKER_DIAGNOSTIC_ROLES,
)
from openvr_input.openvr_tracker_wrapper import OpenVRTrackerWrapper


def test_diagnostic_publisher_is_fixed_shape_and_keeps_raw_and_corrected_apart():
    raw = np.eye(4, dtype=np.float32)
    raw[:3, 3] = [1.0, 2.0, 3.0]
    corrected = np.eye(4, dtype=np.float32)
    corrected[:3, 3] = [4.0, 5.0, 6.0]
    raw_state = {
        "pose": raw,
        "linear_velocity": np.asarray([0.1, 0.2, 0.3]),
        "angular_velocity": np.asarray([0.4, 0.5, 0.6]),
        "connected": True,
        "valid": True,
        "tracking_result": 200,
        "device_index": 7,
    }
    published = []
    node = object.__new__(OpenVRInputNode)
    node.tracker_wrapper = SimpleNamespace(
        get_raw_states=lambda: {"chest": raw_state}
    )
    node._diagnostics_publisher = SimpleNamespace(publish=published.append)

    OpenVRInputNode._publish_tracker_diagnostics(
        node, {"chest": corrected}
    )

    assert len(published) == 1
    rows = np.asarray(published[0].data).reshape(
        len(TRACKER_DIAGNOSTIC_ROLES), TRACKER_DIAGNOSTIC_COLUMNS
    )
    np.testing.assert_allclose(rows[0, 0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rows[0, 7:10], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(rows[0, 14:20], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert rows[0, 20:23].tolist() == [1.0, 1.0, 1.0]
    assert rows[0, 23:26].tolist() == [200.0, 7.0, 1.0]
    # Undetected roles remain explicit invalid rows with sentinel IDs.
    assert rows[1, 20:23].tolist() == [0.0, 0.0, 0.0]
    assert rows[1, 23:25].tolist() == [-1.0, -1.0]


def test_raw_cache_is_copied_before_control_path_applies_offsets():
    matrix = SimpleNamespace(
        m=[
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
        ]
    )

    def vector(values):
        return SimpleNamespace(v=values)

    pose = SimpleNamespace(
        mDeviceToAbsoluteTracking=matrix,
        vVelocity=vector([0.1, 0.2, 0.3]),
        vAngularVelocity=vector([0.4, 0.5, 0.6]),
        bDeviceIsConnected=True,
        bPoseIsValid=True,
        eTrackingResult=200,
    )
    wrapper = object.__new__(OpenVRTrackerWrapper)
    wrapper._vr_system = SimpleNamespace(
        getDeviceToAbsoluteTrackingPose=lambda *_args: [pose, pose, pose, pose]
    )
    wrapper._detected_trackers = {3: "right_wrist"}
    wrapper._latest_raw_states = {}
    wrapper._is_connected = True

    all_poses = OpenVRTrackerWrapper._get_all_raw_poses(wrapper)
    control_matrix = OpenVRTrackerWrapper._extract_pose(
        wrapper, all_poses, 3
    )
    control_matrix[:3, 3] += 10.0
    cached = OpenVRTrackerWrapper.get_raw_states(wrapper)["right_wrist"]

    np.testing.assert_allclose(cached["pose"][:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cached["linear_velocity"], [0.1, 0.2, 0.3])
    assert cached["valid"] is True
