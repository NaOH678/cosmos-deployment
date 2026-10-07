import pytest
import numpy as np

from wuji_data_pipeline.cloud_policy_server import action_vector_to_mapping
from wuji_data_pipeline.deployment_protocol import (
    LatestActionPlan,
    PendingActionChunk,
    PchipActionInterpolator,
    VelocityContinuousBoundaryInterpolator,
    blend_action_prefix,
    calculate_prefetch_lead,
    decode_color_image,
    encode_color_image,
    extract_action_chunk,
    interpolate_action_pair,
    quaternion_from_rotvec_xyzw,
    quaternion_relative_rotvec_xyzw,
    select_pending_action_window,
    smooth_action_chunk_butterworth,
)


def test_jpeg_roundtrip_preserves_shape_and_transport_metadata():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 180

    payload = encode_color_image(
        image, codec="jpeg", jpeg_quality=90, timestamp=12.5
    )
    decoded = decode_color_image(payload)

    assert payload["codec"] == "jpeg"
    assert payload["timestamp"] == 12.5
    assert isinstance(payload["data"], bytes)
    assert decoded.shape == image.shape
    assert decoded.dtype == np.uint8


def test_legacy_action_and_protocol_chunk_are_both_supported():
    legacy = {"arm_action_left": {}, "hand_action_left": []}
    actions, rate = extract_action_chunk(legacy, default_rate_hz=30.0)
    assert actions == [legacy]
    assert rate == 30.0

    chunk = {"action_rate_hz": 20.0, "action_chunk": [{"step": 1}, {"step": 2}]}
    actions, rate = extract_action_chunk(chunk, default_rate_hz=30.0)
    assert [action["step"] for action in actions] == [1, 2]
    assert rate == 20.0


def test_latest_plan_preserves_order_without_latency_or_timer_drops():
    plan = LatestActionPlan(max_actions=50)
    result = plan.install(
        [{"step": index} for index in range(4)],
        observation_created_at=1.0,
        received_at=1.11,
        rate_hz=10.0,
    )

    assert result.latency_dropped == 0
    assert result.accepted == 4
    assert plan.pop_due(1.11).action["step"] == 0
    # A timer delayed by more than one action period consumes exactly one
    # action and realigns the rest.  It neither skips nor bursts the backlog.
    assert plan.pop_due(1.36).action["step"] == 1
    assert plan.pop_due(1.36) is None
    assert plan.pop_due(1.459) is None
    assert plan.pop_due(1.461).action["step"] == 2
    assert plan.pop_due(1.561).action["step"] == 3
    assert plan.remaining() == 0
    assert plan.status()["execution_dropped_actions"] == 0
    assert plan.status()["schedule_realignments"] == 1


def test_latest_plan_exposes_next_waypoint_without_consuming_it():
    plan = LatestActionPlan(max_actions=4)
    plan.install(
        [{"step": 0}, {"step": 1}],
        observation_created_at=1.0,
        received_at=1.0,
        rate_hz=30.0,
    )

    assert plan.pop_due(1.0).action["step"] == 0
    assert plan.peek_next().action["step"] == 1
    assert plan.remaining() == 1


def test_position_only_action_interpolation_uses_linear_and_shortest_slerp():
    start = {}
    end = {}
    for side in ("left", "right"):
        start[f"arm_action_{side}"] = {
            "ee_pos": [0.0, 0.0, 0.0],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
            "zsp": [1.0, 0.0, 0.0],
        }
        end[f"arm_action_{side}"] = {
            "ee_pos": [1.0, 2.0, 3.0],
            "ee_quat": [0.0, 0.0, 1.0, 0.0],
            "zsp": [0.0, 1.0, 0.0],
        }
        start[f"hand_action_{side}"] = [0.0] * 20
        end[f"hand_action_{side}"] = [20.0] * 20
        start[f"arm_joint_action_{side}"] = [0.0] * 7
        end[f"arm_joint_action_{side}"] = [1.0] * 7

    sampled = interpolate_action_pair(start, end, fraction=0.25)

    assert np.allclose(
        sampled["arm_action_right"]["ee_pos"], [0.25, 0.5, 0.75]
    )
    assert np.allclose(
        sampled["arm_action_right"]["ee_quat"],
        [0.0, 0.0, np.sin(np.pi / 8.0), np.cos(np.pi / 8.0)],
    )
    expected_zsp = np.asarray([0.75, 0.25, 0.0])
    expected_zsp /= np.linalg.norm(expected_zsp)
    assert np.allclose(sampled["arm_action_right"]["zsp"], expected_zsp)
    assert np.allclose(sampled["hand_action_right"], [5.0] * 20)
    assert np.allclose(sampled["arm_joint_action_right"], [0.25] * 7)


def test_pchip_action_interpolator_is_c1_and_shape_preserving():
    actions = []
    for value in (0.0, 1.0, 0.0):
        action = {}
        for side in ("left", "right"):
            action[f"arm_action_{side}"] = {
                "ee_pos": [value, 0.0, 0.0],
                "ee_quat": [0.0, 0.0, 0.0, 1.0],
                "zsp": [0.0, 1.0, 0.0],
            }
            action[f"hand_action_{side}"] = [10.0 * value] * 20
            action[f"arm_joint_action_{side}"] = [value] * 7
        actions.append(action)
    interpolator = PchipActionInterpolator(actions)

    samples = [
        interpolator.sample(index / 20.0)
        for index in range(41)
    ]
    positions = np.asarray(
        [sample["arm_action_right"]["ee_pos"][0] for sample in samples]
    )
    hands = np.asarray(
        [sample["hand_action_right"][0] for sample in samples]
    )
    joints = np.asarray(
        [sample["arm_joint_action_right"][0] for sample in samples]
    )
    assert interpolator.pchip_enabled is True
    assert np.min(positions) >= 0.0
    assert np.max(positions) <= 1.0
    assert np.min(hands) >= 0.0
    assert np.max(hands) <= 10.0
    assert np.min(joints) >= 0.0
    assert np.max(joints) <= 1.0
    assert np.isclose(interpolator.sample(1.0)["arm_action_right"]["ee_pos"][0], 1.0)

    epsilon = 1e-4
    before = interpolator.sample(1.0 - epsilon)["arm_action_right"]["ee_pos"][0]
    center = interpolator.sample(1.0)["arm_action_right"]["ee_pos"][0]
    after = interpolator.sample(1.0 + epsilon)["arm_action_right"]["ee_pos"][0]
    derivative_before = (center - before) / epsilon
    derivative_after = (after - center) / epsilon
    assert abs(derivative_before - derivative_after) < 1e-3


def test_pchip_action_interpolator_falls_back_for_two_waypoints():
    actions = []
    for value in (0.0, 1.0):
        action = {}
        for side in ("left", "right"):
            action[f"arm_action_{side}"] = {
                "ee_pos": [value, 0.0, 0.0],
                "ee_quat": [0.0, 0.0, 0.0, 1.0],
                "zsp": [0.0, 1.0, 0.0],
            }
            action[f"hand_action_{side}"] = [value] * 20
        actions.append(action)

    interpolator = PchipActionInterpolator(actions)
    sampled = interpolator.sample(0.25)

    assert interpolator.pchip_enabled is False
    assert np.isclose(
        sampled["arm_action_right"]["ee_pos"][0], 0.25
    )
    assert np.isclose(sampled["hand_action_right"][0], 0.25)


def test_joint_action_interpolation_keeps_arm_and_hand_in_radians():
    actions = []
    for value in (0.0, 0.4, 1.0):
        action = {}
        for side in ("left", "right"):
            action[f"arm_action_{side}"] = {
                "joint_pos": [value] * 7,
            }
            action[f"hand_action_{side}"] = [0.5 * value] * 20
        actions.append(action)

    linear = interpolate_action_pair(
        actions[0],
        actions[2],
        fraction=0.25,
        action_mode="joint",
    )
    assert np.allclose(
        linear["arm_action_right"]["joint_pos"], [0.25] * 7
    )
    assert np.allclose(linear["hand_action_right"], [0.125] * 20)
    assert "ee_pos" not in linear["arm_action_right"]
    assert "ee_quat" not in linear["arm_action_right"]

    interpolator = PchipActionInterpolator(actions, action_mode="joint")
    sampled = interpolator.sample(1.5)
    assert interpolator.pchip_enabled
    assert np.asarray(
        sampled["arm_action_right"]["joint_pos"]
    ).shape == (7,)
    assert np.asarray(sampled["hand_action_right"]).shape == (20,)
    assert np.all(np.isfinite(sampled["arm_action_right"]["joint_pos"]))
    assert np.all(np.isfinite(sampled["hand_action_right"]))


def test_butterworth_chunk_smoothing_removes_waypoint_jitter_without_delay():
    actions = []
    for index in range(50):
        alternating = -1.0 if index % 2 else 1.0
        yaw = 0.01 * index + 0.04 * alternating
        action = {}
        for side in ("left", "right"):
            action[f"arm_action_{side}"] = {
                "ee_pos": [
                    0.5 + 0.002 * index + 0.003 * alternating,
                    -0.2 + 0.002 * alternating,
                    0.3,
                ],
                "ee_quat": [
                    0.0,
                    0.0,
                    np.sin(0.5 * yaw),
                    np.cos(0.5 * yaw),
                ],
                "zsp": [0.0, -1.0, -0.5],
            }
            action[f"hand_action_{side}"] = [
                0.25 * index + 2.0 * alternating
            ] * 20
        actions.append(action)

    smoothed = smooth_action_chunk_butterworth(
        actions,
        rate_hz=30.0,
        cutoff_hz=3.0,
        order=3,
    )

    raw_position = np.asarray(
        [action["arm_action_right"]["ee_pos"] for action in actions]
    )
    smooth_position = np.asarray(
        [action["arm_action_right"]["ee_pos"] for action in smoothed]
    )
    raw_hand = np.asarray(
        [action["hand_action_right"] for action in actions]
    )
    smooth_hand = np.asarray(
        [action["hand_action_right"] for action in smoothed]
    )
    assert np.linalg.norm(np.diff(smooth_position, n=2, axis=0)) < (
        0.1 * np.linalg.norm(np.diff(raw_position, n=2, axis=0))
    )
    assert np.linalg.norm(np.diff(smooth_hand, n=2, axis=0)) < (
        0.1 * np.linalg.norm(np.diff(raw_hand, n=2, axis=0))
    )
    quaternions = np.asarray(
        [action["arm_action_right"]["ee_quat"] for action in smoothed]
    )
    assert np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6)
    assert all(
        action["arm_action_right"]["zsp"] == [0.0, -1.0, -0.5]
        for action in smoothed
    )


def test_latest_plan_keeps_30hz_phase_when_polled_at_100hz():
    plan = LatestActionPlan(max_actions=61)
    plan.install(
        [{"step": index} for index in range(61)],
        observation_created_at=0.0,
        received_at=0.0,
        rate_hz=30.0,
    )

    dispatch_times = []
    for tick in range(301):
        now = tick / 100.0
        action = plan.pop_due(now)
        if action is not None:
            dispatch_times.append(now)

    assert len(dispatch_times) == 61
    measured_rate = (len(dispatch_times) - 1) / (
        dispatch_times[-1] - dispatch_times[0]
    )
    assert 29.5 <= measured_rate <= 30.5
    status = plan.status()
    assert status["schedule_realignments"] == 0
    assert 29.5 <= status["rolling_dispatch_rate_hz"] <= 30.5
    assert status["dispatched_actions"] == 61


def test_latest_plan_does_not_burst_after_a_long_executor_pause():
    plan = LatestActionPlan(max_actions=4)
    plan.install(
        [{"step": index} for index in range(4)],
        observation_created_at=0.0,
        received_at=0.0,
        rate_hz=30.0,
    )

    assert plan.pop_due(0.0).action["step"] == 0
    assert plan.pop_due(0.5).action["step"] == 1
    assert plan.pop_due(0.5) is None
    assert plan.pop_due(0.532) is None
    assert plan.pop_due(0.534).action["step"] == 2
    assert plan.status()["schedule_realignments"] == 1


def test_prefetch_lead_uses_bounded_complete_rtt_p99():
    common = {
        "action_rate_hz": 30.0,
        "safety_actions": 2,
        "minimum_actions": 3,
        "maximum_actions": 10,
        "initial_actions": 5,
        "minimum_samples": 5,
    }

    lead, p99_ms = calculate_prefetch_lead([], **common)
    assert lead == 5
    assert p99_ms == 0.0

    # Before the window is warm, retain the conservative configured lead.
    lead, p99_ms = calculate_prefetch_lead([20.0, 40.0, 60.0], **common)
    assert lead == 5
    assert p99_ms == 60.0

    # P99=75 ms -> ceil(2.25 action periods) + 2 safety actions = 5.
    lead, p99_ms = calculate_prefetch_lead(
        [20.0, 30.0, 40.0, 50.0, 75.0], **common
    )
    assert lead == 5
    assert p99_ms == 75.0

    lead, _ = calculate_prefetch_lead([1000.0] * 5, **common)
    assert lead == 10


def test_pending_chunk_skips_elapsed_prefix_and_keeps_25_actions():
    pending = PendingActionChunk(
        actions=tuple({"step": index} for index in range(50)),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.06,
        align_to_observation=True,
    )

    actions, skipped = select_pending_action_window(
        pending,
        activation_at=10.0 + 5.0 / 30.0,
        horizon=25,
    )

    assert skipped == 5
    assert len(actions) == 25
    assert actions[0]["step"] == 5
    assert actions[-1]["step"] == 29


def test_non_prefetched_chunk_starts_from_zero_and_can_keep_future_schedule():
    pending = PendingActionChunk(
        actions=tuple({"step": index} for index in range(50)),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.08,
        align_to_observation=False,
    )
    actions, skipped = select_pending_action_window(
        pending, activation_at=10.08, horizon=25
    )
    assert skipped == 0

    plan = LatestActionPlan(max_actions=25)
    plan.install(
        actions,
        observation_created_at=pending.observation_created_at,
        received_at=pending.response_received_at,
        rate_hz=pending.action_rate_hz,
        schedule_start_at=10.10,
    )
    assert plan.pop_due(10.099) is None
    assert plan.pop_due(10.10).action["step"] == 0


def test_pending_chunk_rejects_alignment_without_full_horizon():
    pending = PendingActionChunk(
        actions=tuple({"step": index} for index in range(50)),
        action_rate_hz=30.0,
        observation_created_at=0.0,
        response_received_at=0.1,
        align_to_observation=True,
    )
    with np.testing.assert_raises_regex(TimeoutError, "full aligned horizon"):
        select_pending_action_window(
            pending,
            activation_at=26.0 / 30.0,
            horizon=25,
        )


def test_full_chunk_horizon_can_explicitly_disable_time_alignment():
    pending = PendingActionChunk(
        actions=tuple({"step": index} for index in range(50)),
        action_rate_hz=30.0,
        observation_created_at=10.0,
        response_received_at=10.1,
        align_to_observation=True,
    )

    actions, skipped = select_pending_action_window(
        pending,
        activation_at=10.2,
        horizon=50,
        align_to_observation=False,
    )

    assert len(actions) == 50
    assert skipped == 0
    assert actions[0]["step"] == 0
    assert actions[-1]["step"] == 49


def test_boundary_blend_advances_aligned_targets_without_time_delay():
    actions = []
    for index in range(8):
        action = {}
        for side in ("left", "right"):
            action[f"arm_action_{side}"] = {
                "ee_pos": [0.06 * (index + 1), 0.0, 0.5],
                "ee_quat": [0.0, 0.0, 0.3826834, 0.9238795],
                "zsp": [1.0, 0.0, 0.0],
            }
            action[f"hand_action_{side}"] = [6.0 * (index + 1)] * 20
        actions.append(action)
    anchors_pose = {
        side: np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        for side in ("left", "right")
    }
    anchors_hand = {
        side: np.zeros(20, dtype=np.float32)
        for side in ("left", "right")
    }
    anchors_zsp = {
        side: np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for side in ("left", "right")
    }

    blended = blend_action_prefix(
        actions,
        anchor_poses=anchors_pose,
        anchor_hands=anchors_hand,
        anchor_zsp=anchors_zsp,
        blend_steps=6,
    )

    first_alpha = (1.0 / 6.0) ** 2 * (3.0 - 2.0 / 6.0)
    assert np.isclose(
        blended[0]["arm_action_right"]["ee_pos"][0],
        first_alpha * 0.06,
    )
    assert np.isclose(
        blended[0]["hand_action_right"][0], first_alpha * 6.0
    )
    assert np.isclose(
        np.linalg.norm(blended[0]["arm_action_right"]["zsp"]), 1.0
    )
    # The bridge endpoint is the aligned model's sixth action, then the
    # original seventh action follows immediately. No six-frame delay is added.
    assert np.allclose(
        blended[5]["arm_action_right"]["ee_pos"],
        actions[5]["arm_action_right"]["ee_pos"],
    )
    assert blended[6] is actions[6]
    assert actions[0]["arm_action_right"]["ee_pos"][0] == 0.06


def test_joint_boundary_blend_preserves_joint_and_degree_hand_semantics():
    actions = []
    for index in range(8):
        value = float(index + 1)
        action = {}
        for side in ("left", "right"):
            action[f"arm_action_{side}"] = {
                "joint_pos": np.radians([value] * 7).tolist()
            }
            action[f"hand_action_{side}"] = [value] * 20
        actions.append(action)
    anchors_joint = {
        side: np.zeros(7, dtype=np.float32)
        for side in ("left", "right")
    }
    anchors_hand = {
        side: np.zeros(20, dtype=np.float32)
        for side in ("left", "right")
    }

    blended = blend_action_prefix(
        actions,
        anchor_poses=anchors_joint,
        anchor_hands=anchors_hand,
        anchor_zsp={side: None for side in ("left", "right")},
        blend_steps=6,
        action_mode="joint",
    )

    alpha = (1.0 / 6.0) ** 2 * (3.0 - 2.0 / 6.0)
    assert np.isclose(
        blended[0]["arm_action_right"]["joint_pos"][0],
        alpha * np.radians(1.0),
    )
    assert np.isclose(blended[0]["hand_action_right"][0], alpha)
    assert "ee_pos" not in blended[0]["arm_action_right"]
    assert np.allclose(
        blended[5]["arm_action_right"]["joint_pos"],
        actions[5]["arm_action_right"]["joint_pos"],
    )


def test_velocity_continuous_boundary_matches_both_endpoint_velocities():
    rate_hz = 30.0
    publish_dt = 1.0 / 120.0
    linear_velocity = 0.09
    angular_velocity = 0.12
    hand_velocity = 3.0
    actions = []
    for index in range(10):
        action = {}
        for side in ("left", "right"):
            action[f"arm_action_{side}"] = {
                "ee_pos": [0.02 + 0.003 * index, 0.0, 0.5],
                "ee_quat": quaternion_from_rotvec_xyzw(
                    np.array([0.0, 0.0, 0.02 + 0.004 * index])
                ).tolist(),
                "zsp": [1.0, 0.0, 0.0],
            }
            action[f"hand_action_{side}"] = [0.5 + 0.1 * index] * 20
        actions.append(action)
    anchors_pose = {
        side: np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        for side in ("left", "right")
    }
    previous_pose = {
        side: np.concatenate(
            [
                np.array([-linear_velocity * publish_dt, 0.0, 0.5]),
                quaternion_from_rotvec_xyzw(
                    np.array([0.0, 0.0, -angular_velocity * publish_dt])
                ),
            ]
        )
        for side in ("left", "right")
    }
    anchors_hand = {
        side: np.zeros(20, dtype=np.float32)
        for side in ("left", "right")
    }
    previous_hand = {
        side: np.full(20, -hand_velocity * publish_dt, dtype=np.float32)
        for side in ("left", "right")
    }
    anchors_zsp = {
        side: np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for side in ("left", "right")
    }

    bridge = VelocityContinuousBoundaryInterpolator(
        actions,
        anchor_poses=anchors_pose,
        anchor_hands=anchors_hand,
        anchor_zsp=anchors_zsp,
        previous_poses=previous_pose,
        previous_hands=previous_hand,
        previous_dt_s=publish_dt,
        blend_steps=6,
        rate_hz=rate_hz,
    )

    epsilon = 1e-5
    start = bridge.sample(0.0)
    start_plus = bridge.sample(epsilon)
    end_minus = bridge.sample(bridge.duration_s - epsilon)
    end = bridge.sample(bridge.duration_s)
    start_position_velocity = (
        np.asarray(start_plus["arm_action_right"]["ee_pos"])
        - np.asarray(start["arm_action_right"]["ee_pos"])
    ) / epsilon
    end_position_velocity = (
        np.asarray(end["arm_action_right"]["ee_pos"])
        - np.asarray(end_minus["arm_action_right"]["ee_pos"])
    ) / epsilon
    start_hand_velocity = (
        np.asarray(start_plus["hand_action_right"])
        - np.asarray(start["hand_action_right"])
    ) / epsilon
    end_hand_velocity = (
        np.asarray(end["hand_action_right"])
        - np.asarray(end_minus["hand_action_right"])
    ) / epsilon
    start_rotation_velocity = quaternion_relative_rotvec_xyzw(
        start["arm_action_right"]["ee_quat"],
        start_plus["arm_action_right"]["ee_quat"],
    ) / epsilon
    end_rotation_velocity = quaternion_relative_rotvec_xyzw(
        end_minus["arm_action_right"]["ee_quat"],
        end["arm_action_right"]["ee_quat"],
    ) / epsilon

    assert np.isclose(start_position_velocity[0], linear_velocity, atol=2e-3)
    assert np.isclose(end_position_velocity[0], linear_velocity, atol=2e-3)
    assert np.isclose(start_hand_velocity[0], hand_velocity, atol=2e-2)
    assert np.isclose(end_hand_velocity[0], hand_velocity, atol=2e-2)
    assert np.isclose(start_rotation_velocity[2], angular_velocity, atol=2e-3)
    assert np.isclose(end_rotation_velocity[2], angular_velocity, atol=2e-3)
    assert np.allclose(
        end["arm_action_right"]["ee_pos"],
        actions[5]["arm_action_right"]["ee_pos"],
    )
    assert np.isclose(
        abs(
            np.dot(
                end["arm_action_right"]["ee_quat"],
                actions[5]["arm_action_right"]["ee_quat"],
            )
        ),
        1.0,
        atol=1e-6,
    )
    assert np.isclose(
        np.linalg.norm(end["arm_action_right"]["zsp"]), 1.0
    )


def test_latest_plan_uses_configured_25_step_prefix_of_cloud_chunk():
    plan = LatestActionPlan(max_actions=25)
    result = plan.install(
        [{"step": index} for index in range(50)],
        observation_created_at=10.0,
        received_at=10.01,
        rate_hz=30.0,
    )

    assert result.input_count == 50
    assert result.accepted == 25
    assert plan.remaining() == 25


def test_latest_plan_refuses_to_replace_unconsumed_chunk():
    plan = LatestActionPlan(max_actions=50)
    plan.install(
        [{"step": 0}, {"step": 1}],
        observation_created_at=1.0,
        received_at=1.0,
        rate_hz=30.0,
    )

    with np.testing.assert_raises_regex(
        RuntimeError, "cannot replace an unconsumed"
    ):
        plan.install(
            [{"step": 99}],
            observation_created_at=2.0,
            received_at=2.0,
            rate_hz=30.0,
        )


def test_cloud_server_converts_canonical_dual_54d_action():
    action = np.zeros(54, dtype=np.float32)
    action[3:7] = [0.0, 0.0, 0.0, 1.0]
    action[30:34] = [0.0, 0.0, 0.0, 1.0]

    mapped = action_vector_to_mapping(action)

    assert set(mapped) == {
        "arm_action_left",
        "hand_action_left",
        "arm_action_right",
        "hand_action_right",
    }
    assert len(mapped["hand_action_left"]) == 20
    assert len(mapped["hand_action_right"]) == 20


@pytest.mark.parametrize("prefetched,expected", [(True, 6), (False, 0)])
def test_fixed_skip_preserves_window_and_bootstrap(prefetched, expected):
    pending = PendingActionChunk(actions=tuple({"step": i} for i in range(32)),
        action_rate_hz=15.0, observation_created_at=1.0, response_received_at=1.6,
        align_to_observation=prefetched)
    actions, skipped = select_pending_action_window(pending, activation_at=1.8,
        horizon=24, align_to_observation=False, fixed_skip_actions=6)
    assert skipped == expected
    assert [x["step"] for x in actions] == list(range(expected, expected + 24))
    if prefetched:
        with pytest.raises(TimeoutError):
            select_pending_action_window(pending, activation_at=1.8, horizon=24,
                align_to_observation=False, fixed_skip_actions=9)
        with pytest.raises(ValueError):
            select_pending_action_window(pending, activation_at=1.8, horizon=24,
                align_to_observation=True, fixed_skip_actions=6)
