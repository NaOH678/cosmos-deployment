# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy
import hashlib
import http.client
import pickle
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cosmos_framework.inference.robot_policy.adapters import (
    AdapterOutput,
    HoldAdapter,
    JointHoldAdapter,
    _compose_right_wrist_over_head,
    _decode_jpeg_rgb,
    _verify_checkpoint_sha256,
    build_native_right_joint_state,
    build_native_right_state,
    create_model_adapter,
    native_right_actions_to_wire,
    native_right_joint_actions_to_wire,
)
from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig, load_robot_policy_config
from cosmos_framework.inference.robot_policy.protocol import ProtocolError, validate_hello, validate_observation
from cosmos_framework.inference.robot_policy.server import PolicyApplication, RobotPolicyHTTPServer, SessionRegistry


def _robot_layout() -> dict[str, Any]:
    return {
        "arm_dof": 14,
        "hand_dof": 40,
        "arm_dof_per_side": 7,
        "hand_dof_per_side": 20,
        "eef_dof_per_side": 7,
        "state_dim": 54,
        "action_dim": 54,
        "eef_dim": 14,
        "n_sides": 2,
        "sides": ["left", "right"],
        "action_layout": [
            {"side": "left", "eef": [0, 7], "hand": [7, 27]},
            {"side": "right", "eef": [27, 34], "hand": [34, 54]},
        ],
        "qpos_layout": [
            {"side": "left", "arm": [0, 7], "hand": [7, 27]},
            {"side": "right", "arm": [27, 34], "hand": [34, 54]},
        ],
        "units": {
            "qpos.arm": "radian",
            "qpos.hand": "radian",
            "action.eef.position": "metre",
            "action.eef.quaternion": "xyzw",
            "action.hand": "degree",
        },
    }


def _joint_robot_layout() -> dict[str, Any]:
    layout = _robot_layout()
    layout["action_layout"] = [
        {"side": "left", "arm_joint": [0, 7], "hand": [7, 27]},
        {"side": "right", "arm_joint": [27, 34], "hand": [34, 54]},
    ]
    layout["units"] = {
        "qpos.arm": "radian",
        "qpos.hand": "radian",
        "action.arm_joint": "radian",
        "action.hand": "degree",
    }
    return layout


def _config(**service_overrides: Any) -> RobotPolicyConfig:
    service = {
        "host": "127.0.0.1",
        "port": 18080,
        "max_request_bytes": 1024 * 1024,
        "max_response_bytes": 1024 * 1024,
        "max_sessions": 2,
        "session_ttl_s": 60.0,
        "max_inflight_inferences": 1,
    }
    service.update(service_overrides)
    return RobotPolicyConfig.model_validate(
        {
            "deployment": {
                "model_id": "cosmos-test-v1",
                "camera_names": ["head", "right_wrist"],
                "robot_layout": _robot_layout(),
            },
            "model": {"service_mode": "hold", "warmup": False},
            "service": service,
        }
    )


def _joint_config(**service_overrides: Any) -> RobotPolicyConfig:
    service = {
        "host": "127.0.0.1",
        "port": 18080,
        "max_request_bytes": 1024 * 1024,
        "max_response_bytes": 1024 * 1024,
        "max_sessions": 2,
        "session_ttl_s": 60.0,
        "max_inflight_inferences": 1,
    }
    service.update(service_overrides)
    return RobotPolicyConfig.model_validate(
        {
            "deployment": {
                "model_id": "cosmos-joint-test-v1",
                "arm_command_mode": "joint",
                "action_space": "joint",
                "camera_names": ["head", "right_wrist"],
                "robot_layout": _joint_robot_layout(),
                "startup_handoff": True,
            },
            "model": {"service_mode": "hold", "warmup": False},
            "service": service,
        }
    )


def _arm_state(position: tuple[float, float, float] = (0.1, 0.2, 0.3)) -> dict[str, np.ndarray]:
    quaternion = np.asarray([0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    position_array = np.asarray(position, dtype=np.float32)
    return {
        "joint_pos": np.zeros(7, dtype=np.float32),
        "joint_vel": np.zeros(7, dtype=np.float32),
        "joint_torque": np.zeros(7, dtype=np.float32),
        "ee_pos": position_array.copy(),
        "ee_quat": quaternion.copy(),
        "eef": np.concatenate([position_array, quaternion]),
    }


def _hand_state(position_rad: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "joint_pos": np.full(20, position_rad, dtype=np.float32),
        "joint_vel": np.zeros(20, dtype=np.float32),
        "joint_torque": np.zeros(20, dtype=np.float32),
    }


def _hello(request_id: int = 1, session_id: str = "session-a") -> dict[str, Any]:
    return {
        "protocol_version": 2,
        "message_type": "hello",
        "session_id": session_id,
        "request_id": request_id,
        "robot_layout": _robot_layout(),
        "camera_names": ["head", "right_wrist"],
    }


def _joint_hello(request_id: int = 1, session_id: str = "session-a") -> dict[str, Any]:
    request = _hello(request_id, session_id)
    request.update(
        {
            "robot_layout": _joint_robot_layout(),
            "arm_command_mode": "joint",
            "action_space": "joint",
        }
    )
    return request


def _observation(request_id: int = 2, session_id: str = "session-a") -> dict[str, Any]:
    timestamp = 1000.0
    image = {
        "codec": "jpeg",
        "shape": [2, 3, 3],
        "color_space": "bgr8",
        "timestamp": timestamp,
        "data": b"fixture-jpeg",
    }
    return {
        "protocol_version": 2,
        "schema_version": 2,
        "message_type": "observation",
        "session_id": session_id,
        "request_id": request_id,
        "timestamp": timestamp,
        "client_monotonic": 10.0,
        "arms": ["left", "right"],
        "robot_layout": _robot_layout(),
        "active_hand_sides": ["right"],
        "zero_filled_hand_sides": ["left"],
        "source_timestamps": {"camera_head": timestamp, "camera_right_wrist": timestamp},
        "arm_state_left": _arm_state((-0.1, 0.2, 0.3)),
        "hand_state_left": _hand_state(),
        "arm_state_right": _arm_state(),
        "hand_state_right": _hand_state(),
        "images": {"head": copy.deepcopy(image), "right_wrist": copy.deepcopy(image)},
    }


def _joint_observation(request_id: int = 2, session_id: str = "session-a") -> dict[str, Any]:
    request = _observation(request_id, session_id)
    request["robot_layout"] = _joint_robot_layout()
    return request


class _CountingHoldAdapter(HoldAdapter):
    def __init__(self, config: RobotPolicyConfig) -> None:
        super().__init__(config)
        self.inference_count = 0

    def infer(self, observation: Mapping[str, Any]) -> AdapterOutput:
        self.inference_count += 1
        return super().infer(observation)


class _BlockingHoldAdapter(_CountingHoldAdapter):
    def __init__(self, config: RobotPolicyConfig) -> None:
        super().__init__(config)
        self.started = threading.Event()
        self.release = threading.Event()

    def infer(self, observation: Mapping[str, Any]) -> AdapterOutput:
        self.started.set()
        assert self.release.wait(timeout=5.0)
        return super().infer(observation)


def _post(
    connection: http.client.HTTPConnection,
    payload: Mapping[str, Any] | bytes,
    *,
    api_key: str = "test-key",
) -> tuple[http.client.HTTPResponse, bytes]:
    body = payload if isinstance(payload, bytes) else pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    connection.request(
        "POST",
        "/v1/robot-policy",
        body=body,
        headers={
            "Content-Type": "application/octet-stream",
            "Accept": "application/octet-stream",
            "Authorization": f"Bearer {api_key}",
            "X-OpenPI-API-Key": api_key,
        },
    )
    response = connection.getresponse()
    response_body = response.read()
    return response, response_body


def test_example_config_is_valid() -> None:
    root = Path(__file__).parents[3]
    config = load_robot_policy_config(root / "examples/deployment/cosmos_singlerighthand_protocol_v2.yaml")
    assert config.deployment.camera_names == ["head", "right_wrist"]
    assert config.deployment.robot_layout == _robot_layout()
    assert config.coordinate_frames.model_to_wire_transform == "identity"


def test_nano_example_config_is_valid() -> None:
    root = Path(__file__).parents[3]
    config = load_robot_policy_config(root / "examples/deployment/cosmos_singlerighthand_nano_protocol_v2.yaml")
    assert config.deployment.model_id == "singlerighthand-nano-policy-droid-iter-000027500"
    assert config.model.experiment == "action_policy_singlerighthand_nano"
    assert config.model.max_action_dim == 64
    assert config.model.use_ema_weights is True


def test_dropper_edge_example_config_is_valid() -> None:
    root = Path(__file__).parents[3]
    config = load_robot_policy_config(root / "examples/deployment/cosmos_singlerighthand_dropper_edge_protocol_v2.yaml")
    assert config.deployment.model_id == "singlerighthand-dropper-edge-droid-50k-aot-iter-000030000"
    assert config.deployment.camera_names == ["head", "right_wrist"]
    assert config.deployment.arm_command_mode == "joint"
    assert config.deployment.action_space == "joint"
    assert config.deployment.robot_layout == _joint_robot_layout()
    assert config.deployment.startup_handoff is True
    assert config.model.experiment == "action_policy_singlerighthand_edge"
    assert config.model.task == "draw liquid from the beaker with a dropper and dispense it into the test tube"
    assert config.model.use_ema_weights is True


def test_joint_hello_requires_explicit_matching_action_profile() -> None:
    config = _joint_config()
    assert validate_hello(_joint_hello(), config) == ("session-a", 1)

    missing_mode = _joint_hello()
    del missing_mode["arm_command_mode"]
    with pytest.raises(ProtocolError, match="arm_command_mode must be 'joint'"):
        validate_hello(missing_mode, config)

    wrong_space = _joint_hello()
    wrong_space["action_space"] = "eef"
    with pytest.raises(ProtocolError, match="action_space must be 'joint'"):
        validate_hello(wrong_space, config)


def test_checkpoint_sha256_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"fixed-model-artifact")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    _verify_checkpoint_sha256(str(artifact), expected)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verify_checkpoint_sha256(str(artifact), "0" * 64)


def test_observation_rejects_missing_camera_and_nonfinite_state() -> None:
    config = _config()
    missing_camera = _observation()
    del missing_camera["images"]["right_wrist"]
    with pytest.raises(ProtocolError, match="missing required cameras"):
        validate_observation(missing_camera, config)

    nonfinite = _observation()
    nonfinite["arm_state_right"]["eef"][0] = np.nan
    with pytest.raises(ProtocolError, match="non-finite"):
        validate_observation(nonfinite, config)


def test_native_right_action_conversion_holds_left_and_converts_hand_to_degrees() -> None:
    config = _config()
    observation = _observation()
    native = np.zeros((32, 27), dtype=np.float32)
    native[:, :3] = observation["arm_state_right"]["eef"][:3]
    native[:, 6] = 1.0
    native[:, 7:] = 0.1

    actions = native_right_actions_to_wire(observation, native, config)

    assert len(actions) == 32
    assert actions[0]["arm_action_left"]["ee_pos"] == pytest.approx([-0.1, 0.2, 0.3])
    assert actions[0]["arm_action_left"]["ee_quat"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert actions[0]["hand_action_left"] == pytest.approx([0.0] * 20)
    assert actions[0]["hand_action_right"] == pytest.approx([np.rad2deg(0.1)] * 20)


def test_joint_action_conversion_uses_joint_envelope_and_holds_left() -> None:
    config = _joint_config()
    observation = _observation()
    observation["arm_state_left"]["joint_pos"] = np.arange(7, dtype=np.float32) / 100.0
    observation["arm_state_right"]["joint_pos"] = np.arange(7, dtype=np.float32) / 50.0
    native = np.zeros((32, 27), dtype=np.float32)
    native[:, :7] = observation["arm_state_right"]["joint_pos"] + 0.01
    native[:, 7:] = 0.1

    actions = native_right_joint_actions_to_wire(observation, native, config)

    assert len(actions) == 32
    assert set(actions[0]) == {
        "arm_joint_action_left",
        "hand_action_left",
        "arm_joint_action_right",
        "hand_action_right",
    }
    assert actions[0]["arm_joint_action_left"] == pytest.approx(np.arange(7) / 100.0)
    assert actions[-1]["arm_joint_action_left"] == pytest.approx(actions[0]["arm_joint_action_left"])
    assert actions[0]["arm_joint_action_right"] == pytest.approx(np.arange(7) / 50.0 + 0.01)
    assert actions[0]["hand_action_left"] == pytest.approx([0.0] * 20)
    assert actions[0]["hand_action_right"] == pytest.approx([np.rad2deg(0.1)] * 20)


def test_joint_hold_adapter_is_selected_without_changing_eef_hold_adapter() -> None:
    assert isinstance(create_model_adapter(_joint_config()), JointHoldAdapter)
    assert type(create_model_adapter(_config())) is HoldAdapter


def test_binomial_smoothing_reduces_zigzag_and_preserves_final_grasp_target() -> None:
    raw_config = _config().model_dump()
    raw_config["model"].update(
        {
            "service_mode": "full",
            "checkpoint_path": "/fixture/checkpoint",
            "trajectory_smoothing": "none",
        }
    )
    unsmoothed_config = RobotPolicyConfig.model_validate(raw_config)
    raw_config["model"]["trajectory_smoothing"] = "binomial5"
    smoothed_config = RobotPolicyConfig.model_validate(raw_config)
    observation = _observation()

    native = np.zeros((32, 27), dtype=np.float32)
    progress = np.linspace(0.0, 1.0, len(native), dtype=np.float32)
    alternating = np.where(np.arange(len(native)) % 2 == 0, 1.0, -1.0).astype(np.float32)
    native[:, :3] = observation["arm_state_right"]["eef"][:3]
    native[:, 0] += 0.12 * progress + 0.02 * alternating
    native[:, 6] = 1.0
    native[:, 7:] = (0.8 * progress + 0.08 * alternating)[:, None]

    unsmoothed = native_right_actions_to_wire(observation, native, unsmoothed_config)
    smoothed = native_right_actions_to_wire(observation, native, smoothed_config)
    raw_position = np.asarray([step["arm_action_right"]["ee_pos"] for step in unsmoothed])
    filtered_position = np.asarray([step["arm_action_right"]["ee_pos"] for step in smoothed])
    raw_hand = np.asarray([step["hand_action_right"] for step in unsmoothed])
    filtered_hand = np.asarray([step["hand_action_right"] for step in smoothed])

    assert (
        np.linalg.norm(np.diff(filtered_position, axis=0), axis=1).mean()
        < 0.25 * np.linalg.norm(np.diff(raw_position, axis=0), axis=1).mean()
    )
    assert np.abs(np.diff(filtered_hand, axis=0)).mean() < 0.25 * np.abs(np.diff(raw_hand, axis=0)).mean()
    assert filtered_position[-1] == pytest.approx(raw_position[-1])
    assert filtered_hand[-1] == pytest.approx(raw_hand[-1])


def test_native_state_layout_normalizes_xyzw_and_keeps_hand_radians() -> None:
    observation = _observation()
    observation["hand_state_right"]["joint_pos"] = np.arange(20, dtype=np.float32) / 100.0

    state = build_native_right_state(observation)

    assert state.shape == (27,)
    assert state[:3] == pytest.approx([0.1, 0.2, 0.3])
    assert state[3:7] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert state[7:] == pytest.approx(np.arange(20, dtype=np.float32) / 100.0)


def test_native_joint_state_uses_only_arm_and_hand_joints_in_radians() -> None:
    observation = _observation()
    observation["arm_state_right"]["joint_pos"] = np.arange(7, dtype=np.float32) / 10.0
    observation["hand_state_right"]["joint_pos"] = np.arange(20, dtype=np.float32) / 100.0
    del observation["arm_state_right"]["eef"]
    del observation["arm_state_right"]["ee_pos"]
    del observation["arm_state_right"]["ee_quat"]

    state = build_native_right_joint_state(observation)

    assert state.shape == (27,)
    assert state[:7] == pytest.approx(np.arange(7, dtype=np.float32) / 10.0)
    assert state[7:] == pytest.approx(np.arange(20, dtype=np.float32) / 100.0)


def test_jpeg_bgr_to_rgb_and_camera_composition_order() -> None:
    cv2 = pytest.importorskip("cv2")
    head_bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    head_bgr[..., 0] = 240
    ok, encoded = cv2.imencode(".jpg", head_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    head_rgb = _decode_jpeg_rgb(
        {"data": encoded.tobytes(), "shape": [4, 6, 3]},
        "head",
    )
    assert head_rgb[..., 2].mean() > 220
    assert head_rgb[..., 0].mean() < 20

    wrist_rgb = np.zeros_like(head_rgb)
    wrist_rgb[..., 1] = 200
    composed = _compose_right_wrist_over_head(head_rgb, wrist_rgb).numpy()
    assert composed.shape == (3, 8, 6)
    assert composed[1, :4].mean() == pytest.approx(200.0)
    assert composed[2, 4:].mean() > 220


def test_small_motion_limits_right_targets() -> None:
    raw = _config().model_dump()
    raw["model"].update({"service_mode": "small_motion", "checkpoint_path": "/fixture/checkpoint"})
    config = RobotPolicyConfig.model_validate(raw)
    observation = _observation()
    native = np.zeros((32, 27), dtype=np.float32)
    native[:, :3] = observation["arm_state_right"]["eef"][:3] + np.asarray([1.0, 0.0, 0.0])
    native[:, 5] = 1.0
    native[:, 7:] = 1.0

    action = native_right_actions_to_wire(observation, native, config)[0]

    assert action["arm_action_right"]["ee_pos"][0] == pytest.approx(0.11)
    assert action["hand_action_right"] == pytest.approx([5.0] * 20)


def test_hello_does_not_infer_and_observation_preserves_identity() -> None:
    config = _config()
    adapter = _CountingHoldAdapter(config)
    application = PolicyApplication(config, adapter)

    hello_response, handle = application.handle_request(_hello(11), "connection-a", None, time.time())
    assert adapter.inference_count == 0
    assert hello_response["message_type"] == "hello_ack"
    assert hello_response["session_id"] == "session-a"
    assert hello_response["request_id"] == 11

    response, _ = application.handle_request(_observation(12), "connection-a", handle, time.time())
    assert adapter.inference_count == 1
    assert response["message_type"] == "action_chunk"
    assert response["session_id"] == "session-a"
    assert response["request_id"] == 12


def test_joint_hello_and_observation_return_only_joint_action_envelope() -> None:
    config = _joint_config()
    application = PolicyApplication(config, JointHoldAdapter(config))

    hello_response, handle = application.handle_request(_joint_hello(21), "connection-joint", None, time.time())
    assert hello_response["message_type"] == "hello_ack"

    response, _ = application.handle_request(_joint_observation(22), "connection-joint", handle, time.time())
    assert response["message_type"] == "action_chunk"
    assert response["request_id"] == 22
    assert len(response["action_chunk"]) == 32
    assert set(response["action_chunk"][0]) == {
        "arm_joint_action_left",
        "hand_action_left",
        "arm_joint_action_right",
        "hand_action_right",
    }


def test_reconnect_discards_an_inflight_old_generation() -> None:
    config = _config(max_inflight_inferences=2)
    adapter = _BlockingHoldAdapter(config)
    application = PolicyApplication(config, adapter)
    _, old_handle = application.handle_request(_hello(), "connection-old", None, time.time())
    result: list[dict[str, Any]] = []

    def run_old_request() -> None:
        response, _ = application.handle_request(_observation(), "connection-old", old_handle, time.time())
        result.append(response)

    worker = threading.Thread(target=run_old_request)
    worker.start()
    assert adapter.started.wait(timeout=5.0)
    hello_response, new_handle = application.handle_request(_hello(3), "connection-new", None, time.time())
    assert hello_response["message_type"] == "hello_ack"
    assert new_handle is not None
    adapter.release.set()
    worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert result[0]["error_code"] == "COSMOS_STALE_SESSION"
    assert result[0]["fatal_session"] is True


def test_inference_timeout_is_a_retryable_pickle_error() -> None:
    config = _config(max_inflight_inferences=1, inference_timeout_s=0.01)
    adapter = _BlockingHoldAdapter(config)
    application = PolicyApplication(config, adapter)
    _, handle = application.handle_request(_hello(), "connection-a", None, time.time())

    response, returned_handle = application.handle_request(_observation(), "connection-a", handle, time.time())
    adapter.release.set()

    assert response["error_code"] == "COSMOS_INFERENCE_TIMEOUT"
    assert response["fatal_session"] is False
    assert returned_handle == handle


def test_session_registry_enforces_capacity() -> None:
    registry = SessionRegistry(max_sessions=1, ttl_s=60.0)
    first = registry.begin("first", "connection-a")
    second = registry.begin("second", "connection-b")
    assert not registry.is_current(first)
    assert registry.is_current(second)


def test_http_pickle_protocol_persists_connection_and_reports_bad_pickle() -> None:
    config = _config()
    adapter = _CountingHoldAdapter(config)
    server = RobotPolicyHTTPServer(("127.0.0.1", 0), PolicyApplication(config, adapter), "test-key")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5.0)
    try:
        hello_http, hello_body = _post(connection, _hello(41))
        assert hello_http.status == 200
        assert hello_http.getheader("Content-Type") == "application/octet-stream"
        assert int(hello_http.getheader("Content-Length")) == len(hello_body)
        assert pickle.loads(hello_body)["request_id"] == 41
        socket_after_hello = connection.sock

        action_http, action_body = _post(connection, _observation(42))
        assert action_http.status == 200
        assert connection.sock is socket_after_hello
        assert int(action_http.getheader("Content-Length")) == len(action_body)
        action_response = pickle.loads(action_body)
        assert action_response["message_type"] == "action_chunk"
        assert action_response["request_id"] == 42

        bad_http, bad_body = _post(connection, b"not-a-pickle")
        assert bad_http.status == 200
        assert pickle.loads(bad_body)["error_code"] == "COSMOS_BAD_PICKLE"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_http_rejects_bad_authorization_without_reflecting_it() -> None:
    config = _config()
    server = RobotPolicyHTTPServer(
        ("127.0.0.1", 0),
        PolicyApplication(config, _CountingHoldAdapter(config)),
        "test-key",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5.0)
    try:
        response, body = _post(connection, _hello(), api_key="secret-that-must-not-be-reflected")
        assert response.status == 401
        assert body == b""
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
