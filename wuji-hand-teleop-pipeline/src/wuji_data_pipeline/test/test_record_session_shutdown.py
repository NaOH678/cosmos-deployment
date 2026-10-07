import signal
import subprocess
import threading
from types import SimpleNamespace

import pytest

from wuji_data_pipeline import record_session


class _FakeProcess:
    def __init__(self, wait_results=None):
        self.pid = 4321
        self._wait_results = list(wait_results or [0])

    def poll(self):
        return None

    def wait(self, timeout):
        result = self._wait_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_launch_graph_starts_in_an_owned_process_group(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(record_session.subprocess, "Popen", fake_popen)

    record_session._launch_graph(
        ["ros2", "launch", "package", "file.launch.py"],
        str(tmp_path / "children.log"),
    )

    assert captured["start_new_session"] is True
    assert captured["stderr"] is subprocess.STDOUT
    assert captured["stdout"].name == str(tmp_path / "children.log")


def test_shutdown_signals_the_entire_launch_group(monkeypatch):
    signals = []
    monkeypatch.setattr(
        record_session.os,
        "killpg",
        lambda process_group, sig: signals.append((process_group, sig)),
    )
    process = _FakeProcess()

    record_session._shutdown_launch_graph(process)

    assert signals == [(process.pid, signal.SIGINT)]


def test_shutdown_escalates_a_stuck_launch_group(monkeypatch):
    signals = []
    monkeypatch.setattr(
        record_session.os,
        "killpg",
        lambda process_group, sig: signals.append((process_group, sig)),
    )
    process = _FakeProcess(
        [
            subprocess.TimeoutExpired(
                "ros2 launch", record_session.LAUNCH_SIGINT_TIMEOUT_S
            ),
            subprocess.TimeoutExpired(
                "ros2 launch", record_session.LAUNCH_SIGTERM_TIMEOUT_S
            ),
            0,
        ]
    )

    record_session._shutdown_launch_graph(process)

    assert signals == [
        (process.pid, signal.SIGINT),
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]


def test_required_nodes_follow_documented_external_openvr_and_single_hand():
    nodes = record_session.required_control_nodes("right")

    assert "/openvr_input" in nodes
    assert "/right_hand/wujihand_driver" in nodes
    assert "/wujihand_controller_right" in nodes
    assert "/left_hand/wujihand_driver" not in nodes
    assert "/tianji_right_tf" in nodes
    assert "/tianji_left_tf" not in nodes
    assert "/manus_data_publisher" in nodes


def test_right_arm_enable_uses_only_the_right_tracker_tf_chain():
    assert record_session.active_arm_tf_pairs("right") == (
        ("right_chest", "tianji_right"),
        ("right_chest", "right_arm"),
    )


def test_recovery_preflight_requires_controller_and_selected_hand_driver():
    captured = {}

    def recovery_preflight_nodes(required_nodes, **kwargs):
        captured["required_nodes"] = required_nodes
        captured.update(kwargs)
        return []

    fake_session = SimpleNamespace(
        recovery_preflight_nodes=recovery_preflight_nodes
    )

    errors = record_session.SessionClient.recovery_preflight(
        fake_session,
        active_hand="right",
        active_arm="right",
    )

    assert errors == []
    assert captured["required_nodes"] == {
        "/tianji_arm_controller",
        "/right_hand/wujihand_driver",
    }
    assert captured["tf_pairs"] == ()


def test_enable_preflight_requires_the_selected_teleop_graph_and_tf():
    captured = {}

    def enable_preflight_nodes(required_nodes, **kwargs):
        captured["required_nodes"] = required_nodes
        captured.update(kwargs)
        return []

    fake_session = SimpleNamespace(
        enable_preflight_nodes=enable_preflight_nodes
    )

    errors = record_session.SessionClient.enable_preflight(
        fake_session,
        active_hand="right",
        active_arm="right",
    )

    assert errors == []
    assert "/openvr_input" in captured["required_nodes"]
    assert "/manus_data_publisher" in captured["required_nodes"]
    assert "/right_hand/wujihand_driver" in captured["required_nodes"]
    assert "/left_hand/wujihand_driver" not in captured["required_nodes"]
    assert captured["tf_pairs"] == (
        ("right_chest", "tianji_right"),
        ("right_chest", "right_arm"),
    )


def test_enable_preflight_checks_nodes_before_waiting_for_tf():
    fake_session = SimpleNamespace(
        visible_nodes=lambda: {"/tianji_arm_controller"},
        missing_recovery_transforms=lambda **kwargs: pytest.fail(
            "TF check must not run while required nodes are missing"
        ),
    )

    errors = record_session.SessionClient.enable_preflight_nodes(
        fake_session,
        {
            "/tianji_arm_controller",
            "/openvr_input",
        },
        tf_pairs=(("right_chest", "tianji_right"),),
    )

    assert errors == ["missing ROS node /openvr_input"]


def test_single_hand_enable_targets_all_joints_on_the_selected_side():
    calls = []

    def call(name, request, **kwargs):
        calls.append((name, request, kwargs))
        return True, "ok"

    fake_session = SimpleNamespace(call=call)

    ok, message = record_session.SessionClient.set_hands_enabled(
        fake_session,
        "right",
        True,
    )

    assert ok is True
    assert message == "WujiHand enabled: right"
    assert len(calls) == 1
    name, request, _ = calls[0]
    assert name == "hand_enable_right"
    assert request.finger_id == 255
    assert request.joint_id == 255
    assert request.enabled is True


def test_partial_dual_hand_enable_rolls_back_the_enabled_side():
    calls = []

    def call(name, request, **kwargs):
        calls.append((name, request.enabled))
        if name == "hand_enable_right" and request.enabled:
            return False, "right hand offline"
        return True, "ok"

    fake_session = SimpleNamespace(call=call)

    ok, message = record_session.SessionClient.set_hands_enabled(
        fake_session,
        "both",
        True,
    )

    assert ok is False
    assert "right hand offline" in message
    assert calls == [
        ("hand_enable_left", True),
        ("hand_enable_right", True),
        ("hand_enable_left", False),
    ]


def test_hand_disable_attempts_both_sides_even_after_one_failure():
    calls = []

    def call(name, request, **kwargs):
        calls.append((name, request.enabled))
        return (name != "hand_enable_left"), "reply"

    fake_session = SimpleNamespace(call=call)

    ok, message = record_session.SessionClient.set_hands_enabled(
        fake_session,
        "both",
        False,
    )

    assert ok is False
    assert "left" in message
    assert calls == [
        ("hand_enable_left", False),
        ("hand_enable_right", False),
    ]


def test_hand_recovery_starts_only_the_selected_side():
    calls = []
    fake_session = SimpleNamespace(
        hand_recovery_states={"left": 2, "right": 2},
        call=lambda name, request, **kwargs: (
            calls.append((name, type(request).__name__, kwargs)) or (True, "ok")
        ),
    )

    ok, message = record_session.SessionClient.start_hand_recovery(
        fake_session,
        "right",
    )

    assert ok is True
    assert message == "WujiHand Recovery started: right"
    assert fake_session.hand_recovery_states["right"] is None
    assert [call[0] for call in calls] == ["hand_recover_right"]


def test_partial_dual_hand_recovery_disables_the_accepted_side():
    calls = []

    def call(name, request, **kwargs):
        calls.append((name, getattr(request, "enabled", None)))
        if name == "hand_recover_right":
            return False, "initial pose missing"
        return True, "ok"

    fake_session = SimpleNamespace(
        hand_recovery_states={"left": 0, "right": 0},
        call=call,
    )

    ok, message = record_session.SessionClient.start_hand_recovery(
        fake_session,
        "both",
    )

    assert ok is False
    assert "initial pose missing" in message
    assert calls == [
        ("hand_recover_left", None),
        ("hand_recover_right", None),
        ("hand_enable_left", False),
    ]


def test_hand_recovery_result_requires_every_selected_hand_ready():
    fake_session = SimpleNamespace(
        hand_recovery_states={"left": record_session.HAND_RECOVERY_READY,
                              "right": record_session.HAND_RECOVERY_MOVING}
    )
    assert record_session.SessionClient.hand_recovery_result(
        fake_session, "both"
    ) == "pending"

    fake_session.hand_recovery_states["right"] = record_session.HAND_RECOVERY_READY
    assert record_session.SessionClient.hand_recovery_result(
        fake_session, "both"
    ) == "ready"

    fake_session.hand_recovery_states["left"] = record_session.HAND_RECOVERY_FAILED
    assert record_session.SessionClient.hand_recovery_result(
        fake_session, "both"
    ) == "failed"


def test_arm_status_must_be_clean_standby_before_recovery():
    clean = {
        side: {"state": 0, "err_code": 0, "servo_errors": ["0x00000000"] * 7}
        for side in ("left", "right")
    }
    assert record_session.arm_status_errors(clean) == []

    clean["left"]["state"] = 1
    clean["right"]["servo_errors"][2] = "0x8613"
    errors = record_session.arm_status_errors(clean)
    assert "left:state=1 (required 0)" in errors
    assert any("right:servo_faults" in error for error in errors)


def test_shutdown_confirmation_checks_state_zero_not_fault_cleanup():
    status = {
        side: {
            "state": 0,
            "err_code": 42,
            "servo_errors": ["0x8613"] * 7,
        }
        for side in ("left", "right")
    }

    assert record_session.arm_standby_errors(status) == []

    status["right"]["state"] = 3
    assert record_session.arm_standby_errors(status) == [
        "right:state=3 (required 0)"
    ]


def test_recovery_tf_pairs_match_the_documented_tf2_echo_checks():
    assert record_session.RECOVERY_TF_PAIRS == (
        ("left_chest", "tianji_left"),
        ("right_chest", "tianji_right"),
        ("left_chest", "left_arm"),
        ("right_chest", "right_arm"),
    )


def test_recovery_tf_wait_spins_before_declaring_frames_missing(monkeypatch):
    class _Buffer:
        ready = False

        def can_transform(self, target, source, when):
            return self.ready

    buffer = _Buffer()
    fake_session = SimpleNamespace(_tf_buffer=buffer)

    def fake_spin_once(node, timeout_sec):
        buffer.ready = True

    monkeypatch.setattr(record_session.rclpy, "spin_once", fake_spin_once)

    errors = record_session.SessionClient.missing_recovery_transforms(
        fake_session, timeout_s=0.1
    )

    assert errors == []


def test_documented_environment_is_enforced(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "0")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    with pytest.raises(SystemExit, match="ROS_DOMAIN_ID must be 112"):
        record_session._validate_documented_environment()

    monkeypatch.setenv("ROS_DOMAIN_ID", "112")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    monkeypatch.delenv("CYCLONEDDS_URI", raising=False)
    record_session._validate_documented_environment()


def test_service_discovery_can_fail_fast_independently_of_response_timeout():
    class _Client:
        srv_name = "/recorder/stop"

        def __init__(self):
            self.wait_timeout = None

        def wait_for_service(self, timeout_sec):
            self.wait_timeout = timeout_sec
            return False

    client = _Client()
    fake_session = SimpleNamespace(_service_clients={"stop": client})

    ok, message = record_session.SessionClient.call(
        fake_session,
        "stop",
        object(),
        timeout_s=60.0,
        service_wait_s=0.2,
    )

    assert ok is False
    assert message == "service unavailable: /recorder/stop"
    assert 0.0 <= client.wait_timeout <= 0.05


def test_ctrl_c_interrupts_an_inflight_recorder_service_call(monkeypatch):
    abort_event = threading.Event()

    class _Future:
        cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True

    future = _Future()

    class _Client:
        srv_name = "/wuji_teleop_recorder/stop"

        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, request):
            return future

    def fake_spin_once(node, timeout_sec):
        abort_event.set()

    monkeypatch.setattr(record_session.rclpy, "spin_once", fake_spin_once)
    fake_session = SimpleNamespace(_service_clients={"stop": _Client()})

    ok, message = record_session.SessionClient.call(
        fake_session,
        "stop",
        object(),
        timeout_s=60.0,
        abort_event=abort_event,
    )

    assert ok is False
    assert message == (
        "service call interrupted by shutdown: /wuji_teleop_recorder/stop"
    )
    assert future.cancelled is True


def test_shutdown_requests_disable_then_confirms_both_arm_states(monkeypatch):
    calls = []

    class _Session:
        def call(self, name, request, **kwargs):
            calls.append(name)
            if name == "enable":
                return True, "Arms stopped"
            return True, (
                '{"left":{"state":0,"err_code":0},'
                '"right":{"state":0,"err_code":0}}'
            )

    ok, message = record_session._request_tianji_standby(_Session())

    assert ok is True
    assert message == "both Tianji arms confirmed state=0"
    assert calls == ["enable", "arm_status"]


def test_shutdown_fails_closed_when_state_zero_is_not_confirmed(monkeypatch):
    class _Session:
        def call(self, name, request, **kwargs):
            if name == "enable":
                return False, "set_standby timed out"
            return True, (
                '{"left":{"state":3,"err_code":0},'
                '"right":{"state":3,"err_code":0}}'
            )

    monkeypatch.setattr(record_session.time, "sleep", lambda seconds: None)
    ok, message = record_session._request_tianji_standby(
        _Session(), verify_timeout_s=0.0
    )

    assert ok is False
    assert "left:state=3 (required 0)" in message
    assert "right:state=3 (required 0)" in message
