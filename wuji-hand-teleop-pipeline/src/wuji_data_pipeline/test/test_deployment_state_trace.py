import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from wuji_data_pipeline.deployment_state_trace import (
    DeploymentStateTraceNode,
    StateTraceWriter,
    StreamStats,
    _parse_args,
    _finite_vector,
    _stale_limits_ms,
)
from wuji_data_pipeline.plot_deployment_trace import (
    build_summary,
    create_eef_comparison_csv,
    create_joint_comparison_csv,
    create_pdf_report,
    find_nearest_deployment_trace,
    quaternion_error_degrees,
)


def _stream(data, receive_ns=1_000_000_000, sequence=1):
    return {
        "source_stamp_ns": receive_ns,
        "received_monotonic_ns": receive_ns,
        "receive_sequence": sequence,
        "age_ms": 0.0,
        "data": data,
    }


def _snapshot(sequence, time_ns):
    pose_actual = {
        "type": "pose",
        "frame_id": "right_chest",
        "position_m": [0.4 + 0.001 * sequence, 0.0, 0.2],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    pose_target = {
        **pose_actual,
        "position_m": [0.405 + 0.001 * sequence, 0.0, 0.2],
    }
    joint_actual = {
        "type": "joint",
        "unit": "degree",
        "position": [float(sequence)] * 7,
        "velocity": [],
        "effort": [],
    }
    joint_target = {**joint_actual, "position": [float(sequence) + 0.5] * 7}
    hand_actual = {
        "type": "joint",
        "unit": "radian",
        "position": [0.1] * 20,
        "velocity": [],
        "effort": [],
    }
    hand_target = {**hand_actual, "position": [0.2] * 20}
    return {
        "event": "snapshot",
        "sample_sequence": sequence,
        "monotonic_ns": time_ns,
        "ready": True,
        "complete": True,
        "streams": {
            "right.arm_actual_eef": _stream(pose_actual, time_ns, sequence),
            "right.arm_external_target": _stream(pose_target, time_ns, sequence),
            "right.arm_controller_target": _stream(pose_target, time_ns, sequence),
            "right.arm_joint_state": _stream(joint_actual, time_ns, sequence),
            "right.arm_joint_command": _stream(joint_target, time_ns, sequence),
            "right.hand_state": _stream(hand_actual, time_ns, sequence),
            "right.hand_command": _stream(hand_target, time_ns, sequence),
        },
    }


def test_state_trace_writer_drains_and_writes_summary(tmp_path):
    writer = StateTraceWriter(tmp_path, queue_size=16, flush_interval_s=0.01)
    writer.record({"event": "snapshot", "sample_sequence": 1})
    writer.record({"event": "snapshot", "sample_sequence": 2})
    writer.close({"sample_count": 2}, timeout_s=2.0)

    events = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "snapshot",
        "snapshot",
        "trace_summary",
    ]
    assert events[-1]["sample_count"] == 2
    assert events[-1]["writer"]["written_events"] == 2
    assert events[-1]["writer"]["dropped_events"] == 0
    assert writer.writer_error == ""


def test_stream_stats_reports_gaps_and_nonmonotonic_stamps():
    stats = StreamStats()
    assert stats.observe(10, 100) == 1
    assert stats.observe(20, 103) == 2
    assert stats.observe(30, 103) == 3
    assert stats.received == 3
    assert stats.max_source_gap_ms == pytest.approx(3e-6)
    assert stats.nonmonotonic_source_stamps == 1


def test_finite_vector_rejects_bad_data():
    assert _finite_vector([1, 2, 3], 3, "value") == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="3 values"):
        _finite_vector([1, 2], 3, "value")
    with pytest.raises(ValueError, match="NaN"):
        _finite_vector([1, math.nan, 3], 3, "value")


def test_joint_external_target_uses_action_stale_limit():
    limits = _stale_limits_ms({"state_trace_action_max_age_ms": 37.5})

    assert limits["arm_external_target"] == pytest.approx(37.5)
    assert limits["arm_external_joint_target"] == pytest.approx(37.5)


def test_state_trace_cli_removes_ros_launch_arguments():
    args = _parse_args(
        [
            "--config",
            "/tmp/pipeline.yaml",
            "--active-side",
            "right",
            "--arm-command-mode",
            "joint",
            "--ros-args",
            "-r",
            "__node:=wuji_deployment_state_trace",
        ]
    )
    assert args.config == "/tmp/pipeline.yaml"
    assert args.active_side == "right"
    assert args.arm_command_mode == "joint"


def test_deployment_uses_cosmos_config_without_renaming_recording_default():
    package_root = Path(__file__).parents[1]
    deployment_launch = (package_root / "launch" / "deployment.launch.py").read_text()
    deployment_session = (
        package_root / "wuji_data_pipeline" / "deployment_session.py"
    ).read_text()
    record_launch = (package_root / "launch" / "record.launch.py").read_text()
    cosmos_config = yaml.safe_load(
        (package_root / "config" / "cosmos_protocol_v2.yaml").read_text()
    )

    assert '"cosmos_protocol_v2.yaml"' in deployment_launch
    assert '/ "cosmos_protocol_v2.yaml"' in deployment_session
    assert '"pipeline.yaml"' in record_launch
    assert '"--arm-command-mode"' in deployment_launch
    assert "arm_command_mode," in deployment_launch
    assert cosmos_config["deployment"]["protocol_mode"] == "protocol_v2"
    assert int(cosmos_config["deployment"]["open_loop_horizon"]) > 0


def test_cosmos_protocol_v2_config_matches_server_manifest():
    package_root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (package_root / "config" / "cosmos_protocol_v2.yaml").read_text()
    )
    deployment = config["deployment"]

    assert deployment["protocol_mode"] == "protocol_v2"
    assert deployment["server"] == "http://127.0.0.1:8000"
    assert deployment["policy_http_path"] == "/v1/robot-policy"
    assert deployment["policy_http_api_key_env"] == "COSMOS_POLICY_API_KEY"
    assert deployment["policy_http_expected_model_id"] == (
        "singlerighthand-dropper-edge-droid-50k-aot-iter-000030000"
    )
    assert deployment["policy_http_expected_chunk_size"] == 32
    assert deployment["arm_command_mode"] == "joint"
    assert deployment["action_space"] == "joint"
    assert deployment["camera_names"] == ["head", "right_wrist"]
    assert deployment["action_rate_hz"] == pytest.approx(15.0)
    assert deployment["startup_handoff_gate_enabled"] is True
    assert deployment["action_interpolation_method"] == "linear_joint"
    assert deployment["action_smoothing_method"] == "none"
    assert deployment["boundary_blend_steps"] == 4
    assert deployment["joint_step_velocity_validation_enabled"] is False
    assert "small_motion_max_arm_joint_step_rad" not in deployment
    assert (
        "acceleration_limit_deg_s2" not in deployment["joint_safety"]
    )
    assert deployment["max_observation_age_s"] == pytest.approx(1.06)
    assert deployment["policy_start_timeout_s"] == pytest.approx(3.0)
    assert deployment["open_loop_horizon"] == 16
    assert deployment["prefetch_min_lead_actions"] == 7
    assert deployment["prefetch_initial_lead_actions"] == 8
    assert deployment["prefetch_max_lead_actions"] == 9

    chunk_size = int(deployment["policy_http_expected_chunk_size"])
    horizon = int(deployment["open_loop_horizon"])
    maximum_lead = int(deployment["prefetch_max_lead_actions"])
    assert chunk_size >= horizon + maximum_lead + 1


def test_state_trace_required_streams_follow_arm_command_mode():
    node = object.__new__(DeploymentStateTraceNode)
    node._active_sides = ("right",)
    node._arm_command_mode = "eef"
    eef_streams = node._required_streams()
    assert "right.arm_external_target" in eef_streams
    assert "right.arm_controller_target" in eef_streams
    assert "right.arm_external_joint_target" not in eef_streams

    node._arm_command_mode = "joint"
    joint_streams = node._required_streams()
    assert "right.arm_external_joint_target" in joint_streams
    assert "right.arm_external_target" not in joint_streams
    assert "right.arm_controller_target" not in joint_streams


def test_quaternion_error_uses_shortest_rotation():
    actual = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    same_negative_sign = -actual
    assert quaternion_error_degrees(actual, same_negative_sign)[0] == pytest.approx(0.0)


def test_build_summary_marks_lossless_trace_complete():
    state_events = [
        {"event": "trace_start", "monotonic_ns": 1_000_000_000},
        _snapshot(1, 1_000_000_000),
        _snapshot(2, 1_008_333_333),
        {
            "event": "trace_summary",
            "sample_count": 2,
            "ready_sample_count": 2,
            "incomplete_ready_sample_count": 0,
            "stream_stats": {},
            "writer": {"written_events": 3, "dropped_events": 0},
        },
    ]
    deployment_events = [
        {
            "event": "policy_response_ready",
            "complete_rtt_ms": 40.0,
        }
    ]
    summary = build_summary(state_events, deployment_events, ("right",))
    assert summary["complete"] is True
    assert summary["sample_interval_ms"]["mean"] == pytest.approx(8.333333)
    assert summary["policy_complete_rtt_ms"]["p99"] == pytest.approx(40.0)
    assert summary["sides"]["right"]["eef_position_error_rms_m"] == pytest.approx(0.005)
    assert summary["sides"]["right"]["arm_joint_error_max_deg"] == pytest.approx(0.5)


def test_build_summary_fails_completeness_without_footer():
    summary = build_summary(
        [_snapshot(1, 1_000_000_000)], [], ("right",)
    )
    assert summary["complete"] is False
    assert summary["trace_summary_present"] is False
    assert "trace_summary_missing" in summary["completeness_failures"]


def test_build_summary_understands_joint_replay_source():
    snapshot = _snapshot(1, 1_000_000_000)
    snapshot["streams"].pop("right.arm_external_target")
    snapshot["streams"].pop("right.arm_controller_target")
    snapshot["streams"]["right.arm_external_joint_target"] = _stream({
        "type": "joint",
        "unit": "radian",
        "position": [math.radians(1.5)] * 7,
        "velocity": [],
        "effort": [],
    })
    state_events = [
        {"event": "trace_start", "arm_command_mode": "joint"},
        snapshot,
        {
            "event": "trace_summary",
            "sample_rate_hz": 120.0,
            "sample_count": 1,
            "ready_sample_count": 1,
            "incomplete_ready_sample_count": 0,
            "stream_stats": {},
            "writer": {"written_events": 2, "dropped_events": 0},
        },
    ]

    summary = build_summary(state_events, [], ("right",))

    assert summary["complete"] is True
    assert summary["arm_command_mode"] == "joint"
    assert summary["sides"]["right"]["source_joint_error_rms_deg"] == pytest.approx(0.5)
    assert summary["sides"]["right"]["eef_position_error_rms_m"] is None


def test_deployment_trace_matching_prefers_exact_status_session(tmp_path):
    session_id = "12345678abcdef"
    expected = tmp_path / "deployment_trace_new_12345678.jsonl"
    older = tmp_path / "deployment_trace_old_deadbeef.jsonl"
    expected.write_text(json.dumps({
        "event": "session_start",
        "session_id": session_id,
        "wall_time": 20.0,
    }) + "\n")
    older.write_text(json.dumps({
        "event": "session_start",
        "session_id": "deadbeef",
        "wall_time": 10.0,
    }) + "\n")
    state_events = [
        {"event": "trace_start", "wall_time_ns": 10_000_000_000},
        {
            "event": "deployment_status",
            "status": {
                "session_id": session_id,
                "diagnostic_trace": {
                    "path": f"/container/path/{expected.name}"
                },
            },
        },
    ]

    assert find_nearest_deployment_trace(
        tmp_path / "deployment_state.jsonl", state_events
    ) == expected


def test_create_eef_comparison_csv(tmp_path):
    output = create_eef_comparison_csv(
        tmp_path / "eef.csv",
        [_snapshot(1, 1_000_000_000), _snapshot(2, 1_008_333_333)],
        ("right",),
    )
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["side"] == "right"
    assert float(rows[0]["actual_input_position_error_mm"]) == pytest.approx(5.0)
    assert float(rows[0]["actual_input_rotation_error_deg"]) == pytest.approx(0.0)
    assert float(rows[0]["controller_input_position_error_mm"]) == pytest.approx(0.0)


def test_create_joint_comparison_csv_converts_source_radians(tmp_path):
    snapshot = _snapshot(1, 1_000_000_000)
    snapshot["streams"]["right.arm_external_joint_target"] = _stream({
        "type": "joint",
        "unit": "radian",
        "position": [math.pi / 2.0] * 7,
        "velocity": [],
        "effort": [],
    })
    output = create_joint_comparison_csv(
        tmp_path / "joints.csv", [snapshot], ("right",)
    )
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 1
    assert float(rows[0]["external_j1_deg"]) == pytest.approx(90.0)
    assert float(rows[0]["actual_minus_command_j1_deg"]) == pytest.approx(-0.5)


def test_create_pdf_report_from_state_and_action_trace(tmp_path):
    # The robot container currently carries NumPy 2 with Ubuntu's older
    # Matplotlib binary. Plotting is intentionally performed offline by the
    # host wrapper; all data capture/summary tests still run in the container.
    if int(np.__version__.split(".", 1)[0]) >= 2:
        pytest.skip("PDF rendering is verified in the host analysis environment")
    state_events = [
        _snapshot(1, 1_000_000_000),
        _snapshot(2, 1_008_333_333),
        {
            "event": "trace_summary",
            "sample_count": 2,
            "ready_sample_count": 2,
            "incomplete_ready_sample_count": 0,
            "stream_stats": {},
            "writer": {"written_events": 2, "dropped_events": 0},
        },
    ]
    deployment_events = [
        {
            "event": "command_publish",
            "monotonic_time": 1.0,
            "chunk_id": 1,
        }
    ]
    summary = build_summary(state_events, deployment_events, ("right",))
    output = create_pdf_report(
        tmp_path / "report.pdf",
        state_events,
        deployment_events,
        ("right",),
        summary,
    )
    assert output.is_file()
    assert output.stat().st_size > 1000
