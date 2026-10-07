from copy import deepcopy
import pickle

import numpy as np
import pytest

from wuji_data_pipeline.fdm_async import (
    FDM_PROTOCOL_MODE,
    FeedbackAccumulator,
    FeedbackQueueFull,
    FdmFeedbackWorker,
    FdmProtocolConfig,
    FdmProtocolError,
    FdmSessionError,
    FdmSessionLedger,
    NativeActionRechunker,
    build_action_request,
    build_bootstrap,
    build_hello,
    native_spans_for_wire,
    parse_action_chunk,
    robot_layout_for_action_mode,
    validate_hello_ack,
)
from wuji_data_pipeline.schema import RobotLayout


MODEL_ID = "lingbot-va/singlerighthand/sandwich-100/step-40000"


def fdm_config(**overrides):
    values = {
        "protocol_version": 3,
        "model_id": MODEL_ID,
        "action_mode": "eef",
        "action_rate_hz": 30.0,
        "wire_chunk_size": 48,
        "native_first_horizon": 48,
        "native_horizon": 64,
        "feedback_stride": 4,
        "state_history_enabled": False,
        "camera_names": ("head", "right_wrist"),
        "supports_selective_grounding": True,
        "feedback_http_path": "/v1/robot-policy",
        "feedback_queue_size": 8,
        "feedback_retry_limit": 2,
        "feedback_retry_backoff_s": 0.001,
        "keyframe_wait_timeout_s": 0.1,
        "pending_miss_policy": "hold_last",
        "pending_miss_timeout_s": 0.2,
        "action_retry_backoff_s": 0.001,
        "actual_action_semantics": "final_30hz_waypoint",
        "keyframe_policy": "first_complete_snapshot_after_stride",
        "single_active_session": True,
        "feedback_ack_global_start_field": "accepted_global_action_start",
        "feedback_ack_action_count_field": "accepted_action_count",
        "feedback_ack_required_fields": (
            "native_chunk_id",
            "received_batches",
            "required_batches",
            "grounding_triggered",
            "grounded_frontier",
        ),
    }
    values.update(overrides)
    config = FdmProtocolConfig(**values)
    config.validate()
    return config


def test_pending_miss_policy_must_be_explicit_hold_last():
    with pytest.raises(ValueError, match="pending_miss_policy"):
        fdm_config(pending_miss_policy="")

    with pytest.raises(ValueError, match="pending_miss_policy"):
        fdm_config(pending_miss_policy="reset_immediately")


def action_chunk_response(config, request, actions=None):
    wire_id = int(request["wire_chunk_id"])
    global_start = int(request["global_action_start"])
    if actions is None:
        actions = [{"index": global_start + index} for index in range(48)]
    return {
        "protocol_version": config.protocol_version,
        "protocol_mode": FDM_PROTOCOL_MODE,
        "message_type": "action_chunk",
        "session_id": request["session_id"],
        "request_id": request["request_id"],
        "model_id": config.model_id,
        "action_rate_hz": config.action_rate_hz,
        "wire_chunk_id": wire_id,
        "global_action_start": global_start,
        "action_chunk": actions,
        "native_spans": [
            span.as_mapping() for span in native_spans_for_wire(wire_id, config)
        ],
        "server_timing": {},
    }


def test_native_64_stream_is_rechunked_across_fixed_wire_48_boundaries():
    config = fdm_config()
    mapper = NativeActionRechunker(config)

    w0 = mapper.push_native(0, list(range(48)))
    w1 = mapper.push_native(1, list(range(1000, 1064)))
    w2 = mapper.push_native(2, list(range(2000, 2064)))

    assert len(w0) == 1
    assert w0[0].actions == tuple(range(48))
    assert len(w1) == 1
    assert w1[0].actions == tuple(range(1000, 1048))
    assert mapper.buffered_actions == 32
    assert len(w2) == 1
    assert w2[0].wire_chunk_id == 2
    assert w2[0].actions == tuple(range(1048, 1064)) + tuple(range(2000, 2032))
    assert [span.as_mapping() for span in w2[0].native_spans] == [
        {
            "native_chunk_id": 1,
            "native_action_start": 48,
            "wire_action_start": 0,
            "length": 16,
        },
        {
            "native_chunk_id": 2,
            "native_action_start": 0,
            "wire_action_start": 16,
            "length": 32,
        },
    ]


def test_hello_requires_exact_fdm_capabilities():
    config = fdm_config()
    request = build_hello(
        config,
        session_id="session-a",
        request_id=1,
        robot_layout={"action_dim": 54},
    )
    response = dict(request)
    response["message_type"] = "hello_ack"
    validate_hello_ack(
        response,
        config,
        session_id="session-a",
        request_id=1,
    )

    response["native_horizon"] = 48
    with pytest.raises(FdmProtocolError, match="native_horizon"):
        validate_hello_ack(
            response,
            config,
            session_id="session-a",
            request_id=1,
        )


def test_joint_hello_advertises_radian_joint_action_layout():
    config = fdm_config(action_mode="joint", state_history_enabled=True)
    request = build_hello(
        config,
        session_id="session-joint",
        request_id=1,
        robot_layout=RobotLayout().metadata(),
    )

    assert request["action_mode"] == "joint"
    assert request["robot_layout"]["action_layout"] == [
        {"side": "left", "arm": [0, 7], "hand": [7, 27]},
        {"side": "right", "arm": [27, 34], "hand": [34, 54]},
    ]
    assert request["robot_layout"]["units"] == {
        "qpos.arm": "radian",
        "qpos.hand": "radian",
        "action.arm": "radian",
        "action.hand": "radian",
    }
    assert request["robot_layout"]["eef_dof_per_side"] == 7
    assert request["robot_layout"]["eef_dim"] == 14

    response = dict(request)
    response["message_type"] = "hello_ack"
    validate_hello_ack(
        response,
        config,
        session_id="session-joint",
        request_id=1,
    )
    response.pop("action_mode")
    with pytest.raises(FdmProtocolError, match="action_mode"):
        validate_hello_ack(
            response,
            config,
            session_id="session-joint",
            request_id=1,
        )


def test_joint_action_mode_requires_state_history():
    with pytest.raises(ValueError, match="state_history"):
        fdm_config(action_mode="joint", state_history_enabled=False)


def test_eef_robot_layout_remains_unchanged():
    layout = RobotLayout().metadata()
    assert robot_layout_for_action_mode(layout, "eef") == layout


def test_bootstrap_preserves_initial_observation_and_requests_w0():
    config = fdm_config()
    observation = {
        "images": {"head": {}, "right_wrist": {}},
        "arm_state_left": {"joint_pos": [0] * 7},
        "arm_state_right": {"joint_pos": [0] * 7},
        "hand_state_left": {"joint_pos": [0] * 20},
        "hand_state_right": {"joint_pos": [0] * 20},
        "active_hand_sides": ["right"],
        "zero_filled_hand_sides": ["left"],
    }
    request = build_bootstrap(
        observation,
        config,
        session_id="session-a",
        request_id=2,
    )

    assert request["message_type"] == "bootstrap"
    assert request["wire_chunk_id"] == 0
    assert request["global_action_start"] == 0
    assert request["images"] is observation["images"]
    chunk = parse_action_chunk(
        action_chunk_response(config, request),
        config,
        session_id="session-a",
        request_id=2,
        wire_chunk_id=0,
        global_action_start=0,
    )
    assert len(chunk.actions) == 48
    assert chunk.native_spans == native_spans_for_wire(0, config)


def test_action_chunk_identity_order_and_duplicate_content_are_checked():
    config = fdm_config()
    ledger = FdmSessionLedger(config)
    ledger.reset("session-a", 7)
    request = build_action_request(
        config,
        session_id="session-a",
        request_id=2,
        wire_chunk_id=0,
        global_action_start=0,
    )
    chunk = parse_action_chunk(
        action_chunk_response(config, request),
        config,
        session_id="session-a",
        request_id=2,
        wire_chunk_id=0,
        global_action_start=0,
    )
    assert ledger.accept_action_chunk(chunk)
    assert not ledger.accept_action_chunk(chunk)

    changed = action_chunk_response(config, request)
    changed["action_chunk"][0] = {"index": 999}
    changed_chunk = parse_action_chunk(
        changed,
        config,
        session_id="session-a",
        request_id=2,
        wire_chunk_id=0,
        global_action_start=0,
    )
    with pytest.raises(FdmSessionError, match="changed content"):
        ledger.accept_action_chunk(changed_chunk)


def test_feedback_is_created_exactly_every_four_executed_actions():
    config = fdm_config()
    next_request = iter(range(10, 20))
    accumulator = FeedbackAccumulator(
        config, lambda: ("session-a", next(next_request))
    )
    accumulator.reset("session-a", 3)

    drafts = []
    for index in range(8):
        draft = accumulator.record(
            session_id="session-a",
            generation=3,
            global_action_index=index,
            action={"actual": index},
            executed_at=100.0 + index,
            keyframe_not_before=100.0 + index,
        )
        if draft is not None:
            drafts.append(draft)

    assert [draft.global_action_start for draft in drafts] == [0, 4]
    assert [draft.feedback_seq for draft in drafts] == [0, 1]
    assert [draft.request_id for draft in drafts] == [10, 11]
    assert [[action["actual"] for action in draft.executed_actions] for draft in drafts] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert all(not draft.qpos_history for draft in drafts)
    assert all(not draft.qpos_timestamps for draft in drafts)


def test_enabled_state_history_collects_all_four_measured_qpos_samples():
    config = fdm_config(state_history_enabled=True)
    accumulator = FeedbackAccumulator(config, lambda: ("session-a", 10))
    accumulator.reset("session-a", 3)

    draft = None
    for index in range(4):
        draft = accumulator.record(
            session_id="session-a",
            generation=3,
            global_action_index=index,
            action={"actual": index},
            executed_at=100.0 + index / 30.0,
            keyframe_not_before=100.0 + index / 30.0,
            qpos=np.full(54, index, dtype=np.float32),
            qpos_timestamp=99.999 + index / 30.0,
        )

    assert draft is not None
    assert np.asarray(draft.qpos_history).shape == (4, 54)
    assert np.allclose(np.asarray(draft.qpos_history)[:, 0], [0, 1, 2, 3])
    assert len(draft.qpos_timestamps) == 4


def complete_snapshot(draft):
    timestamp = draft.keyframe_not_before + 0.01
    return {
        "images": {"head": {"jpeg": b"h"}, "right_wrist": {"jpeg": b"r"}},
        "source_timestamps": {
            "camera_head": timestamp,
            "camera_right_wrist": timestamp,
        },
        "robot_layout": {"action_dim": 54},
        "active_hand_sides": ["right"],
        "zero_filled_hand_sides": ["left"],
        "arm_state_left": {},
        "hand_state_left": {},
        "arm_state_right": {},
        "hand_state_right": {},
    }


class RetryFeedbackTransport:
    def __init__(self):
        self.calls = 0
        self.reconnects = 0
        self.payloads = []

    def exchange(self, request):
        self.calls += 1
        self.payloads.append(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
        if self.calls == 1:
            raise TimeoutError("injected feedback timeout")
        return {
            "protocol_version": request["protocol_version"],
            "message_type": "feedback_ack",
            "session_id": request["session_id"],
            "request_id": request["request_id"],
            "feedback_id": request["feedback_id"],
            "feedback_seq": request["feedback_seq"],
            "accepted_global_action_start": request["global_action_start"],
            "accepted_action_count": request["action_count"],
            "native_chunk_id": 0,
            "received_batches": 1,
            "required_batches": 12,
            "grounding_triggered": False,
            "grounded_frontier": 0,
        }

    def reconnect(self):
        self.reconnects += 1


def feedback_draft(accumulator):
    draft = None
    for index in range(4):
        draft = accumulator.record(
            session_id="session-a",
            generation=3,
            global_action_index=index,
            action={"actual": index},
            executed_at=10.0 + index,
            keyframe_not_before=10.0 + index,
        )
    assert draft is not None
    return draft


def test_feedback_retry_reuses_exact_payload_and_submit_deduplicates():
    config = fdm_config()
    accumulator = FeedbackAccumulator(config, lambda: ("session-a", 20))
    accumulator.reset("session-a", 3)
    draft = feedback_draft(accumulator)
    fatal = []
    worker = FdmFeedbackWorker(
        config,
        transport_factory=lambda: None,
        snapshot_builder=complete_snapshot,
        on_fatal=fatal.append,
        autostart=False,
    )
    worker.reset_session("session-a", 3)
    assert worker.submit(draft)
    queued = worker._queue.get_nowait()
    transport = RetryFeedbackTransport()
    try:
        assert worker.process_one(transport, queued)
    finally:
        worker._queue.task_done()

    assert transport.calls == 2
    assert transport.reconnects == 1
    assert transport.payloads[0] == transport.payloads[1]
    message = pickle.loads(transport.payloads[0])
    assert "qpos_history" not in message
    assert "qpos_timestamps" not in message
    assert worker.submit(draft) is False
    assert fatal == []
    assert worker.status()["acknowledged"] == 1


def test_enabled_state_history_is_added_to_execution_feedback():
    config = fdm_config(state_history_enabled=True)
    accumulator = FeedbackAccumulator(config, lambda: ("session-a", 20))
    accumulator.reset("session-a", 3)
    draft = None
    for index in range(4):
        draft = accumulator.record(
            session_id="session-a",
            generation=3,
            global_action_index=index,
            action={"actual": index},
            executed_at=10.0 + index,
            keyframe_not_before=10.0 + index,
            qpos=np.full(54, index + 0.25, dtype=np.float32),
            qpos_timestamp=9.9 + index,
        )
    assert draft is not None
    worker = FdmFeedbackWorker(
        config,
        transport_factory=lambda: None,
        snapshot_builder=complete_snapshot,
        on_fatal=lambda _reason: None,
        autostart=False,
    )
    worker.reset_session("session-a", 3)
    transport = RetryFeedbackTransport()

    assert worker.process_one(transport, draft)
    message = pickle.loads(transport.payloads[0])
    assert message["qpos_history"].shape == (4, 54)
    assert message["qpos_history"].dtype == np.float32
    assert np.allclose(message["qpos_history"][:, 0], [0.25, 1.25, 2.25, 3.25])
    assert np.allclose(message["qpos_timestamps"], [9.9, 10.9, 11.9, 12.9])


def test_feedback_conflict_and_old_session_are_rejected_after_reset():
    config = fdm_config()
    accumulator = FeedbackAccumulator(config, lambda: ("session-a", 20))
    accumulator.reset("session-a", 3)
    draft = feedback_draft(accumulator)
    worker = FdmFeedbackWorker(
        config,
        transport_factory=lambda: None,
        snapshot_builder=complete_snapshot,
        on_fatal=lambda _reason: None,
        autostart=False,
    )
    worker.reset_session("session-a", 3)
    assert worker.submit(draft)
    conflicting = deepcopy(draft)
    object.__setattr__(conflicting, "executed_actions", ({"actual": 99},) * 4)
    with pytest.raises(FdmSessionError, match="conflicting"):
        worker.submit(conflicting)

    worker.reset_session("session-b", 4)
    with pytest.raises(FdmSessionError, match="old-session"):
        worker.submit(draft)


def test_feedback_queue_overflow_is_fatal_instead_of_dropping_a_batch():
    config = fdm_config(feedback_queue_size=1)
    requests = iter((20, 21))
    accumulator = FeedbackAccumulator(
        config, lambda: ("session-a", next(requests))
    )
    accumulator.reset("session-a", 3)
    drafts = []
    for index in range(8):
        draft = accumulator.record(
            session_id="session-a",
            generation=3,
            global_action_index=index,
            action={"actual": index},
            executed_at=10.0 + index,
            keyframe_not_before=10.0 + index,
        )
        if draft is not None:
            drafts.append(draft)
    fatal = []
    worker = FdmFeedbackWorker(
        config,
        transport_factory=lambda: None,
        snapshot_builder=complete_snapshot,
        on_fatal=fatal.append,
        autostart=False,
    )
    worker.reset_session("session-a", 3)

    assert worker.submit(drafts[0])
    with pytest.raises(FeedbackQueueFull):
        worker.submit(drafts[1])
    assert fatal == ["lossless FDM feedback queue is full"]


def test_session_reset_clears_action_and_feedback_frontiers():
    config = fdm_config()
    ledger = FdmSessionLedger(config)
    ledger.reset("session-a", 1)
    ledger.record_execution("session-a", 0)
    ledger.reset("session-b", 2)

    assert ledger.status()["executed_frontier"] == 0
    with pytest.raises(FdmSessionError, match="old-session"):
        ledger.record_execution("session-a", 0)
    assert ledger.record_execution("session-b", 0) == 1
