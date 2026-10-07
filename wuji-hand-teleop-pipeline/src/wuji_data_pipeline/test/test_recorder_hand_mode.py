import threading
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from wuji_data_pipeline.recorder_node import (
    TeleopRecorderNode,
    active_hand_sides,
    camera_names_for_active_arm,
    zero_hand_sample,
)
from wuji_data_pipeline.schema import RobotLayout, build_training_frame
from wuji_data_pipeline.sync import TimedRingBuffer, TimedSample
from stereocamera.shared_frames import FrameBatch, SharedFrame


def _recorder_stub(active_hand, active_arm="both"):
    node = object.__new__(TeleopRecorderNode)
    node.active_hand_sides = active_hand_sides(active_hand)
    node.active_arm_sides = active_hand_sides(active_arm)
    node.camera_names = []
    return node


def test_active_hand_sides_are_explicit_and_validated():
    assert active_hand_sides("both") == ("left", "right")
    assert active_hand_sides("left") == ("left",)
    assert active_hand_sides("right") == ("right",)
    with pytest.raises(ValueError, match="active_hand"):
        active_hand_sides("auto")


def test_camera_names_follow_the_active_arm_mode():
    configured = ["head", "left_wrist", "right_wrist"]

    assert camera_names_for_active_arm(configured, "right") == [
        "head",
        "right_wrist",
    ]
    assert camera_names_for_active_arm(configured, "left") == [
        "head",
        "left_wrist",
    ]
    assert camera_names_for_active_arm(configured, "both") == configured


def test_inactive_hand_topics_are_not_required():
    node = _recorder_stub("left")
    keys = TeleopRecorderNode._required_source_keys(node)

    assert "hand_state_left" in keys
    assert "hand_command_left" in keys
    assert "hand_state_right" not in keys
    assert "hand_command_right" not in keys
    assert "arm_state_left" in keys
    assert "arm_state_right" in keys
    assert not any(key.startswith("aux_camera_") for key in keys)


def test_inactive_arm_only_requires_measured_state_and_eef():
    node = _recorder_stub("right", "right")
    keys = TeleopRecorderNode._required_source_keys(node)

    assert "arm_state_left" in keys
    assert "arm_actual_eef_left" in keys
    assert "arm_command_left" not in keys
    assert "arm_target_eef_left" not in keys
    assert "arm_zsp_left" not in keys
    assert "arm_state_right" in keys
    assert "arm_command_right" in keys
    assert "arm_target_eef_right" in keys
    assert "arm_zsp_right" in keys
    assert "hand_state_left" not in keys
    assert "hand_state_right" in keys


def test_inactive_arm_frame_uses_measured_hold_pose_defaults():
    node = _recorder_stub("right", "right")
    node._joint_velocity = SimpleNamespace(
        measure=lambda *_args: np.zeros(7)
    )
    left_position = np.arange(7, dtype=np.float32)
    left_eef = np.asarray(
        [0.4, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    matched = {
        "arm_state_left": TimedSample(
            1.0,
            SimpleNamespace(
                position=left_position,
                velocity=[],
                effort=[],
            ),
        ),
        "arm_actual_eef_left": TimedSample(1.0, left_eef),
    }

    arm, source_times = TeleopRecorderNode._arm_frame_sample(
        node, "left", matched
    )
    frame = build_training_frame(
        RobotLayout(),
        {
            "left": arm,
            "right": {
                "joint_pos_deg": np.zeros(7),
                "actual_eef": [0.4, -0.2, 0.3, 0, 0, 0, 1],
            },
        },
        {
            "left": zero_hand_sample(),
            "right": zero_hand_sample(),
        },
    )

    assert set(source_times) == {
        "arm_state_left",
        "arm_actual_eef_left",
    }
    assert np.allclose(frame["commanded_eef"][:7], left_eef)
    assert np.allclose(
        frame["arm_joint_command"][:7], np.radians(left_position)
    )
    assert np.allclose(frame["zsp"][:3], 0.0)


def test_right_arm_is_used_as_no_camera_sync_anchor():
    node = _recorder_stub("right", "right")

    assert TeleopRecorderNode._alignment_anchor_key(node) == "arm_state_right"


def test_absent_hand_is_zero_in_qpos_and_action_without_changing_54d_schema():
    layout = RobotLayout()
    arms = {
        side: {
            "joint_pos_deg": np.zeros(7),
            "actual_eef": [0.4, 0.1 if side == "left" else -0.1, 0.3, 0, 0, 0, 1],
        }
        for side in layout.sides
    }
    hands = {
        "left": {
            "actual_q_rad": np.full(20, 0.1),
            "target_q_rad": np.full(20, 0.2),
        },
        "right": zero_hand_sample(layout.hand_dof),
    }

    frame = build_training_frame(layout, arms, hands)

    assert frame["qpos"].shape == (54,)
    assert frame["action"].shape == (54,)
    assert np.allclose(frame["qpos"][34:54], 0.0)
    assert np.allclose(frame["action"][34:54], 0.0)


def test_start_service_reports_writer_errors_without_crashing_the_node():
    messages = []
    fake_node = SimpleNamespace(
        start_recording=lambda: (_ for _ in ()).throw(PermissionError("read-only")),
        get_logger=lambda: SimpleNamespace(error=messages.append),
    )
    response = SimpleNamespace(success=None, message="")

    result = TeleopRecorderNode._start_callback(fake_node, object(), response)

    assert result is response
    assert response.success is False
    assert response.message == "failed to start episode: read-only"
    assert messages == [response.message]


def test_process_shutdown_discards_unsaved_episode():
    messages = []
    fake_node = SimpleNamespace(
        discard_unsaved=lambda: (True, "/tmp/episode_0000.inprogress"),
        get_logger=lambda: SimpleNamespace(warning=messages.append),
    )

    TeleopRecorderNode.shutdown(fake_node)

    assert messages == [
        "Recorder shutdown discarded unsaved data: "
        "/tmp/episode_0000.inprogress"
    ]


class _FakeWriter:
    def __init__(self, path="/tmp/episode_0000.inprogress", steps=12):
        self.episode_dir = Path(path)
        self.step_count = steps
        self.discarded = False
        self.finalized_with = None
        self.closed_incomplete = False

    def discard(self):
        self.discarded = True
        return self.episode_dir

    def finalize(self, metadata):
        self.finalized_with = dict(metadata)
        return Path(str(self.episode_dir).removesuffix(".inprogress"))

    def close_incomplete(self):
        self.closed_incomplete = True


def _state_machine_stub(writer=None, pending=None):
    return SimpleNamespace(
        _record_lock=threading.Lock(),
        _writer=writer,
        _pending_writer=pending,
        require_cameras=False,
        _finalizing=False,
        _record_started_at=1.0 if writer is not None else None,
        _sync_skip_count=3,
        _sync_error_sum=0.12,
        _sync_error_max=0.03,
        get_logger=lambda: SimpleNamespace(
            info=lambda _message: None,
            warning=lambda _message: None,
            error=lambda _message: None,
        ),
    )


def test_finish_then_save_is_a_two_phase_commit():
    writer = _FakeWriter()
    node = _state_machine_stub(writer=writer)

    ok, message = TeleopRecorderNode.finish_recording(node)

    assert ok is True
    assert "awaiting save" in message
    assert node._writer is None
    assert node._pending_writer is writer
    assert node._capture_has_finished is True
    assert writer.finalized_with is None

    ok, message = TeleopRecorderNode.save_pending(node)

    assert ok is True
    assert "episode_0000" in message
    assert node._pending_writer is None
    assert writer.finalized_with["sync_skip_count"] == 3
    assert writer.finalized_with["frame_sync_vqe"] == pytest.approx(12 / 15)


def test_discard_unsaved_removes_active_and_pending_without_finalizing():
    active = _FakeWriter("/tmp/active.inprogress")
    pending = _FakeWriter("/tmp/pending.inprogress")
    node = _state_machine_stub(writer=active, pending=pending)

    ok, message = TeleopRecorderNode.discard_unsaved(node)

    assert ok is True
    assert "active.inprogress" in message
    assert "pending.inprogress" in message
    assert active.discarded is True
    assert pending.discarded is True
    assert active.finalized_with is None
    assert pending.finalized_with is None


def test_new_capture_keeps_pending_episode_if_sources_are_not_ready():
    pending = _FakeWriter("/tmp/pending.inprogress")
    node = _state_machine_stub(pending=pending)
    node.readiness_errors = lambda: ["camera_head:missing"]

    ok, message = TeleopRecorderNode.start_recording(node)

    assert ok is False
    assert "sources not ready" in message
    assert node._pending_writer is pending
    assert pending.discarded is False


def test_camera_capture_selects_only_the_current_online_subset():
    pending = _FakeWriter("/tmp/pending.inprogress")
    node = _state_machine_stub(pending=pending)
    node.require_cameras = True
    node.camera_names = ["head", "left_wrist", "right_wrist"]
    node.online_camera_names = lambda: ["left_wrist"]
    selected_during_readiness = []

    def readiness_errors():
        selected_during_readiness.extend(node.camera_names)
        return ["arm_state_left:missing"]

    node.readiness_errors = readiness_errors

    ok, message = TeleopRecorderNode.start_recording(node)

    assert ok is False
    assert "sources not ready" in message
    assert selected_during_readiness == ["left_wrist"]
    assert node.camera_names == ["left_wrist"]
    assert node._pending_writer is pending
    assert pending.discarded is False


def test_camera_capture_requires_at_least_one_online_camera():
    pending = _FakeWriter("/tmp/pending.inprogress")
    node = _state_machine_stub(pending=pending)
    node.require_cameras = True
    node.camera_names = ["head", "left_wrist", "right_wrist"]
    node.online_camera_names = lambda: []

    ok, message = TeleopRecorderNode.start_recording(node)

    assert ok is False
    assert "no configured camera" in message
    assert node.camera_names == []
    assert node._pending_writer is pending
    assert pending.discarded is False


def test_direct_camera_poll_drains_bounded_ring_into_recorder_buffer():
    image = np.full((3, 4, 3), 17, dtype=np.uint8)
    frame = SharedFrame(
        camera_name="head",
        sequence=8,
        monotonic_ns=1_000,
        system_ns=2_000_000_000,
        image=image,
    )
    reader = SimpleNamespace(
        read_since=lambda _sequence: FrameBatch(
            frames=(frame,),
            overwritten=2,
            producer_generation=99,
            capture_failures=3,
        )
    )
    buffers = {}

    def buffer_for(key):
        return buffers.setdefault(
            key, TimedRingBuffer(maxlen=8, max_age_s=3.0)
        )

    node = SimpleNamespace(
        _camera_readers={"head": reader},
        _camera_sequences={"head": 0},
        _camera_generations={"head": 0},
        _camera_overwritten={"head": 0},
        _camera_capture_failures={"head": 0},
        _buffer=buffer_for,
        _warn_diagnostic=lambda _message: None,
    )

    TeleopRecorderNode._poll_direct_cameras(node)

    sample = buffers["camera_head"].latest()
    assert sample.timestamp == 2.0
    assert np.array_equal(sample.value, image)
    assert node._camera_sequences["head"] == 8
    assert node._camera_generations["head"] == 99
    assert node._camera_overwritten["head"] == 2
    assert node._camera_capture_failures["head"] == 3


def test_direct_auxiliary_poll_is_namespaced_and_never_required():
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
    frame = SharedFrame(
        camera_name="head_depth",
        sequence=5,
        monotonic_ns=1_000,
        system_ns=3_000_000_000,
        image=depth,
    )
    reader = SimpleNamespace(
        read_since=lambda _sequence: FrameBatch(
            frames=(frame,),
            overwritten=1,
            producer_generation=88,
            capture_failures=2,
        )
    )
    buffers = {}

    def buffer_for(key):
        return buffers.setdefault(
            key, TimedRingBuffer(maxlen=8, max_age_s=3.0)
        )

    node = SimpleNamespace(
        _camera_readers={},
        _camera_sequences={},
        _camera_generations={},
        _camera_overwritten={},
        _camera_capture_failures={},
        _auxiliary_camera_readers={"head_depth": reader},
        _auxiliary_camera_sequences={"head_depth": 0},
        _auxiliary_camera_generations={"head_depth": 0},
        _auxiliary_camera_overwritten={"head_depth": 0},
        _auxiliary_camera_capture_failures={"head_depth": 0},
        _buffer=buffer_for,
        _warn_diagnostic=lambda _message: None,
    )

    TeleopRecorderNode._poll_direct_cameras(node)

    sample = buffers["aux_camera_head_depth"].latest()
    assert sample.timestamp == 3.0
    assert np.array_equal(sample.value["image"], depth)
    assert sample.value["sequence"] == 5
    assert node._auxiliary_camera_overwritten["head_depth"] == 1
    assert node._auxiliary_camera_capture_failures["head_depth"] == 2


def test_auxiliary_alignment_is_best_effort_and_preserves_dtype():
    node = SimpleNamespace(
        auxiliary_camera_enabled=True,
        auxiliary_stream_names=("head_depth", "head_ir_left"),
        auxiliary_sync_tolerance_s=0.06,
        buffers={
            "aux_camera_head_depth": TimedRingBuffer(
                maxlen=8, max_age_s=3.0
            ),
        },
    )
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
    node.buffers["aux_camera_head_depth"].append(
        10.02,
        {"image": depth, "sequence": 9},
    )

    images, timestamps, sequences = (
        TeleopRecorderNode._aligned_auxiliary_camera(node, 10.0)
    )

    assert np.array_equal(images["head_depth"], depth)
    assert images["head_depth"].dtype == np.uint16
    assert timestamps == {"head_depth": 10.02}
    assert sequences == {"head_depth": 9}
    assert "head_ir_left" not in images
