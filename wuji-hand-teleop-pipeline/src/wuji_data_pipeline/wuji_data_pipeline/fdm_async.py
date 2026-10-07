"""LingBot-VA asynchronous FDM protocol and robot-side state machines.

This module deliberately does not import ROS.  The deployment node owns robot
I/O, while these classes own the FDM wire identities, native/wire timeline,
feedback batching, retries, and session isolation.  Pi protocol-v2 continues
to use :mod:`deployment_protocol` unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import pickle
import queue
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


PI_PROTOCOL_MODE = "pi_v2"
FDM_PROTOCOL_MODE = "fdm_async"
FDM_DEFAULT_PROTOCOL_VERSION = 3
FDM_ACTION_MODES = ("eef", "joint")

# These choices are local integration baselines, not claims that the cloud
# implementation has already accepted them.  They are also repeated in the
# dedicated profile and handoff document so a protocol review has one place
# to change each value.
FDM_OPEN_PROTOCOL_ITEMS = {
    "protocol_version": "configured; local baseline is 3",
    "feedback_transport": "second persistent connection on configured path",
    "actual_action": "final local 30 Hz waypoint after local processing",
    "boundary_blend": "disabled in the initial profile",
    "keyframe": "first complete named-camera snapshot after action four",
    "feedback_queue": "bounded and lossless; overflow fails the session",
    "pending_miss": "hold without advancing, then reset at configured timeout",
    "single_active_session": "required by the initial profile",
}


class FdmProtocolError(ValueError):
    """A peer message violates the negotiated FDM contract."""


class FdmSessionError(RuntimeError):
    """An FDM message or action belongs to the wrong session/frontier."""


class FeedbackQueueFull(FdmSessionError):
    """The lossless feedback queue cannot accept another batch."""


@dataclass(frozen=True)
class FdmProtocolConfig:
    """Explicit negotiated values and local failure-policy choices."""

    protocol_version: int
    model_id: str
    action_mode: str
    action_rate_hz: float
    wire_chunk_size: int
    native_first_horizon: int
    native_horizon: int
    feedback_stride: int
    state_history_enabled: bool
    camera_names: tuple[str, ...]
    supports_selective_grounding: bool
    feedback_http_path: str
    feedback_queue_size: int
    feedback_retry_limit: int
    feedback_retry_backoff_s: float
    keyframe_wait_timeout_s: float
    pending_miss_policy: str
    pending_miss_timeout_s: float
    action_retry_backoff_s: float
    actual_action_semantics: str
    keyframe_policy: str
    single_active_session: bool
    feedback_ack_global_start_field: str
    feedback_ack_action_count_field: str
    feedback_ack_required_fields: tuple[str, ...]

    @classmethod
    def from_deployment_config(
        cls, deployment: Mapping[str, Any]
    ) -> "FdmProtocolConfig":
        raw = deployment.get("fdm_async")
        if not isinstance(raw, Mapping):
            raise ValueError(
                "deployment.fdm_async must be an explicit mapping when "
                "protocol_mode=fdm_async"
            )
        state_history = raw.get("state_history", {})
        if not isinstance(state_history, Mapping):
            raise ValueError(
                "deployment.fdm_async.state_history must be a mapping"
            )
        state_history_enabled = state_history.get("enabled", False)
        if not isinstance(state_history_enabled, bool):
            raise ValueError(
                "deployment.fdm_async.state_history.enabled must be boolean"
            )
        expected_model_id = str(
            deployment.get("policy_http_expected_model_id", "")
        ).strip()
        legacy_model_id = str(raw.get("model_id", "")).strip()
        if (
            expected_model_id
            and legacy_model_id
            and expected_model_id != legacy_model_id
        ):
            raise ValueError(
                "deployment.policy_http_expected_model_id must equal the "
                "legacy deployment.fdm_async.model_id"
            )
        config = cls(
            protocol_version=int(raw.get("protocol_version", 0)),
            # Pi and FDM profiles share one deployment-level model identity.
            # Keep the nested field as a read-only compatibility fallback for
            # older handoff profiles, but do not require duplicate settings.
            model_id=expected_model_id or legacy_model_id,
            action_mode=str(raw.get("action_mode", "eef")).strip().lower(),
            action_rate_hz=float(raw.get("action_rate_hz", 0.0)),
            wire_chunk_size=int(raw.get("wire_chunk_size", 0)),
            native_first_horizon=int(raw.get("native_first_horizon", 0)),
            native_horizon=int(raw.get("native_horizon", 0)),
            feedback_stride=int(raw.get("feedback_stride", 0)),
            state_history_enabled=state_history_enabled,
            camera_names=tuple(str(name) for name in raw.get("camera_names", ())),
            supports_selective_grounding=bool(
                raw.get("supports_selective_grounding", False)
            ),
            feedback_http_path=str(raw.get("feedback_http_path", "")).strip(),
            feedback_queue_size=int(raw.get("feedback_queue_size", 0)),
            feedback_retry_limit=int(raw.get("feedback_retry_limit", -1)),
            feedback_retry_backoff_s=float(
                raw.get("feedback_retry_backoff_s", 0.0)
            ),
            keyframe_wait_timeout_s=float(
                raw.get("keyframe_wait_timeout_s", 0.0)
            ),
            pending_miss_policy=str(
                raw.get("pending_miss_policy", "")
            ).strip(),
            pending_miss_timeout_s=float(
                raw.get("pending_miss_timeout_s", 0.0)
            ),
            action_retry_backoff_s=float(
                raw.get("action_retry_backoff_s", 0.0)
            ),
            actual_action_semantics=str(
                raw.get("actual_action_semantics", "")
            ).strip(),
            keyframe_policy=str(raw.get("keyframe_policy", "")).strip(),
            single_active_session=bool(raw.get("single_active_session", False)),
            feedback_ack_global_start_field=str(
                raw.get("feedback_ack_global_start_field", "")
            ).strip(),
            feedback_ack_action_count_field=str(
                raw.get("feedback_ack_action_count_field", "")
            ).strip(),
            feedback_ack_required_fields=tuple(
                str(name)
                for name in raw.get(
                    "feedback_ack_required_fields",
                    (
                        "native_chunk_id",
                        "received_batches",
                        "required_batches",
                        "grounding_triggered",
                        "grounded_frontier",
                    ),
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.protocol_version <= 0:
            raise ValueError("FDM protocol_version must be explicitly positive")
        if not self.model_id:
            raise ValueError("FDM model_id must not be empty")
        if self.action_mode not in FDM_ACTION_MODES:
            raise ValueError(
                f"FDM action_mode must be one of {FDM_ACTION_MODES}"
            )
        if self.action_mode == "joint" and not self.state_history_enabled:
            raise ValueError(
                "LingBot-VA joint action mode requires "
                "state_history.enabled=true"
            )
        if not np.isclose(self.action_rate_hz, 30.0):
            raise ValueError("LingBot-VA FDM action_rate_hz must be 30")
        if self.wire_chunk_size != 48:
            raise ValueError("LingBot-VA FDM wire_chunk_size must be 48")
        if self.native_first_horizon != 48 or self.native_horizon != 64:
            raise ValueError("LingBot-VA native horizons must be first=48, later=64")
        if self.feedback_stride != 4:
            raise ValueError("LingBot-VA feedback_stride must be 4")
        if self.camera_names != ("head", "right_wrist"):
            raise ValueError(
                "LingBot-VA camera_names must be [head, right_wrist] in order"
            )
        if not self.supports_selective_grounding:
            raise ValueError("FDM requires selective grounding capability")
        if not self.feedback_http_path.startswith("/"):
            raise ValueError("FDM feedback_http_path must be an absolute path")
        if self.feedback_queue_size <= 0:
            raise ValueError("FDM feedback_queue_size must be positive")
        if self.feedback_retry_limit < 0:
            raise ValueError("FDM feedback_retry_limit must be non-negative")
        if min(
            self.feedback_retry_backoff_s,
            self.keyframe_wait_timeout_s,
            self.pending_miss_timeout_s,
            self.action_retry_backoff_s,
        ) <= 0.0:
            raise ValueError("FDM retry/keyframe/pending timeouts must be positive")
        if self.pending_miss_policy != "hold_last":
            raise ValueError(
                "initial FDM pending_miss_policy must be explicitly hold_last"
            )
        if self.actual_action_semantics != "final_30hz_waypoint":
            raise ValueError(
                "FDM actual_action_semantics must be final_30hz_waypoint"
            )
        if self.keyframe_policy != "first_complete_snapshot_after_stride":
            raise ValueError(
                "FDM keyframe_policy must be "
                "first_complete_snapshot_after_stride"
            )
        if not self.single_active_session:
            raise ValueError("initial FDM profile requires single_active_session=true")
        if not self.feedback_ack_global_start_field:
            raise ValueError("feedback ACK global-start field must be configured")
        if not self.feedback_ack_action_count_field:
            raise ValueError("feedback ACK action-count field must be configured")
        if not self.feedback_ack_required_fields:
            raise ValueError("feedback_ack_required_fields must not be empty")


@dataclass(frozen=True)
class NativeSpan:
    native_chunk_id: int
    native_action_start: int
    wire_action_start: int
    length: int

    def as_mapping(self) -> dict[str, int]:
        return {
            "native_chunk_id": self.native_chunk_id,
            "native_action_start": self.native_action_start,
            "wire_action_start": self.wire_action_start,
            "length": self.length,
        }


@dataclass(frozen=True)
class FdmActionChunk:
    session_id: str
    request_id: int
    wire_chunk_id: int
    global_action_start: int
    actions: tuple[Mapping[str, Any], ...]
    native_spans: tuple[NativeSpan, ...]
    server_timing: Mapping[str, Any]


@dataclass(frozen=True)
class RechunkedWireChunk:
    wire_chunk_id: int
    global_action_start: int
    actions: tuple[Any, ...]
    native_spans: tuple[NativeSpan, ...]


def native_horizon(native_chunk_id: int, config: FdmProtocolConfig) -> int:
    if native_chunk_id < 0:
        raise ValueError("native_chunk_id must be non-negative")
    return (
        config.native_first_horizon
        if native_chunk_id == 0
        else config.native_horizon
    )


def native_global_start(native_chunk_id: int, config: FdmProtocolConfig) -> int:
    if native_chunk_id < 0:
        raise ValueError("native_chunk_id must be non-negative")
    if native_chunk_id == 0:
        return 0
    return config.native_first_horizon + (native_chunk_id - 1) * config.native_horizon


def native_spans_for_range(
    global_action_start: int,
    action_count: int,
    config: FdmProtocolConfig,
) -> tuple[NativeSpan, ...]:
    """Return exact native coverage for one continuous global range."""

    if global_action_start < 0 or action_count <= 0:
        raise ValueError("global range must be non-negative and non-empty")
    cursor = int(global_action_start)
    end = cursor + int(action_count)
    wire_offset = 0
    if cursor < config.native_first_horizon:
        native_id = 0
    else:
        native_id = 1 + (
            (cursor - config.native_first_horizon) // config.native_horizon
        )
    spans = []
    while cursor < end:
        native_start = native_global_start(native_id, config)
        horizon = native_horizon(native_id, config)
        offset = cursor - native_start
        if not 0 <= offset < horizon:
            raise AssertionError("native/global frontier calculation is inconsistent")
        length = min(end - cursor, horizon - offset)
        spans.append(
            NativeSpan(
                native_chunk_id=native_id,
                native_action_start=offset,
                wire_action_start=wire_offset,
                length=length,
            )
        )
        cursor += length
        wire_offset += length
        native_id += 1
    return tuple(spans)


def native_spans_for_wire(
    wire_chunk_id: int, config: FdmProtocolConfig
) -> tuple[NativeSpan, ...]:
    if wire_chunk_id < 0:
        raise ValueError("wire_chunk_id must be non-negative")
    return native_spans_for_range(
        wire_chunk_id * config.wire_chunk_size,
        config.wire_chunk_size,
        config,
    )


class NativeActionRechunker:
    """Reference 48/64 native stream to fixed 48-step wire mapping.

    The local client normally receives already assembled wire chunks.  Keeping
    this small reference mapper next to response validation prevents either
    endpoint's tests from accidentally treating every wire chunk as a native
    generation.
    """

    def __init__(self, config: FdmProtocolConfig) -> None:
        self._config = config
        self._next_native_id = 0
        self._next_wire_id = 0
        self._buffer: list[tuple[Any, int, int]] = []

    def push_native(
        self, native_chunk_id: int, actions: Sequence[Any]
    ) -> tuple[RechunkedWireChunk, ...]:
        if int(native_chunk_id) != self._next_native_id:
            raise FdmSessionError(
                f"expected native_chunk_id={self._next_native_id}, "
                f"received {native_chunk_id}"
            )
        expected = native_horizon(native_chunk_id, self._config)
        if len(actions) != expected:
            raise FdmProtocolError(
                f"native chunk {native_chunk_id} must contain {expected} "
                f"executable actions, received {len(actions)}"
            )
        self._buffer.extend(
            (action, int(native_chunk_id), index)
            for index, action in enumerate(actions)
        )
        self._next_native_id += 1
        ready = []
        size = self._config.wire_chunk_size
        while len(self._buffer) >= size:
            records = self._buffer[:size]
            del self._buffer[:size]
            actions_out = tuple(record[0] for record in records)
            spans = native_spans_for_wire(self._next_wire_id, self._config)
            # Cross-check the formula against the actual native provenance.
            expanded = []
            for span in spans:
                expanded.extend(
                    (span.native_chunk_id, span.native_action_start + offset)
                    for offset in range(span.length)
                )
            actual = [(record[1], record[2]) for record in records]
            if actual != expanded:
                raise AssertionError("native stream provenance does not match wire spans")
            ready.append(
                RechunkedWireChunk(
                    wire_chunk_id=self._next_wire_id,
                    global_action_start=self._next_wire_id * size,
                    actions=actions_out,
                    native_spans=spans,
                )
            )
            self._next_wire_id += 1
        return tuple(ready)

    @property
    def buffered_actions(self) -> int:
        return len(self._buffer)


def _identity(
    response: Mapping[str, Any],
    *,
    config: FdmProtocolConfig,
    session_id: str,
    request_id: int,
) -> None:
    try:
        protocol_version = int(response.get("protocol_version"))
    except (TypeError, ValueError) as exc:
        raise FdmProtocolError("response has invalid protocol_version") from exc
    if protocol_version != config.protocol_version:
        raise FdmProtocolError(
            f"expected protocol_version={config.protocol_version}, "
            f"received {protocol_version}"
        )
    if response.get("session_id") != session_id:
        raise FdmSessionError("response session_id does not match request")
    if response.get("request_id") != request_id:
        raise FdmProtocolError("response request_id does not match request")
    if response.get("message_type") == "error" or response.get("error"):
        code = str(response.get("error_code", "FDM_SERVER_ERROR"))
        message = str(response.get("message", response.get("error", "")))[:1000]
        fatal = bool(response.get("fatal_session", False))
        raise FdmSessionError(f"{code}: {message}; fatal_session={fatal}")


def build_hello(
    config: FdmProtocolConfig,
    *,
    session_id: str,
    request_id: int,
    robot_layout: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": config.protocol_version,
        "protocol_mode": FDM_PROTOCOL_MODE,
        "message_type": "hello",
        "session_id": session_id,
        "request_id": request_id,
        "model_id": config.model_id,
        "action_mode": config.action_mode,
        "action_rate_hz": config.action_rate_hz,
        "wire_chunk_size": config.wire_chunk_size,
        "native_first_horizon": config.native_first_horizon,
        "native_horizon": config.native_horizon,
        "feedback_stride": config.feedback_stride,
        "camera_names": list(config.camera_names),
        "supports_selective_grounding": config.supports_selective_grounding,
        "robot_layout": robot_layout_for_action_mode(
            robot_layout, config.action_mode
        ),
    }


def validate_hello_ack(
    response: Mapping[str, Any],
    config: FdmProtocolConfig,
    *,
    session_id: str,
    request_id: int,
) -> None:
    _identity(
        response,
        config=config,
        session_id=session_id,
        request_id=request_id,
    )
    if response.get("message_type") != "hello_ack":
        raise FdmProtocolError("FDM hello was not acknowledged")
    expected = {
        "protocol_mode": FDM_PROTOCOL_MODE,
        "model_id": config.model_id,
        "wire_chunk_size": config.wire_chunk_size,
        "native_first_horizon": config.native_first_horizon,
        "native_horizon": config.native_horizon,
        "feedback_stride": config.feedback_stride,
        "supports_selective_grounding": True,
    }
    if config.action_mode == "joint" or "action_mode" in response:
        expected["action_mode"] = config.action_mode
    for name, value in expected.items():
        if response.get(name) != value:
            raise FdmProtocolError(
                f"hello capability {name} mismatch: expected {value!r}, "
                f"received {response.get(name)!r}"
            )
    try:
        action_rate_hz = float(response.get("action_rate_hz"))
    except (TypeError, ValueError) as exc:
        raise FdmProtocolError("hello_ack action_rate_hz is invalid") from exc
    if not np.isclose(action_rate_hz, config.action_rate_hz):
        raise FdmProtocolError("hello_ack action_rate_hz mismatch")
    if tuple(response.get("camera_names", ())) != config.camera_names:
        raise FdmProtocolError("hello_ack camera_names/order mismatch")


def robot_layout_for_action_mode(
    robot_layout: Mapping[str, Any], action_mode: str
) -> dict[str, Any]:
    """Return protocol metadata for EEF or joint action semantics.

    Dataset/observation qpos always stays in the existing 54D radian layout.
    Joint-action FDM reuses those offsets for its 54D action and deliberately
    removes every EEF/degree action unit from the advertised contract.
    """

    mode = str(action_mode).strip().lower()
    if mode not in FDM_ACTION_MODES:
        raise ValueError(f"unsupported FDM action_mode={action_mode!r}")
    layout = deepcopy(dict(robot_layout))
    if mode == "eef":
        return layout

    qpos_layout = layout.get("qpos_layout")
    if not isinstance(qpos_layout, Sequence) or len(qpos_layout) != 2:
        raise ValueError("joint FDM robot_layout requires two-side qpos_layout")
    action_layout = []
    for raw_side in qpos_layout:
        if not isinstance(raw_side, Mapping):
            raise ValueError("joint FDM qpos_layout entries must be mappings")
        side = str(raw_side.get("side", ""))
        arm = list(raw_side.get("arm", ()))
        hand = list(raw_side.get("hand", ()))
        if side not in ("left", "right") or len(arm) != 2 or len(hand) != 2:
            raise ValueError("joint FDM qpos_layout entry is invalid")
        action_layout.append({"side": side, "arm": arm, "hand": hand})

    layout["action_layout"] = action_layout
    layout["action_dim"] = int(layout.get("state_dim", 54))
    # These two fields describe the fixed robot-state profile and remain in
    # the FDM hello even when policy actions use joint targets.  Joint mode
    # changes only action_layout and the action units below.
    layout["units"] = {
        "qpos.arm": "radian",
        "qpos.hand": "radian",
        "action.arm": "radian",
        "action.hand": "radian",
    }
    return layout


def build_bootstrap(
    observation: Mapping[str, Any],
    config: FdmProtocolConfig,
    *,
    session_id: str,
    request_id: int,
) -> dict[str, Any]:
    request = dict(observation)
    request.update(
        {
            "protocol_version": config.protocol_version,
            "protocol_mode": FDM_PROTOCOL_MODE,
            "message_type": "bootstrap",
            "session_id": session_id,
            "request_id": request_id,
            "model_id": config.model_id,
            "wire_chunk_id": 0,
            "global_action_start": 0,
        }
    )
    return request


def build_action_request(
    config: FdmProtocolConfig,
    *,
    session_id: str,
    request_id: int,
    wire_chunk_id: int,
    global_action_start: int,
) -> dict[str, Any]:
    if global_action_start != wire_chunk_id * config.wire_chunk_size:
        raise FdmSessionError("wire/global request frontier is inconsistent")
    return {
        "protocol_version": config.protocol_version,
        "protocol_mode": FDM_PROTOCOL_MODE,
        "message_type": "action_request",
        "session_id": session_id,
        "request_id": request_id,
        "model_id": config.model_id,
        "wire_chunk_id": int(wire_chunk_id),
        "global_action_start": int(global_action_start),
    }


def parse_action_chunk(
    response: Mapping[str, Any],
    config: FdmProtocolConfig,
    *,
    session_id: str,
    request_id: int,
    wire_chunk_id: int,
    global_action_start: int,
) -> FdmActionChunk:
    _identity(
        response,
        config=config,
        session_id=session_id,
        request_id=request_id,
    )
    if response.get("message_type") != "action_chunk":
        raise FdmProtocolError("FDM action response is not action_chunk")
    if response.get("model_id") != config.model_id:
        raise FdmProtocolError("FDM action_chunk model_id mismatch")
    try:
        response_rate = float(response.get("action_rate_hz"))
    except (TypeError, ValueError) as exc:
        raise FdmProtocolError("action_chunk action_rate_hz is invalid") from exc
    if not np.isclose(response_rate, config.action_rate_hz):
        raise FdmProtocolError("action_chunk action_rate_hz mismatch")
    if response.get("wire_chunk_id") != wire_chunk_id:
        raise FdmSessionError("action_chunk wire_chunk_id is out of order")
    if response.get("global_action_start") != global_action_start:
        raise FdmSessionError("action_chunk global_action_start is out of order")
    raw_actions = response.get("action_chunk")
    if (
        not isinstance(raw_actions, Sequence)
        or isinstance(raw_actions, (str, bytes, bytearray))
        or len(raw_actions) != config.wire_chunk_size
    ):
        raise FdmProtocolError(
            f"action_chunk must contain exactly {config.wire_chunk_size} actions"
        )
    if any(not isinstance(action, Mapping) for action in raw_actions):
        raise FdmProtocolError("each wire action must be a mapping")
    raw_spans = response.get("native_spans")
    if not isinstance(raw_spans, Sequence) or not raw_spans:
        raise FdmProtocolError("action_chunk native_spans must be non-empty")
    try:
        spans = tuple(
            NativeSpan(
                native_chunk_id=int(span["native_chunk_id"]),
                native_action_start=int(span["native_action_start"]),
                wire_action_start=int(span["wire_action_start"]),
                length=int(span["length"]),
            )
            for span in raw_spans
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FdmProtocolError("action_chunk has invalid native_spans") from exc
    expected_spans = native_spans_for_range(
        global_action_start, config.wire_chunk_size, config
    )
    if spans != expected_spans:
        raise FdmProtocolError(
            "action_chunk native_spans do not match the continuous 48/64 timeline"
        )
    timing = response.get("server_timing", {})
    if not isinstance(timing, Mapping):
        raise FdmProtocolError("action_chunk server_timing must be a mapping")
    return FdmActionChunk(
        session_id=session_id,
        request_id=request_id,
        wire_chunk_id=wire_chunk_id,
        global_action_start=global_action_start,
        actions=tuple(raw_actions),
        native_spans=spans,
        server_timing=dict(timing),
    )


def _fingerprint(value: Any) -> str:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def _feedback_draft_fingerprint(draft: "FeedbackDraft") -> str:
    """Hash a small draft without pickle; this runs in the control callback."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, Mapping):
            digest.update(b"{")
            for key in sorted(value, key=lambda item: str(item)):
                update(str(key))
                update(value[key])
            digest.update(b"}")
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            digest.update(b"[")
            for item in value:
                update(item)
            digest.update(b"]")
        elif isinstance(value, bytes):
            digest.update(value)
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"|")

    update(
        (
            draft.session_id,
            draft.generation,
            draft.request_id,
            draft.feedback_seq,
            draft.feedback_id,
            draft.global_action_start,
            draft.executed_actions,
            draft.executed_action_timestamps,
            draft.qpos_history,
            draft.qpos_timestamps,
            draft.keyframe_not_before,
        )
    )
    return digest.hexdigest()


class FdmSessionLedger:
    """Thread-safe delivered/executed frontiers with duplicate checks."""

    def __init__(self, config: FdmProtocolConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self.reset("", 0)

    def reset(self, session_id: str, generation: int) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._session_id = str(session_id)
            self._generation = int(generation)
            self._next_wire_id = 0
            self._next_global_delivery = 0
            self._next_global_execution = 0
            self._delivered: dict[int, str] = {}

    def next_delivery(self) -> tuple[str, int, int, int]:
        with self._lock:
            return (
                self._session_id,
                self._generation,
                self._next_wire_id,
                self._next_global_delivery,
            )

    def accept_action_chunk(self, chunk: FdmActionChunk) -> bool:
        digest = _fingerprint(
            (
                chunk.session_id,
                chunk.wire_chunk_id,
                chunk.global_action_start,
                chunk.actions,
                chunk.native_spans,
            )
        )
        with self._lock:
            if chunk.session_id != self._session_id:
                raise FdmSessionError("old-session action_chunk rejected")
            previous = self._delivered.get(chunk.wire_chunk_id)
            if previous is not None:
                if previous != digest:
                    raise FdmSessionError("duplicate wire_chunk_id changed content")
                return False
            if chunk.wire_chunk_id != self._next_wire_id:
                raise FdmSessionError("wire_chunk_id has a gap or reordering")
            if chunk.global_action_start != self._next_global_delivery:
                raise FdmSessionError("delivered global action range has a gap")
            self._delivered[chunk.wire_chunk_id] = digest
            self._next_wire_id += 1
            self._next_global_delivery += self._config.wire_chunk_size
            return True

    def record_execution(self, session_id: str, global_action_index: int) -> int:
        with self._lock:
            if session_id != self._session_id:
                raise FdmSessionError("old-session action execution rejected")
            if global_action_index != self._next_global_execution:
                raise FdmSessionError(
                    f"expected executed global index {self._next_global_execution}, "
                    f"received {global_action_index}"
                )
            self._next_global_execution += 1
            return self._next_global_execution

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self._session_id,
                "generation": self._generation,
                "next_wire_chunk_id": self._next_wire_id,
                "delivered_frontier": self._next_global_delivery,
                "executed_frontier": self._next_global_execution,
            }


@dataclass(frozen=True)
class FeedbackDraft:
    session_id: str
    generation: int
    request_id: int
    feedback_seq: int
    feedback_id: str
    global_action_start: int
    executed_actions: tuple[Mapping[str, Any], ...]
    executed_action_timestamps: tuple[float, ...]
    qpos_history: tuple[np.ndarray, ...]
    qpos_timestamps: tuple[float, ...]
    keyframe_not_before: float


class FeedbackAccumulator:
    """Create one immutable feedback draft after every executed stride."""

    def __init__(
        self,
        config: FdmProtocolConfig,
        request_identity: Callable[[], tuple[str, int]],
    ) -> None:
        self._config = config
        self._request_identity = request_identity
        self._lock = threading.Lock()
        self.reset("", 0)

    def reset(self, session_id: str, generation: int) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._session_id = str(session_id)
            self._generation = int(generation)
            self._next_global_index = 0
            self._next_feedback_seq = 0
            self._actions: list[Mapping[str, Any]] = []
            self._timestamps: list[float] = []
            self._qpos_history: list[np.ndarray] = []
            self._qpos_timestamps: list[float] = []

    def record(
        self,
        *,
        session_id: str,
        generation: int,
        global_action_index: int,
        action: Mapping[str, Any],
        executed_at: float,
        keyframe_not_before: float,
        qpos: Optional[np.ndarray] = None,
        qpos_timestamp: Optional[float] = None,
    ) -> Optional[FeedbackDraft]:
        with self._lock:
            if session_id != self._session_id or generation != self._generation:
                raise FdmSessionError("execution belongs to an obsolete FDM session")
            if global_action_index != self._next_global_index:
                raise FdmSessionError(
                    f"feedback expected global index {self._next_global_index}, "
                    f"received {global_action_index}"
                )
            if self._config.state_history_enabled:
                if qpos is None or qpos_timestamp is None:
                    raise FdmSessionError(
                        "state-history feedback requires qpos for every action"
                    )
                qpos_array = np.asarray(qpos, dtype=np.float32).reshape(-1)
                if qpos_array.shape != (54,) or not np.all(
                    np.isfinite(qpos_array)
                ):
                    raise FdmProtocolError(
                        "state-history qpos must be a finite 54D vector"
                    )
                qpos_stamp = float(qpos_timestamp)
                if not np.isfinite(qpos_stamp):
                    raise FdmProtocolError(
                        "state-history qpos timestamp must be finite"
                    )
                self._qpos_history.append(qpos_array.copy())
                self._qpos_timestamps.append(qpos_stamp)
            self._actions.append(deepcopy(dict(action)))
            self._timestamps.append(float(executed_at))
            self._next_global_index += 1
            if len(self._actions) < self._config.feedback_stride:
                return None
            global_start = self._next_global_index - self._config.feedback_stride
            identity_session, request_id = self._request_identity()
            if identity_session != self._session_id:
                raise FdmSessionError("session changed while allocating feedback request")
            feedback_seq = self._next_feedback_seq
            self._next_feedback_seq += 1
            draft = FeedbackDraft(
                session_id=self._session_id,
                generation=self._generation,
                request_id=request_id,
                feedback_seq=feedback_seq,
                feedback_id=f"{self._session_id}:{global_start}",
                global_action_start=global_start,
                executed_actions=tuple(self._actions),
                executed_action_timestamps=tuple(self._timestamps),
                qpos_history=tuple(self._qpos_history),
                qpos_timestamps=tuple(self._qpos_timestamps),
                keyframe_not_before=float(keyframe_not_before),
            )
            self._actions = []
            self._timestamps = []
            self._qpos_history = []
            self._qpos_timestamps = []
            return draft

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "next_global_feedback_index": self._next_global_index,
                "next_feedback_seq": self._next_feedback_seq,
                "partial_action_count": len(self._actions),
                "partial_qpos_count": len(self._qpos_history),
            }


class FdmFeedbackWorker:
    """Own JPEG/snapshot construction and one independent feedback transport."""

    def __init__(
        self,
        config: FdmProtocolConfig,
        *,
        transport_factory: Callable[[], Any],
        snapshot_builder: Callable[[FeedbackDraft], Optional[Mapping[str, Any]]],
        on_fatal: Callable[[str], None],
        on_event: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        autostart: bool = True,
    ) -> None:
        self._config = config
        self._transport_factory = transport_factory
        self._snapshot_builder = snapshot_builder
        self._on_fatal = on_fatal
        self._on_event = on_event
        self._queue: queue.Queue[FeedbackDraft] = queue.Queue(
            maxsize=config.feedback_queue_size
        )
        self._stop = threading.Event()
        self._session_lock = threading.Lock()
        self._session_id = ""
        self._generation = 0
        self._known: dict[str, str] = {}
        self._acked: dict[str, Mapping[str, Any]] = {}
        self._submitted = 0
        self._acknowledged = 0
        self._retries = 0
        self._failures = 0
        self._last_rtt_ms = 0.0
        self._last_error = ""
        self._thread: Optional[threading.Thread] = None
        if autostart:
            self.start()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="wuji-fdm-feedback",
            daemon=True,
        )
        self._thread.start()

    def reset_session(self, session_id: str, generation: int) -> None:
        with self._session_lock:
            self._session_id = str(session_id)
            self._generation = int(generation)
            self._known.clear()
            self._acked.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()

    def submit(self, draft: FeedbackDraft) -> bool:
        digest = _feedback_draft_fingerprint(draft)
        with self._session_lock:
            if (
                draft.session_id != self._session_id
                or draft.generation != self._generation
            ):
                raise FdmSessionError("cannot queue old-session feedback")
            known = self._known.get(draft.feedback_id)
            if known is not None:
                if known != digest:
                    raise FdmSessionError(
                        "duplicate feedback_id has conflicting content"
                    )
                return False
            self._known[draft.feedback_id] = digest
        try:
            self._queue.put_nowait(draft)
        except queue.Full as exc:
            with self._session_lock:
                self._known.pop(draft.feedback_id, None)
            self._failures += 1
            error = "lossless FDM feedback queue is full"
            self._last_error = error
            self._on_fatal(error)
            raise FeedbackQueueFull(error) from exc
        self._submitted += 1
        return True

    def _current(self, draft: FeedbackDraft) -> bool:
        with self._session_lock:
            return (
                draft.session_id == self._session_id
                and draft.generation == self._generation
            )

    def _await_snapshot(self, draft: FeedbackDraft) -> Mapping[str, Any]:
        deadline = time.monotonic() + self._config.keyframe_wait_timeout_s
        while self._current(draft) and not self._stop.is_set():
            snapshot = self._snapshot_builder(draft)
            if snapshot is not None:
                return snapshot
            if time.monotonic() >= deadline:
                break
            self._stop.wait(0.005)
        raise TimeoutError(
            "timed out waiting for a complete post-action FDM keyframe"
        )

    def _build_message(
        self, draft: FeedbackDraft, snapshot: Mapping[str, Any]
    ) -> dict[str, Any]:
        source_timestamps = snapshot.get("source_timestamps", {})
        if not isinstance(source_timestamps, Mapping):
            raise FdmProtocolError("feedback snapshot lacks source_timestamps")
        camera_timestamps = [
            float(source_timestamps[f"camera_{name}"])
            for name in self._config.camera_names
        ]
        if any(stamp < draft.keyframe_not_before for stamp in camera_timestamps):
            raise FdmProtocolError("feedback keyframe precedes the fourth action")
        message = {
            "protocol_version": self._config.protocol_version,
            "protocol_mode": FDM_PROTOCOL_MODE,
            "message_type": "execution_feedback",
            "session_id": draft.session_id,
            "request_id": draft.request_id,
            "feedback_id": draft.feedback_id,
            "feedback_seq": draft.feedback_seq,
            "global_action_start": draft.global_action_start,
            "action_count": len(draft.executed_actions),
            "executed_actions": list(draft.executed_actions),
            "executed_action_timestamps": list(
                draft.executed_action_timestamps
            ),
            # min means every required named camera is at or after this point.
            "keyframe_timestamp": min(camera_timestamps),
            "images": snapshot["images"],
            "source_timestamps": dict(source_timestamps),
            "robot_layout": snapshot["robot_layout"],
            "active_hand_sides": snapshot["active_hand_sides"],
            "zero_filled_hand_sides": snapshot["zero_filled_hand_sides"],
        }
        for side in ("left", "right"):
            message[f"arm_state_{side}"] = snapshot[f"arm_state_{side}"]
            message[f"hand_state_{side}"] = snapshot[f"hand_state_{side}"]
        if self._config.state_history_enabled:
            if (
                len(draft.qpos_history) != self._config.feedback_stride
                or len(draft.qpos_timestamps) != self._config.feedback_stride
            ):
                raise FdmProtocolError(
                    "state-history feedback must match feedback_stride"
                )
            qpos_history = np.stack(draft.qpos_history).astype(
                np.float32, copy=False
            )
            qpos_timestamps = np.asarray(
                draft.qpos_timestamps, dtype=np.float64
            )
            if qpos_history.shape != (self._config.feedback_stride, 54):
                raise FdmProtocolError(
                    "state-history feedback must have shape (feedback_stride, 54)"
                )
            if (
                not np.all(np.isfinite(qpos_history))
                or not np.all(np.isfinite(qpos_timestamps))
            ):
                raise FdmProtocolError("state-history feedback must be finite")
            message["qpos_history"] = qpos_history
            message["qpos_timestamps"] = qpos_timestamps
        return message

    def _validate_ack(
        self, response: Mapping[str, Any], draft: FeedbackDraft
    ) -> None:
        _identity(
            response,
            config=self._config,
            session_id=draft.session_id,
            request_id=draft.request_id,
        )
        if response.get("message_type") != "feedback_ack":
            raise FdmProtocolError("execution_feedback was not acknowledged")
        if response.get("feedback_id") != draft.feedback_id:
            raise FdmProtocolError("feedback_ack feedback_id mismatch")
        if response.get("feedback_seq") != draft.feedback_seq:
            raise FdmProtocolError("feedback_ack feedback_seq mismatch")
        if (
            response.get(self._config.feedback_ack_global_start_field)
            != draft.global_action_start
        ):
            raise FdmProtocolError("feedback_ack accepted range start mismatch")
        if (
            response.get(self._config.feedback_ack_action_count_field)
            != len(draft.executed_actions)
        ):
            raise FdmProtocolError("feedback_ack accepted action count mismatch")
        missing = [
            name
            for name in self._config.feedback_ack_required_fields
            if name not in response
        ]
        if missing:
            raise FdmProtocolError(
                f"feedback_ack missing configured fields: {missing}"
            )

    @staticmethod
    def _response(exchange: Any) -> Mapping[str, Any]:
        response = getattr(exchange, "response", exchange)
        if not isinstance(response, Mapping):
            raise FdmProtocolError("feedback transport returned a non-mapping")
        return response

    def process_one(self, transport: Any, draft: FeedbackDraft) -> bool:
        """Process one draft; public for deterministic fake-transport tests."""

        if not self._current(draft):
            return False
        snapshot = self._await_snapshot(draft)
        message = self._build_message(draft, snapshot)
        # Build once. Retries intentionally reuse the exact images, actions,
        # request_id, and feedback_seq rather than taking a newer keyframe.
        attempts = self._config.feedback_retry_limit + 1
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            if not self._current(draft):
                return False
            started = time.monotonic()
            try:
                response = self._response(transport.exchange(message))
                self._validate_ack(response, draft)
                self._last_rtt_ms = (time.monotonic() - started) * 1000.0
                with self._session_lock:
                    if (
                        draft.session_id != self._session_id
                        or draft.generation != self._generation
                    ):
                        return False
                    self._acked[draft.feedback_id] = dict(response)
                self._acknowledged += 1
                self._last_error = ""
                if self._on_event is not None:
                    self._on_event(
                        "fdm_feedback_ack",
                        {
                            "request_id": draft.request_id,
                            "feedback_id": draft.feedback_id,
                            "feedback_seq": draft.feedback_seq,
                            "global_action_start": draft.global_action_start,
                            "action_count": len(draft.executed_actions),
                            "rtt_ms": self._last_rtt_ms,
                            "native_chunk_id": response.get("native_chunk_id"),
                            "grounded_frontier": response.get("grounded_frontier"),
                            "grounding_triggered": response.get(
                                "grounding_triggered"
                            ),
                        },
                    )
                return True
            except Exception as exc:  # retry exact immutable message
                last_error = exc
                self._last_error = str(exc)
                if attempt + 1 >= attempts:
                    break
                self._retries += 1
                if self._on_event is not None:
                    self._on_event(
                        "fdm_feedback_retry",
                        {
                            "request_id": draft.request_id,
                            "feedback_id": draft.feedback_id,
                            "feedback_seq": draft.feedback_seq,
                            "attempt": attempt + 1,
                            "error": str(exc),
                        },
                    )
                try:
                    transport.reconnect()
                except Exception:
                    pass
                self._stop.wait(self._config.feedback_retry_backoff_s)
        self._failures += 1
        error = f"FDM feedback retry limit exceeded: {last_error}"
        self._last_error = error
        if self._on_event is not None:
            self._on_event(
                "fdm_feedback_failed",
                {
                    "request_id": draft.request_id,
                    "feedback_id": draft.feedback_id,
                    "feedback_seq": draft.feedback_seq,
                    "error": error,
                },
            )
        if self._current(draft):
            self._on_fatal(error)
        return False

    def _run(self) -> None:
        transport = None
        try:
            while not self._stop.is_set():
                try:
                    draft = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if transport is None:
                        transport = self._transport_factory()
                    self.process_one(transport, draft)
                except Exception as exc:
                    self._failures += 1
                    self._last_error = str(exc)
                    if self._current(draft):
                        self._on_fatal(f"FDM feedback worker failed: {exc}")
                    if transport is not None:
                        try:
                            transport.close()
                        except Exception:
                            pass
                        transport = None
                finally:
                    self._queue.task_done()
        finally:
            if transport is not None:
                transport.close()

    def status(self) -> dict[str, Any]:
        with self._session_lock:
            return {
                "session_id": self._session_id,
                "generation": self._generation,
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._config.feedback_queue_size,
                "submitted": self._submitted,
                "acknowledged": self._acknowledged,
                "retries": self._retries,
                "failures": self._failures,
                "last_rtt_ms": self._last_rtt_ms,
                "last_error": self._last_error,
            }

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.0, float(timeout_s)))
