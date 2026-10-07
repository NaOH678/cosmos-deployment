# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HTTP/1.1 transport and bounded session state for robot-policy protocol-v2."""

from __future__ import annotations

import hmac
import json
import logging
import pickle
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from cosmos_framework.inference.robot_policy.adapters import AdapterOutput, ModelAdapter
from cosmos_framework.inference.robot_policy.config import RobotPolicyConfig
from cosmos_framework.inference.robot_policy.protocol import (
    ProtocolError,
    error_envelope,
    validate_hello,
    validate_observation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionHandle:
    """Opaque identity tying a session generation to one HTTP connection."""

    session_id: str
    generation: int
    connection_id: str


@dataclass
class _SessionRecord:
    handle: SessionHandle
    last_access_monotonic: float


class SessionRegistry:
    """Thread-safe, TTL-bounded registry used to reject stale model results."""

    def __init__(self, max_sessions: int, ttl_s: float) -> None:
        self._max_sessions = max_sessions
        self._ttl_s = ttl_s
        self._records: OrderedDict[str, _SessionRecord] = OrderedDict()
        self._generation = 0
        self._lock = threading.Lock()

    def _prune_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, record in self._records.items()
            if now - record.last_access_monotonic > self._ttl_s
        ]
        for session_id in expired:
            del self._records[session_id]

    def begin(self, session_id: str, connection_id: str) -> SessionHandle:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            self._generation += 1
            handle = SessionHandle(session_id, self._generation, connection_id)
            self._records[session_id] = _SessionRecord(handle, now)
            self._records.move_to_end(session_id)
            while len(self._records) > self._max_sessions:
                self._records.popitem(last=False)
            return handle

    def is_current(self, handle: SessionHandle, *, touch: bool = False) -> bool:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            record = self._records.get(handle.session_id)
            if record is None or record.handle != handle:
                return False
            if touch:
                record.last_access_monotonic = now
                self._records.move_to_end(handle.session_id)
            return True

    def invalidate(self, handle: SessionHandle | None) -> None:
        if handle is None:
            return
        with self._lock:
            record = self._records.get(handle.session_id)
            if record is not None and record.handle == handle:
                del self._records[handle.session_id]


class PolicyApplication:
    """Protocol state machine independent from the HTTP request handler."""

    def __init__(self, config: RobotPolicyConfig, adapter: ModelAdapter) -> None:
        self.config = config
        self.adapter = adapter
        self.sessions = SessionRegistry(
            max_sessions=int(config.service.max_sessions),
            ttl_s=float(config.service.session_ttl_s),
        )
        self._inference_slots = threading.BoundedSemaphore(int(config.service.max_inflight_inferences))
        self._executor = ThreadPoolExecutor(
            max_workers=int(config.service.max_inflight_inferences),
            thread_name_prefix="cosmos-policy-inference",
        )

    def _run_inference(self, request: Mapping[str, Any]) -> AdapterOutput:
        try:
            return self.adapter.infer(request)
        finally:
            self._inference_slots.release()

    def close(self) -> None:
        """Stop accepting inference work; running GPU calls finish without being returned."""

        self._executor.shutdown(wait=False, cancel_futures=True)

    def _handle_hello(
        self,
        request: Mapping[str, Any],
        connection_id: str,
        previous_handle: SessionHandle | None,
    ) -> tuple[dict[str, Any], SessionHandle]:
        session_id, request_id = validate_hello(request, self.config)
        if not self.adapter.is_ready:
            raise ProtocolError("COSMOS_NOT_READY", "Cosmos model is not ready", fatal_session=True)

        self.sessions.invalidate(previous_handle)
        if previous_handle is not None:
            self.adapter.reset_session(previous_handle.session_id)
        handle = self.sessions.begin(session_id, connection_id)
        self.adapter.reset_session(session_id)
        return (
            {
                "protocol_version": self.config.deployment.protocol_version,
                "message_type": "hello_ack",
                "session_id": session_id,
                "request_id": request_id,
                "model_id": self.config.deployment.model_id,
                "action_rate_hz": float(self.config.deployment.action_rate_hz),
            },
            handle,
        )

    def _handle_observation(
        self,
        request: Mapping[str, Any],
        handle: SessionHandle | None,
        received_wall_time: float,
    ) -> tuple[dict[str, Any], SessionHandle]:
        session_id, request_id = validate_observation(request, self.config)
        if handle is None or handle.session_id != session_id or not self.sessions.is_current(handle, touch=True):
            raise ProtocolError(
                "COSMOS_SESSION_REQUIRED",
                "a successful hello is required on this connection",
                fatal_session=True,
            )
        if not self.adapter.is_ready:
            raise ProtocolError("COSMOS_NOT_READY", "Cosmos model is not ready", fatal_session=True)
        if not self._inference_slots.acquire(blocking=False):
            raise ProtocolError("COSMOS_BUSY", "Cosmos inference capacity is temporarily busy")

        inference_started = time.time()
        try:
            future = self._executor.submit(self._run_inference, request)
        except Exception:
            self._inference_slots.release()
            raise
        try:
            output = future.result(timeout=float(self.config.service.inference_timeout_s))
        except FutureTimeoutError as exc:
            raise ProtocolError(
                "COSMOS_INFERENCE_TIMEOUT",
                "Cosmos inference exceeded the configured timeout",
            ) from exc
        inference_finished = time.time()

        if not self.sessions.is_current(handle, touch=True):
            raise ProtocolError(
                "COSMOS_STALE_SESSION",
                "the session was replaced while inference was running",
                fatal_session=True,
            )
        return (
            {
                "protocol_version": self.config.deployment.protocol_version,
                "message_type": "action_chunk",
                "session_id": session_id,
                "request_id": request_id,
                "model_id": self.config.deployment.model_id,
                "action_rate_hz": float(self.config.deployment.action_rate_hz),
                "action_chunk": output.action_chunk,
                "server_timing": {
                    "received_wall_time": received_wall_time,
                    "inference_started_wall_time": inference_started,
                    "inference_finished_wall_time": inference_finished,
                    "inference_ms": float(output.inference_ms),
                },
            },
            handle,
        )

    def handle_request(
        self,
        request: Mapping[str, Any],
        connection_id: str,
        handle: SessionHandle | None,
        received_wall_time: float,
    ) -> tuple[dict[str, Any], SessionHandle | None]:
        """Dispatch one decoded request and convert every model/protocol failure to an envelope."""

        try:
            message_type = request.get("message_type")
            if message_type == "hello":
                return self._handle_hello(request, connection_id, handle)
            if message_type == "observation":
                return self._handle_observation(request, handle, received_wall_time)
            raise ProtocolError("COSMOS_MESSAGE_TYPE", "message_type must be hello or observation")
        except ProtocolError as error:
            if error.fatal_session:
                self.sessions.invalidate(handle)
                handle = None
            return error_envelope(request, error), handle
        except Exception:
            logger.exception("Unhandled robot policy request failure (request body omitted)")
            error = ProtocolError(
                "COSMOS_INFERENCE_FAILED",
                "Cosmos inference failed; inspect sanitized server logs",
                fatal_session=True,
            )
            self.sessions.invalidate(handle)
            return error_envelope(request, error), None


class RobotPolicyHTTPServer(ThreadingHTTPServer):
    """Typed server container holding immutable deployment dependencies."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: PolicyApplication,
        api_key: str,
    ) -> None:
        self.application = application
        self.api_key = api_key
        super().__init__(server_address, RobotPolicyRequestHandler)

    def server_close(self) -> None:
        self.application.close()
        super().server_close()


class RobotPolicyRequestHandler(BaseHTTPRequestHandler):
    """Sequential-per-connection HTTP/1.1 handler for trusted pickle requests."""

    protocol_version = "HTTP/1.1"
    server: RobotPolicyHTTPServer

    def setup(self) -> None:
        super().setup()
        self.connection_id = uuid.uuid4().hex
        self.session_handle: SessionHandle | None = None

    def log_message(self, format_string: str, *args: Any) -> None:
        logger.info("robot-policy peer=%s %s", self.client_address[0], format_string % args)

    def _send_empty(self, status: HTTPStatus, *, close: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        if status == HTTPStatus.UNAUTHORIZED:
            self.send_header("WWW-Authenticate", "Bearer")
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_pickle(self, payload: Mapping[str, Any], request: Mapping[str, Any] | None) -> None:
        body = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
        maximum = int(self.server.application.config.service.max_response_bytes)
        if len(body) > maximum:
            body = pickle.dumps(
                error_envelope(
                    request,
                    ProtocolError("COSMOS_RESPONSE_TOO_LARGE", "serialized response exceeds the service limit"),
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if len(body) > maximum:
            self._send_empty(HTTPStatus.INTERNAL_SERVER_ERROR, close=True)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.api_key
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return False
        if not hmac.compare_digest(authorization[len(prefix) :], expected):
            return False
        secondary = self.headers.get("X-OpenPI-API-Key")
        return secondary is None or hmac.compare_digest(secondary, expected)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "alive"})
            return
        if self.path == "/readyz":
            ready = self.server.application.adapter.is_ready
            self._send_json(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE, {"ready": ready})
            return
        self._send_empty(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        config = self.server.application.config
        if self.path != config.service.endpoint:
            self._send_empty(HTTPStatus.NOT_FOUND, close=True)
            return
        if not self._authorized():
            self._send_empty(HTTPStatus.UNAUTHORIZED, close=True)
            return
        if self.headers.get_content_type() not in {
            "application/octet-stream",
            "application/x-python-pickle",
        }:
            self._send_empty(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, close=True)
            return

        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            content_length = -1
        if content_length < 0:
            self._send_empty(HTTPStatus.LENGTH_REQUIRED, close=True)
            return
        if content_length > config.service.max_request_bytes:
            self._send_empty(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, close=True)
            return

        received_wall_time = time.time()
        body = self.rfile.read(content_length)
        request: Mapping[str, Any] | None = None
        if len(body) != content_length:
            self._send_empty(HTTPStatus.BAD_REQUEST, close=True)
            return
        try:
            decoded = pickle.loads(body)  # noqa: S301 - authenticated trusted-network protocol contract.
            if not isinstance(decoded, Mapping):
                raise TypeError("decoded request is not a mapping")
            request = decoded
        except Exception:
            error = ProtocolError("COSMOS_BAD_PICKLE", "request body is not a valid pickle mapping")
            self._send_pickle(error_envelope(None, error), None)
            return

        response, self.session_handle = self.server.application.handle_request(
            request,
            self.connection_id,
            self.session_handle,
            received_wall_time,
        )
        self._send_pickle(response, request)


def create_http_server(
    config: RobotPolicyConfig,
    adapter: ModelAdapter,
    api_key: str,
) -> RobotPolicyHTTPServer:
    """Create, but do not start, the policy HTTP server."""

    application = PolicyApplication(config, adapter)
    return RobotPolicyHTTPServer((config.service.host, int(config.service.port)), application, api_key)


__all__ = [
    "PolicyApplication",
    "RobotPolicyHTTPServer",
    "SessionHandle",
    "SessionRegistry",
    "create_http_server",
]
