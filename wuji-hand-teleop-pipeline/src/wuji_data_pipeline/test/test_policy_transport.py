from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import pickle
import threading
import time

import numpy as np
import pytest

from wuji_data_pipeline.policy_transport import (
    HttpPolicyTransport,
    PolicyAuthenticationError,
    ZmqPolicyTransport,
    create_policy_transport,
    policy_transport_kind,
)
from wuji_data_pipeline.deployment_protocol import LatestActionPlan, extract_action_chunk


API_KEY = "unit-test-secret-that-must-not-be-logged"


class _PolicyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def do_GET(self):
        self.server.get_requests += 1
        body = b"unexpected redirect target"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length)
        self.server.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": request_body,
            }
        )
        mode = self.server.mode
        if mode == "timeout":
            time.sleep(0.15)

        status = {
            "redirect": 302,
            "unauthorized": 401,
            "unavailable": 503,
        }.get(mode, 200)
        if status != 200:
            body = b"request rejected"
            self.send_response(status)
            if status == 302:
                self.send_header("Location", "/redirect-target")
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if mode == "bad_pickle":
            body = b"not a pickle"
        else:
            request = pickle.loads(request_body)
            response_mapping = {
                    "protocol_version": 2,
                    "message_type": "hello_ack",
                    "session_id": request.get("session_id"),
                    "request_id": request.get("request_id"),
                    "model_id": "checkpoint-test",
                    "action_rate_hz": 30.0,
                }
            if self.server.response_builder is not None:
                response_mapping = self.server.response_builder(request)
            body = pickle.dumps(
                response_mapping,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        self.send_response(200)
        content_type = (
            "text/plain" if mode == "bad_content_type" else "application/octet-stream"
        )
        self.send_header("Content-Type", content_type)
        if mode != "missing_length":
            if mode == "oversize":
                response_length = 1025
            elif mode == "short_body":
                response_length = len(body) + 7
            else:
                response_length = len(body)
            self.send_header("Content-Length", str(response_length))
        if mode in ("short_body", "missing_length", "oversize"):
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if mode != "oversize":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass


class _CountingHTTPServer(HTTPServer):
    def __init__(self, *args, **kwargs):
        self.accepted_connections = 0
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, address = super().get_request()
        self.accepted_connections += 1
        return request, address


@contextmanager
def _fake_policy_server(mode="ok", response_builder=None):
    server = _CountingHTTPServer(("127.0.0.1", 0), _PolicyHandler)
    server.mode = mode
    server.requests = []
    server.get_requests = 0
    server.response_builder = response_builder
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _transport(url, *, timeout_ms=500, max_response_bytes=1024):
    return HttpPolicyTransport(
        url,
        timeout_ms=timeout_ms,
        policy_path="/v1/robot-policy",
        api_key=API_KEY,
        max_response_bytes=max_response_bytes,
    )


def _hello():
    return {
        "protocol_version": 2,
        "message_type": "hello",
        "session_id": "session-1",
        "request_id": 7,
        "array": np.arange(4, dtype=np.float32),
    }


def test_endpoint_scheme_selects_zmq_or_http_and_rejects_unknown():
    assert policy_transport_kind("tcp://127.0.0.1:5555") == "zmq"
    assert policy_transport_kind("http://policy.local") == "http"
    assert policy_transport_kind("https://policy.local") == "http"
    with pytest.raises(ValueError, match="unsupported policy endpoint"):
        policy_transport_kind("ftp://policy.local")


def test_factory_retains_original_zmq_transport_for_tcp_endpoint():
    transport = create_policy_transport(
        "tcp://127.0.0.1:5555",
        timeout_ms=100,
        http_policy_path="/v1/robot-policy",
        http_api_key="",
        http_max_response_bytes=1024,
    )
    try:
        assert isinstance(transport, ZmqPolicyTransport)
    finally:
        transport.close()


def test_http_preserves_pickle_body_sends_both_auth_headers_and_ignores_proxy(
    monkeypatch,
):
    # http.client does not consult these variables.  If the transport did,
    # this intentionally dead proxy would make the local exchange fail.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    with _fake_policy_server() as (server, url):
        transport = _transport(url)
        try:
            result = transport.exchange(_hello())
        finally:
            transport.close()

    received = server.requests[0]
    decoded = pickle.loads(received["body"])
    assert received["path"] == "/v1/robot-policy"
    assert decoded["session_id"] == "session-1"
    assert np.array_equal(decoded["array"], _hello()["array"])
    assert received["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert received["headers"]["X-OpenPI-API-Key"] == API_KEY
    assert received["headers"]["Content-Type"] == "application/octet-stream"
    assert result.response["request_id"] == 7
    assert result.request_bytes == len(received["body"])
    assert result.response_bytes > 0
    assert transport.last_http_status == 200


def test_http_reuses_one_persistent_connection_for_sequential_requests():
    with _fake_policy_server() as (server, url):
        transport = _transport(url)
        try:
            transport.exchange(_hello())
            transport.exchange({**_hello(), "request_id": 8})
            assert server.accepted_connections == 1
        finally:
            transport.close()


def test_http_rejects_two_simultaneous_round_trips_on_one_connection():
    with _fake_policy_server("timeout") as (server, url):
        transport = _transport(url, timeout_ms=500)
        first_result = []

        def first_request():
            first_result.append(transport.exchange(_hello()))

        worker = threading.Thread(target=first_request)
        worker.start()
        deadline = time.monotonic() + 1.0
        while not server.requests and time.monotonic() < deadline:
            time.sleep(0.005)
        assert server.requests
        with pytest.raises(RuntimeError, match="concurrent HTTP policy"):
            transport.exchange({**_hello(), "request_id": 8})
        worker.join(timeout=2.0)
        try:
            assert len(first_result) == 1
        finally:
            transport.close()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing_length", "Content-Length"),
        ("short_body", "shorter than Content-Length"),
        ("bad_content_type", "Content-Type"),
        ("bad_pickle", "pickle"),
        ("oversize", "size limit"),
    ],
)
def test_http_rejects_malformed_success_responses(mode, message):
    with _fake_policy_server(mode) as (_server, url):
        transport = _transport(url)
        try:
            with pytest.raises(Exception, match=message):
                transport.exchange(_hello())
            assert transport._connection is None
        finally:
            transport.close()


def test_http_auth_error_does_not_leak_api_key():
    with _fake_policy_server("unauthorized") as (_server, url):
        transport = _transport(url)
        try:
            with pytest.raises(PolicyAuthenticationError) as captured:
                transport.exchange(_hello())
            assert "401" in str(captured.value)
            assert API_KEY not in str(captured.value)
            assert transport._connection is None
        finally:
            transport.close()


@pytest.mark.parametrize("mode", ["unavailable", "timeout"])
def test_http_transient_failure_closes_connection_for_reconnect(mode):
    with _fake_policy_server(mode) as (_server, url):
        transport = _transport(url, timeout_ms=50)
        try:
            with pytest.raises((RuntimeError, TimeoutError, OSError)):
                transport.exchange(_hello())
            assert transport._connection is None
            transport.reconnect()
            assert transport.reconnects == 1
        finally:
            transport.close()


def test_http_redirect_is_not_followed():
    with _fake_policy_server("redirect") as (server, url):
        transport = _transport(url)
        try:
            with pytest.raises(RuntimeError, match="302"):
                transport.exchange(_hello())
            assert server.get_requests == 0
        finally:
            transport.close()


def test_http_endpoint_rejects_embedded_credentials_and_path():
    with pytest.raises(ValueError, match="credentials"):
        _transport("http://user:password@policy.local")
    with pytest.raises(ValueError, match="base URL"):
        _transport("http://policy.local/v1/robot-policy")


def test_http_action_chunk_enters_existing_bounded_action_plan():
    action = {
        "arm_action_left": {
            "ee_pos": [0.4, 0.2, 0.5],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
        },
        "hand_action_left": [0.0] * 20,
        "arm_action_right": {
            "ee_pos": [0.4, -0.2, 0.5],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
        },
        "hand_action_right": [0.0] * 20,
    }

    def response_builder(request):
        return {
            "protocol_version": 2,
            "message_type": "action_chunk",
            "session_id": request["session_id"],
            "request_id": request["request_id"],
            "model_id": "checkpoint-test",
            "action_rate_hz": 30.0,
            "action_chunk": [action for _ in range(50)],
        }

    with _fake_policy_server(response_builder=response_builder) as (_server, url):
        transport = _transport(url, max_response_bytes=1024 * 1024)
        try:
            exchange = transport.exchange(_hello())
        finally:
            transport.close()

    actions, rate_hz = extract_action_chunk(
        exchange.response, default_rate_hz=30.0
    )
    plan = LatestActionPlan(max_actions=25)
    result = plan.install(
        actions,
        observation_created_at=10.0,
        received_at=10.01,
        rate_hz=rate_hz,
    )
    assert len(actions) == 50
    assert result.accepted == 25
    assert plan.remaining() == 25


def test_http_transport_preserves_pi_joint_action_space_and_full_mapping():
    action = {
        "arm_action_left": {"joint_pos": [0.0] * 7},
        "hand_action_left": [0.0] * 20,
        "arm_action_right": {"joint_pos": [0.1] * 7},
        "hand_action_right": [12.0] * 20,
    }

    def response_builder(request):
        return {
            "protocol_version": 2,
            "message_type": "action_chunk",
            "session_id": request["session_id"],
            "request_id": request["request_id"],
            "model_id": "dropper-joint-99999",
            "action_rate_hz": 30.0,
            "arm_action_space": "joint_position",
            "action_chunk": [action for _ in range(50)],
            "server_timing": {"inference_ms": 123.0},
        }

    with _fake_policy_server(response_builder=response_builder) as (_server, url):
        transport = _transport(url, max_response_bytes=1024 * 1024)
        try:
            exchange = transport.exchange(_hello())
        finally:
            transport.close()

    assert exchange.response["arm_action_space"] == "joint_position"
    assert exchange.response["server_timing"]["inference_ms"] == 123.0
    assert len(exchange.response["action_chunk"]) == 50
    assert np.allclose(
        exchange.response["action_chunk"][0]
        ["arm_action_right"]["joint_pos"],
        [0.1] * 7,
    )
