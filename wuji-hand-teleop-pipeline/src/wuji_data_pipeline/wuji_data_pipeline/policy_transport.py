"""Synchronous policy transports used by the non-real-time deployment worker.

Both transports carry the exact same protocol-v2 pickle payload.  HTTP is
implemented with :mod:`http.client` so robot-side proxy environment variables
are never consulted and redirects are never followed implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import pickle
import socket
import threading
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit


HTTP_CONTENT_TYPES = (
    "application/octet-stream",
    "application/x-python-pickle",
)
SUPPORTED_POLICY_SCHEMES = ("tcp", "http", "https")


class PolicyAuthenticationError(RuntimeError):
    """The policy endpoint rejected its configured credentials."""


def policy_transport_kind(endpoint: str) -> str:
    """Return ``zmq`` or ``http`` from a validated endpoint scheme."""

    scheme = urlsplit(str(endpoint).strip()).scheme.lower()
    if scheme == "tcp":
        return "zmq"
    if scheme in ("http", "https"):
        return "http"
    supported = ", ".join(f"{item}://" for item in SUPPORTED_POLICY_SCHEMES)
    raise ValueError(
        f"unsupported policy endpoint scheme {scheme or '<missing>'!r}; "
        f"expected one of {supported}"
    )


@dataclass(frozen=True)
class PolicyExchange:
    """One decoded policy response and its wire byte counts."""

    response: Mapping[str, Any]
    request_bytes: int
    response_bytes: int


class PolicyTransport:
    """Small synchronous interface owned by one deployment worker thread."""

    kind = "unknown"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = str(endpoint).strip()
        self.last_http_status: Optional[int] = None
        self.reconnects = 0

    def exchange(self, request: Mapping[str, Any]) -> PolicyExchange:
        raise NotImplementedError

    def reconnect(self) -> None:
        self.reconnects += 1
        self.close_connection()

    def close_connection(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.close_connection()


class ZmqPolicyTransport(PolicyTransport):
    """Existing pickle + ZMQ REQ transport, retained for replay/LAN use."""

    kind = "zmq"

    def __init__(self, endpoint: str, *, timeout_ms: int) -> None:
        super().__init__(endpoint)
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError(
                "pyzmq is required; rebuild the project Docker image after "
                "the dependency update"
            ) from exc
        self._zmq = zmq
        self._timeout_ms = int(timeout_ms)
        self._context = zmq.Context()
        self._socket = None

    def _ensure_socket(self):
        if self._socket is None:
            socket_ = self._context.socket(self._zmq.REQ)
            socket_.setsockopt(self._zmq.LINGER, 0)
            socket_.connect(self.endpoint)
            self._socket = socket_
        return self._socket

    def exchange(self, request: Mapping[str, Any]) -> PolicyExchange:
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        socket_ = self._ensure_socket()
        socket_.send(payload)
        if socket_.poll(self._timeout_ms, self._zmq.POLLIN) == 0:
            raise TimeoutError(
                f"policy server timed out after {self._timeout_ms}ms"
            )
        response_payload = socket_.recv()
        response = pickle.loads(response_payload)
        if not isinstance(response, Mapping):
            raise ValueError("policy response is not a mapping")
        return PolicyExchange(response, len(payload), len(response_payload))

    def close_connection(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def close(self) -> None:
        self.close_connection()
        self._context.term()


class HttpPolicyTransport(PolicyTransport):
    """Persistent authenticated HTTP/1.1 transport for the inference service."""

    kind = "http"

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_ms: int,
        policy_path: str,
        api_key: str,
        max_response_bytes: int,
    ) -> None:
        super().__init__(endpoint)
        parsed = urlsplit(self.endpoint)
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("HTTP transport requires an http:// or https:// endpoint")
        if not parsed.hostname:
            raise ValueError("HTTP policy endpoint is missing a hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("HTTP policy endpoint must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("HTTP policy endpoint must not contain query or fragment")
        if parsed.path not in ("", "/"):
            raise ValueError(
                "HTTP policy endpoint must be a base URL; configure the policy "
                "path separately"
            )
        normalized_path = str(policy_path).strip()
        if not normalized_path.startswith("/") or "?" in normalized_path:
            raise ValueError("HTTP policy path must be an absolute path without query")
        if not str(api_key):
            raise ValueError("HTTP policy API key is empty")
        if int(timeout_ms) <= 0 or int(max_response_bytes) <= 0:
            raise ValueError("HTTP timeout and response size limit must be positive")

        self._scheme = parsed.scheme.lower()
        self._host = parsed.hostname
        self._port = parsed.port
        self._path = normalized_path
        self._api_key = str(api_key)
        self._timeout_s = int(timeout_ms) / 1000.0
        self._max_response_bytes = int(max_response_bytes)
        self._connection: Optional[http.client.HTTPConnection] = None
        self._round_trip_lock = threading.Lock()

    def _new_connection(self) -> http.client.HTTPConnection:
        connection_class = (
            http.client.HTTPSConnection
            if self._scheme == "https"
            else http.client.HTTPConnection
        )
        return connection_class(
            self._host,
            port=self._port,
            timeout=self._timeout_s,
        )

    def _ensure_connection(self) -> http.client.HTTPConnection:
        if self._connection is None:
            self._connection = self._new_connection()
        return self._connection

    @staticmethod
    def _content_type(response: http.client.HTTPResponse) -> str:
        value = response.getheader("Content-Type", "")
        return value.split(";", 1)[0].strip().lower()

    def _read_success_body(self, response: http.client.HTTPResponse) -> bytes:
        raw_length = response.getheader("Content-Length")
        if raw_length is None:
            raise ValueError("policy HTTP response is missing Content-Length")
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise ValueError("policy HTTP response has invalid Content-Length") from exc
        if length <= 0:
            raise ValueError("policy HTTP response has empty Content-Length")
        if length > self._max_response_bytes:
            raise ValueError(
                "policy HTTP response exceeds configured size limit "
                f"({length} > {self._max_response_bytes} bytes)"
            )
        content_type = self._content_type(response)
        if content_type not in HTTP_CONTENT_TYPES:
            raise ValueError(
                f"unsupported policy HTTP Content-Type {content_type or '<missing>'!r}"
            )
        try:
            body = response.read()
        except http.client.IncompleteRead as exc:
            raise ValueError(
                "policy HTTP response body is shorter than Content-Length"
            ) from exc
        if len(body) != length:
            raise ValueError(
                "policy HTTP response body length does not match Content-Length"
            )
        return body

    def exchange(self, request: Mapping[str, Any]) -> PolicyExchange:
        if not self._round_trip_lock.acquire(blocking=False):
            raise RuntimeError(
                "concurrent HTTP policy round-trip on one connection is forbidden"
            )
        try:
            return self._exchange_exclusive(request)
        finally:
            self._round_trip_lock.release()

    def _exchange_exclusive(self, request: Mapping[str, Any]) -> PolicyExchange:
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-OpenPI-API-Key": self._api_key,
            "Content-Type": "application/octet-stream",
            "Accept": "application/octet-stream",
            "Content-Length": str(len(payload)),
        }
        connection = self._ensure_connection()
        response = None
        self.last_http_status = None
        try:
            connection.request("POST", self._path, body=payload, headers=headers)
            response = connection.getresponse()
            self.last_http_status = int(response.status)
            if response.status != 200:
                # Drain only a small prefix, then close the connection.  Error
                # bodies are intentionally omitted from the exception so a
                # gateway can never reflect credentials into robot logs.
                response.read(4096)
                if response.status in (401, 403):
                    raise PolicyAuthenticationError(
                        "policy HTTP authentication/configuration failed with "
                        f"status {response.status}"
                    )
                raise RuntimeError(
                    f"policy HTTP request failed with status {response.status}"
                )
            response_payload = self._read_success_body(response)
            if response.will_close:
                raise ConnectionError(
                    "policy HTTP service closed the required persistent connection"
                )
            try:
                decoded = pickle.loads(response_payload)
            except (pickle.PickleError, EOFError, AttributeError, ImportError) as exc:
                raise ValueError("failed to decode policy HTTP pickle response") from exc
            if not isinstance(decoded, Mapping):
                raise ValueError("policy response is not a mapping")
            return PolicyExchange(decoded, len(payload), len(response_payload))
        except (
            OSError,
            socket.timeout,
            http.client.HTTPException,
            RuntimeError,
            ValueError,
            pickle.PickleError,
            EOFError,
        ):
            self.close_connection()
            raise
        finally:
            if response is not None:
                response.close()

    def close_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def create_policy_transport(
    endpoint: str,
    *,
    timeout_ms: int,
    http_policy_path: str,
    http_api_key: str,
    http_max_response_bytes: int,
) -> PolicyTransport:
    """Create the transport selected by the endpoint scheme."""

    kind = policy_transport_kind(endpoint)
    if kind == "zmq":
        return ZmqPolicyTransport(endpoint, timeout_ms=timeout_ms)
    return HttpPolicyTransport(
        endpoint,
        timeout_ms=timeout_ms,
        policy_path=http_policy_path,
        api_key=http_api_key,
        max_response_bytes=http_max_response_bytes,
    )
