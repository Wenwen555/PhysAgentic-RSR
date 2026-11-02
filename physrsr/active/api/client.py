"""Small JSON RPC transport. No retries on potentially physical actions."""
from __future__ import annotations

import json
from typing import Protocol
from urllib.request import Request, urlopen
from urllib.parse import urlparse


class Endpoint(Protocol):
    def call(self, method: str, params: dict) -> dict: ...


class APIError(RuntimeError):
    pass


class ModelVersionError(ValueError):
    """An acknowledgement from another model state cannot certify this session."""

    def __init__(self, *, session_id, method, expected, received):
        self.session_id, self.method = session_id, method
        self.expected, self.received = expected, received
        super().__init__(f"Predictive model version mismatch: session={session_id}, method={method}, "
            f"expected={expected}, received={received}")


class RecordedCalls:
    """Record a single RPC attempt. Uncertain physical calls are never repeated."""

    def __init__(self, recorder):
        self.recorder = recorder

    def call(self, endpoint, method, params):
        self.recorder.event("api_request", method=method, params=params)
        try:
            result = endpoint.call(method, params)
        except Exception as exc:
            self.recorder.event("api_error", method=method, error_type=type(exc).__name__, error=str(exc))
            raise
        self.recorder.event("api_response", method=method, result=result)
        return result

    def checked_version(self, response, *, session_id, method, expected):
        received = response.get("model_version")
        if isinstance(received, bool) or not isinstance(received, int) or received != expected:
            error = ModelVersionError(session_id=session_id, method=method, expected=expected, received=received)
            self.recorder.event("model_version_mismatch", session_id=session_id, method=method,
                expected=expected, received=received)
            raise error
        return response


class HTTPClient:
    def __init__(self, url, timeout=600):
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("API endpoint must be HTTP(S)")
        self.url, self.timeout = url, timeout

    def call(self, method, params):
        body = json.dumps(dict(method=method, params=params), allow_nan=False).encode()
        request = Request(self.url, data=body, headers={"Content-Type": "application/json"})
        # An uncertain acknowledgement must abort, never repeat an actuation.
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read(16 * 1024 * 1024))
        if not isinstance(payload, dict) or "error" in payload or "result" not in payload:
            raise APIError(str(payload.get("error", "Malformed API reply")) if isinstance(payload, dict) else "Malformed API reply")
        return payload["result"]


class Router:
    """The same dispatch contract is useful for in-process service tests."""
    def __init__(self, **services):
        self.services = services

    def call(self, method, params):
        namespace = method.partition(".")[0]
        if namespace not in self.services:
            raise APIError(f"Unknown service: {namespace}")
        return self.services[namespace].call(method, params)
