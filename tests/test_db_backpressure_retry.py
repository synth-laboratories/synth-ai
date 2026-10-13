"""Bounded retry for the backend's typed DB backpressure refusal.

The backend sheds DB work before starting it with HTTP 429 and the body from
``core/data/db/admission.py::db_backpressure_payload``. The shared transport
retries only that refusal, only for replay-safe requests (GET/HEAD/OPTIONS or an
Idempotency-Key), honoring ``retry_after`` under a cap, and raises a typed
exhaustion error carrying the last refusal.
"""

from __future__ import annotations

import base64

import httpx
import pytest
from synth_ai.core.errors import SynthError
from synth_ai.core.http.retry import (
    DbBackpressureRetryExhaustedError,
    DbBackpressureRetryPolicy,
)
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.errors import ResearchApiError
from synth_ai.sdk.research.session.client import ResearchSession

UPLOAD_PATH = "/smr/projects/p1/workspace/code-bundle:upload"


def _refusal(retry_after: int = 2, component: str = "api", work_class: str = "write"):
    return httpx.Response(
        429,
        headers={"Retry-After": str(retry_after)},
        json={
            "error": "db_backpressure",
            "error_code": "db_backpressure",
            "message": "Database work was rejected before starting because the service is near capacity.",
            "component": component,
            "work_class": work_class,
            "retry_after": retry_after,
            "operator_action": "retry_after_backoff_or_reduce_db_request_pressure",
        },
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _wire(transport: HttpTransport, handler, clock: _Clock) -> HttpTransport:
    transport.client.close()
    transport.client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://backend.test"
    )
    transport.sleep = clock.sleep
    transport.monotonic = clock.monotonic
    return transport


def _core(handler, clock, policy=None) -> HttpTransport:
    transport = HttpTransport("https://backend.test", {})
    if policy is not None:
        transport.db_backpressure_policy = policy
    return _wire(transport, handler, clock)


def _sequence(*responses):
    calls: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return handler, calls


def test_idempotent_get_retries_backpressure_then_succeeds_honoring_retry_after():
    clock = _Clock()
    handler, calls = _sequence(_refusal(3), _refusal(3), httpx.Response(200, json={"ok": True}))
    policy = DbBackpressureRetryPolicy(jitter_ratio=0.0)
    result = _core(handler, clock, policy).request_json("GET", "/smr/projects/p1")
    assert result == {"ok": True}
    assert len(calls) == 3
    assert clock.sleeps == [3.0, 3.0]


def test_retry_after_is_capped_and_jitter_never_waits_less_than_retry_after():
    clock = _Clock()
    handler, _ = _sequence(_refusal(10), _refusal(500), httpx.Response(200, json={}))
    policy = DbBackpressureRetryPolicy(retry_after_cap_seconds=30.0, jitter_ratio=0.25)
    _core(handler, clock, policy).request_json("GET", "/x")
    assert 10.0 <= clock.sleeps[0] <= 12.5
    assert 30.0 <= clock.sleeps[1] <= 37.5


def test_exhaustion_is_typed_and_carries_last_refusal():
    clock = _Clock()
    handler, calls = _sequence(_refusal(1, component="api", work_class="write"))
    policy = DbBackpressureRetryPolicy(attempts_max=4, jitter_ratio=0.0)
    with pytest.raises(DbBackpressureRetryExhaustedError) as caught:
        _core(handler, clock, policy).request_json("GET", "/x")
    error = caught.value
    assert len(calls) == 4
    assert error.attempts == 4
    assert error.component == "api"
    assert error.work_class == "write"
    assert error.retry_after_seconds == 1.0
    assert error.refusal.operator_action == "retry_after_backoff_or_reduce_db_request_pressure"
    assert error.retryable is False
    assert "db_backpressure" in str(error)


def test_total_wait_budget_bounds_retries():
    clock = _Clock()
    handler, calls = _sequence(_refusal(25))
    policy = DbBackpressureRetryPolicy(
        attempts_max=10, total_wait_seconds_max=60.0, jitter_ratio=0.0
    )
    with pytest.raises(DbBackpressureRetryExhaustedError) as caught:
        _core(handler, clock, policy).request_json("GET", "/x")
    assert clock.sleeps == [25.0, 25.0]
    assert len(calls) == 3
    assert caught.value.waited_seconds == 50.0


def test_unkeyed_post_is_not_retried():
    clock = _Clock()
    handler, calls = _sequence(_refusal(1), httpx.Response(200, json={}))
    with pytest.raises(SynthError) as caught:
        _core(handler, clock).request_json("POST", "/x", json_body={"a": 1})
    assert not isinstance(caught.value, DbBackpressureRetryExhaustedError)
    assert len(calls) == 1
    assert clock.sleeps == []


def test_keyed_post_is_retried():
    clock = _Clock()
    handler, calls = _sequence(_refusal(1), httpx.Response(200, json={"ok": 1}))
    result = _core(handler, clock).request_json(
        "POST", "/x", json_body={"a": 1}, headers={"Idempotency-Key": "k1"}
    )
    assert result == {"ok": 1}
    assert len(calls) == 2


def test_other_429_is_not_treated_as_backpressure():
    clock = _Clock()
    handler, calls = _sequence(
        httpx.Response(429, json={"detail": {"error_code": "rate_limited"}}),
        httpx.Response(200, json={}),
    )
    with pytest.raises(SynthError):
        _core(handler, clock).request_json("GET", "/x")
    assert len(calls) == 1


def test_research_code_bundle_upload_retries_with_content_derived_key(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle-bytes")
    clock = _Clock()
    handler, calls = _sequence(
        _refusal(2),
        httpx.Response(200, json={"kind": "uploaded_bundle", "status": "ready"}),
    )
    session = ResearchSession(api_key="test-key", backend_base="https://backend.test")
    _wire(session._transport, handler, clock)
    session._transport.db_backpressure_policy = DbBackpressureRetryPolicy(jitter_ratio=0.0)
    result = session.upload_project_code_bundle("p1", bundle)
    assert result["status"] == "ready"
    assert len(calls) == 2
    keys = {request.headers.get("Idempotency-Key") for request in calls}
    assert len(keys) == 1
    key = keys.pop()
    assert key is not None and key.startswith("code-bundle-upload:")
    other = ResearchSession(api_key="test-key", backend_base="https://backend.test")
    seen: list[httpx.Request] = []
    _wire(other._transport, lambda r: seen.append(r) or httpx.Response(200, json={}), clock)
    bundle.write_bytes(b"different-bytes")
    other.upload_project_code_bundle("p1", bundle)
    assert seen[0].headers["Idempotency-Key"] != key
    assert clock.sleeps == [2.0]
    assert (
        base64.b64decode(__import__("json").loads(calls[0].content)["content"]) == b"bundle-bytes"
    )


def test_research_upload_exhaustion_is_typed(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"x")
    clock = _Clock()
    handler, calls = _sequence(_refusal(1, component="smr", work_class="write"))
    session = ResearchSession(api_key="test-key", backend_base="https://backend.test")
    _wire(session._transport, handler, clock)
    with pytest.raises(DbBackpressureRetryExhaustedError) as caught:
        session.upload_project_code_bundle("p1", bundle)
    assert caught.value.component == "smr"
    assert len(calls) == DbBackpressureRetryPolicy().attempts_max
    assert not isinstance(caught.value, ResearchApiError) or caught.value.retryable is False
