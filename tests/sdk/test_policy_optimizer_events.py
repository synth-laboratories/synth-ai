from __future__ import annotations

import asyncio

import pytest

from synth_ai.sdk.optimization.policy import v1 as policy_v1


class _StubHttpClient:
    last_get_path: str | None = None
    last_get_params: dict | None = None

    def __init__(self, _base_url: str, _api_key: str, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _StubHttpClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, path: str, *, params=None):  # noqa: ANN001
        _StubHttpClient.last_get_path = path
        _StubHttpClient.last_get_params = dict(params or {})
        return {"ok": True, "path": path, "params": params or {}}


def test_optimizer_events_uses_optimizer_events_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.optimizer_events_async(
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
            limit=50,
            run_id="run_1",
            event_family="proposer",
            stream_id="run_1:proposer:sess_1",
            sequence="19",
            payload_redacted=True,
        )
    )
    assert payload.get("ok") is True
    assert _StubHttpClient.last_get_path == "/v2/optimizer/events"
    assert _StubHttpClient.last_get_params == {
        "limit": 50,
        "run_id": "run_1",
        "event_family": "proposer",
        "stream_id": "run_1:proposer:sess_1",
        "sequence": "19",
        "payload_redacted": True,
    }


def test_failure_events_uses_failures_query_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.failure_events_async(
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
            reason_code="upstream_timeout",
            error_type="timeout",
            source="proposer",
            source_session_id="sess_1",
            source_sequence="11",
        )
    )
    assert payload.get("ok") is True
    assert _StubHttpClient.last_get_path == "/v2/failures/query"
    assert _StubHttpClient.last_get_params == {
        "limit": 200,
        "reason_code": "upstream_timeout",
        "error_type": "timeout",
        "source": "proposer",
        "source_session_id": "sess_1",
        "source_sequence": "11",
    }


def test_admin_optimizer_events_uses_admin_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.admin_optimizer_events_async(
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
            org_id="org_1",
            trial_id="trial_9",
            runtime_tick_id="tick_2",
        )
    )
    assert payload.get("ok") is True
    assert _StubHttpClient.last_get_path == "/v2/admin/optimizer/events"
    assert _StubHttpClient.last_get_params == {
        "limit": 200,
        "org_id": "org_1",
        "trial_id": "trial_9",
        "runtime_tick_id": "tick_2",
    }


def test_admin_failure_events_uses_admin_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.admin_failure_events_async(
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
            org_id="org_1",
            reason_code="candidate_validation_blocked",
            event_family="candidate",
        )
    )
    assert payload.get("ok") is True
    assert _StubHttpClient.last_get_path == "/v2/admin/failures/query"
    assert _StubHttpClient.last_get_params == {
        "limit": 200,
        "org_id": "org_1",
        "event_family": "candidate",
        "reason_code": "candidate_validation_blocked",
    }


def test_admin_victoria_logs_query_uses_admin_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.admin_victoria_logs_query_async(
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
            q='event_family:"proposer"',
            limit=25,
            redact=True,
        )
    )
    assert payload.get("ok") is True
    assert _StubHttpClient.last_get_path == "/v2/admin/victoria-logs/query"
    assert _StubHttpClient.last_get_params == {
        "q": 'event_family:"proposer"',
        "redact": True,
        "limit": 25,
    }
