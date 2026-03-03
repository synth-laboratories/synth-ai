from __future__ import annotations

import asyncio

import pytest

from synth_ai.sdk.optimization.policy import v1 as policy_v1


class _StubHttpClient:
    last_get_path: str | None = None
    last_get_params: dict | None = None
    last_post_path: str | None = None
    last_post_json: dict | None = None

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

    async def post_json(self, path: str, *, json):  # noqa: ANN001
        _StubHttpClient.last_post_path = path
        _StubHttpClient.last_post_json = dict(json)
        return {"ok": True, "path": path, "payload": json}


class _StubPatchResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):  # noqa: ANN201
        return {"ok": True}


class _StubAsyncClient:
    last_patch_url: str | None = None
    last_patch_params: dict | None = None
    last_patch_json: dict | None = None

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _StubAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def patch(self, url: str, *, params=None, json=None, headers=None):  # noqa: ANN001
        _StubAsyncClient.last_patch_url = url
        _StubAsyncClient.last_patch_params = dict(params or {})
        _StubAsyncClient.last_patch_json = dict(json or {})
        return _StubPatchResponse()


def _session() -> policy_v1.PolicyOptimizationOnlineSession:
    return policy_v1.PolicyOptimizationOnlineSession(
        session_id="sess-1",
        system_id="sys-1",
        backend_url="http://localhost:8080",
        api_key="test-key",
        api_version="v2",
    )


def _session_without_system_id() -> policy_v1.PolicyOptimizationOnlineSession:
    return policy_v1.PolicyOptimizationOnlineSession(
        session_id="sess-1",
        system_id=None,
        backend_url="http://localhost:8080",
        api_key="test-key",
        api_version="v2",
    )


def test_runtime_queue_trials_uses_runtime_system_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        _session().runtime_queue_trials_async(
            actor="proposer",
            algorithm="gepa",
            status="queued",
            limit=25,
        )
    )
    assert payload.get("ok") is True
    assert _StubHttpClient.last_get_path == "/v2/runtime/systems/sys-1/queue/trials"
    assert _StubHttpClient.last_get_params == {
        "actor": "proposer",
        "algorithm": "gepa",
        "status": "queued",
        "limit": 25,
    }


def test_runtime_queue_trials_uses_runtime_session_route_when_system_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(_session_without_system_id().runtime_queue_trials_async())
    assert payload.get("ok") is True
    assert _StubHttpClient.last_get_path == "/v2/runtime/sessions/sess-1/queue/trials"


def test_runtime_queue_contract_uses_runtime_session_route_when_system_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(_session_without_system_id().runtime_queue_contract_async())
    assert payload.get("ok") is True
    assert _StubHttpClient.last_get_path == "/v2/runtime/sessions/sess-1/queue/contract"


def test_runtime_queue_create_trial_appends_query_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        _session().runtime_queue_create_trial_async(
            candidate_id="cand-9",
            seed=4,
            actor="runtime",
            algorithm="mipro",
        )
    )
    assert payload.get("ok") is True
    assert _StubHttpClient.last_post_path == "/v2/runtime/systems/sys-1/queue/trials?actor=runtime&algorithm=mipro"
    assert _StubHttpClient.last_post_json == {"candidate_id": "cand-9", "seed": 4}


def test_runtime_queue_patch_rollout_uses_patch_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "httpx", type("_HTTPX", (), {"AsyncClient": _StubAsyncClient}))
    payload = asyncio.run(
        _session().runtime_queue_patch_rollout_async(
            "rollout-3",
            status="completed",
            now_ms=1234,
            actor="runtime",
            algorithm="gepa",
        )
    )
    assert payload.get("ok") is True
    assert _StubAsyncClient.last_patch_url == "http://localhost:8080/api/v2/runtime/systems/sys-1/queue/rollouts/rollout-3"
    assert _StubAsyncClient.last_patch_params == {"actor": "runtime", "algorithm": "gepa"}
    assert _StubAsyncClient.last_patch_json == {"status": "completed", "now_ms": 1234}


def test_runtime_queue_patch_contract_uses_patch_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "httpx", type("_HTTPX", (), {"AsyncClient": _StubAsyncClient}))
    payload = asyncio.run(
        _session().runtime_queue_patch_contract_async(
            patch={"rollout_queue": {"retry": {"max_attempts": 5}}},
            actor="operator",
            algorithm="mipro",
        )
    )
    assert payload.get("ok") is True
    assert (
        _StubAsyncClient.last_patch_url
        == "http://localhost:8080/api/v2/runtime/systems/sys-1/queue/contract"
    )
    assert _StubAsyncClient.last_patch_params == {"actor": "operator", "algorithm": "mipro"}
    assert _StubAsyncClient.last_patch_json == {
        "patch": {"rollout_queue": {"retry": {"max_attempts": 5}}}
    }


def test_runtime_queue_create_trial_passes_expected_state_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        _session().runtime_queue_create_trial_async(
            candidate_id="cand-10",
            expected_state_revision=9,
            actor="proposer",
            algorithm="gepa",
        )
    )
    assert payload.get("ok") is True
    assert (
        _StubHttpClient.last_post_path
        == "/v2/runtime/systems/sys-1/queue/trials?actor=proposer&algorithm=gepa&expected_state_revision=9"
    )


def test_runtime_queue_patch_contract_passes_expected_state_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "httpx", type("_HTTPX", (), {"AsyncClient": _StubAsyncClient}))
    payload = asyncio.run(
        _session().runtime_queue_patch_contract_async(
            patch={"trial_queue": {"default_priority": 2}},
            expected_state_revision=13,
            actor="operator",
            algorithm="mipro",
        )
    )
    assert payload.get("ok") is True
    assert _StubAsyncClient.last_patch_params == {
        "actor": "operator",
        "algorithm": "mipro",
        "expected_state_revision": 13,
    }


def test_runtime_container_rollout_checkpoint_dump_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.runtime_container_rollout_checkpoint_dump_async(
            "container-1",
            "rollout-9",
            payload={"reason": "manual"},
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
        )
    )
    assert payload.get("ok") is True
    assert (
        _StubHttpClient.last_post_path
        == "/v2/runtime/containers/container-1/rollouts/rollout-9/checkpoint/dump"
    )
    assert _StubHttpClient.last_post_json == {"reason": "manual"}


def test_runtime_container_rollout_checkpoint_restore_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.runtime_container_rollout_checkpoint_restore_async(
            "container-1",
            "rollout-9",
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
        )
    )
    assert payload.get("ok") is True
    assert (
        _StubHttpClient.last_post_path
        == "/v2/runtime/containers/container-1/rollouts/rollout-9/checkpoint/restore"
    )
    assert _StubHttpClient.last_post_json == {}
