from __future__ import annotations

import asyncio

import pytest

from synth_ai.sdk.optimization.policy import v1 as policy_v1


class _StubHttpClient:
    last_get_path: str | None = None
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
        del headers
        _StubAsyncClient.last_patch_url = url
        _StubAsyncClient.last_patch_params = dict(params or {})
        _StubAsyncClient.last_patch_json = dict(json or {})
        return _StubPatchResponse()


def _job() -> policy_v1.PolicyOptimizationOfflineJob:
    return policy_v1.PolicyOptimizationOfflineJob(
        job_id="pl_123",
        backend_url="http://localhost:8080",
        api_key="test-key",
        api_version="v2",
    )


def test_get_rollout_dispatch_metrics_uses_expected_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(_job().get_rollout_dispatch_metrics_async())
    assert payload["ok"] is True
    assert _StubHttpClient.last_get_path == "/v2/offline/jobs/pl_123/queue/rollouts/metrics"


def test_get_rollout_limiter_status_uses_expected_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(_job().get_rollout_limiter_status_async())
    assert payload["ok"] is True
    assert _StubHttpClient.last_get_path == "/v2/offline/jobs/pl_123/queue/rollouts/limiter-status"


def test_retry_rollout_dispatch_uses_expected_path_and_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        _job().retry_rollout_dispatch_async(
            "trial_0001:0",
            algorithm_kind="gepa",
        )
    )
    assert payload["ok"] is True
    assert (
        _StubHttpClient.last_post_path
        == "/v2/offline/jobs/pl_123/queue/rollouts/trial_0001:0/retry?algorithm_kind=gepa"
    )
    assert _StubHttpClient.last_post_json == {}


def test_retry_rollout_dispatch_rejects_blank_dispatch_id() -> None:
    with pytest.raises(ValueError, match="dispatch_id is required"):
        asyncio.run(_job().retry_rollout_dispatch_async("   "))


def test_drain_rollout_queue_uses_expected_path_and_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        _job().drain_rollout_queue_async(
            cancel_queued=True,
            algorithm_kind="mipro",
        )
    )
    assert payload["ok"] is True
    assert (
        _StubHttpClient.last_post_path
        == "/v2/offline/jobs/pl_123/queue/rollouts/drain?algorithm_kind=mipro"
    )
    assert _StubHttpClient.last_post_json == {"cancel_queued": True}


def test_set_rollout_queue_policy_uses_patch_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "httpx", type("_HTTPX", (), {"AsyncClient": _StubAsyncClient}))
    payload = asyncio.run(
        _job().set_rollout_queue_policy_async(
            policy_patch={"retry_policy": {"max_attempts": 5}},
            algorithm_kind="gepa",
        )
    )
    assert payload["ok"] is True
    assert (
        _StubAsyncClient.last_patch_url
        == "http://localhost:8080/api/v2/offline/jobs/pl_123/queue/rollouts/policy"
    )
    assert _StubAsyncClient.last_patch_params == {"algorithm_kind": "gepa"}
    assert _StubAsyncClient.last_patch_json == {"retry_policy": {"max_attempts": 5}}
