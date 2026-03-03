from __future__ import annotations

import asyncio

import pytest

from synth_ai.sdk.optimization.internal.learning import prompt_learning_client as plc_module


class _StubHttpClient:
    get_calls: list[str] = []
    post_calls: list[tuple[str, dict]] = []

    def __init__(self, _base_url: str, _api_key: str, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _StubHttpClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, path: str, *, params=None):  # noqa: ANN001
        del params
        _StubHttpClient.get_calls.append(path)
        return {"ok": True, "path": path}

    async def post_json(self, path: str, *, json):  # noqa: ANN001
        _StubHttpClient.post_calls.append((path, dict(json)))
        return {"ok": True, "path": path, "payload": json}


class _StubBadGetClient(_StubHttpClient):
    async def get(self, path: str, *, params=None):  # noqa: ANN001
        del path, params
        return ["unexpected"]


def _client() -> plc_module.PromptLearningClient:
    return plc_module.PromptLearningClient(
        base_url="http://localhost:8080",
        api_key="test-key",
        api_version="v2",
    )


def test_prompt_learning_client_rollout_metrics_and_limiter_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubHttpClient.get_calls = []
    _StubHttpClient.post_calls = []
    monkeypatch.setattr(plc_module, "RustCoreHttpClient", _StubHttpClient)

    metrics_payload = asyncio.run(_client().get_rollout_dispatch_metrics("pl_123"))
    limiter_payload = asyncio.run(_client().get_rollout_limiter_status("pl_123"))

    assert metrics_payload["ok"] is True
    assert limiter_payload["ok"] is True
    assert _StubHttpClient.get_calls == [
        "/v2/offline/jobs/pl_123/queue/rollouts/metrics",
        "/v2/offline/jobs/pl_123/queue/rollouts/limiter-status",
    ]


def test_prompt_learning_client_retry_and_drain_rollout_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubHttpClient.get_calls = []
    _StubHttpClient.post_calls = []
    monkeypatch.setattr(plc_module, "RustCoreHttpClient", _StubHttpClient)

    retry_payload = asyncio.run(
        _client().retry_rollout_dispatch(
            "pl_123",
            "trial_0001:0",
            algorithm_kind="gepa",
        )
    )
    drain_payload = asyncio.run(
        _client().drain_rollout_queue(
            "pl_123",
            cancel_queued=True,
            algorithm_kind="mipro",
        )
    )

    assert retry_payload["ok"] is True
    assert drain_payload["ok"] is True
    assert _StubHttpClient.post_calls == [
        ("/v2/offline/jobs/pl_123/queue/rollouts/trial_0001:0/retry?algorithm_kind=gepa", {}),
        (
            "/v2/offline/jobs/pl_123/queue/rollouts/drain?algorithm_kind=mipro",
            {"cancel_queued": True},
        ),
    ]


def test_prompt_learning_client_retry_rollout_dispatch_rejects_blank_id() -> None:
    with pytest.raises(ValueError, match="dispatch_id is required"):
        asyncio.run(_client().retry_rollout_dispatch("pl_123", "   "))


def test_prompt_learning_client_rollout_metrics_requires_object_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plc_module, "RustCoreHttpClient", _StubBadGetClient)

    with pytest.raises(ValueError, match="Unexpected response structure from rollout metrics endpoint"):
        asyncio.run(_client().get_rollout_dispatch_metrics("pl_123"))
