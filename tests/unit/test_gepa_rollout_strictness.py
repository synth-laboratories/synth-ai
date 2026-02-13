from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from synth_ai.gepa.api import _call_llm
from synth_ai.sdk.container._impl.contracts import RolloutRequest, RolloutResponse
from synth_ai.sdk.container._impl.server import ContainerConfig, create_container


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.com")
            response = httpx.Response(
                self.status_code,
                request=request,
                json=self._payload,
            )
            raise httpx.HTTPStatusError("upstream error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: dict, headers: dict[str, str]) -> _FakeResponse:
        self._calls.append(url)
        if "api.openai.com" in url:
            return _FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "fallback should not be used"}}]},
            )
        return _FakeResponse(status_code=503, payload={"error": "temporary failure"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gepa_call_llm_does_not_bypass_configured_inference_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "synth_ai.gepa.api.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(calls),
    )

    with pytest.raises(RuntimeError, match="configured inference_url"):
        await _call_llm(
            system_prompt="system",
            user_prompt="user",
            policy_config={
                "inference_url": "https://api.usesynth.ai/v1/chat/completions",
                "model": "gpt-4.1-mini",
            },
            request_headers={},
        )

    assert calls == ["https://api.usesynth.ai/v1/chat/completions"]


@pytest.mark.unit
def test_rollout_request_and_response_require_non_empty_trace_correlation_id() -> None:
    with pytest.raises(ValidationError):
        RolloutRequest(trace_correlation_id="   ", env={}, policy={})
    with pytest.raises(ValidationError):
        RolloutResponse(trace_correlation_id="", reward_info={"outcome_reward": 1.0})

    request = RolloutRequest(trace_correlation_id=" corr-123 ", env={}, policy={})
    response = RolloutResponse(
        trace_correlation_id=" corr-456 ",
        reward_info={"outcome_reward": 1.0},
    )
    assert request.trace_correlation_id == "corr-123"
    assert response.trace_correlation_id == "corr-456"


@pytest.mark.unit
def test_rollout_endpoint_rejects_invalid_mapping_response() -> None:
    async def _rollout(_rollout_request, _request):
        return {"trace_correlation_id": "corr-1"}

    app = create_container(
        ContainerConfig(
            app_id="unit-test-app",
            name="Unit Test App",
            description="Container for rollout strictness tests.",
            provide_taskset_description=lambda: {"name": "demo"},
            provide_task_instances=lambda _seeds: [],
            rollout=_rollout,
            require_api_key=False,
            ensure_container_auth=False,
            expose_debug_env=False,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/rollout",
        json={
            "trace_correlation_id": "corr-1",
            "env": {},
            "policy": {},
        },
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]["error"]["code"] == "invalid_rollout_response"
