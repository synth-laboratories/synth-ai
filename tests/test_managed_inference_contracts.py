from __future__ import annotations

import httpx
import pytest
from synth_ai.sdk.research.contracts.factory_operations import FactoryCreateRequest
from synth_ai.sdk.research.contracts.managed_inference import (
    ManagedInference,
    ManagedInferenceLimits,
)
from synth_ai.sdk.research.contracts.swarms import SwarmSpec
from synth_ai.sdk.research.managed_inference import ManagedInferenceClient


def test_managed_inference_serializes_across_run_and_factory_contracts() -> None:
    inference = ManagedInference(
        model="gpt-5.3-codex",
        wire_apis=("responses",),
        limits=ManagedInferenceLimits(max_calls=4, max_tokens=100_000),
    )
    expected = {
        "mode": "synth_managed",
        "model": "gpt-5.3-codex",
        "wire_apis": ["responses"],
        "limits": {"max_calls": 4, "max_tokens": 100_000},
    }
    assert SwarmSpec(objective="research", inference=inference).to_wire()["inference"] == expected
    assert FactoryCreateRequest(name="factory", default_child_inference=inference).to_wire()[
        "default_child_inference"
    ] == expected


def test_managed_inference_rejects_invalid_gateway_contracts() -> None:
    with pytest.raises(ValueError, match="model"):
        ManagedInference(model="")
    with pytest.raises(ValueError, match="wire_apis"):
        ManagedInference(model="gpt-5.3-codex", wire_apis=())
    with pytest.raises(ValueError, match="positive"):
        ManagedInferenceLimits(max_calls=0)


def test_workload_client_uses_only_injected_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMR_METERED_INFERENCE_BASE_URL", "https://gateway.example/smr/inference/v1")
    monkeypatch.setenv("SMR_METERED_INFERENCE_API_KEY", "capability")
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured.update(url=url, **kwargs)
        return httpx.Response(200, json={"id": "response-1"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    result = ManagedInferenceClient.from_environment().responses(
        {"model": "gpt-5.3-codex", "input": "hi", "max_output_tokens": 10},
        idempotency_key="request-1",
    )
    assert result["id"] == "response-1"
    assert captured["url"] == "https://gateway.example/smr/inference/v1/responses"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer capability"
    assert headers["Idempotency-Key"] == "request-1"
