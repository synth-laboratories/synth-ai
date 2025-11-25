from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from synth_ai.sdk.inference.client import InferenceClient

pytestmark = pytest.mark.unit


class DummyHTTPClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    async def __aenter__(self) -> "DummyHTTPClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post_json(self, url: str, json: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((url, json))
        return {"id": "chatcmpl-123"}


@pytest.mark.asyncio
async def test_create_chat_completion_valid_model(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_clients: List[DummyHTTPClient] = []

    def _factory(*args: Any, **kwargs: Any) -> DummyHTTPClient:
        client = DummyHTTPClient(*args, **kwargs)
        dummy_clients.append(client)
        return client

    monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", _factory)

    client = InferenceClient(base_url="https://synth", api_key="sk-test")
    response = await client.create_chat_completion(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response == {"id": "chatcmpl-123"}
    assert dummy_clients, "HTTP client should be instantiated"
    url, payload = dummy_clients[0].calls[0]
    assert url == "/api/inference/v1/chat/completions"
    assert payload["model"] == "Qwen/Qwen3-0.6B"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["thinking_budget"] == 256


@pytest.mark.asyncio
async def test_create_chat_completion_rejects_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def _factory(*args: Any, **kwargs: Any) -> DummyHTTPClient:  # pragma: no cover - should not run
        raise AssertionError("HTTP client should not be constructed for invalid model")

    monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", _factory)

    client = InferenceClient(base_url="https://synth", api_key="sk-test")
    with pytest.raises(ValueError):
        await client.create_chat_completion(
            model="Unknown/Model",
            messages=[{"role": "user", "content": "hi"}],
        )

