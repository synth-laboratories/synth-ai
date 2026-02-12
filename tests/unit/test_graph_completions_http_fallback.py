from __future__ import annotations

import types

import pytest

from synth_ai.sdk.graphs.completions import (
    GraphCompletionsAsyncClient,
    GraphCompletionsSyncClient,
)


class _FakeRustParseErrorClient:
    def graph_complete(self, _payload):
        raise ValueError(
            "internal error: json parse error: invalid type: map, expected i64"
        )


class _FakeAsyncResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _FakeAsyncHttpClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeAsyncHttpClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, *_args, **_kwargs) -> _FakeAsyncResponse:
        return _FakeAsyncResponse(self._payload)


class _FakeSyncResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _FakeSyncHttpClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeSyncHttpClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, *_args, **_kwargs) -> _FakeSyncResponse:
        return _FakeSyncResponse(self._payload)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_run_falls_back_to_http_on_rust_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_payload = {
        "output": {"outcome_review": {"total": 0.5}},
        "usage": [
            {
                "model": "gpt-4.1-mini",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        ],
    }

    client = GraphCompletionsAsyncClient("https://api.usesynth.ai", "sk-test")
    client._rust = types.SimpleNamespace(
        resolve_graph_job_id=lambda job_id, graph: job_id,
        SynthClient=lambda *args, **kwargs: _FakeRustParseErrorClient(),
    )
    monkeypatch.setattr(
        "synth_ai.sdk.graphs.completions.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncHttpClient(fallback_payload),
    )

    result = await client.run(input_data={"foo": "bar"}, job_id="graph-1")
    assert isinstance(result, dict)
    assert result.get("output", {}).get("outcome_review", {}).get("total") == 0.5


@pytest.mark.unit
def test_sync_run_falls_back_to_http_on_rust_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_payload = {
        "output": {"outcome_review": {"total": 0.25}},
        "usage": [
            {
                "model": "gpt-4.1-mini",
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
            }
        ],
    }

    client = GraphCompletionsSyncClient("https://api.usesynth.ai", "sk-test")
    client._rust = types.SimpleNamespace(
        resolve_graph_job_id=lambda job_id, graph: job_id,
        SynthClient=lambda *args, **kwargs: _FakeRustParseErrorClient(),
    )
    monkeypatch.setattr(
        "synth_ai.sdk.graphs.completions.httpx.Client",
        lambda *args, **kwargs: _FakeSyncHttpClient(fallback_payload),
    )

    result = client.run(input_data={"foo": "bar"}, job_id="graph-2")
    assert result.output.get("outcome_review", {}).get("total") == 0.25
