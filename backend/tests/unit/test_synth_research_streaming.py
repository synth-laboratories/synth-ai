from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
import pytest

from app.routes import synth_research


class _DummyResponse:
    def __init__(self, data: Dict[str, Any], status: int = 200) -> None:
        self._data = data
        self.status_code = status
        self.headers: Dict[str, str] = {}

    def json(self) -> Dict[str, Any]:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=None, response=httpx.Response(self.status_code)
            )


class _DummyStream:
    def __init__(self, lines: List[str], status: int = 200) -> None:
        self._lines = lines
        self.status_code = status
        self.headers: Dict[str, str] = {}

    async def __aenter__(self) -> "_DummyStream":
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=None, response=httpx.Response(self.status_code)
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=None, response=httpx.Response(self.status_code)
            )

    async def aiter_lines(self):
        for line in self._lines:
            await asyncio.sleep(0)
            yield line


class _DummyAsyncClient:
    def __init__(self, response: _DummyResponse | None = None, stream: _DummyStream | None = None) -> None:
        self._response = response
        self._stream = stream

    async def __aenter__(self) -> "_DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, *args, **kwargs) -> _DummyResponse:
        assert self._response is not None, "Response not configured"
        return self._response

    def stream(self, *args, **kwargs) -> _DummyStream:
        assert self._stream is not None, "Stream not configured"
        return self._stream


@pytest.mark.asyncio
async def test_call_llm_api_normalizes_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        synth_research.SUPPORTED_MODELS,
        "synth-small",
        {
            "backend": "gpt-5-nano",
            "provider": "openai",
            "api_url": "https://example.test/chat/completions",
        },
    )

    dummy = _DummyResponse({"model": "gpt-5-nano", "choices": []})

    monkeypatch.setattr(
        synth_research.httpx, "AsyncClient", lambda *args, **kwargs: _DummyAsyncClient(response=dummy)
    )

    result = await synth_research._call_llm_api(
        {"model": "synth-small", "messages": [], "max_tokens": 128},
        api_key="sk-test",
        model_name="synth-small",
        timeout_s=5,
    )

    assert result["model"] == "synth-small"


@pytest.mark.asyncio
async def test_stream_llm_api_rewrites_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        synth_research.SUPPORTED_MODELS,
        "synth-small",
        {
            "backend": "gpt-5-nano",
            "provider": "openai",
            "api_url": "https://example.test/chat/completions",
        },
    )

    payload_chunks = [
        "data: {\"id\":\"abc\",\"model\":\"gpt-5-nano\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n",
        "data: [DONE]\n",
    ]

    stream = _DummyStream(payload_chunks)
    monkeypatch.setattr(
        synth_research.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _DummyAsyncClient(stream=stream),
    )

    chunks = []
    async for chunk in synth_research._stream_llm_api(
        {"model": "synth-small", "messages": [], "stream": True},
        api_key="sk-test",
        model_name="synth-small",
        timeout_s=5,
    ):
        chunks.append(chunk)

    assert any('"model":"synth-small"' in c for c in chunks)

