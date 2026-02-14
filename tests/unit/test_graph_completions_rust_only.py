from __future__ import annotations

import types

import pytest

from synth_ai.core.errors import PaymentRequiredError
from synth_ai.sdk.graphs.completions import (
    GraphCompletionsAsyncClient,
    GraphCompletionsSyncClient,
)


class _FakeRustClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def graph_complete(self, _payload):  # type: ignore[no-untyped-def]
        return self._payload


class _FakeRust402Client:
    def graph_complete(self, _payload):  # type: ignore[no-untyped-def]
        raise PaymentRequiredError(
            status=402,
            url="https://api.usesynth.ai/api/graphs/completions",
            message="payment_required",
            body_snippet=None,
            detail={"detail": {"x402": {"challenge": {"claims": {"challenge_id": "abc"}}}}},
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_run_uses_rust_only(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GraphCompletionsAsyncClient("https://api.usesynth.ai", "sk-test")
    client._rust = types.SimpleNamespace(
        resolve_graph_job_id=lambda job_id, graph: job_id,
        SynthClient=lambda *args, **kwargs: _FakeRustClient(
            {"output": {"outcome_review": {"total": 0.5}}}
        ),
    )

    # If any httpx fallback still exists, fail loudly.
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    result = await client.run(input_data={"foo": "bar"}, job_id="graph-1")
    assert isinstance(result, dict)
    assert result.get("output", {}).get("outcome_review", {}).get("total") == 0.5


@pytest.mark.unit
def test_sync_run_uses_rust_only(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GraphCompletionsSyncClient("https://api.usesynth.ai", "sk-test")
    client._rust = types.SimpleNamespace(
        resolve_graph_job_id=lambda job_id, graph: job_id,
        SynthClient=lambda *args, **kwargs: _FakeRustClient(
            {"output": {"outcome_review": {"total": 0.25}}}
        ),
    )

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    result = client.run(input_data={"foo": "bar"}, job_id="graph-2")
    assert result.output.get("outcome_review", {}).get("total") == 0.25


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_run_propagates_payment_required() -> None:
    client = GraphCompletionsAsyncClient("https://api.usesynth.ai", "sk-test")
    client._rust = types.SimpleNamespace(
        resolve_graph_job_id=lambda job_id, graph: job_id,
        SynthClient=lambda *args, **kwargs: _FakeRust402Client(),
    )

    with pytest.raises(PaymentRequiredError) as exc_info:
        await client.run(input_data={"foo": "bar"}, job_id="graph-3")

    assert exc_info.value.status == 402
    assert isinstance(exc_info.value.challenge, dict)


@pytest.mark.unit
def test_sync_run_propagates_payment_required() -> None:
    client = GraphCompletionsSyncClient("https://api.usesynth.ai", "sk-test")
    client._rust = types.SimpleNamespace(
        resolve_graph_job_id=lambda job_id, graph: job_id,
        SynthClient=lambda *args, **kwargs: _FakeRust402Client(),
    )

    with pytest.raises(PaymentRequiredError) as exc_info:
        client.run(input_data={"foo": "bar"}, job_id="graph-4")

    assert exc_info.value.status == 402
    assert isinstance(exc_info.value.challenge, dict)

