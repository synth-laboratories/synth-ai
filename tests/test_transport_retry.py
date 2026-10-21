"""Infrastructure faults are classified before bounded Synth body decoding."""

from hashlib import sha256

import httpx
import pytest
from synth_ai.errors import ContractError, OutcomeUncertainError, SynthError, TransportError
from tests.http_fixtures import Exchanges, invoke, page_response

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
async def test_headerless_infrastructure_read_retries_then_succeeds(async_mode, status):
    handler = Exchanges(httpx.Response(status, text="<html>proxy failure</html>"), page_response())
    page = await invoke(async_mode, handler)
    assert page.complete and page.metadata.client_timing.attempts == 2
    assert len(handler.requests) == 2 and len(handler.capability_requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_exhausted_read_reports_retryable_infrastructure_status(async_mode):
    handler = Exchanges(httpx.Response(503, text="deploying", headers={"Retry-After": "0"}))
    with pytest.raises(TransportError) as caught:
        await invoke(async_mode, handler)
    error = caught.value
    assert (error.status, error.retryable, error.stage, error.code) == (
        503,
        True,
        "infrastructure",
        None,
    )
    assert "index.searches.list" in str(error) and "503" in str(error)
    assert error.retry_after_seconds == 0 and len(handler.requests) == 3


@pytest.mark.parametrize("async_mode", [False, True])
async def test_retry_after_is_honored_without_wall_clock_sleep(monkeypatch, async_mode):
    delays = []

    async def asleep(delay):
        delays.append(delay)

    monkeypatch.setattr("synth_ai.core.http.client.time.sleep", delays.append)
    monkeypatch.setattr("synth_ai.core.http.async_client.asyncio.sleep", asleep)
    handler = Exchanges(httpx.Response(429, headers={"Retry-After": "1"}), page_response())
    await invoke(async_mode, handler)
    assert delays == [1]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
async def test_mutation_preserves_exact_identity_and_never_retries(async_mode, status):
    handler = Exchanges(httpx.Response(status, text="upstream failure"))
    with pytest.raises(OutcomeUncertainError) as caught:
        await invoke(async_mode, handler, mutation=True)
    error = caught.value
    assert error.reason == "infrastructure_status" and error.status == status
    assert len(handler.requests) == len(handler.capability_requests) == 1
    request = handler.requests[0]
    assert error.identity.key == request.headers["Idempotency-Key"] == "key"
    assert error.identity.operation == "index.search"
    assert error.identity.payload_sha256 == sha256(request.content).hexdigest()


@pytest.mark.parametrize("async_mode", [False, True])
async def test_synth_typed_503_keeps_typed_refusal_after_retry_budget(async_mode):
    handler = Exchanges(
        *[
            httpx.Response(
                503,
                headers={"X-Synth-Contract": "index.v1"},
                json={
                    "error": {"code": "dependency_unavailable", "request_id": "server-request"},
                },
            )
            for _ in range(3)
        ]
    )
    with pytest.raises(SynthError) as caught:
        await invoke(async_mode, handler)
    assert not isinstance(caught.value, TransportError)
    assert caught.value.code == "dependency_unavailable"
    assert caught.value.request_id == "server-request" and len(handler.requests) == 3


@pytest.mark.parametrize("async_mode", [False, True])
async def test_typed_500_is_not_retried(async_mode):
    handler = Exchanges(
        httpx.Response(
            500,
            headers={"X-Synth-Contract": "index.v1"},
            json={
                "error": {"code": "internal_error", "request_id": "server-request"},
            },
        )
    )
    with pytest.raises(SynthError) as caught:
        await invoke(async_mode, handler)
    assert caught.value.code == "internal_error" and len(handler.requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("mutation", [False, True])
async def test_headerless_200_remains_contract_skew_without_retry(async_mode, mutation):
    handler = Exchanges(httpx.Response(200, json={"unexpected": "body"}))
    expected = OutcomeUncertainError if mutation else ContractError
    with pytest.raises(expected) as caught:
        await invoke(async_mode, handler, mutation=mutation)
    if mutation:
        assert caught.value.reason == "contract_header_missing"
    else:
        assert "upgrade synth-ai" in str(caught.value)
    assert len(handler.requests) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_matching_header_with_invalid_503_body_is_infrastructure(async_mode):
    handler = Exchanges(
        httpx.Response(
            503,
            headers={"X-Synth-Contract": "index.v1"},
            json={
                "unexpected": "not an error receipt",
            },
        ),
        page_response(),
    )
    page = await invoke(async_mode, handler)
    assert page.metadata.client_timing.attempts == 2


class UntrustedBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __iter__(self):
        raise AssertionError("Infrastructure bodies must not be read")
        yield  # pragma: no cover

    async def __aiter__(self):
        raise AssertionError("Infrastructure bodies must not be read")
        yield  # pragma: no cover


@pytest.mark.parametrize("async_mode", [False, True])
async def test_compressed_infrastructure_is_retried_without_reading_body(async_mode):
    handler = Exchanges(
        httpx.Response(502, headers={"Content-Encoding": "gzip"}, stream=UntrustedBody()),
        page_response(),
    )
    page = await invoke(async_mode, handler)
    assert page.metadata.client_timing.attempts == 2


@pytest.mark.parametrize("async_mode", [False, True])
async def test_contract_body_decode_error_is_not_retried(async_mode):
    handler = Exchanges(
        httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json={"invalid": "page"})
    )
    with pytest.raises(ContractError):
        await invoke(async_mode, handler)
    assert len(handler.requests) == 1
