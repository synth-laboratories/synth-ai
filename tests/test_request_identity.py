"""Support identity survives retries, preflight refusals, and server errors."""

import inspect
import pickle
import re

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient, errors
from synth_ai.contracts.operations import LIST_SEARCHES
from synth_ai.errors import RateLimitError, SynthError, TransportError, UnsupportedOperationError
from synth_ai.options import ClientOptions, ReadOptions
from tests.http_fixtures import Exchanges, invoke, page_response

pytestmark = pytest.mark.unit

ERRORS = [
    kind for _, kind in inspect.getmembers(errors, inspect.isclass) if issubclass(kind, SynthError)
]


@pytest.mark.parametrize("kind", ERRORS)
@pytest.mark.parametrize("details", [False, True])
def test_error_text_repr_and_pickle_preserve_safe_details(kind, details):
    fields = (
        {
            "code": "failed_check",
            "status": 409,
            "request_id": "server-request",
            "client_request_id": "client-request",
            "operation": "index.search",
            "stage": "admission",
            "cause": "rights",
        }
        if details
        else {}
    )
    error = kind("Useful failure message", **fields)
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is kind and restored.args == error.args
    assert restored.__dict__ == error.__dict__
    assert "Useful failure message" in str(error) and "Useful failure message" in repr(error)
    assert repr(error).startswith(kind.__name__ + "(")
    if details:
        for name, value in fields.items():
            if name != "client_request_id":
                assert f"{name}={value}" in str(error)
                assert f"{name}={value!r}" in repr(error)
    else:
        assert str(error) == "Useful failure message"
        assert repr(error) == f"{kind.__name__}('Useful failure message')"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_request_id_is_stable_across_retries_with_attempt_and_agent(async_mode):
    handler = Exchanges(httpx.Response(503), httpx.Response(503), page_response())
    await invoke(async_mode, handler)
    ids = [request.headers["X-Request-Id"] for request in handler.requests]
    assert len(set(ids)) == 1 and re.fullmatch(r"[0-9a-f]{32}", ids[0])
    assert [r.headers["X-Synth-Attempt"] for r in handler.requests] == ["1", "2", "3"]
    for request in [*handler.requests, *handler.capability_requests]:
        assert re.fullmatch(
            r"synth-ai-python/\d+\.\d+\.\d+\S* python/3\.\d+\.\d+", request.headers["User-Agent"]
        )
    assert handler.capability_requests[0].headers["X-Request-Id"] != ids[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_explicit_request_id_and_borrowed_user_agent_win(async_mode):
    handler = Exchanges(httpx.Response(503), page_response())
    await invoke(
        async_mode,
        handler,
        read_options=ReadOptions(request_id="caller-id"),
        network_headers={"User-Agent": "my-explicit-client/1"},
    )
    assert all(r.headers["X-Request-Id"] == "caller-id" for r in handler.requests)
    assert all(
        r.headers["User-Agent"] == "my-explicit-client/1"
        for r in [*handler.requests, *handler.capability_requests]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_server_and_client_ids_survive_typed_error(async_mode):
    handler = Exchanges(
        *[
            httpx.Response(
                429,
                headers={"X-Synth-Contract": "index.v1"},
                json={
                    "error": {"code": "too_many_requests", "request_id": "server-request"},
                },
            )
            for _ in range(3)
        ]
    )
    with pytest.raises(RateLimitError) as caught:
        await invoke(async_mode, handler)
    error = caught.value
    assert error.request_id == "server-request"
    assert error.client_request_id == handler.requests[0].headers["X-Request-Id"]
    assert error.operation == "index.searches.list"
    assert "index.searches.list refused: too_many_requests" in str(error)
    for value in ["too_many_requests", "429", "server-request", "index.searches.list"]:
        assert value in str(error) and value in repr(error)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_infrastructure_error_keeps_original_wire_identity(async_mode):
    handler = Exchanges(httpx.Response(503))
    with pytest.raises(TransportError) as caught:
        await invoke(async_mode, handler)
    error = caught.value
    assert (
        error.request_id == error.client_request_id == handler.requests[0].headers["X-Request-Id"]
    )
    assert error.operation == "index.searches.list"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_capability_exchange_failure_names_its_actual_wire_identity(async_mode):
    requests = []

    def fail(request):
        requests.append(request)
        return httpx.Response(503)

    with pytest.raises(TransportError) as caught:
        await invoke(async_mode, fail)
    assert all(r.url.path.endswith("/capabilities") for r in requests)
    assert caught.value.operation == "capabilities.get"
    assert caught.value.client_request_id == requests[0].headers["X-Request-Id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_unavailable_operation_preflight_is_identified(async_mode):
    import json

    from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
    from tests.http_fixtures import capabilities_body

    body = capabilities_body()
    body["operations"] = []
    snapshot = {k: v for k, v in body.items() if k not in {"metadata", "capability_sha256"}}
    body["capability_sha256"] = snapshot_sha256(
        CapabilitySnapshot.model_validate_json(json.dumps(snapshot))
    )
    requests = []

    def refuse(request):
        requests.append(request)
        return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=body)

    with pytest.raises(UnsupportedOperationError) as caught:
        await invoke(async_mode, refuse)
    assert caught.value.operation == "index.searches.list"
    assert re.fullmatch(r"[0-9a-f]{32}", caught.value.client_request_id)
    assert "index.searches.list" in str(caught.value)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_stream_exchange_error_has_operation_and_request_id(async_mode):
    handler = Exchanges(httpx.Response(200, content=b"body without negotiated deployment headers"))
    from synth_ai.errors import ContractError

    if async_mode:
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(handler)) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            with pytest.raises(ContractError) as caught:
                await client._transport._open_stream(
                    LIST_SEARCHES,
                    path=LIST_SEARCHES.path,
                    query={},
                    remaining_seconds=1,
                    digest="a" * 64,
                )
    else:
        with (
            httpx.Client(transport=httpx.MockTransport(handler)) as network,
            SynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(http_client=network),
            ) as client,
            pytest.raises(ContractError) as caught,
        ):
            client._transport._open_stream(
                LIST_SEARCHES,
                path=LIST_SEARCHES.path,
                query={},
                remaining_seconds=1,
                digest="a" * 64,
            )
    assert caught.value.operation == LIST_SEARCHES.id
    assert caught.value.request_id == handler.requests[0].headers["X-Request-Id"]
    assert caught.value.client_request_id == caught.value.request_id
