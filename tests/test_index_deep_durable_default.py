"""SYN-4999: keyed DEEP uses one durable admission and bounded observation."""

import asyncio

import pytest
from synth_ai.errors import RequestTimeoutError
from synth_ai.options import MutationOptions
from test_index_retention import (
    LEGACY_ZDR,
    SEARCH,
    Recorder,
    async_client,
    options,
    reply,
    result_body,
    snapshot_body,
    sync_client,
)
from test_strict_event_stream import reply as stream_reply
from test_strict_event_stream import wire


@pytest.mark.parametrize("asynchronous", [False, True])
def test_deep_search_creates_then_reads_the_same_durable_search(asynchronous):
    admission = snapshot_body(mode="deep", retention=LEGACY_ZDR)
    completed = snapshot_body("completed", mode="deep", available=True, retention=LEGACY_ZDR)
    result = result_body(LEGACY_ZDR)
    result["mode"] = "deep"
    recorder = Recorder(reply(200, admission), reply(200, completed), reply(200, result))
    selected = options(mode="deep", deadline_seconds=300)
    if asynchronous:

        async def run():
            client = async_client(recorder)
            try:
                return await client.search(
                    "q", options=selected, mutation=MutationOptions(idempotency_key="original")
                )
            finally:
                await client._transport.aclose()

        output = asyncio.run(run())
    else:
        client = sync_client(recorder)
        try:
            output = client.search(
                "q", options=selected, mutation=MutationOptions(idempotency_key="original")
            )
        finally:
            client._transport.close()
    assert output.search_id == SEARCH and output.response == "answer"
    assert [(request.method, request.url.path) for request in recorder.requests] == [
        ("POST", "/api/v1/index/searches"),
        ("GET", f"/api/v1/index/searches/{SEARCH}"),
        ("GET", f"/api/v1/index/searches/{SEARCH}/result"),
    ]
    assert recorder.requests[0].headers["Idempotency-Key"] == "original"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_strict_deep_search_uses_single_live_delivery_stream(asynchronous):
    strict = dict(LEGACY_ZDR, mode="zdr_strict", content_state="unavailable")
    recorder = Recorder(
        reply(200, snapshot_body(mode="deep", retention=strict)), stream_reply(wire())
    )
    selected = options(mode="deep", retention="zdr_strict")
    if asynchronous:

        async def run():
            client = async_client(recorder)
            try:
                return await client.search(
                    "private intent",
                    options=selected,
                    mutation=MutationOptions(idempotency_key="original"),
                )
            finally:
                await client._transport.aclose()

        output = asyncio.run(run())
    else:
        client = sync_client(recorder)
        try:
            output = client.search(
                "private intent",
                options=selected,
                mutation=MutationOptions(idempotency_key="original"),
            )
        finally:
            client._transport.close()
    assert output.search_id == SEARCH and output.response == "answer"
    assert [(request.method, request.url.path) for request in recorder.requests] == [
        ("POST", "/api/v1/index/searches"),
        ("POST", f"/api/v1/index/searches/{SEARCH}/events"),
    ]
    assert recorder.requests[0].headers["Idempotency-Key"] == "original"


def test_observation_timeout_keeps_durable_search_identity(monkeypatch):
    client = sync_client(Recorder(reply(200, snapshot_body(mode="deep", retention=LEGACY_ZDR))))
    from synth_ai.sdk.index.search_run import SearchRun

    def timed_out(self, *, observation, raise_on_failure=True):
        raise RequestTimeoutError("timed out")

    monkeypatch.setattr(SearchRun, "wait", timed_out)
    try:
        with pytest.raises(RequestTimeoutError, match=SEARCH):
            client.search("q", options=options(mode="deep"))
    finally:
        client._transport.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_failed_deep_search_returns_the_terminal_result(asynchronous):
    from test_search_wait_terminal import result, snapshot

    admission = snapshot_body(mode="deep", retention=LEGACY_ZDR)
    failed = snapshot("failed").model_dump(mode="json")
    terminal = result(
        "failed", failure={"code": "agents_unavailable", "retryable": True}
    ).model_dump(mode="json")
    recorder = Recorder(reply(200, admission), reply(200, failed), reply(200, terminal))
    if asynchronous:

        async def run():
            client = async_client(recorder)
            try:
                return await client.search("q", options=options(mode="deep"))
            finally:
                await client._transport.aclose()

        output = asyncio.run(run())
    else:
        client = sync_client(recorder)
        try:
            output = client.search("q", options=options(mode="deep"))
        finally:
            client._transport.close()
    assert output.status == "failed" and output.failure.code == "agents_unavailable"


def test_preexisting_sync_mutation_identity_keeps_original_route():
    from synth_ai.contracts.operations import SEARCH as SEARCH_OPERATION
    from synth_ai.core.http.request import encode_body
    from synth_ai.sdk.index.search import search_request
    from synth_ai.types.identity import MutationIdentity

    selected = options(mode="deep")
    body = search_request("q", selected, True)
    identity = MutationIdentity.create(
        SEARCH_OPERATION.id, encode_body(body), "/api/v1/index/search", key="existing"
    )
    result = result_body(LEGACY_ZDR)
    result["mode"] = "deep"
    recorder = Recorder(reply(200, result))
    client = sync_client(recorder)
    try:
        output = client.search("q", options=selected, mutation=MutationOptions(identity=identity))
    finally:
        client._transport.close()
    assert output.search_id == SEARCH
    assert [(request.method, request.url.path) for request in recorder.requests] == [
        ("POST", "/api/v1/index/search")
    ]
    assert recorder.requests[0].headers["Idempotency-Key"] == "existing"


def test_observation_respects_explicit_client_timeout():
    import time

    from synth_ai.sdk.index.client import _deep_observation

    observation = _deep_observation(
        options(mode="deep", deadline_seconds=300), None, time.monotonic(), SEARCH, 20
    )
    assert 0 < observation.timeout_seconds <= 20
