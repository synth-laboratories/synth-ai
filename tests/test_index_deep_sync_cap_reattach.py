"""SYN-4999: the synchronous DEEP cap refusal reattaches to the durable Search."""

import asyncio

import httpx
import pytest
from synth_ai.errors import SynthError
from synth_ai.options import MutationOptions
from synth_ai.sdk.index import SyncCapReattached
from test_index_retention import (
    HEADERS,
    LEGACY_ZDR,
    SEARCH,
    Recorder,
    async_client,
    error_body,
    options,
    reply,
    result_body,
    snapshot_body,
    sync_client,
)

CAP_REASON = "index_deep_sync_wait_cap_exceeded"


def cap_refusal(*, search_id: str | None = SEARCH, cause: str = CAP_REASON) -> httpx.Response:
    """Wire shape of backend #3599: 504 typed body plus the reattach headers."""
    headers = dict(
        HEADERS,
        **{
            "X-Synth-Refusal-Cause": cause,
            "X-Synth-Refusal-Stage": "index_deep_sync_wait",
            "X-Synth-Deep-Sync-Cap-Seconds": "90",
        },
    )
    if search_id is not None:
        headers["X-Synth-Durable-Search-Id"] = search_id
    return httpx.Response(504, headers=headers, json=error_body("deadline_exceeded"))


def sync_identity(selected):
    """An explicitly bound synchronous ``index.search`` identity (legacy route)."""
    from synth_ai.contracts.operations import SEARCH as SEARCH_OPERATION
    from synth_ai.core.http.request import encode_body
    from synth_ai.sdk.index.search import search_request
    from synth_ai.types.identity import MutationIdentity

    body = search_request("q", selected, True)
    return MutationOptions(
        identity=MutationIdentity.create(
            SEARCH_OPERATION.id, encode_body(body), "/api/v1/index/search", key="existing"
        )
    )


def deep_result() -> dict:
    result = result_body(LEGACY_ZDR)
    result["mode"] = "deep"
    return result


def run_search(recorder: Recorder, selected, mutation, asynchronous: bool):
    if asynchronous:

        async def run():
            client = async_client(recorder)
            try:
                return await client.search("q", options=selected, mutation=mutation)
            finally:
                await client._transport.aclose()

        return asyncio.run(run())
    client = sync_client(recorder)
    try:
        return client.search("q", options=selected, mutation=mutation)
    finally:
        client._transport.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_cap_refusal_reattaches_to_the_named_durable_search(asynchronous):
    completed = snapshot_body("completed", mode="deep", available=True, retention=LEGACY_ZDR)
    recorder = Recorder(
        cap_refusal(), reply(200, completed), reply(200, completed), reply(200, deep_result())
    )
    selected = options(mode="deep", deadline_seconds=300)
    output = run_search(recorder, selected, sync_identity(selected), asynchronous)
    assert output.search_id == SEARCH and output.response == "answer"
    assert output.sync_cap_reattached == SyncCapReattached(
        search_id=SEARCH, cap_seconds=90.0, cause=CAP_REASON, request_id="req-err"
    )
    assert [(request.method, request.url.path) for request in recorder.requests] == [
        ("POST", "/api/v1/index/search"),
        ("GET", f"/api/v1/index/searches/{SEARCH}"),
        ("GET", f"/api/v1/index/searches/{SEARCH}"),
        ("GET", f"/api/v1/index/searches/{SEARCH}/result"),
    ]
    # Reattachment is a read: nothing is resubmitted.
    assert sum(request.method == "POST" for request in recorder.requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_cap_refusal_without_the_search_id_header_raises_the_typed_error(asynchronous):
    recorder = Recorder(cap_refusal(search_id=None))
    selected = options(mode="deep", deadline_seconds=300)
    with pytest.raises(SynthError) as caught:
        run_search(recorder, selected, sync_identity(selected), asynchronous)
    error = caught.value
    assert (error.status, error.code) == (504, "deadline_exceeded")
    assert (error.cause, error.stage) == (CAP_REASON, "index_deep_sync_wait")
    assert error.durable_search_id is None and error.sync_cap_seconds == 90.0
    assert len(recorder.requests) == 1


def test_malformed_search_id_header_is_dropped_and_the_error_propagates():
    recorder = Recorder(cap_refusal(search_id="not-a-uuid"))
    selected = options(mode="deep", deadline_seconds=300)
    with pytest.raises(SynthError) as caught:
        run_search(recorder, selected, sync_identity(selected), False)
    assert caught.value.durable_search_id is None and len(recorder.requests) == 1


def test_other_504_causes_do_not_reattach():
    recorder = Recorder(cap_refusal(cause="index_deep_admission_timeout"))
    selected = options(mode="deep", deadline_seconds=300)
    with pytest.raises(SynthError) as caught:
        run_search(recorder, selected, sync_identity(selected), False)
    assert caught.value.durable_search_id == SEARCH and len(recorder.requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_fast_search_is_unaffected_by_the_reattach_headers(asynchronous):
    recorder = Recorder(cap_refusal())
    selected = options(mode="fast")
    with pytest.raises(SynthError) as caught:
        run_search(recorder, selected, MutationOptions(idempotency_key="k"), asynchronous)
    assert caught.value.status == 504 and len(recorder.requests) == 1
    assert recorder.requests[0].url.path == "/api/v1/index/search"


def test_plain_result_carries_no_reattach_note():
    recorder = Recorder(reply(200, deep_result()))
    selected = options(mode="deep")
    output = run_search(recorder, selected, sync_identity(selected), False)
    assert output.sync_cap_reattached is None
