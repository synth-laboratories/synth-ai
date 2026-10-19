"""Single-request strict events keep original identity and never reconnect."""

import asyncio
import json

import httpx
import pytest
from synth_ai.contracts.operations import DELIVER_SEARCH_EVENTS
from synth_ai.core.http.stream import FrameDecoder
from synth_ai.errors import ContractError, OutcomeUncertainError, PermissionDeniedError
from synth_ai.options import MutationOptions
from synth_ai.sdk.index.types.search import KeyedSearchRequest, StrictSearchEventRequest
from test_index_retention import (
    LEGACY_ZDR,
    METADATA,
    SEARCH,
    Recorder,
    async_client,
    options,
    result_body,
    sync_client,
)


def intent():
    return StrictSearchEventRequest(
        creation_key="original",
        request=KeyedSearchRequest(
            query="private intent", options=options(mode="deep", retention="zdr_strict")
        ),
    )


def wire(*, terminal=True, defect=None):
    progress = {
        "search_id": SEARCH,
        "event_id": SEARCH + ".2",
        "sequence": 2,
        "kind": "progress",
        "stage": "working",
        "search_calls": 1,
        "read_calls": 0,
        "input_tokens": 5,
        "output_tokens": 2,
    }
    result = result_body(dict(LEGACY_ZDR, mode="zdr_strict"))
    result["mode"] = "deep"
    event = {
        "event_id": SEARCH + ".5",
        "search_id": SEARCH,
        "sequence": 5,
        "kind": "completed",
        "result": result,
    }
    page = {"metadata": METADATA, "items": [event], "complete": True, "next_cursor": None}
    if defect == "search":
        event["search_id"] = "foreign"
    if defect == "sequence":
        event["sequence"] = 1
    if defect == "content":
        progress["query"] = "private"
    frames = [("progress", progress)] + ([("terminal", page)] if terminal else [])
    return b"".join(
        b"event: "
        + kind.encode()
        + b"\ndata: "
        + json.dumps(value, separators=(",", ":")).encode()
        + b"\n\n"
        for kind, value in frames
    )


def reply(body):
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream", "X-Synth-Contract": "index.v1"},
        content=body,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_typed_progress_and_terminal_share_one_post(asynchronous):
    recorder = Recorder(reply(wire()))
    if asynchronous:

        async def run():
            client = async_client(recorder)
            run = await _async_handle(client)
            async with run.deliver_events(
                original_request=intent().request, creation_key="original"
            ) as events:
                values = [event async for event in events]
                assert [value.kind for value in values] == ["progress", "completed"]
                assert events.result.response == "answer"
            assert not client._transport._streams and not client._transport._responses
            await client._transport.aclose()

        asyncio.run(run())
    else:
        client = sync_client(recorder)
        from synth_ai.sdk.index.search_run import SearchRun

        run = SearchRun(client._transport, SEARCH, expected_retention="zdr_strict")
        with run.deliver_events(
            original_request=intent().request, creation_key="original"
        ) as events:
            assert [value.kind for value in events] == ["progress", "completed"]
            assert events.result.response == "answer"
        assert not client._transport._streams and not client._transport._responses
        client._transport.close()
    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST" and request.headers["Accept"] == "text/event-stream"
    assert request.headers["Idempotency-Key"]
    assert request.headers["X-Synth-Intent-SHA256"]
    assert json.loads(request.content)["creation_key"] == "original"


async def _async_handle(client):
    from synth_ai.sdk.index.search_run import AsyncSearchRun

    return AsyncSearchRun(client._transport, SEARCH, expected_retention="zdr_strict")


@pytest.mark.parametrize("defect", [None, "search", "sequence", "content"])
def test_lost_or_malformed_stream_never_reconnects_or_regenerates(defect):
    recorder = Recorder(reply(wire(terminal=defect is not None, defect=defect)))
    client = sync_client(recorder)
    from synth_ai.sdk.index.search_run import SearchRun

    run = SearchRun(client._transport, SEARCH, expected_retention="zdr_strict")
    with (
        pytest.raises((OutcomeUncertainError, ContractError)) as error,
        run.deliver_events(
            original_request=intent().request,
            creation_key="original",
            mutation=MutationOptions(idempotency_key="delivery"),
        ) as events,
    ):
        list(events)
    assert error.value.identity.key == "delivery"
    assert len(recorder.requests) == 1
    assert not client._transport._responses and not client._transport._streams
    client._transport.close()


def test_definite_pre_stream_refusal_preserves_typed_error():
    recorder = Recorder(
        httpx.Response(
            403,
            headers={"X-Synth-Contract": "index.v1"},
            json={"error": {"code": "denied", "request_id": "req", "outcome": "not_accepted"}},
        )
    )
    client = sync_client(recorder)
    with (
        pytest.raises(PermissionDeniedError),
        client._transport.mutation_events(
            DELIVER_SEARCH_EVENTS, path=f"/index/searches/{SEARCH}/events", body=intent()
        ) as events,
    ):
        list(events)
    assert len(recorder.requests) == 1
    client._transport.close()


def test_decoder_accepts_split_frames_and_rejects_oversize_and_extra_fields():
    decoder = FrameDecoder(100)
    chunks = [b": keepalive\n\nevent: pro", b'gress\ndata: {"n":1}\n', b"\n"]
    assert [(frame.event, frame.data) for chunk in chunks for frame in decoder.feed(chunk)] == [
        ("progress", b'{"n":1}')
    ]
    with pytest.raises(ContractError):
        list(FrameDecoder(10).feed(b"x" * 11))
    with pytest.raises(ContractError):
        list(FrameDecoder(100).feed(b"event: progress\nid: reconnect\ndata: {}\n\n"))


def test_deployment_pin_change_keeps_exact_delivery_identity_and_no_retry():
    from types import SimpleNamespace

    recorder = Recorder(reply(wire()))
    client = sync_client(recorder)
    pins = SimpleNamespace(
        schema_sha256="a" * 64, capability_sha256="b" * 64, release_manifest_sha256="c" * 64
    )
    client._transport._negotiate = lambda *args: pins
    with (
        pytest.raises(OutcomeUncertainError) as error,
        client._transport.mutation_events(
            DELIVER_SEARCH_EVENTS,
            path=f"/index/searches/{SEARCH}/events",
            body=intent(),
            options=MutationOptions(idempotency_key="delivery"),
        ) as events,
    ):
        list(events)
    assert error.value.identity.key == "delivery" and len(recorder.requests) == 1
    assert not client._transport._responses
    client._transport.close()


def test_cleanup_failure_preserves_primary_typed_refusal_and_recovery_identity():
    recorder = Recorder(
        httpx.Response(
            403,
            headers={"X-Synth-Contract": "index.v1"},
            json={"error": {"code": "denied", "request_id": "req", "outcome": "not_accepted"}},
        )
    )
    client = sync_client(recorder)
    original = client._transport._release_response

    def release(response):
        original(response)
        raise OSError("fixture cleanup failure")

    client._transport._release_response = release
    with (
        pytest.raises(PermissionDeniedError) as error,
        client._transport.mutation_events(
            DELIVER_SEARCH_EVENTS,
            path=f"/index/searches/{SEARCH}/events",
            body=intent(),
            options=MutationOptions(idempotency_key="delivery"),
        ) as events,
    ):
        list(events)
    assert "HTTP response cleanup also failed" in error.value.__notes__
    assert not client._transport._responses
    client._transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_async_cancellation_closes_response_and_retains_mutation_identity(cancel):
    from synth_ai.errors import MutationCancelledError

    closed, waiting = asyncio.Event(), asyncio.Event()

    class Bytes(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield wire(terminal=False)
            waiting.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    recorder = Recorder(
        httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream", "X-Synth-Contract": "index.v1"},
            stream=Bytes(),
        )
    )
    client = async_client(recorder)
    async with client._transport.mutation_events(
        DELIVER_SEARCH_EVENTS,
        path=f"/index/searches/{SEARCH}/events",
        body=intent(),
        options=MutationOptions(idempotency_key="delivery"),
    ) as events:
        iterator = events.__aiter__()
        assert (await anext(iterator)).event == "progress"
        owner = asyncio.create_task(anext(iterator))
        await waiting.wait()
        if cancel:
            owner.cancel()
        else:
            await client._transport.aclose()
        with pytest.raises(MutationCancelledError) as error:
            await owner
        assert error.value.identity.key == "delivery"
    assert closed.is_set() and not client._transport._responses and not client._transport._streams
    assert len(recorder.requests) == 1
    await client._transport.aclose()
