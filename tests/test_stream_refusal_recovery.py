"""Server-marked retryable feed refusals reconnect from the checkpoint; others propagate."""

import json

import httpx
import pytest
from test_observer import Feed, client, response
from test_routing_lifecycle import event, snapshot

from synth_ai import SynthClient
from synth_ai._retry import RetryPolicy, stream_refusal_retryable
from synth_ai._stream import StreamCheckpoint
from synth_ai.errors import SwarmsAPIError, TransportFailureError, local_problem


def problem(code="event_owner_unavailable", status=503, **extra):
    return dict(
        type="about:blank",
        title=code,
        detail=code,
        status=status,
        instance="urn:test",
        code=code,
        service="core",
        stage="events",
        outcome="not_accepted",
        **extra,
    )


def data(item, cursor=None):
    prefix = "id: " + cursor + "\n" if cursor is not None else ""
    return (prefix + "data: " + json.dumps(item) + "\n\n").encode()


def error_frame(body):
    return ("event: error\ndata: " + json.dumps(body) + "\n\n").encode()


def refusing(first, refusal, where):
    """A feed that delivers `first`, then refuses in-stream or on the next open."""
    feed = Feed()
    feed.queue.put_nowait(first)
    if where == "in-stream":
        feed.queue.put_nowait(error_frame(refusal))
    else:
        feed.queue.put_nowait(httpx.ReadError("lost"))
    return feed


RETRYABLE = [
    pytest.param(problem(), id="legacy-owner-unavailable"),
    pytest.param(problem("owner_handoff", 502, retryable=True), id="explicit-retryable"),
]


def test_only_server_marked_refusals_are_retryable():
    from synth_ai._generated.models import Problem

    assert stream_refusal_retryable(Problem.model_validate(problem()))
    assert stream_refusal_retryable(Problem.model_validate(problem("x", 500, retryable=True)))
    assert not stream_refusal_retryable(Problem.model_validate(problem(retryable=False)))
    assert not stream_refusal_retryable(Problem.model_validate(problem("refused", 503)))
    assert not stream_refusal_retryable(Problem.model_validate(problem(status=502)))
    assert not stream_refusal_retryable(local_problem("event_owner_unavailable", "events"))


@pytest.mark.parametrize("refusal", RETRYABLE)
@pytest.mark.parametrize("where", ["in-stream", "open"])
async def test_wait_survives_retryable_refusal_and_rereads(contract, refusal, where):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    cursors, reads_after_reconnect = [], []
    first = refusing(data(event(state["run_id"], "A"), "8"), refusal, where)

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            cursors.append(request.headers.get("last-event-id"))
            if len(cursors) == 1:
                return response(first)
            if where == "open" and len(cursors) == 2:
                return httpx.Response(refusal["status"], json=refusal)
            state.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
            return response(Feed())
        if len(cursors) > 1:
            reads_after_reconnect.append(dict(state))
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http, attempts=3) as api:
            finished = await api.swarms.wait(state["run_id"], poll_interval=0.001)
    assert finished.finished
    # The milestone is proven by a snapshot read after the reconnect.
    assert reads_after_reconnect and reads_after_reconnect[-1]["cleanup_complete"]
    assert cursors[0] is None and all(cursor == "8" for cursor in cursors[1:])
    assert first.closed


@pytest.mark.parametrize("refusal", RETRYABLE)
async def test_follow_survives_retryable_refusal_without_duplicates(contract, refusal):
    state = snapshot(contract, execution_closed=False)
    one, two = event(state["run_id"], "A"), event(state["run_id"], "B")
    cursors = []

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            cursors.append(request.headers.get("last-event-id"))
            if len(cursors) == 1:
                feed = Feed()
                feed.queue.put_nowait(data(one, "8"))
                feed.queue.put_nowait(error_frame(refusal))
                return response(feed)
            state.update(execution_closed=True)
            # The owner replays the committed frame; the checkpoint suppresses it.
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=data(one, "8") + data(two, "9"),
            )
        return httpx.Response(200, json=state)

    checkpoint = StreamCheckpoint()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http, attempts=3) as api:
            received = [
                item.event_id
                async for item in api.swarms.follow(state["run_id"], checkpoint=checkpoint)
            ]
    assert received == [one["event_id"], two["event_id"]]
    assert cursors == [None, "8"] and checkpoint.last_event_id == "9"


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param(problem("refused", 503), id="unmarked-503"),
        pytest.param(problem(retryable=False), id="explicitly-not-retryable"),
        pytest.param(problem("cursor_expired", 410), id="cursor-expired"),
    ],
)
@pytest.mark.parametrize("method", ["wait", "follow"])
async def test_non_retryable_in_stream_refusal_raises_at_once(contract, refusal, method):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    subscriptions = 0

    def handle(request):
        nonlocal subscriptions
        if request.headers["accept"] == "text/event-stream":
            subscriptions += 1
            feed = Feed()
            feed.queue.put_nowait(error_frame(refusal))
            return response(feed)
        return httpx.Response(200, json=state)

    # A wait outlasts an unmarked gateway refusal (a deploy) until its own timeout,
    # then raises that refusal; follow and every other refusal raise at once.
    outlasted = method == "wait" and refusal["code"] == "refused"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http, attempts=3) as api:
            with pytest.raises(SwarmsAPIError) as caught:
                if method == "wait":
                    await api.swarms.wait(
                        state["run_id"], timeout=0.5 if outlasted else 600, poll_interval=0.001
                    )
                else:
                    [item async for item in api.swarms.follow(state["run_id"])]
    assert not isinstance(caught.value, TransportFailureError)
    assert caught.value.code == refusal["code"]
    assert subscriptions >= 2 if outlasted else subscriptions == 1
    assert caught.value.checkpoint is not None


@pytest.mark.parametrize("method", ["wait", "follow"])
async def test_exhausted_reconnects_report_attempts_checkpoint_and_snapshot(contract, method):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    feeds = []

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            feed = Feed()
            if not feeds:
                feed.queue.put_nowait(data(event(state["run_id"], "A"), "8"))
            # Not a gateway status: a deploy's 502/503/504 never spends a wait's attempts.
            feed.queue.put_nowait(error_frame(problem("owner_handoff", 500, retryable=True)))
            feeds.append(feed)
            return response(feed)
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http, attempts=2) as api:
            with pytest.raises(TransportFailureError) as caught:
                if method == "wait":
                    await api.swarms.wait(state["run_id"], poll_interval=0.001)
                else:
                    [item async for item in api.swarms.follow(state["run_id"])]
    error = caught.value
    assert error.code == "event_stream_interrupted"
    assert "2 connection attempts (1 reconnect)" in str(error)
    assert error.connection_attempts == 2 and len(feeds) == 2
    assert error.checkpoint is not None and error.checkpoint.last_event_id == "8"
    assert isinstance(error.__cause__, SwarmsAPIError)
    assert error.__cause__.code == "owner_handoff"
    if method == "wait":
        assert error.snapshot is not None and error.snapshot.run_id == state["run_id"]
    assert all(feed.closed for feed in feeds)


def test_sync_wait_survives_in_stream_owner_handoff(contract):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    cursors = []

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            cursors.append(request.headers.get("last-event-id"))
            if len(cursors) == 1:
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=data(event(state["run_id"], "A"), "8") + error_frame(problem()),
                )
            state.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
            return response(Feed())
        return httpx.Response(200, json=state)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    with SynthClient(
        "token",
        base_url="https://example.test",
        http_client=http,
        retry_policy=RetryPolicy(attempts=3, backoff_seconds=0, jitter_seconds=0),
    ) as api:
        assert api.swarms.wait(state["run_id"], poll_interval=0.001).finished
    assert cursors == [None, "8"]
