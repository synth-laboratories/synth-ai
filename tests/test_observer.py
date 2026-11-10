import asyncio
import json

import httpx
import pytest
from test_routing_lifecycle import event, snapshot

from synth_ai import AsyncSynthClient, _observer
from synth_ai._transport import RetryPolicy
from synth_ai.errors import (
    ResponseContractError,
    RunHistoryUnavailableError,
    SwarmsAPIError,
    TransportFailureError,
    UpstreamUnavailableError,
    WaitTimeoutError,
)


class Feed(httpx.AsyncByteStream):
    def __init__(self):
        self.queue = asyncio.Queue()
        self.closed = False

    async def __aiter__(self):
        while True:
            item = await self.queue.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                return
            yield item

    async def aclose(self):
        self.closed = True


def response(feed):
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=feed)


def client(http, attempts=3):
    return AsyncSynthClient(
        "token",
        base_url="https://example.test",
        http_client=http,
        retry_policy=RetryPolicy(attempts=attempts, backoff_seconds=0, jitter_seconds=0),
    )


async def test_subscribe_before_read_catches_change_without_event(contract):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    feed, calls = Feed(), []

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            calls.append("subscribe")
            state.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
            return response(feed)
        calls.append("read")
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            assert (await api.swarms.wait(state["run_id"])).finished
    assert calls == ["subscribe", "read"]
    assert feed.closed


async def test_notifications_are_wakeups_and_idle_feed_does_not_poll(contract):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    feed, reads = Feed(), 0
    initial = asyncio.Event()

    def handle(request):
        nonlocal reads
        if request.headers["accept"] == "text/event-stream":
            return response(feed)
        reads += 1
        initial.set()
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            waiting = asyncio.create_task(api.swarms.wait(state["run_id"], poll_interval=0.001))
            await initial.wait()
            await asyncio.sleep(0.03)
            assert reads == 1 and not waiting.done()
            state.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
            feed.queue.put_nowait(
                ("id: 1\ndata: " + json.dumps(event(state["run_id"], "A")) + "\n\n").encode()
            )
            assert (await asyncio.wait_for(waiting, 1)).finished
    assert reads == 2 and feed.closed


async def test_lost_notification_reconciles_on_open_feed(contract, monkeypatch):
    monkeypatch.setattr(_observer, "_RECONCILE_SECONDS", 0.01)
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    feed, reads = Feed(), 0

    def handle(request):
        nonlocal reads
        if request.headers["accept"] == "text/event-stream":
            return response(feed)
        reads += 1
        if reads == 2:
            state.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            assert (await api.swarms.wait(state["run_id"], timeout=1, poll_interval=0.001)).finished
    assert reads == 2 and feed.closed


@pytest.mark.parametrize("finish", ["timeout", "cancel"])
async def test_observation_closes_pending_stream_without_mutating_run(contract, finish):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    feed, calls = Feed(), []
    ready = asyncio.Event()

    def handle(request):
        calls.append(request.method)
        if request.headers["accept"] == "text/event-stream":
            return response(feed)
        ready.set()
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            waiting = asyncio.create_task(api.swarms.wait(state["run_id"], timeout=0.05))
            await ready.wait()
            if finish == "cancel":
                waiting.cancel()
            with pytest.raises(WaitTimeoutError if finish == "timeout" else asyncio.CancelledError):
                await waiting
    assert feed.closed and calls == ["GET", "GET"]


async def test_disconnect_resumes_checkpoint_and_rereads(contract):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    feed, cursors = Feed(), []
    feed.queue.put_nowait(
        ("id: 8\ndata: " + json.dumps(event(state["run_id"], "A")) + "\n\n").encode()
    )
    feed.queue.put_nowait(httpx.ReadError("lost"))

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            cursors.append(request.headers.get("last-event-id"))
            if len(cursors) > 1:
                state.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
                return response(Feed())
            return response(feed)
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            assert (await api.swarms.wait(state["run_id"], poll_interval=0.001)).finished
    assert cursors == [None, "8"] and feed.closed


@pytest.mark.parametrize("status", [401, 403, 404, 410, 503])
async def test_feed_refusals_do_not_silently_fallback(contract, status):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            status,
            json=dict(
                type="about:blank",
                title="refused",
                detail="refused",
                status=status,
                instance="urn:test",
                code="expired" if status == 410 else "refused",
                service="core",
                stage="events",
                outcome="not_accepted",
                # A gateway status is transient unless the server explicitly says not.
                **({"retryable": False} if status == 503 else {}),
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(SwarmsAPIError) as caught:
                await api.swarms.wait(snapshot(contract)["run_id"])
    assert caught.value.status == status and len(calls) == 1


def gateway_problem(status):
    return httpx.Response(
        status,
        json=dict(
            type="about:blank",
            title="unavailable",
            detail="Bad Gateway",
            status=status,
            instance="urn:test",
            code="upstream_refused",
            service="core",
            stage="getRun",
            outcome="not_accepted",
        ),
    )


@pytest.mark.parametrize("failing", ["read", "subscribe"])
async def test_wait_outlasts_deploy_gateway_failures(contract, monkeypatch, failing):
    """A Core deploy's 502/503/504 on reads or the feed delays a wait; it does not end it."""
    monkeypatch.setattr(_observer, "_RECONCILE_SECONDS", 0.01)
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    outages = [503, 502, 504, 503]
    calls = []

    def handle(request):
        kind = "subscribe" if request.headers["accept"] == "text/event-stream" else "read"
        calls.append(kind)
        if kind == failing and outages:
            return gateway_problem(outages.pop(0))
        if kind == "subscribe":
            return response(Feed())
        if not outages:
            state.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http, attempts=1) as api:
            done = await api.swarms.wait(state["run_id"], timeout=5, poll_interval=0.001)
    assert done.finished and not outages
    assert calls.count(failing) >= 5


async def test_wait_raises_original_gateway_error_when_outage_outlasts_timeout(contract):
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        if request.headers["accept"] == "text/event-stream":
            return response(Feed())
        return gateway_problem(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(UpstreamUnavailableError) as caught:
                await api.swarms.wait(snapshot(contract)["run_id"], timeout=0.2, poll_interval=0.01)
    assert caught.value.status == 503 and caught.value.code == "upstream_refused"
    assert isinstance(caught.value.__cause__, TimeoutError) and calls >= 3


async def test_launch_is_never_retried_on_gateway_failure(contract):
    calls = []

    def handle(request):
        calls.append(request.method)
        return gateway_problem(502)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(UpstreamUnavailableError):
                await api.swarms.pause("run_x")
    assert calls == ["POST"]


async def test_unavailable_feed_has_bounded_explicit_fallback(contract, monkeypatch):
    monkeypatch.setattr(_observer, "_FALLBACK_MIN_SECONDS", 0.01)
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    subscriptions, reads = 0, 0

    def handle(request):
        nonlocal subscriptions, reads
        if request.headers["accept"] == "text/event-stream":
            subscriptions += 1
            return httpx.Response(
                501,
                json=dict(
                    type="about:blank",
                    title="unsupported",
                    detail="unsupported",
                    status=501,
                    instance="urn:test",
                    code="unsupported",
                    service="core",
                    stage="events",
                    outcome="not_accepted",
                ),
            )
        reads += 1
        if reads == 2:
            state.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            assert (await api.swarms.wait(state["run_id"], poll_interval=0.001)).finished
    assert subscriptions == 1 and reads == 2


async def test_eof_is_not_completion_and_reconnects_are_bounded(contract):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    feeds = []

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            feed = Feed()
            feed.queue.put_nowait(None)
            feeds.append(feed)
            return response(feed)
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http, attempts=2) as api:
            with pytest.raises(TransportFailureError, match="event_stream_interrupted"):
                await api.swarms.wait(state["run_id"], poll_interval=0.001)
    assert len(feeds) == 2 and all(feed.closed for feed in feeds)


async def test_historical_and_malformed_feed_fail_clearly(contract):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    for historical in [True, False]:
        state["metadata"] = {"imported_history": True} if historical else {}

        def handle(request, historical=historical):
            if request.headers["accept"] == "text/event-stream":
                if historical:
                    return response(Feed())
                return httpx.Response(200, headers={"Content-Type": "application/json"}, json={})
            return httpx.Response(200, json=state)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            async with client(http) as api:
                with pytest.raises(
                    RunHistoryUnavailableError if historical else ResponseContractError
                ):
                    await api.swarms.wait(state["run_id"])


async def test_execution_closed_eof_polls_finality_without_reconnecting(contract, monkeypatch):
    monkeypatch.setattr(_observer, "_FALLBACK_MIN_SECONDS", 0.01)
    state = snapshot(
        contract, execution_closed=True, accounting_final=False, cleanup_complete=False
    )
    subscriptions, reads = 0, 0

    def handle(request):
        nonlocal subscriptions, reads
        if request.headers["accept"] == "text/event-stream":
            subscriptions += 1
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=b"")
        reads += 1
        if reads == 4:
            state.update(accounting_final=True, cleanup_complete=True)
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            assert (await api.swarms.wait(state["run_id"], timeout=1, poll_interval=0.001)).finished
    assert subscriptions == 1 and reads == 4


async def test_snapshot_refusal_cannot_be_mistaken_for_unavailable_feed(contract):
    calls = []
    feed = Feed()

    def handle(request):
        calls.append(request)
        if request.headers["accept"] == "text/event-stream":
            return response(feed)
        return httpx.Response(
            501,
            json=dict(
                type="about:blank",
                title="refused",
                detail="refused",
                status=501,
                instance="urn:test",
                code="snapshot_unavailable",
                service="core",
                stage="getRun",
                outcome="not_accepted",
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(SwarmsAPIError, match="snapshot_unavailable"):
                await api.swarms.wait(snapshot(contract)["run_id"])
    assert len(calls) == 2 and feed.closed


async def test_overall_deadline_covers_reconnect_delay(contract):
    state = snapshot(
        contract, execution_closed=False, accounting_final=False, cleanup_complete=False
    )
    subscriptions = 0

    def handle(request):
        nonlocal subscriptions
        if request.headers["accept"] == "text/event-stream":
            subscriptions += 1
            raise httpx.ConnectError("no connection")
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(WaitTimeoutError) as caught:
                await api.swarms.wait(state["run_id"], timeout=0.01, poll_interval=1)
    assert caught.value.snapshot is None and subscriptions == 1

@pytest.mark.parametrize("until", ["finished", "paused"])
async def test_paused_returns_explicitly_or_raises_with_snapshot(contract, until):
    from synth_ai import CompletionMilestone, RunPausedError
    state = snapshot(contract, execution_closed=False, accounting_final=False, cleanup_complete=False)
    state["state"] = "paused"
    feed = Feed()
    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            return response(feed)
        return httpx.Response(200, json=state)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            if until == "paused":
                result = await api.swarms.wait(state["run_id"], until=CompletionMilestone.PAUSED)
                assert result.state == "paused" and not result.finished
            else:
                with pytest.raises(RunPausedError) as caught:
                    await api.swarms.wait(state["run_id"])
                assert caught.value.snapshot.state == "paused"
    assert feed.closed
