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


@pytest.mark.parametrize("status", [401, 403, 404, 406, 410, 415, 501, 503])
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
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(SwarmsAPIError) as caught:
                await api.swarms.wait(snapshot(contract)["run_id"])
    assert caught.value.status == status and len(calls) == 1


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


def milestone(run_id, name, level, sequence=8, final=False, **flags):
    data = dict(
        milestone=name,
        run_id=run_id,
        state="completed",
        execution_closed=True,
        accounting_final=level >= 2,
        cleanup_complete=level >= 3,
        result_available=True,
        failure=None,
        final=final,
    )
    data.update(flags)
    return (
        "id: "
        + str(sequence)
        + "."
        + str(level)
        + "\nevent: milestone\ndata: "
        + json.dumps(data)
        + "\n\n"
    ).encode()


async def test_closed_run_follows_milestones_without_polling(contract, monkeypatch):
    monkeypatch.setattr(_observer, "_RECONCILE_SECONDS", 5)
    state = snapshot(
        contract, execution_closed=True, accounting_final=False, cleanup_complete=False
    )
    feed, subscriptions, reads = Feed(), 0, 0
    run_id = state["run_id"]
    feed.queue.put_nowait(milestone(run_id, "execution_closed", 1))

    def handle(request):
        nonlocal subscriptions, reads
        if request.headers["accept"] == "text/event-stream":
            subscriptions += 1
            return response(feed)
        reads += 1
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            waiting = asyncio.create_task(api.swarms.wait(run_id, timeout=2, poll_interval=0.001))
            await asyncio.sleep(0.05)
            # Closed but not settled: the observer waits on the feed, not on getRun.
            assert reads == 2 and not waiting.done()
            state.update(accounting_final=True)
            feed.queue.put_nowait(milestone(run_id, "accounting_final", 2))
            await asyncio.sleep(0.05)
            assert reads == 3 and not waiting.done()
            state.update(cleanup_complete=True)
            feed.queue.put_nowait(milestone(run_id, "cleanup_complete", 3, final=True))
            assert (await asyncio.wait_for(waiting, 1)).finished
    assert subscriptions == 1 and reads == 4 and feed.closed


async def test_feed_cut_before_final_milestone_resumes_past_it(contract):
    state = snapshot(
        contract, execution_closed=True, accounting_final=False, cleanup_complete=False
    )
    run_id, cursors = state["run_id"], []

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            cursors.append(request.headers.get("last-event-id"))
            feed = Feed()
            if len(cursors) == 1:
                feed.queue.put_nowait(milestone(run_id, "execution_closed", 1))
                feed.queue.put_nowait(None)  # lifetime cap: no final milestone
            else:
                state.update(accounting_final=True, cleanup_complete=True)
                feed.queue.put_nowait(milestone(run_id, "accounting_final", 2))
                feed.queue.put_nowait(milestone(run_id, "cleanup_complete", 3, final=True))
            return response(feed)
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            assert (await api.swarms.wait(run_id, timeout=1, poll_interval=0.001)).finished
    assert cursors == [None, "8.1"]


async def test_final_milestone_without_requested_state_fails_clearly(contract):
    state = snapshot(
        contract, execution_closed=True, accounting_final=False, cleanup_complete=False
    )
    run_id, subscriptions = state["run_id"], 0

    def handle(request):
        nonlocal subscriptions
        if request.headers["accept"] == "text/event-stream":
            subscriptions += 1
            feed = Feed()
            feed.queue.put_nowait(milestone(run_id, "cleanup_complete", 3, final=True))
            feed.queue.put_nowait(None)
            return response(feed)
        return httpx.Response(200, json=state)  # getRun never shows the milestone

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            with pytest.raises(ResponseContractError, match="event_stream_ended_before_milestone"):
                await api.swarms.wait(run_id, timeout=1, poll_interval=0.001)
    assert subscriptions == 1


async def test_follow_yields_events_and_checkpoints_milestones(contract):
    run_id = snapshot(contract)["run_id"]
    feed = Feed()
    feed.queue.put_nowait(("id: 8\ndata: " + json.dumps(event(run_id, "A")) + "\n\n").encode())
    feed.queue.put_nowait(milestone(run_id, "execution_closed", 1))
    feed.queue.put_nowait(milestone(run_id, "cleanup_complete", 3, final=True))
    feed.queue.put_nowait(None)
    state = snapshot(contract, execution_closed=True, accounting_final=True, cleanup_complete=True)

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            return response(feed)
        return httpx.Response(200, json=state)

    from synth_ai._stream import StreamCheckpoint

    checkpoint = StreamCheckpoint()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            items = [item async for item in api.swarms.follow(run_id, checkpoint=checkpoint)]
    assert [item.event_id for item in items] == ["evt_" + "A" * 22]
    assert checkpoint.last_event_id == "8.3"


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
