import json

import httpx
import pytest
from test_observer import Feed, client, response
from test_routing_lifecycle import event, snapshot

from synth_ai import AsyncSynthClient
from synth_ai._limits import CHECKPOINT_MAX_EVENTS
from synth_ai._stream import StreamCheckpoint
from synth_ai.errors import ResponseContractError


def frame(item, cursor=None):
    return (
        ("id: " + cursor + "\n" if cursor is not None else "")
        + "data: "
        + json.dumps(item)
        + "\n\n"
    ).encode()


@pytest.mark.parametrize("final_frame", [False, True])
async def test_checkpoint_at_yield_means_received_and_close_releases_connection(
    contract, final_frame
):
    state = snapshot(contract)
    item = event(state["run_id"], "A")
    feed = Feed()
    feed.queue.put_nowait(frame(item, "8" if final_frame else None))
    checkpoint = StreamCheckpoint(run_id=state["run_id"], last_event_id="7")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: response(feed))) as http:
        async with client(http) as api:
            stream = api.swarms.follow(state["run_id"], checkpoint=checkpoint)
            assert (await anext(stream)).event_id == item["event_id"]
            restored = StreamCheckpoint.from_data(checkpoint.to_data())
            assert restored.last_event_id == ("8" if final_frame else "7")
            assert restored.pending_event_ids == (set() if final_frame else {item["event_id"]})
            await stream.aclose()
    assert feed.closed


async def test_restart_partial_expansion_and_committed_duplicate(contract):
    state = snapshot(contract, execution_closed=True)
    one, two = event(state["run_id"], "A"), event(state["run_id"], "B")
    checkpoint = StreamCheckpoint.from_dict(
        dict(run_id=state["run_id"], last_event_id="7", pending_event_ids=[one["event_id"]])
    )
    transcript = frame(one) + frame(two, "8") + frame(two, "8")

    def handle(request):
        if request.headers["accept"] == "text/event-stream":
            assert request.headers["last-event-id"] == "7"
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=transcript
            )
        return httpx.Response(200, json=state)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with client(http) as api:
            received = [
                item async for item in api.swarms.follow(state["run_id"], checkpoint=checkpoint)
            ]
    assert [item.event_id for item in received] == [two["event_id"]]
    assert checkpoint.last_event_id == "8" and not checkpoint.pending_event_ids


def test_checkpoint_copies_caller_and_serialized_collections():
    owned = {"event-one"}
    checkpoint = StreamCheckpoint(pending_event_ids=owned)
    owned.add("unreceived")
    assert checkpoint.pending_event_ids == {"event-one"}
    data = checkpoint.to_data()
    data.pending_event_ids.append("unsaved")
    assert checkpoint.pending_event_ids == {"event-one"}


async def test_checkpoint_overflow_leaves_recoverable_prior_state(contract):
    state = snapshot(contract)
    checkpoint = StreamCheckpoint(
        run_id=state["run_id"],
        last_event_id="7",
        pending_event_ids={f"received-{i}" for i in range(CHECKPOINT_MAX_EVENTS)},
    )
    prior = checkpoint.to_dict()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=frame(event(state["run_id"], "A")),
            )
        )
    ) as http:
        async with AsyncSynthClient(
            "token", base_url="https://example.test", http_client=http
        ) as api:
            with pytest.raises(ResponseContractError):
                await anext(api.swarms.follow(state["run_id"], checkpoint=checkpoint))
    assert checkpoint.to_dict() == prior
