"""Exercise message bodies, idempotency, pagination and feed-driven replies."""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum

import httpx
import pytest
from pydantic import ValidationError

from synth_ai import AsyncSynthClient, RetryPolicy, SynthClient
from synth_ai.errors import (
    MessageReplyTimeoutError,
    MessagingUnavailableError,
    RunMessageRateLimitedError,
)
from synth_ai.mcp.server import call_tool
from synth_ai.models import (
    BodyUnavailable,
    ManagerMessageState,
    MessageDisposition,
    MessageKind,
    PostedMessageKind,
    RunEvent,
    RunMessage,
    RunMessageEventPayload,
    RunMessagePage,
)

RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"
MESSAGE = "msg_AAAAAAAAAAAAAAAAAAAAAA"
REPLY = "msg_AAAAAAAAAAAAAAAAAAAAAB"
THREAD = "thr_AAAAAAAAAAAAAAAAAAAAAA"


def message(*, reply=False, retained=True):
    return {
        "run_id": RUN,
        "message_id": REPLY if reply else MESSAGE,
        "thread_id": THREAD,
        "author": "manager" if reply else "customer",
        "kind": "reply" if reply else "question",
        "sequence": 2 if reply else 1,
        "state": {"kind": "posted" if reply else "queued"},
        "body": {"kind": "available", "text": "fixture"}
        if retained
        else {"kind": "unavailable", "reason": "retention"},
        "created_at": "2026-11-17T00:00:00Z",
        "updated_at": "2026-11-17T00:00:01Z",
        **({"reply_to_message_id": MESSAGE} if reply else {}),
    }


def page(items, *, cursor=None, more=False, available=True):
    return {
        "items": items,
        "next_cursor": cursor,
        "has_more": more,
        "availability": {"kind": "available"}
        if available
        else {"kind": "unavailable", "reason": "messaging_unavailable"},
    }


def event(*, kind="run.message.posted", reply_to=MESSAGE):
    return {
        "run_id": RUN,
        "event_id": "evt_AAAAAAAAAAAAAAAAAAAAAA",
        "sequence": 2,
        "kind": kind,
        "original_kind": kind,
        "usage": None,
        "approval_id": None,
        "observed_at": "2026-11-17T00:00:01Z",
        "payload": {
            "message_id": REPLY,
            "state": {"kind": "posted"},
            "author": "manager",
            "reply_to_message_id": reply_to,
        },
    }


def test_contract_named_enums_and_retention_body_are_typed():
    assert issubclass(MessageKind, StrEnum)
    assert issubclass(ManagerMessageState, StrEnum)
    assert issubclass(MessageDisposition, StrEnum)
    item = RunMessage.model_validate(message(reply=True, retained=False))
    assert item.kind == PostedMessageKind.REPLY
    assert isinstance(item.body, BodyUnavailable)
    assert item.body.reason == "retention"
    parsed = RunEvent.model_validate(event())
    assert isinstance(parsed.payload, RunMessageEventPayload)


def test_empty_unavailable_history_preserves_the_typed_reason():
    history = RunMessagePage.model_validate(page([], available=False))
    assert history.items == []
    assert history.availability.kind == "unavailable"
    assert history.availability.reason == "messaging_unavailable"


@pytest.mark.parametrize(
    "state",
    [
        {"kind": "delivered"},
        {"kind": "queued", "decision_step": 1},
        {"kind": "posted", "decision_step": 1},
        {"kind": "disposed", "decision_step": 1},
        {"kind": "disposed", "decision_step": 1, "disposition": {"kind": "declined"}},
        {
            "kind": "disposed",
            "decision_step": 1,
            "disposition": {"kind": "applied", "reason_message_id": REPLY},
        },
    ],
)
def test_generated_constraints_refuse_impossible_message_states(state):
    with pytest.raises(ValidationError):
        RunMessage.model_validate({**message(), "state": state})


async def test_send_snapshot_and_key_are_retained_across_retry():
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={
                    "type": "about:blank",
                    "title": "run_message_rate_limited",
                    "status": 429,
                    "code": "run_message_rate_limited",
                    "detail": "fixture",
                    "instance": "urn:request:fixture",
                    "service": "synth-backend",
                    "stage": "run_messages",
                    "outcome": "not_accepted",
                    "retryable": True,
                },
            )
        return httpx.Response(202, json=message())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture",
            base_url="https://example.test",
            http_client=http,
            retry_policy=RetryPolicy(backoff_seconds=0, jitter_seconds=0),
        ) as client:
            result = await client.swarms.send_message(
                RUN, "fixture", kind=MessageKind.QUESTION, idempotency_key="same-message"
            )
    assert result.state.kind == ManagerMessageState.QUEUED
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    assert requests[0].headers["Idempotency-Key"] == "same-message"
    assert requests[1].headers["Idempotency-Key"] == "same-message"
    assert json.loads(requests[0].content) == {"kind": "question", "text": "fixture"}


async def test_false_retryable_refusal_is_never_retried():
    requests = []

    def handle(request):
        requests.append(request)
        from synth_ai.errors import local_problem

        problem = local_problem("run_message_rate_limited", "run_messages", status=429)
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json=problem.model_copy(
                update={"service": "synth-backend", "retryable": False}
            ).model_dump(mode="json"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(RunMessageRateLimitedError):
                await client.swarms.send_message(RUN, "fixture", idempotency_key="once")
    assert len(requests) == 1


async def test_byte_limit_refuses_before_transport():
    requests = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: requests.append(request))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(ValidationError):
                await client.swarms.send_message(RUN, "é" * 4097, idempotency_key="large")
    assert requests == []


async def test_pagination_and_single_read_use_the_public_routes():
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/" + REPLY):
            return httpx.Response(200, json=message(reply=True))
        return httpx.Response(
            200,
            json=page([message(reply=True)])
            if request.url.params.get("cursor") == "next"
            else page([message()], cursor="next", more=True),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        with SynthClient("fixture", base_url="https://example.test", http_client=http) as client:
            assert [item.message_id for item in client.swarms.iter_messages(RUN)] == [
                MESSAGE,
                REPLY,
            ]
            assert client.swarms.message(RUN, REPLY).reply_to_message_id == MESSAGE
    assert len(requests) == 3


async def test_iterator_does_not_silently_turn_unavailable_history_into_no_messages():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=page([], available=False))
        )
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(MessagingUnavailableError):
                await anext(client.swarms.iter_messages(RUN))


async def test_reply_wait_opens_feed_before_history_then_reads_only_on_message_wake():
    requests = []
    posted = event()
    sse = ("id: reply\ndata: " + json.dumps(posted) + "\n\n").encode()

    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/events"):
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=sse)
        if request.url.path.endswith("/" + REPLY):
            return httpx.Response(200, json=message(reply=True))
        return httpx.Response(200, json=page([message()]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            result = await client.swarms.wait_for_reply(RUN, MESSAGE, timeout=1)
    assert result.message_id == REPLY
    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "events",
        "messages",
        REPLY,
    ]
    assert all(request.method == "GET" for request in requests)


async def test_reply_timeout_closes_feed_without_cancelling_message():
    requests = []

    class WaitingFeed(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            await asyncio.sleep(3600)
            yield b""

        async def aclose(self):
            self.closed = True

    feed = WaitingFeed()

    def handle(request):
        requests.append(request)
        if request.url.path.endswith("/events"):
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=feed)
        return httpx.Response(200, json=page([]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(MessageReplyTimeoutError) as caught:
                await client.swarms.wait_for_reply(RUN, MESSAGE, timeout=0.01)
    assert caught.value.message_id == MESSAGE
    assert feed.closed
    assert len(requests) == 2
    assert all(request.method == "GET" for request in requests)


async def test_mcp_send_list_get_use_the_same_typed_sdk_surface():
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json=message())
        if request.url.path.endswith("/" + MESSAGE):
            return httpx.Response(200, json=message())
        return httpx.Response(200, json=page([message()]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            sent = json.loads(
                await call_tool(
                    client,
                    "swarms_run_message_send",
                    {
                        "run_id": RUN,
                        "body": {"kind": "question", "text": "fixture"},
                        "idempotency_key": "mcp-message",
                    },
                )
            )
            history = json.loads(await call_tool(client, "swarms_run_messages", {"run_id": RUN}))
            single = json.loads(
                await call_tool(
                    client,
                    "swarms_run_message",
                    {
                        "run_id": RUN,
                        "message_id": MESSAGE,
                    },
                )
            )
    assert sent["message_id"] == single["message_id"] == MESSAGE
    assert len(history["items"]) == 1
    assert requests[0].headers["Idempotency-Key"] == "mcp-message"
    assert [(item.method, item.url.path) for item in requests] == [
        ("POST", f"/v1/runs/{RUN}/messages"),
        ("GET", f"/v1/runs/{RUN}/messages"),
        ("GET", f"/v1/runs/{RUN}/messages/{MESSAGE}"),
    ]


async def test_typechecked_quickstart_sends_mid_run_then_returns_correlated_reply():
    from examples.messaging import ask_during_run

    def handle(request):
        if request.method == "POST":
            return httpx.Response(202, json=message())
        if request.url.path.endswith("/events"):
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=b"")
        return httpx.Response(200, json=page([message(reply=True)]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            reply = await ask_during_run(
                client, RUN, text="fixture", idempotency_key="quickstart", timeout=1
            )
    assert reply.message_id == REPLY
