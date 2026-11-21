"""A send before the run is admitted retries Core's typed 503 under one key (SYN-5744)."""

from __future__ import annotations

import json

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient, cli
from synth_ai._retry import ADMISSION_RETRY_BUDGET_SECONDS
from synth_ai._transport import Transport
from synth_ai.errors import UpstreamUnavailableError
from synth_ai.mcp.server import call_tool
from synth_ai.models import MessageKind
from synth_ai.swarms import AsyncSwarms

RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"
MESSAGE = "msg_AAAAAAAAAAAAAAAAAAAAAA"
PENDING = "run_message_admission_pending"


def accepted():
    return httpx.Response(
        202,
        json={
            "run_id": RUN,
            "message_id": MESSAGE,
            "thread_id": "thr_AAAAAAAAAAAAAAAAAAAAAA",
            "author": "customer",
            "kind": "guidance",
            "sequence": 1,
            "state": {"kind": "queued"},
            "body": {"kind": "available", "text": "fixture"},
            "created_at": "2026-11-19T00:00:00Z",
            "updated_at": "2026-11-19T00:00:01Z",
        },
    )


def refusal(code=PENDING, *, retry_after: str | None = "1"):
    """Core's wire body for a send before the admission commits (messages.go sendBinding)."""
    return httpx.Response(
        503,
        headers={} if retry_after is None else {"Retry-After": retry_after},
        json={
            "type": "about:blank",
            "title": code,
            "status": 503,
            "detail": "The run is not admitted yet. Send the message again shortly.",
            "instance": "urn:request:fixture",
            "code": code,
            "service": "core",
            "stage": "run_messages",
            "outcome": "not_accepted",
            "retryable": True,
            "retry_after_seconds": "1",
        },
    )


class Clock:
    """Fake monotonic clock that advances only when the retry sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def script(*responses):
    requests: list[httpx.Request] = []
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return requests, handle


async def send(handle, clock: Clock, *, idempotency_key=None):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        transport = Transport(
            "https://example.test", "fixture", client=http, sleep=clock.sleep, clock=clock
        )
        if idempotency_key is None:
            # The SDK method requires a key; the shared layer generates one for raw invokes.
            return await transport.invoke(
                "sendRunMessage", path={"run_id": RUN}, body={"kind": "guidance", "text": "fixture"}
            )
        swarms = AsyncSwarms(transport)
        return await swarms.send_message(RUN, "fixture", idempotency_key=idempotency_key)


async def test_more_pending_refusals_than_policy_attempts_then_accepted_under_one_key():
    clock = Clock()
    requests, handle = script(*(refusal() for _ in range(4)), accepted())
    result = await send(handle, clock, idempotency_key="same-message")
    assert result.message_id == MESSAGE
    assert len(requests) == 5
    assert {r.headers["Idempotency-Key"] for r in requests} == {"same-message"}
    assert len({r.content for r in requests}) == 1
    assert clock.sleeps == [1.0, 1.0, 1.0, 1.0]  # exactly Retry-After; no backoff or jitter


async def test_two_pending_then_accepted_generates_one_key_when_caller_gave_none():
    clock = Clock()
    requests, handle = script(refusal(), refusal(), accepted())
    result = await send(handle, clock)
    assert result.message_id == MESSAGE
    keys = {r.headers["Idempotency-Key"] for r in requests}
    assert len(requests) == 3 and len(keys) == 1 and "" not in keys


@pytest.mark.parametrize(
    "header,wait",
    [("7", 7.0), ("600", 60.0), (None, 1.0), ("soon", 1.0), ("0", 0.0)],
)
async def test_retry_after_is_honoured_capped_and_defaulted(header, wait):
    clock = Clock()
    _, handle = script(refusal(retry_after=header), accepted())
    await send(handle, clock, idempotency_key="k")
    assert clock.sleeps == [wait]


async def test_budget_exhaustion_raises_the_typed_refusal_with_attempts_and_elapsed():
    clock = Clock()
    requests, handle = script(refusal(retry_after="50"))
    with pytest.raises(UpstreamUnavailableError) as caught:
        await send(handle, clock, idempotency_key="k")
    error = caught.value
    assert error.code == PENDING and error.status == 503
    assert error.problem.detail.startswith("The run is not admitted yet")
    assert error.retry_after_seconds == 50
    assert error.idempotency_key == "k"
    # Waits of 50 + 50 fit the budget; a third would cross it.
    assert clock.sleeps == [50.0, 50.0]
    assert error.attempts == len(requests) == 3
    assert error.elapsed_seconds == 100.0 <= ADMISSION_RETRY_BUDGET_SECONDS


@pytest.mark.parametrize("code", ["messaging_unavailable", "event_owner_unavailable"])
async def test_other_503_codes_are_not_admission_retried(code):
    clock = Clock()
    response = refusal(code)
    body = json.loads(response.content)
    body["retryable"] = False
    requests, handle = script(httpx.Response(503, headers={"Retry-After": "1"}, json=body))
    with pytest.raises(UpstreamUnavailableError) as caught:
        await send(handle, clock, idempotency_key="k")
    assert len(requests) == 1 and clock.sleeps == []
    assert caught.value.attempts is None


async def test_sync_client_uses_the_same_retry():
    requests, handle = script(refusal(retry_after="0"), refusal(retry_after="0"), accepted())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        with SynthClient("fixture", base_url="https://example.test", http_client=http) as client:
            result = client.swarms.send_message(
                RUN, "fixture", kind=MessageKind.GUIDANCE, idempotency_key="sync"
            )
    assert result.message_id == MESSAGE
    assert [r.headers["Idempotency-Key"] for r in requests] == ["sync"] * 3


def test_cli_message_retries_through_the_shared_layer(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("SYNTH_STATE_DIR", str(tmp_path / "state"))
    requests, handle = script(*(refusal(retry_after="0") for _ in range(4)), accepted())
    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    code = cli.main(["message", RUN, "fixture"], http_client=http)
    capsys.readouterr()
    assert code == 0
    assert len(requests) == 5
    assert len({r.headers["Idempotency-Key"] for r in requests}) == 1


async def test_mcp_send_retries_through_the_shared_layer():
    requests, handle = script(*(refusal(retry_after="0") for _ in range(4)), accepted())
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
                        "body": {"kind": "guidance", "text": "fixture"},
                        "idempotency_key": "mcp-message",
                    },
                )
            )
    assert sent["message_id"] == MESSAGE
    assert [r.headers["Idempotency-Key"] for r in requests] == ["mcp-message"] * 5
