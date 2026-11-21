"""Core's own inbound-admission 503 (SYN-5721/SYN-5723) surfaces as a typed, retryable refusal.

The fixture holds Core's exact wire: the pre-fix capture from B14 (synth-backend
c4739811, a private ``{"error": {...}}`` envelope) and the body Core's bulkhead
writes after f5311d62, captured from the Go handler itself.
"""

import json
from pathlib import Path

import httpx
import pytest
from test_cli import api, invoke  # noqa: F401  (api is the CLI fixture)

from synth_ai import AsyncSynthClient, RetryPolicy, cli
from synth_ai.errors import MalformedErrorResponseError, SwarmsAPIError

WIRE = json.loads(
    (Path(__file__).parent / "fixtures" / "core_inbound_overload_503.json").read_text()
)


def reply(capture):
    return httpx.Response(
        capture["status"], headers=capture["headers"], content=capture["body"].encode()
    )


async def account_refusal(capture, *, attempts):
    requests = []

    def handle(request):
        requests.append(request)
        return reply(capture)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture",
            base_url="https://example.test",
            http_client=http,
            retry_policy=RetryPolicy(attempts=attempts, backoff_seconds=0, jitter_seconds=0),
        ) as client:
            with pytest.raises(SwarmsAPIError) as caught:
                await client.account.get()
    return caught.value, requests


async def test_core_inbound_overload_is_a_typed_retryable_refusal_with_retry_after():
    error, requests = await account_refusal(WIRE["fixed"], attempts=1)
    assert not isinstance(error, MalformedErrorResponseError)
    assert error.code == "overloaded" and error.status == 503
    assert error.stage == "inbound_admission" and error.outcome == "not_accepted"
    assert error.problem.retryable is True
    assert error.retry_after_seconds == 2
    assert len(requests) == 1


async def test_core_inbound_overload_read_is_retried_after_the_server_delay(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("synth_ai._operations.asyncio.sleep", sleep)
    error, requests = await account_refusal(WIRE["fixed"], attempts=2)
    assert error.code == "overloaded" and error.retry_after_seconds == 2
    assert len(requests) == 2 and delays and delays[0] >= 2


async def test_pre_fix_core_envelope_was_malformed_and_lost_retry_after():
    """The B14 control: the same header with Core's private envelope decodes as malformed."""
    error, _ = await account_refusal(WIRE["pre_fix_c4739811"], attempts=1)
    assert isinstance(error, MalformedErrorResponseError)
    assert error.code == "malformed_error_response" and error.retry_after_seconds is None


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_surfaces_core_inbound_overload_retry_after(api, capsys, monkeypatch, as_json):  # noqa: F811
    async def sleep(delay):
        pass

    monkeypatch.setattr("synth_ai._operations.asyncio.sleep", sleep)
    api.overrides["getAccount"] = lambda request: reply(WIRE["fixed"])
    code, out, err = invoke(api, capsys, "whoami", *(["--json"] if as_json else []))
    assert code == cli.EXIT_SERVER_REFUSAL and out == ""
    if as_json:
        document = json.loads(err)
        assert document["code"] == "overloaded" and document["status"] == 503
        assert document["retryable"] is True and document["retry_after_seconds"] == 2
    else:
        assert "overloaded" in err and "retry after 2 seconds" in err
