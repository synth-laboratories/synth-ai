"""Explicit retryability is authoritative; quota refusals must not loop."""

import httpx
import pytest
from test_cli import example

from synth_ai import AsyncSynthClient, RetryPolicy
from synth_ai.errors import SwarmsAPIError, local_problem


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("retryable", [False, None])
async def test_unretryable_read_refusal_surfaces_immediately(status, retryable):
    requests = []
    problem = local_problem("quota_exhausted", "account", status=status).model_dump(mode="json")
    problem["retryable"] = retryable

    def handle(request):
        requests.append(request)
        return httpx.Response(status, headers={"Retry-After": "0"}, json=problem)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture",
            base_url="https://example.test",
            http_client=http,
            retry_policy=RetryPolicy(attempts=3, backoff_seconds=0, jitter_seconds=0),
        ) as client:
            with pytest.raises(SwarmsAPIError) as caught:
                await client.account.get()
    assert caught.value.code == "quota_exhausted"
    assert len(requests) == 1


@pytest.mark.parametrize("status", [429, 503])
async def test_explicit_retry_waits_server_delay_and_preserves_read(contract, monkeypatch, status):
    requests, delays = [], []
    problem = local_problem("overloaded", "account", status=status).model_dump(mode="json")
    problem["retryable"] = True
    _, account = example(contract, "getAccount")

    async def sleep(delay):
        delays.append(delay)

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status, headers={"Retry-After": "2"}, json=problem)
        return httpx.Response(200, json=account)

    monkeypatch.setattr("synth_ai._operations.asyncio.sleep", sleep)
    monkeypatch.setattr("synth_ai._retry.random.uniform", lambda minimum, maximum: maximum)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture",
            base_url="https://example.test",
            http_client=http,
            retry_policy=RetryPolicy(attempts=3, backoff_seconds=0, jitter_seconds=0.25),
        ) as client:
            await client.account.get()
    assert delays == [2.25]
    assert len(requests) == 2
    assert requests[0].url == requests[1].url
