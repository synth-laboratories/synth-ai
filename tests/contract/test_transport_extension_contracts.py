"""Positive transport controls for the extension laws. No network or provider effects."""

import asyncio

import httpx
import pytest
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.retry import RetryPolicy
from synth_ai.core.http.transport import _retry_after_seconds
from test_transport_extension_law import invoke, operation


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_explicit_retryable_response_can_recover(mode):
    requests = []

    def response(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                503, json={"detail": {"error_code": "temporarily_busy", "retryable": True}}
            )
        return httpx.Response(200, json={"recovered": True})

    result = asyncio.run(
        invoke(
            mode, response, lambda transport: transport.execute(HttpRequest(operation(), "/probe"))
        )
    )
    assert result == {"recovered": True}
    assert len(requests) == 2


@pytest.mark.parametrize("header", ["NaN", "inf", "-1"])
def test_invalid_retry_after_is_ignored(header):
    response = httpx.Response(503, headers={"retry-after": header})
    assert _retry_after_seconds(response, sources=()) is None


def test_retry_policy_valid_boundaries():
    policy = RetryPolicy(attempts_max=1, delay_seconds_initial=0, delay_seconds_max=0)
    assert policy.delay_seconds(0, retry_after_seconds=None) == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_finite_json_and_numeric_looking_strings_remain_valid(mode):
    payload = {"number": 0.5, "missing": None, "flag": True, "literal": "NaN"}

    def response(request):
        return httpx.Response(200, json=payload)

    result = asyncio.run(
        invoke(mode, response, lambda transport: transport.request_json("GET", "/probe"))
    )
    assert result == payload
