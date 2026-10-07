"""Verified first-class retry directives; retains EX-01 assertions unchanged."""

import asyncio

import httpx
import pytest
from synth_ai.core.errors import SynthError
from synth_ai.core.http.request import HttpMethod, HttpRequest
from test_transport_extension_fixes_contracts import invoke, operation


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status", [409, 429, 503])
@pytest.mark.parametrize("method", [HttpMethod.GET, HttpMethod.POST])
def test_explicit_refusal_is_attempted_once__EX01(mode, status, method):
    requests = []

    def response(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={
                "detail": {
                    "error_code": "permanent_refusal",
                    "retryable": False,
                    "message": "blocked by declared policy",
                }
            },
        )

    with pytest.raises(SynthError) as refusal:
        asyncio.run(
            invoke(
                mode,
                response,
                lambda transport: transport.execute(
                    HttpRequest(
                        operation(method), "/probe", headers={"Idempotency-Key": "original-intent"}
                    )
                ),
            )
        )
    assert refusal.value.error_code == "permanent_refusal", "EX-01: refusal identity lost"
    assert refusal.value.retryable is False, "EX-01: explicit retryability lost"
    assert len(requests) == 1, (
        f"EX-01: explicit retryable=false was attempted {len(requests)} times"
    )
