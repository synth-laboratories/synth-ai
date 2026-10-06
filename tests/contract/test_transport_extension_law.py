"""Additional first-class error laws; see core_research_migration.md and tigerstyle.md.

EX-01: explicit non-retryable refusals override status fallback.
EX-02: HTTP idempotency headers have case-insensitive semantics.
EX-03: bounded retry configuration must be finite and typed.
EX-04: redirect responses without a destination cannot become successful data.
EX-05: empty 200 JSON responses cannot become a fabricated empty object.
"""

import asyncio
import math

import httpx
import pytest
from synth_ai.core.errors import ContractMismatchError, SynthError
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpMethod, HttpRequest, OperationId, OperationMetadata
from synth_ai.core.http.retry import RetryPolicy, idempotency_key_from_request
from synth_ai.core.http.transport import HttpTransport


def operation(method=HttpMethod.GET):
    return OperationMetadata(
        OperationId("offline_probe"),
        method,
        "/probe",
        mutation=method is not HttpMethod.GET,
        idempotent=True,
    )


async def invoke(mode, handler, action):
    policy = RetryPolicy(attempts_max=3, delay_seconds_initial=0, delay_seconds_max=0)
    if mode == "sync":
        transport = HttpTransport("http://offline.invalid", {}, retry_policy=policy)
        transport.client.close()
        transport.client = httpx.Client(
            base_url="http://offline.invalid", transport=httpx.MockTransport(handler)
        )
        try:
            return action(transport)
        finally:
            transport.close()
    transport = AsyncHttpTransport("http://offline.invalid", {}, retry_policy=policy)
    await transport.client.aclose()
    transport.client = httpx.AsyncClient(
        base_url="http://offline.invalid", transport=httpx.MockTransport(handler)
    )
    try:
        return await action(transport)
    finally:
        await transport.close()


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


@pytest.mark.parametrize("header", ["IDEMPOTENCY-KEY", "iDeMpOtEnCy-KeY"])
def test_idempotency_header_is_case_insensitive__EX02(header):
    request = HttpRequest(operation(HttpMethod.POST), "/probe", headers={header: "original-intent"})
    assert httpx.Headers(request.headers)["idempotency-key"] == "original-intent"
    assert idempotency_key_from_request(request) == "original-intent", (
        "EX-02: valid HTTP header casing loses idempotency identity"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delay_seconds_initial": math.nan},
        {"delay_seconds_max": math.nan},
        {"delay_seconds_max": math.inf},
        {"attempts_max": 2.5},
        {"attempts_max": True},
    ],
)
def test_retry_configuration_is_finite_and_typed__EX03(kwargs):
    try:
        RetryPolicy(**kwargs)
    except (TypeError, ValueError):
        return
    pytest.fail(f"EX-03: non-finite or non-integer retry configuration accepted: {kwargs}")


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("operation_kind", ["json", "bytes"])
def test_redirect_without_location_is_not_success__EX04(mode, operation_kind):
    def response(request):
        return httpx.Response(302, json={"message": "redirect without Location"})

    def action(transport):
        method = transport.request_json if operation_kind == "json" else transport.request_bytes
        return method("GET", "/probe", operation_id="offline_probe")

    try:
        asyncio.run(invoke(mode, response, action))
    except SynthError as refusal:
        assert refusal.operation == "offline_probe", "EX-04: refusal operation identity lost"
        return
    pytest.fail("EX-04: HTTP 302 with no Location accepted as successful response data")


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_empty_json_success_is_contract_failure__EX05(mode):
    def response(request):
        return httpx.Response(200, content=b"", headers={"content-type": "application/json"})

    try:
        asyncio.run(
            invoke(
                mode,
                response,
                lambda transport: transport.request_json(
                    "GET", "/probe", operation_id="offline_probe"
                ),
            )
        )
    except ContractMismatchError as refusal:
        assert refusal.operation == "offline_probe", "EX-05: empty response failure loses operation"
        assert refusal.__cause__ is not None, "EX-05: parsing cause must survive translation"
        return
    pytest.fail("EX-05: empty HTTP 200 JSON response fabricated as {}")


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_number_is_contract_failure__EX07(mode, token):
    def response(request):
        return httpx.Response(
            200,
            content=('{"measurement":' + token + "}").encode(),
            headers={"content-type": "application/json"},
        )

    try:
        asyncio.run(
            invoke(
                mode,
                response,
                lambda transport: transport.request_json(
                    "GET", "/probe", operation_id="offline_probe"
                ),
            )
        )
    except ContractMismatchError as refusal:
        assert refusal.operation == "offline_probe", "EX-07: invalid JSON failure loses operation"
        assert refusal.retryable is False, "EX-07: malformed JSON is not transient"
        assert refusal.__cause__ is not None, "EX-07: parsing cause must survive translation"
        return
    pytest.fail(f"EX-07: non-JSON numeric token {token} passed strict JSON boundary")
