import json

import httpx
import pytest

from synth_ai._generated.models import RunLabelsRequest
from synth_ai._transport import RetryPolicy, Transport
from synth_ai.errors import (
    ResponseContractError,
    SwarmsAPIError,
    TransportFailureError,
    local_problem,
)


@pytest.mark.parametrize(
    "operation,path",
    [
        ("updateRunLabels", {"run_id": "run_one"}),
        ("updateApiKey", {"key_id": "key_one"}),
        ("cancelRun", {"run_id": "run_one"}),
        ("createUpload", None),
        ("finalizeUpload", {"upload_id": "upload_one"}),
        ("createCheckout", None),
        ("createApiKey", None),
        ("rotateApiKey", {"key_id": "key_one"}),
    ],
)
async def test_malformed_mutation_success_never_claims_no_effect(operation, path):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    ) as http:
        transport = Transport("https://example.test", "token", client=http)
        with pytest.raises(ResponseContractError) as caught:
            await transport.invoke(operation, path=path, idempotency_key="retained")
    assert caught.value.outcome == "unknown"
    assert caught.value.request_id


@pytest.mark.parametrize(
    "fault,unknown",
    [
        (httpx.ConnectError, False),
        (httpx.ConnectTimeout, False),
        (httpx.PoolTimeout, False),
        (httpx.WriteError, True),
        (httpx.ReadError, True),
        (httpx.ReadTimeout, True),
    ],
)
async def test_transport_phase_classifies_uncertainty_without_retry(fault, unknown):
    calls = []

    def handle(request):
        calls.append(request)
        raise fault("private transport context", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        transport = Transport("https://example.test", "token", client=http)
        with pytest.raises(TransportFailureError) as caught:
            await transport.invoke(
                "cancelRun", path={"run_id": "run_one"}, idempotency_key="retained"
            )
    assert (caught.value.outcome == "unknown") == unknown
    assert caught.value.idempotency_key == "retained"
    assert caught.value.request_id == calls[0].headers["x-request-id"]
    assert len(calls) == 1 and "private" not in str(caught.value)


@pytest.mark.parametrize("body", [{}, {"outcome": "unknown"}, {"outcome": "not_accepted"}])
async def test_unverified_mutation_refusal_is_not_retried(body):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        transport = Transport(
            "https://example.test",
            "token",
            client=http,
            retry_policy=RetryPolicy(backoff_seconds=0, jitter_seconds=0),
        )
        with pytest.raises(SwarmsAPIError) as caught:
            await transport.invoke(
                "cancelRun", path={"run_id": "run_one"}, idempotency_key="retained"
            )
    assert len(calls) == 1
    assert caught.value.idempotency_key == "retained" and caught.value.request_id


async def test_retry_freezes_nested_payload_and_identity():
    # Use an open generated-independent payload to exercise the serialization boundary.
    body = {"metadata": {"tags": ["original"]}}
    calls = []

    def handle(request):
        calls.append(request)
        body["metadata"]["tags"].append("changed")
        return httpx.Response(503, json=local_problem("busy", "admission").model_dump(mode="json"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        transport = Transport(
            "https://example.test",
            "token",
            client=http,
            retry_policy=RetryPolicy(attempts=2, backoff_seconds=0, jitter_seconds=0),
        )
        with pytest.raises(SwarmsAPIError):
            await transport.invoke(
                "cancelRun", path={"run_id": "run_one"}, body=body, idempotency_key="retained"
            )
    assert len(calls) == 2
    assert calls[0].content == calls[1].content
    assert json.loads(calls[1].content) == {"metadata": {"tags": ["original"]}}
    assert calls[0].headers["x-request-id"] == calls[1].headers["x-request-id"]
    assert calls[1].headers["idempotency-key"] == "retained"


async def test_model_input_is_snapshotted_before_send():
    labels = RunLabelsRequest(labels={"tag": "original"}, expected_revision=1)

    def handle(request):
        labels.labels["tag"] = "changed"
        assert json.loads(request.content)["labels"]["tag"] == "original"
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        transport = Transport("https://example.test", "token", client=http)
        with pytest.raises(ResponseContractError):
            await transport.invoke("updateRunLabels", path={"run_id": "run_one"}, body=labels)


async def test_legacy_unknown_flag_overrides_a_not_accepted_problem():
    body = local_problem("pending", "admission").model_dump(mode="json")
    body["outcome_unknown"] = True
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        transport = Transport(
            "https://example.test",
            "token",
            client=http,
            retry_policy=RetryPolicy(backoff_seconds=0, jitter_seconds=0),
        )
        with pytest.raises(SwarmsAPIError):
            await transport.invoke("cancelRun", path={"run_id": "run_one"})
    assert len(calls) == 1
