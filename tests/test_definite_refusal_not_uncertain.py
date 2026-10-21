"""A definite typed refusal of a mutation is never "outcome uncertain" (SYN-4346)."""

from __future__ import annotations

import httpx
import pytest
from synth_ai.contracts.swarms import CREATE_PROJECT
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.http.response import decode_response
from synth_ai.errors import (
    ConflictError,
    OutcomeUncertainError,
    PaymentRequiredError,
    PermissionDeniedError,
    RefusedError,
    SynthError,
)
from synth_ai.options import ClientOptions, MutationOptions
from synth_ai.sdk.swarms.types.projects import ProjectCreateRequest, ProjectTemplate
from synth_ai.types.base import ClientTiming, ResponseModel
from synth_ai.types.identity import MutationIdentity

pytestmark = pytest.mark.unit

CONTRACT = CREATE_PROJECT.contract


def _body(code: str, **extra: object) -> dict:
    return {"error": {"code": code, "request_id": "455b3858ede542e894a4dfdeb3107119", **extra}}


def _decode(status: int, payload: dict | bytes, *, mutation: bool = True) -> None:
    response = httpx.Response(
        status,
        headers={"X-Synth-Contract": CONTRACT, "Content-Type": "application/json"},
        **({"content": payload} if isinstance(payload, bytes) else {"json": payload}),
    )
    identity = MutationIdentity.create("op", b"{}", "/api/v1/x", key="k")
    decode_response(
        response,
        response.content,
        ResponseModel,
        ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
        identity,
        mutation,
        CONTRACT,
    )


def test_402_index_payment_required_is_a_definite_typed_refusal():
    with pytest.raises(PaymentRequiredError) as caught:
        _decode(402, _body("index_payment_required"))
    error = caught.value
    assert not isinstance(error, OutcomeUncertainError)
    assert (error.status, error.code, error.retryable) == (402, "index_payment_required", False)
    assert error.request_id == "455b3858ede542e894a4dfdeb3107119"


@pytest.mark.parametrize(
    ("status", "code", "kind"),
    [
        (403, "forbidden_thing", PermissionDeniedError),
        (409, "version_conflict", ConflictError),
        (422, "unprocessable", RefusedError),
        (400, "bad_request", RefusedError),
    ],
)
def test_typed_4xx_mutation_refusals_are_definite(status, code, kind):
    with pytest.raises(kind) as caught:
        _decode(status, _body(code))
    assert not isinstance(caught.value, OutcomeUncertainError)
    assert caught.value.retryable is False and caught.value.code == code


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
def test_ambiguous_statuses_stay_uncertain_for_mutations(status):
    with pytest.raises(OutcomeUncertainError):
        _decode(status, _body("something"))


def test_server_declared_unknown_outcome_wins_over_a_4xx():
    with pytest.raises(OutcomeUncertainError):
        _decode(409, _body("in_flight", outcome="unknown"))


def test_server_declared_not_accepted_makes_a_5xx_definite():
    with pytest.raises(SynthError) as caught:
        _decode(503, _body("shed", outcome="not_accepted"))
    assert not isinstance(caught.value, OutcomeUncertainError)
    assert caught.value.retryable is False


def test_5xx_without_a_typed_body_is_uncertain():
    with pytest.raises(OutcomeUncertainError):
        _decode(502, b'{"detail":"bad gateway"}')


def _transport(handler) -> HttpTransport:
    transport = HttpTransport("https://backend.test/api/v1", "key", ClientOptions())
    transport.client.close()
    transport.client = httpx.Client(transport=httpx.MockTransport(handler))
    transport._negotiate = lambda operation, deadline: None  # type: ignore[method-assign]
    return transport


def _create(transport: HttpTransport) -> None:
    transport.request(
        CREATE_PROJECT,
        ResponseModel,
        body=ProjectCreateRequest(name="n", template=ProjectTemplate.REPORT_NATIVE, objective="o"),
        options=MutationOptions(),
    )


def test_wire_402_raises_payment_required_not_uncertain():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            headers={"X-Synth-Contract": CONTRACT},
            json=_body("index_payment_required"),
        )

    with pytest.raises(PaymentRequiredError) as caught:
        _create(_transport(handler))
    assert caught.value.retryable is False


def test_timeout_after_send_is_still_uncertain():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(OutcomeUncertainError):
        _create(_transport(handler))
