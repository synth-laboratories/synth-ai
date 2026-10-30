"""A definite ``synth.refusal.v1`` never surfaces as an uncertain outcome (SYN-5273).

Root, 2026-10-30 06:46Z: ``index.contributions.attest`` request
6cf0247d03784c528947a43bc3d44343 returned 409 ``contribution_transition_refused``
with a refusal document that said ``retryable=false, in_doubt=false`` while the
legacy ``outcome`` said ``unknown``; the SDK raised ``OutcomeUncertainError``.
"""

from __future__ import annotations

import httpx
import pytest
from synth_ai.core.http.response import decode_response
from synth_ai.errors import ConflictError, OutcomeUncertainError
from synth_ai.types.base import ClientTiming, ResponseModel
from synth_ai.types.identity import MutationIdentity

pytestmark = pytest.mark.unit

REQUEST_ID = "6cf0247d03784c528947a43bc3d44343"
PATH = "/api/v1/index/contributions/f5b8ad3c-e1cf-4692-9675-51d838769872/attest"


def _document(**changes: object) -> dict:
    document = {
        "schema_version": "synth.refusal.v1",
        "code": "contribution_transition_refused",
        "service": "backend",
        "stage": "customer_api",
        "retryable": False,
        "in_doubt": False,
        "operation_id": None,
        "request_id": REQUEST_ID,
        "status": 409,
        "upstream": None,
        "detail": {},
    }
    document.update(changes)
    return document


def _decode(status: int, document: dict, *, outcome: str = "unknown") -> None:
    body = {
        "error": {
            "code": document["code"],
            "request_id": REQUEST_ID,
            "outcome": outcome,
            "retry_after_seconds": None,
            "refusal": document,
        }
    }
    response = httpx.Response(
        status,
        headers={
            "X-Synth-Contract": "index.v1",
            "X-Synth-Refusal-Contract": "synth.refusal.v1",
            "Content-Type": "application/json",
        },
        json=body,
    )
    identity = MutationIdentity.create(
        "index.contributions.attest",
        b"{}",
        PATH,
        key="mlok-craftax-public-20261030-attest-private-3",
    )
    decode_response(
        response,
        response.content,
        ResponseModel,
        ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
        identity,
        True,
        "index.v1",
        operation="index.contributions.attest",
    )


def test_roots_exact_409_refusal_is_a_typed_conflict_not_uncertain():
    with pytest.raises(ConflictError) as caught:
        _decode(409, _document())
    error = caught.value
    assert not isinstance(error, OutcomeUncertainError)
    assert (error.status, error.code, error.retryable) == (
        409,
        "contribution_transition_refused",
        False,
    )
    assert error.request_id == REQUEST_ID
    assert error.refusal is not None and error.refusal.in_doubt is False


def test_typed_cause_detail_reaches_the_caller():
    detail = {
        "cause": "rights_holder_required",
        "cause_kind": "standing_mismatch",
        "operation": "index.contributions.attest",
        "standing": "holds_qa_capability",
    }
    with pytest.raises(ConflictError) as caught:
        _decode(
            409,
            _document(stage="index_contribution_attest", detail=detail),
            outcome="not_accepted",
        )
    assert dict(caught.value.refusal.detail) == detail


def test_document_in_doubt_stays_uncertain():
    with pytest.raises(OutcomeUncertainError):
        _decode(409, _document(in_doubt=True))


def test_document_retryable_stays_uncertain_under_unknown_outcome():
    with pytest.raises(OutcomeUncertainError):
        _decode(409, _document(retryable=True))


@pytest.mark.parametrize("status", [429, 503])
def test_ambiguous_status_document_stays_uncertain(status):
    with pytest.raises(OutcomeUncertainError):
        _decode(status, _document(status=status))


@pytest.mark.parametrize("code", ["contribution_outcome_unavailable", "ledger_outcome_unavailable"])
def test_outcome_unavailable_stays_uncertain_even_when_in_doubt_false(code):
    # Older/prod backends emit in_doubt=false here, but the outcome is uncertain.
    with pytest.raises(OutcomeUncertainError):
        _decode(409, _document(code=code, in_doubt=False))
