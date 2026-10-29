"""An Orchestra-owned lifecycle guard refusal is typed and still uncertain (SYN-4642)."""

from __future__ import annotations

import httpx
import pytest
from synth_ai.contracts.swarms import CREATE_PROJECT
from synth_ai.core.http.response import decode_response
from synth_ai.errors import (
    ConflictError,
    LifecycleWriteRefusedError,
    OutcomeUncertainError,
)
from synth_ai.types.base import ClientTiming, ResponseModel
from synth_ai.types.identity import MutationIdentity

pytestmark = pytest.mark.unit

CONTRACT = CREATE_PROJECT.contract
IDENTITY = MutationIdentity.create("swarms.cancel", b"", "/api/v1/swarms/runs/r/cancel", key="k")


def _decode(status: int, code: str, *, outcome: str | None, mutation: bool = True) -> None:
    error: dict[str, object] = {"code": code, "request_id": "455b3858ede542e894a4dfdeb3107119"}
    if outcome is not None:
        error["outcome"] = outcome
    response = httpx.Response(
        status,
        headers={"X-Synth-Contract": CONTRACT, "Content-Type": "application/json"},
        json={"error": error},
    )
    decode_response(
        response,
        response.content,
        ResponseModel,
        ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
        IDENTITY,
        mutation,
        CONTRACT,
    )


@pytest.mark.parametrize("code", ["orchestra_lifecycle_write_refused", "lifecycle_write_not_owner"])
def test_guard_refusal_is_typed_and_keeps_the_mutation_identity(code):
    with pytest.raises(LifecycleWriteRefusedError) as caught:
        _decode(503, code, outcome="unknown")
    error = caught.value
    # Existing OutcomeUncertainError handlers keep working.
    assert isinstance(error, OutcomeUncertainError)
    assert (error.status, error.code, error.identity) == (503, code, IDENTITY)


def test_other_uncertain_codes_stay_plain():
    with pytest.raises(OutcomeUncertainError) as caught:
        _decode(503, "orchestra_unavailable", outcome="unknown")
    assert type(caught.value) is OutcomeUncertainError


def test_a_definite_guard_refusal_is_not_reported_as_uncertain():
    with pytest.raises(ConflictError) as caught:
        _decode(409, "lifecycle_write_not_owner", outcome="not_accepted")
    assert not isinstance(caught.value, OutcomeUncertainError)
    assert caught.value.code == "lifecycle_write_not_owner"
