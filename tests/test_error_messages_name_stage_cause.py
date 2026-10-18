"""SDK errors name code, status, stage, cause and request id; closure headers are typed."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from synth_ai.contracts.swarms import CREATE_PROJECT
from synth_ai.core.http.response import decode_response
from synth_ai.errors import ConflictError, RequestTimeoutError, SynthError
from synth_ai.types.base import ClientTiming, ResponseModel

CONTRACT = CREATE_PROJECT.contract
REQUEST_ID = "455b3858ede542e894a4dfdeb3107119"


def _error(status: int, code: str, headers: dict[str, str] | None = None) -> SynthError:
    response = httpx.Response(
        status,
        headers={"X-Synth-Contract": CONTRACT, "Content-Type": "application/json", **(headers or {})},
        json={"error": {"code": code, "request_id": REQUEST_ID}},
    )
    with pytest.raises(SynthError) as caught:
        decode_response(
            response,
            response.content,
            ResponseModel,
            ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
            None,
            False,
            CONTRACT,
        )
    return caught.value


def test_408_with_stage_and_cause_renders_both():
    error = _error(
        408,
        "request_deadline_exceeded",
        {"X-Synth-Refusal-Stage": "runtime_binding_place", "X-Synth-Refusal-Cause": "deadline"},
    )
    assert isinstance(error, RequestTimeoutError)
    assert (error.stage, error.cause) == ("runtime_binding_place", "deadline")
    assert str(error) == (
        "RequestTimeoutError: request_deadline_exceeded (status 408) "
        "stage=runtime_binding_place cause=deadline request_id=" + REQUEST_ID
    )


def test_409_closure_headers_set_typed_fields():
    error = _error(
        409,
        "run_admission_expired",
        {
            "X-Synth-Closure-Pending": "deadline",
            "X-Synth-Closure-Expected-By": "2026-10-17T12:00:00+00:00",
        },
    )
    assert isinstance(error, ConflictError)
    assert error.closure_pending == "deadline"
    assert error.closure_expected_by == datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
    assert "closure_pending=deadline closure_expected_by=2026-10-17T12:00:00+00:00" in str(error)


def test_absent_headers_leave_fields_empty_and_message_minimal():
    error = _error(408, "request_deadline_exceeded")
    assert (error.stage, error.cause, error.closure_pending, error.closure_expected_by) == (
        None,
        None,
        None,
        None,
    )
    assert str(error) == (
        f"RequestTimeoutError: request_deadline_exceeded (status 408) request_id={REQUEST_ID}"
    )


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Synth-Closure-Pending": "deadline"},
        {"X-Synth-Closure-Pending": "deadline", "X-Synth-Closure-Expected-By": "2026-10-17T12:00"},
        {"X-Synth-Closure-Pending": "deadline", "X-Synth-Closure-Expected-By": "soon"},
    ],
)
def test_incomplete_or_naive_closure_headers_are_dropped(headers):
    error = _error(409, "run_admission_expired", headers)
    assert (error.closure_pending, error.closure_expected_by) == (None, None)


def test_malformed_stage_is_dropped_and_other_headers_never_rendered():
    error = _error(
        408,
        "request_deadline_exceeded",
        {"X-Synth-Refusal-Stage": "bad stage; secret", "Authorization": "Bearer sk-secret"},
    )
    assert error.stage is None
    assert "secret" not in str(error)


def test_error_without_metadata_keeps_plain_message():
    assert str(SynthError("plain failure")) == "plain failure"
