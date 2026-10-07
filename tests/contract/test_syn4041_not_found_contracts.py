"""SYN-4041: typed scoped misses must survive the public SDK error boundary."""

import httpx
import pytest
from synth_ai.core.errors import ResourceRef, SynthErrorCode
from synth_ai.sdk.research.errors import ResearchNotFoundError
from synth_ai.sdk.research.transport.http import _raise_for_error_response


@pytest.mark.parametrize(
    "code,scope_field,scope,kind",
    [
        ("run_not_found", "run_id", "requested-run", "run"),
        ("intern_async_runtime_not_found", "org_id", "requested-org", "async_runtime"),
        (
            "intern_acceptance_fixture_retention_expired",
            "fixture_id",
            "requested-fixture",
            "acceptance_fixture",
        ),
    ],
)
def test_scoped_404_is_a_first_class_error__syn4041(code, scope_field, scope, kind):
    detail = {
        "error_code": code,
        "message": "Resource unavailable in this scope.",
        "retryable": False,
        scope_field: scope,
    }
    response = httpx.Response(
        404,
        json={"detail": detail},
        request=httpx.Request("GET", "https://example.invalid/smr/runs/requested"),
    )
    with pytest.raises(ResearchNotFoundError) as caught:
        _raise_for_error_response(response, operation_id="scoped.lookup")
    error = caught.value
    assert error.backend_error_code == code, "SYN-4041: backend refusal code lost"
    assert error.error_code == SynthErrorCode(code)
    assert error.resource == ResourceRef(kind, scope)
    assert error.lookup_resource == kind
    assert error.scope_identifier == scope
    assert error.operation_id == "scoped.lookup"
    assert error.failure.operation == "scoped.lookup"
    assert error.retryable is False
    assert error.status_code == 404
    assert error.detail == detail


def test_scoped_404_does_not_invent_a_resource_identity__syn4041():
    response = httpx.Response(
        404,
        json={"detail": {"error_code": "run_not_found", "retryable": False}},
        request=httpx.Request("GET", "https://example.invalid/smr/runs/requested"),
    )
    with pytest.raises(ResearchNotFoundError) as caught:
        _raise_for_error_response(response)
    assert caught.value.resource is None
    assert caught.value.scope_identifier is None
    assert caught.value.backend_error_code == "run_not_found"
