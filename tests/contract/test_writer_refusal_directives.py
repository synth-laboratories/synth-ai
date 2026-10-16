"""SYN-3998: explicit backend fence directives override compatibility defaults."""

import pytest
from synth_ai.core.http.request import HttpMethod, HttpRequest, OperationId, OperationMetadata
from synth_ai.core.http.retry import RetryPolicy, should_retry_failure
from synth_ai.sdk.research.errors import ResearchLaunchRefusalError


@pytest.mark.parametrize("retryable", [False, True])
def test_fence_retains_authoritative_retry_flag__SYN3998(retryable):
    error = ResearchLaunchRefusalError(
        "fenced",
        status_code=409,
        detail={"error_code": "transfer_fence_active", "retryable": retryable},
        operation_id="launch",
    )
    assert error.retryable is retryable
    request = HttpRequest(
        OperationMetadata(
            OperationId("launch"), HttpMethod.POST, "/launch", mutation=True, idempotent=True
        ),
        "/launch",
        headers={"Idempotency-Key": "original"},
    )
    assert should_retry_failure(RetryPolicy(), request, error) is retryable


def test_nonboolean_retry_flag_is_contract_error__SYN3998():
    with pytest.raises(ValueError):
        ResearchLaunchRefusalError(
            "fenced",
            status_code=409,
            detail={"error_code": "transfer_fence_active", "retryable": "false"},
            operation_id="launch",
        )
