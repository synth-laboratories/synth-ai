"""Outputs list carries additive public execution verifications (SYN-4548)."""

import httpx
import pytest
from pydantic import ValidationError
from synth_ai import SynthClient
from synth_ai.sdk.swarms.outputs import Outputs
from synth_ai.sdk.swarms.types.outputs import OutputPage, OutputVerification

pytestmark = pytest.mark.unit
METADATA = {
    "contract": "swarms.v1",
    "request_id": "outputs-fixture",
    "operation_id": None,
    "observed_at": "2026-10-28T20:00:00Z",
}
OPERATION = "a" * 64


def page(**extra):
    return {"metadata": METADATA, "items": [], "next_cursor": None, "complete": True, **extra}


def read(body) -> OutputPage:
    with SynthClient(api_key="fixture-key", base_url="https://backend.test") as client:
        client._transport.client.close()
        client._transport.client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"X-Synth-Contract": "swarms.v1"}, json=body
                )
            )
        )
        client._transport._negotiate = lambda *args: None
        return Outputs(client._transport, "run-1").list(limit=10)


def test_outputs_list_returns_verifications_from_first_page():
    result = read(
        page(
            verifications=[
                {
                    "operation_id": OPERATION,
                    "status": "failed",
                    "attempts": 1,
                    "error": {
                        "code": "public_execution_refused",
                        "refusal_code": "public_execution_unchanged_baseline",
                        "message": "submission did not change the baseline",
                    },
                },
                {
                    "operation_id": "b" * 64,
                    "status": "pending",
                    "reason": "verifier_unavailable_retrying",
                    "attempts": 0,
                },
            ]
        )
    )
    assert isinstance(result, OutputPage)
    failed, pending = result.verifications
    assert failed.error is not None
    assert failed.error.refusal_code == "public_execution_unchanged_baseline"
    assert failed.error.retryable is False
    assert pending.reason == "verifier_unavailable_retrying"


def test_older_server_without_verifications_reads_as_empty():
    assert read(page()).verifications == ()


@pytest.mark.parametrize(
    "value",
    [
        {"operation_id": OPERATION, "status": "pending", "attempts": 0},
        {"operation_id": OPERATION, "status": "passed", "reason": "late", "attempts": 1},
        {"operation_id": OPERATION, "status": "failed", "attempts": 1},
        {
            "operation_id": OPERATION,
            "status": "passed",
            "attempts": 1,
            "error": {"code": "x", "message": "y"},
        },
        {"operation_id": "not-hex", "status": "passed", "attempts": 1},
    ],
)
def test_verification_status_shape_is_closed(value):
    with pytest.raises(ValidationError):
        OutputVerification.model_validate(value)
