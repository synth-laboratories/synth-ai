"""Curated Intern APIs use generated owning schemas and stable typed errors."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from synth_ai.sdk.research.contracts.intern_authority import (
    InternRuntimeRouteSelectionRequest,
    InternTaskOperation,
)
from synth_ai.sdk.research.errors import ResearchApiError
from synth_ai.sdk.research.intern_authority import ResearchInternRuntimeRouteAPI

PRODUCER = json.loads(
    (
        Path(__file__).parents[2] / "synth_ai/sdk/research/contracts/intern_authority_producer.json"
    ).read_text()
)


def test_closed_task_vocabulary_and_route_request_match_generated_producer():
    assert (
        sorted(operation.value for operation in InternTaskOperation) == PRODUCER["task_operations"]
    )
    request = InternRuntimeRouteSelectionRequest(
        operation_id="owned-test", route="mloky", reason="owned acceptance"
    )
    Draft202012Validator(PRODUCER["models"]["InternRuntimeRouteSelectionRequest"]).validate(
        request.to_wire()
    )


def test_bad_runtime_wire_is_typed_and_never_prints_credential_marker():
    marker = "PRIVATE-CREDENTIAL-MARKER"

    class Transport:
        def execute(self, request):
            return {"schema_version": marker, "authorization": marker}

    with pytest.raises(ResearchApiError) as caught:
        ResearchInternRuntimeRouteAPI(Transport()).get()
    assert caught.value.operation_id == "get_intern_runtime_route"
    assert caught.value.failure.category == "contract_mismatch"
    assert caught.value.failure.retry.retryable is False
    assert marker not in str(caught.value)
    assert caught.value.__suppress_context__ is True
