"""SYN-3993/4007: a malformed successful write reply retains uncertain intent."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from synth_ai.sdk.research.contracts.forge.contracts import ExactReference, contract_digest
from synth_ai.sdk.research.contracts.forge.operations import PublicWrite
from synth_ai.sdk.research.contracts.forge.records import Measurement, Result
from synth_ai.sdk.research.errors import ResearchOutcomeUncertainError
from synth_ai.sdk.research.scientific_records import (
    AsyncScientificRecordsAPI,
    ExecutionWriteRequest,
    ScientificRecordsAPI,
)

PROJECT = "00000000-0000-4000-8000-00000000000b"
ORG = "00000000-0000-4000-8000-00000000000a"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("invalid", ["missing", "operation", "project", "digest", "organization"])
def test_invalid_write_receipt_retains_original_operation_and_cause__SYN3993(mode, invalid):
    reference = ExactReference(
        authority="forge", kind="trial", record_id="trial", revision="1", digest_sha256="a" * 64
    )
    payload = Result(
        trial=reference,
        outcome="null",
        measurements=(Measurement(name="reward", value=None, unit="score"),),
        missing_evidence=("measurement unavailable",),
    )
    values = {
        "operation_id": "original-intent",
        "record_id": "result",
        "expected_revision": 0,
        "payload": payload.model_dump(mode="json"),
    }
    request = ExecutionWriteRequest(
        organization_id=ORG,
        admission_id="adm_" + "0" * 26,
        write=PublicWrite(**values, request_digest_sha256=contract_digest(values)),
    )
    valid = {
        "schema_version": "forge.receipt.v1",
        "scope": {"organization_id": ORG, "project_id": PROJECT},
        "operation_id": "original-intent",
        "payload_digest_sha256": "b" * 64,
        "reference": {
            "authority": "forge",
            "kind": "result",
            "record_id": "result",
            "revision": "1",
            "digest_sha256": contract_digest(payload),
        },
        "cursor": 1,
        "recorded_at": "2026-10-06T00:00:00Z",
    }
    document = deepcopy(valid)
    if invalid == "missing":
        document = {}
    elif invalid == "operation":
        document["operation_id"] = "another-intent"
    elif invalid == "project":
        document["scope"]["project_id"] = "another-project"
    elif invalid == "digest":
        document["reference"]["digest_sha256"] = "c" * 64
    else:
        document["scope"]["organization_id"] = "another-organization"
    calls = []

    def submit(*args, **kwargs):
        calls.append(kwargs)
        return document

    async def submit_async(*args, **kwargs):
        return submit(*args, **kwargs)

    with pytest.raises(ResearchOutcomeUncertainError) as caught:
        if mode == "sync":
            ScientificRecordsAPI(SimpleNamespace(request_json=submit)).execution_write(
                PROJECT, request
            )
        else:
            asyncio.run(
                AsyncScientificRecordsAPI(
                    SimpleNamespace(request_json=submit_async)
                ).execution_write(PROJECT, request)
            )
    error = caught.value
    assert error.operation_id == "original-intent"
    assert str(error.failure.code) == "outcome_uncertain"
    assert error.failure.operation == "original-intent"
    assert error.failure.retry.retryable is False
    assert error.detail["cause_code"] == "scientific_receipt_invalid"
    assert error.detail["mutation_applied"] is None
    assert isinstance(error.__cause__, ValueError)
    assert len(calls) == 1 and calls[0]["operation_id"] == "original-intent"
