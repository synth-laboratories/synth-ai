"""Boundary regressions: invalid inputs never turn into plausible wire values."""

import itertools
import json

import httpx
import pytest
from pydantic import ValidationError

from synth_ai import AsyncSynthClient, CompletionMilestone, RunState
from synth_ai._generated.models import (
    LaunchRequest,
    OwnerResourceRequest,
    RunEvent,
    RunSnapshot,
    RunTask,
    RunTaskPage,
    Upload,
    UploadGrant,
)
from synth_ai._stream import StreamCheckpoint, StreamCheckpointData
from synth_ai.errors import ErrorKind, ResponseContractError, WaitTimeoutError
from synth_ai.models import InputFile, RunInputs

RUN_ID = "run_AAAAAAAAAAAAAAAAAAAAAA"
EVENT_ID = "evt_AAAAAAAAAAAAAAAAAAAAAA"


def snapshot():
    return dict(
        run_id=RUN_ID,
        metadata={},
        labels={},
        labels_revision=1,
        objectives=[],
        guidance_coverage=[],
        state="running",
        model="test",
        provider="test",
        result_available=False,
        execution_closed=False,
        accounting_final=False,
        cleanup_complete=False,
        problems=[],
        problems_observed=True,
        created_at="2026-11-08T00:00:00Z",
        started_at=None,
        closed_at=None,
    )


def launch(**fields):
    return {
        "message": "hello",
        "limits": {"maximum_charge": {"amount": "1", "currency": "USD"}},
        **fields,
    }


@pytest.mark.parametrize("value", [True, "60", 60.5])
async def test_wrong_wall_time_type_is_rejected_before_any_request(value):
    sent = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: sent.append(request))
    ) as transport:
        async with AsyncSynthClient("test", http_client=transport) as client:
            with pytest.raises(ValidationError):
                await client.swarms.launch("hello", budget="1", wall_time_seconds=value)
    assert not sent


def test_named_inputs_tasks_and_upload_grants_are_records():
    request = LaunchRequest.model_validate(
        launch(
            inputs={
                "files": [
                    {
                        "file_id": "file_1",
                        "revision_id": "revision_1",
                        "digest_sha256": "a" * 64,
                    }
                ]
            }
        )
    )
    assert isinstance(request.inputs, RunInputs)
    assert isinstance(request.inputs.files[0], InputFile)
    assert isinstance(
        RunTaskPage(items=[{"task_id": "t", "state": "running"}], has_more=False).items[0], RunTask
    )
    assert Upload.model_fields["put"].annotation == UploadGrant | None
    with pytest.raises(ValidationError):
        request.message = "mutated"
    with pytest.raises(ValidationError):
        LaunchRequest.model_validate(launch(labels={"count": 3}))


@pytest.mark.parametrize("extension", [object(), float("nan"), {"nested": object()}])
def test_response_extensions_must_be_finite_json(extension):
    with pytest.raises(ValidationError):
        RunSnapshot.model_validate({**snapshot(), "vendor_extension": extension})


def test_unknown_state_is_preserved_without_claiming_a_known_state():
    record = RunSnapshot.model_validate({**snapshot(), "state": "awaiting_future_feature"})
    assert record.state == "awaiting_future_feature"
    assert record.known_state is None
    assert not record.finished
    assert RunSnapshot.model_validate(snapshot()).known_state is RunState.RUNNING


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_finished_requires_all_three_authoritative_flags(flags):
    record = RunSnapshot.model_validate(
        {
            **snapshot(),
            **dict(
                zip(
                    ["execution_closed", "accounting_final", "cleanup_complete"], flags, strict=True
                )
            ),
        }
    )
    assert record.finished == all(flags)
    assert record.reached(CompletionMilestone.FINISHED) == all(flags)
    assert record.reached(CompletionMilestone.EXECUTION_CLOSED) == flags[0]
    with pytest.raises(ValueError):
        record.reached("typo")


def event(**fields):
    return dict(
        event_id=EVENT_ID,
        run_id=RUN_ID,
        sequence=3,
        kind="objective.progress",
        original_kind="observation",
        usage=None,
        observed_at=None,
        approval_id=None,
        **fields,
    )


def test_event_kind_selects_payload_instead_of_guessing_from_shape():
    claim = {
        "objective_id": "goal",
        "claim_id": "claim",
        "percent_complete": 50,
        "kind": "progress",
        "summary": "halfway",
    }
    with pytest.raises(ValidationError):
        RunEvent.model_validate(event(payload=claim))
    valid = event(payload=claim)
    valid["kind"] = "objective.claimed"
    assert RunEvent.model_validate(valid).payload.claim_id == "claim"
    valid["kind"] = "unknown.future_event"
    valid["payload"] = None
    assert RunEvent.model_validate(valid).kind == "unknown.future_event"


async def test_malformed_success_preserves_request_identity_and_error_category():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={**snapshot(), "execution_closed": "false"})
        )
    ) as transport:
        async with AsyncSynthClient("test", http_client=transport) as client:
            with pytest.raises(ResponseContractError) as caught:
                await client.swarms.launch("hello", budget="1", idempotency_key="same-request")
    assert caught.value.kind is ErrorKind.CONTRACT
    assert caught.value.code == "response_contract_invalid"
    assert caught.value.idempotency_key == "same-request"
    assert caught.value.outcome == "unknown"
    assert caught.value.__cause__ is not None
    assert WaitTimeoutError("waiting").code == "wait_timeout"


def test_checkpoint_has_a_versioned_typed_persistence_format():
    state = StreamCheckpoint(run_id=RUN_ID, last_event_id="10", pending_event_ids={EVENT_ID})
    data = state.to_data()
    assert isinstance(data, StreamCheckpointData)
    assert data.schema_version == 1
    assert StreamCheckpoint.from_dict(json.loads(data.model_dump_json())) == state
    legacy = state.to_dict()
    legacy.pop("schema_version")
    assert StreamCheckpoint.from_dict(legacy) == state
    for version in [2, "1", True]:
        with pytest.raises(ValidationError):
            StreamCheckpoint.from_dict({**state.to_dict(), "schema_version": version})


def test_integer_literal_does_not_accept_boolean_equality():
    resource = {
        "plugin": "pool",
        "version": True,
        "roles": ["worker"],
        "limits": {"spend_micros": 1, "rollouts": 1},
        "selection": {"pool_ids": ["p"]},
    }
    with pytest.raises(ValidationError):
        OwnerResourceRequest.model_validate(resource)
    resource["version"] = 1
    assert OwnerResourceRequest.model_validate(resource).version == 1
