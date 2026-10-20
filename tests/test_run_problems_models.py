"""`run.status().problems` and `run.status().failure.reason_chain` (synth.problems.v1).

The payloads are the customer wire the backend renders from the recorded
owner-wire fixtures (tests/fixtures/orchestra_problems_owner_wire.json).
"""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from synth_ai.errors import OperationFailedError
from synth_ai.sdk.swarms import Problem, ReasonLink, RunSnapshot
from synth_ai.sdk.swarms.types.runs import RunFailure, SwarmResult

pytestmark = pytest.mark.unit

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "orchestra_problems_owner_wire.json").read_text()
)
AT = datetime(2026, 10, 16, 15, 0, tzinfo=UTC)
METADATA = {
    "contract": "swarms.v1",
    "request_id": "req-1",
    "operation_id": None,
    "observed_at": AT,
}


def customer_problem(run: str) -> dict:
    wire = deepcopy(FIXTURE[run]["problems"][0])
    wire["since"] = datetime.fromtimestamp(wire.pop("since_unix_ms") / 1000, UTC)
    wire["last_seen"] = datetime.fromtimestamp(wire.pop("last_seen_unix_ms") / 1000, UTC)
    return wire


def wire(model, payload):
    """Validate the way the SDK does: from the JSON the server sent."""
    return model.model_validate_json(json.dumps(payload, default=lambda v: v.isoformat()))


def snapshot_payload(**extra) -> dict:
    return {
        "metadata": METADATA,
        "run_id": "5ca83a99-b111-4f28-b888-1f35ae3859c3",
        "project_id": "p-1",
        "state": "failed",
        "model": "gpt-6-luna",
        "provider": "openai",
        "result_available": True,
        "execution_closed": True,
        "accounting_final": True,
        "cleanup_complete": True,
        **extra,
    }


def test_status_exposes_problems_and_a_root_cause_failure_chain() -> None:
    problem = customer_problem("5ca83a99-b111-4f28-b888-1f35ae3859c3")
    status = wire(
        RunSnapshot,
        snapshot_payload(
            problems=[problem],
            problems_observed=True,
            failure={
                "code": problem["code"],
                "retryable": False,
                "problem_id": problem["id"],
                "reason_chain": problem["reason_chain"],
            },
        ),
    )
    assert status.problems[0].code == "provider_refused_prompt"
    assert status.problems[0].origin == "provider:openai"
    assert status.failure is not None
    assert status.failure.reason_chain[0].service == "provider:openai"
    assert [link.service for link in status.failure.reason_chain][:3] == [
        "provider:openai",
        "gateway",
        "agents",
    ]
    assert isinstance(status.problems[0].reason_chain[0], ReasonLink)
    assert isinstance(status.problems[0], Problem)


def test_a_response_from_an_older_server_still_validates() -> None:
    status = wire(RunSnapshot, snapshot_payload(state="completed"))
    assert status.problems == () and status.failure is None
    assert status.problems_observed is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["reason_chain"][0].update(cause=None),
        lambda p: p.update(reason_chain=[]),
        lambda p: p.update(origin="somewhere"),
    ],
)
def test_malformed_known_problem_values_are_refused(mutate) -> None:
    problem = customer_problem("f40b6a07-c2c9-4221-b24c-ed763cb81d00")
    mutate(problem)
    with pytest.raises(ValidationError):
        wire(RunSnapshot, snapshot_payload(problems=[problem]))


def test_future_problem_code_and_fields_are_observed_without_interpreting_them():
    from synth_ai.types.unknown import UnknownValue

    problem = customer_problem("f40b6a07-c2c9-4221-b24c-ed763cb81d00")
    problem.update(code="future_problem_code", future_evidence=True)
    result = wire(RunSnapshot, snapshot_payload(problems=[problem]))
    assert isinstance(result.problems[0].code, UnknownValue)
    assert result.problems[0].code == "future_problem_code"


def test_raise_for_failure_names_the_root_cause_not_just_a_code() -> None:
    problem = customer_problem("5ca83a99-b111-4f28-b888-1f35ae3859c3")
    failure = wire(
        RunFailure,
        {
            "code": problem["code"],
            "retryable": False,
            "problem_id": problem["id"],
            "reason_chain": problem["reason_chain"],
        },
    )
    result = SwarmResult.model_construct(
        metadata=wire(RunSnapshot, snapshot_payload()).metadata,
        status="failed",
        failure=failure,
    )
    with pytest.raises(OperationFailedError) as error:
        result.raise_for_failure()
    assert error.value.code == "provider_refused_prompt"
    assert "provider:openai/invalid_prompt" in str(error.value)
