"""Real Core reads parse under the pinned contract.

The fixtures are raw slot1 reads (Core dev, attempt 21) of a run paused on an
actor limit. Under the earlier 1.17.0 pin, pause.limit.reason was an extra
field on a closed schema and every get() of such a run raised
ResponseContractError, which hung the qualification observers.
"""

import json
from pathlib import Path

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai._generated.models import RunLimitsProgress, RunProblem, RunSnapshot

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = json.loads((FIXTURES / "core_limit_paused_run_snapshot.json").read_text())
LIMITS = json.loads((FIXTURES / "core_limit_paused_run_limits.json").read_text())

# A worker turn refused by Agents' HTTP worker guard, as Core's RunProblem
# serializes it (core/internal/products/swarms/run_problems.go, 11c9aba7).
UPSTREAM_PROBLEM = {
    "id": "task:t1:task_worker_failed",
    "code": "task_worker_failed",
    "severity": "error",
    "service": "agents",
    "stage": "worker_turn",
    "upstream_status": 429,
    "upstream_status_basis": "reported",
    "upstream_code": "capacity_exhausted",
    "upstream_service": "agents",
    "upstream_stage": "http_worker_guard",
    "upstream_message": "http worker limit reached",
    "limiting_policy": "http_worker_limit",
    "upstream_detail": {"active": "32", "limit": "32"},
    "cause": "task t1's worker failed: agents http_worker_guard refused 429 capacity_exhausted",
    "in_doubt": False,
    "task_id": "t1",
    "since": "2026-11-12T02:40:00Z",
    "last_seen": "2026-11-12T02:41:00Z",
}


def test_real_limit_paused_snapshot_parses_with_pause_reason():
    run = RunSnapshot.model_validate_json(json.dumps(SNAPSHOT))
    assert run.pause is not None and run.pause.limit is not None
    assert run.pause.limit.reason == "planner_left_park_unfunded"
    assert run.pause.limit.scope == "actor"


def test_real_limits_view_parses():
    limits = RunLimitsProgress.model_validate_json(json.dumps(LIMITS))
    assert limits.run_id == SNAPSHOT["run_id"]


def test_problem_carries_upstream_refusal_chain():
    run = RunSnapshot.model_validate_json(json.dumps({**SNAPSHOT, "problems": [UPSTREAM_PROBLEM]}))
    assert run.problems is not None
    problem = run.problems[0]
    assert problem.limiting_policy == "http_worker_limit"
    assert problem.upstream_detail == {"active": "32", "limit": "32"}
    assert problem.upstream_status == 429


def test_problem_items_stay_open_but_require_code_and_cause():
    # A launch failure's problem carries the failure document's own fields.
    extra = RunProblem.model_validate(
        {"code": "launch_not_dispatched", "cause": "c", "family": "launch"}
    )
    assert extra.model_extra == {"family": "launch"}
    with pytest.raises(ValueError):
        RunProblem.model_validate({"code": "task_worker_failed"})


async def test_get_and_limits_read_real_core_bodies():
    def handle(request):
        body = LIMITS if request.url.path.endswith("/limits") else SNAPSHOT
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            run = await client.swarms.get(SNAPSHOT["run_id"])
            limits = await client.swarms.limits(SNAPSHOT["run_id"])
    assert run.pause is not None and run.pause.limit is not None
    assert run.pause.limit.reason == "planner_left_park_unfunded"
    assert limits.run_id == SNAPSHOT["run_id"]


async def test_transitive_control_progress_is_typed_without_asserting_physical_stop():
    control = {
        "operation_id": "core-operation",
        "runtime_operation_id": "producer-operation",
        "command": "pause",
        "state": "blocked",
        "origin": "compensation",
        "caused_by_operation_id": "expired-resume",
        "generation": 4,
        "requested_at": "2026-11-13T19:00:00Z",
        "deadline_at": "2026-11-13T19:02:00Z",
        "owners": [
            {"owner": "orchestra", "delivery_state": "acknowledged", "failure_code": None},
            {
                "owner": "resources",
                "delivery_state": "blocked",
                "failure_code": "physical_stop_unconfirmed",
            },
        ],
    }
    body = {
        **SNAPSHOT,
        "state": "pausing",
        "control": control,
        "execution_closed": False,
        "cleanup_complete": False,
        "accounting_final": False,
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            run = await client.swarms.get(SNAPSHOT["run_id"])
    assert run.control is not None and run.control.owners is not None
    assert run.control.model_dump(mode="json", exclude_none=True)["generation"] == 4
    assert (
        run.control.origin == "compensation"
        and run.control.caused_by_operation_id == "expired-resume"
    )
    assert run.control.runtime_operation_id == "producer-operation"
    assert run.control.owners[0].delivery_state == "acknowledged"
    assert run.control.owners[1].failure_code == "physical_stop_unconfirmed"
    assert not run.execution_closed and not run.cleanup_complete and not run.accounting_final


def test_control_owner_observation_requires_complete_identity_and_state():
    from synth_ai._generated.models import ControlOwner

    with pytest.raises(ValueError):
        ControlOwner.model_validate({"owner": "resources", "delivery_state": "acknowledged"})
    with pytest.raises(ValueError):
        ControlOwner.model_validate(
            {"owner": "resources", "delivery_state": "physically_stopped", "failure_code": None}
        )


@pytest.mark.parametrize(
    "proof",
    [
        {
            "runtime_quiescence": {
                "producer_sequence": 12,
                "state_digest": "a" * 64,
                "observed_at": "2026-11-13T21:00:00Z",
            }
        },
        {
            "runtime_absence": {
                "operation_id": "cb266ae2-94aa-4a6a-b164-1bb666314cf1",
                "generation": 1,
                "recorded_at_ms": 1794600000000,
            }
        },
    ],
)
async def test_physical_cancellation_preserves_pending_accounting(proof):
    body = {
        **SNAPSHOT,
        "state": "cancelled",
        "execution_closed": True,
        "cleanup_complete": False,
        "accounting_final": False,
        "control": {
            "operation_id": "cb266ae2-94aa-4a6a-b164-1bb666314cf1",
            "generation": 1,
            "command": "cancel",
            "state": "satisfied",
            **proof,
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            run = await client.swarms.get(SNAPSHOT["run_id"])
    assert run.control is not None and run.control.state == "satisfied"
    assert run.execution_closed and not run.accounting_final and not run.cleanup_complete
    assert run.control.runtime_quiescence is not None or run.control.runtime_absence is not None
