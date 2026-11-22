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
from synth_ai._generated.models import LimitMeterAction, RunLimitsProgress, RunProblem, RunSnapshot
from synth_ai.errors import ResponseContractError

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


# Orchestra 7e8fbbab reports planner_turns uncapped under actor limits, with no
# enforcement action (contract 1.24.0); slot2 Core 8a2a37eb served this row.
UNCAPPED_PLANNER_TURNS = {
    "dimension": "planner_turns",
    "unit": "count",
    "scope": "run",
    "effective_cap": None,
    "used": 165,
    "reserved": 0,
    "unknown_reserved": 0,
    "remaining": None,
    "state": "uncapped",
    "action": "none",
}


def _limits_with_planner_turns(row):
    secondary = [
        row if meter["dimension"] == "planner_turns" else meter for meter in LIMITS["secondary"]
    ]
    return {**LIMITS, "secondary": secondary}


async def test_uncapped_meter_action_none_parses_and_unknown_action_is_refused():
    limits = RunLimitsProgress.model_validate(_limits_with_planner_turns(UNCAPPED_PLANNER_TURNS))
    planner = next(m for m in limits.secondary if m.dimension == "planner_turns")
    assert planner.action is LimitMeterAction.NONE
    assert planner.effective_cap is None and planner.state == "uncapped"

    bodies = {"/none": _limits_with_planner_turns(UNCAPPED_PLANNER_TURNS)}
    bodies["/bogus"] = _limits_with_planner_turns({**UNCAPPED_PLANNER_TURNS, "action": "skip"})

    def handle(request):
        key = "/bogus" if request.headers.get("x-case") == "bogus" else "/none"
        return httpx.Response(200, json=bodies[key])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            read = await client.swarms.limits(SNAPSHOT["run_id"])
            assert read.secondary[5].action is LimitMeterAction.NONE
            http.headers["x-case"] = "bogus"
            with pytest.raises(ResponseContractError):
                await client.swarms.limits(SNAPSHOT["run_id"])
