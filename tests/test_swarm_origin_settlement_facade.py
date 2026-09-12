"""Public facade for the E01 journey: find an Intern's swarms, read settlement and usage."""

from __future__ import annotations

from typing import Any

import pytest
from synth_ai.core.http.request import HttpRequest
from synth_ai.sdk.research.contracts.common import ProjectId, SwarmId
from synth_ai.sdk.research.operations import research_operation
from synth_ai.sdk.research.research_intern import ResearchInternAPI
from synth_ai.sdk.research.swarms import SwarmsAPI

NOW = "2026-09-11T12:00:00Z"
SWARM = {
    "run_id": "run-1",
    "project_id": "proj-1",
    "org_id": "org-1",
    "public_state": "running",
    "runbook": "lite",
    "trigger": "intern",
    "created_at": NOW,
    "updated_at": NOW,
    "origin_runtime_kind": "sync",
    "origin_runtime_id": "sync-1",
}
SETTLEMENT = {
    "run_id": "run-1",
    "observed_at": NOW,
    "coverage": "explicit-v1",
    "settled": False,
    "pending": 2,
}
USAGE = {
    "origin_runtime_kind": "sync",
    "origin_runtime_id": "sync-1",
    "org_id": "org-1",
    "spend_cents": 0,
    "token_count": 0,
    "run_count": 1,
    "billing": {
        "billing_state": "pending",
        "stalled_row_count": 0,
        "stalled_spend_cents": 0,
        "billing_failed_row_count": 0,
        "message": None,
        "show_stalled_banner": False,
    },
    "bindings": [],
    "metadata": {"run_ids": ["run-1"]},
}


class _Transport:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.requests: list[HttpRequest] = []

    def execute(self, request: HttpRequest) -> Any:
        self.requests.append(request)
        return self.responses[str(request.operation.operation_id)]


@pytest.mark.parametrize(
    ("operation_id", "path"),
    [
        ("get_run_resource_settlement", "/smr/runs/{run_id}/resource-settlement"),
        (
            "get_intern_sync_session_usage",
            "/smr/research-intern/sync-sessions/{sync_session_id}/usage",
        ),
        (
            "get_intern_async_assignment_usage",
            "/smr/research-intern/async-assignments/{assignment_id}/usage",
        ),
    ],
)
def test_new_operations_are_registered(operation_id: str, path: str) -> None:
    assert research_operation(operation_id).path_template == path


def test_list_swarms_by_intern_origin() -> None:
    transport = _Transport({"list_project_runs": [SWARM]})
    page = SwarmsAPI(transport).list(  # type: ignore[arg-type]
        ProjectId("proj-1"), origin_runtime_kind="sync", origin_runtime_id="sync-1"
    )

    assert [swarm.origin_runtime_id for swarm in page.items] == ["sync-1"]
    assert transport.requests[0].query == {
        "limit": 100,
        "origin_runtime_kind": "sync",
        "origin_runtime_id": "sync-1",
    }


def test_half_origin_filter_fails_before_any_request() -> None:
    transport = _Transport({})
    with pytest.raises(ValueError, match="set together"):
        SwarmsAPI(transport).list(ProjectId("proj-1"), origin_runtime_kind="sync")  # type: ignore[arg-type]
    assert transport.requests == []


def test_swarm_resource_settlement() -> None:
    transport = _Transport({"get_run_resource_settlement": SETTLEMENT})
    settlement = SwarmsAPI(transport).resource_settlement(SwarmId("run-1"))  # type: ignore[arg-type]

    assert settlement.settled is False and settlement.pending == 2
    assert transport.requests[0].path == "/smr/runs/run-1/resource-settlement"


def test_sync_session_usage_checks_identity() -> None:
    sync = ResearchInternAPI(_Transport({"get_intern_sync_session_usage": USAGE})).sync_  # type: ignore[arg-type]
    assert sync.usage("sync-1").run_ids == ("run-1",)

    other = ResearchInternAPI(_Transport({"get_intern_sync_session_usage": USAGE})).sync_  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identity drifted"):
        other.usage("sync-2")
