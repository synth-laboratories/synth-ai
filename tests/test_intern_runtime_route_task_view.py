"""Runtime-route (A07) and Task-authority view (T03) SDK methods; no network."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest
from pydantic import ValidationError
from synth_ai.core.http.request import HttpRequest
from synth_ai.sdk.research.contracts.intern_authority import (
    InternRuntimeRoute,
    InternRuntimeRouteSelectionRequest,
)
from synth_ai.sdk.research.errors import ResearchApiError
from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS
from synth_ai.sdk.research.research_intern import AsyncResearchInternAPI, ResearchInternAPI

ORG = "00000000-0000-0000-0000-000000000001"
INTERN = "d44b13c3-41fa-469b-a2b5-0482abe3b014"
PROJECT = "51072000-0000-0000-0000-00000000b002"
GRANT = "grt_" + "a" * 32
TASK = f"{ORG}:mloky-native-20261005-c6"
SELECT = InternRuntimeRouteSelectionRequest(
    operation_id="a07-route-1", route=InternRuntimeRoute.MLOKY, reason="A07 qualification"
)
ROUTE_VIEW = {
    "schema_version": "intern.runtime_route.v1",
    "org_id": ORG,
    "route": "mloky",
    "stored_route": "mloky",
    "default_route": "legacy_smr_runtime",
    "reason": "A07 qualification",
    "changed_by": "user:u",
    "updated_unix_ms": 1,
    "latest_receipt_id": "rrr_" + "b" * 32,
}
RECEIPT = {
    "schema_version": "intern.runtime_route_receipt.v1",
    "receipt_id": "rrr_" + "b" * 32,
    "org_id": ORG,
    "operation_id": "a07-route-1",
    "principal_ref": "user:u",
    "prior_route": None,
    "route": "mloky",
    "reason": "A07 qualification",
    "changed": True,
    "recorded_unix_ms": 2,
}
TASK_VIEW = {
    "schema_version": "intern.task_view.v1",
    "org_id": ORG,
    "intern_id": INTERN,
    "project_id": PROJECT,
    "sublinear_project_id": "sl-1",
    "grant_id": GRANT,
    "task_id": TASK,
    "title": "c6",
    "snapshot_revision": 7,
    "current_task": {"issue_id": TASK, "organization_id": ORG, "revision": 7},
    "entries": [{"revision": 7, "command": {"kind": "retain_attempt"}}],
    "next_cursor": None,
    "observed_unix_ms": 3,
}
RESPONSES = {
    "get_intern_runtime_route": ROUTE_VIEW,
    "select_intern_runtime_route": RECEIPT,
    "get_intern_task_view": TASK_VIEW,
}


class _Transport:
    def __init__(self, override: dict[str, Any] | None = None) -> None:
        self.requests: list[HttpRequest] = []
        self.responses = {**RESPONSES, **(override or {})}

    def execute(self, request: HttpRequest) -> Any:
        self.requests.append(request)
        return copy.deepcopy(self.responses[str(request.operation.operation_id)])


class _AsyncTransport(_Transport):
    async def execute(self, request: HttpRequest) -> Any:  # type: ignore[override]
        return _Transport.execute(self, request)


def _sent(t: _Transport) -> list[tuple[str, str, dict, Any]]:
    return [(str(r.operation.operation_id), r.path, dict(r.query), r.body) for r in t.requests]


def test_operations_match_backend_routes() -> None:
    ops = {str(k): v for k, v in RESEARCH_OPERATIONS.items()}
    for op, method, path in (
        ("get_intern_runtime_route", "GET", "/smr/research-intern/runtime-route"),
        ("select_intern_runtime_route", "PUT", "/smr/research-intern/runtime-route"),
        (
            "get_intern_task_view",
            "GET",
            "/smr/research-intern/task-grants/{grant_id}/tasks/{task_id}",
        ),
    ):
        assert str(ops[op].method.value).upper() == method
        assert ops[op].path_template == path
        assert ops[op].idempotent


def test_sync_and_async_arms_send_identical_requests() -> None:
    st, at = _Transport(), _AsyncTransport()
    sync = ResearchInternAPI(st)  # type: ignore[arg-type]
    results = [
        sync.runtime_route.get(),
        sync.runtime_route.select(SELECT),
        sync.task_views.get(GRANT, TASK, project_id=PROJECT, snapshot_revision=7, after_revision=1),
    ]

    async def drive() -> list[Any]:
        api = AsyncResearchInternAPI(at)  # type: ignore[arg-type]
        return [
            await api.runtime_route.get(),
            await api.runtime_route.select(SELECT),
            await api.task_views.get(
                GRANT, TASK, project_id=PROJECT, snapshot_revision=7, after_revision=1
            ),
        ]

    assert asyncio.run(drive()) == results
    assert _sent(st) == _sent(at)
    sent = _sent(st)
    assert sent[1][3] == {
        "schema_version": "intern.runtime_route_selection.v1",
        "operation_id": "a07-route-1",
        "route": "mloky",
        "reason": "A07 qualification",
    }
    assert (
        sent[2][1] == f"/smr/research-intern/task-grants/{GRANT}/tasks/{TASK.replace(':', '%3A')}"
    )
    assert sent[2][2] == {
        "project_id": PROJECT,
        "limit": 16,
        "snapshot_revision": 7,
        "after_revision": 1,
    }
    assert results[0].route is InternRuntimeRoute.MLOKY
    assert results[2].current_task["revision"] == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"route": "codex"},
        {"operation_id": "has space"},
        {"reason": ""},
        {"extra": 1},
    ],
)
def test_selection_refuses_locally(kwargs: dict[str, Any]) -> None:
    base = {"operation_id": "op", "route": "mloky", "reason": "r"}
    with pytest.raises(ValidationError):
        InternRuntimeRouteSelectionRequest(**{**base, **kwargs})


def test_task_view_cursor_and_limit_bounds_refuse_before_io() -> None:
    t = _Transport()
    api = ResearchInternAPI(t)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        api.task_views.get(GRANT, TASK, project_id=PROJECT, snapshot_revision=7)
    with pytest.raises(ValueError):
        api.task_views.get(GRANT, TASK, project_id=PROJECT, limit=101)
    assert t.requests == []


@pytest.mark.parametrize(
    ("override", "call"),
    [
        (
            {"select_intern_runtime_route": {**RECEIPT, "route": "legacy_smr_runtime"}},
            lambda api: api.runtime_route.select(SELECT),
        ),
        (
            {"get_intern_task_view": {**TASK_VIEW, "task_id": "other"}},
            lambda api: api.task_views.get(GRANT, TASK, project_id=PROJECT),
        ),
        (
            {"get_intern_task_view": {**TASK_VIEW, "current_task": {"issue_id": "other"}}},
            lambda api: api.task_views.get(GRANT, TASK, project_id=PROJECT),
        ),
        (
            {"get_intern_runtime_route": {**ROUTE_VIEW, "unknown": 1}},
            lambda api: api.runtime_route.get(),
        ),
    ],
)
def test_identity_drift_and_unknown_fields_fail_closed(override: dict, call: Any) -> None:
    with pytest.raises(ResearchApiError) as caught:
        call(ResearchInternAPI(_Transport(override)))  # type: ignore[arg-type]
    assert caught.value.failure.code == "schema_integrity_conflict"
    assert caught.value.operation_id == next(iter(override))
    assert caught.value.failure.retry.retryable is False
