"""Intern identity, Task grant and project Sublinear Task SDK methods (A07/T03).

Wire fixtures: Task grant documents are recorded from slot2
(``tests/fixtures/intern_authority/slot2_recorded_grants.json``); identity and
Sublinear Task payloads are built from the backend route DTOs they mirror. No
network and no model calls.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from synth_ai.core.http.request import HttpRequest
from synth_ai.sdk.research.contracts.intern_authority import (
    BackendContextBindGrantDeclaration,
    InternIdentityProvisionKind,
    InternIdentityProvisionRequest,
    InternTaskGrant,
    InternTaskGrantDeclaration,
    InternTaskGrantStatus,
)
from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS
from synth_ai.sdk.research.research_intern import AsyncResearchInternAPI, ResearchInternAPI

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures/intern_authority/slot2_recorded_grants.json").read_text()
)
TASK_GRANT = FIXTURES["task_grant_revoked"]
BIND_GRANT = FIXTURES["context_bind_grant_revoked"]
ORG = TASK_GRANT["organization_id"]
INTERN = TASK_GRANT["intern_id"]
SMR_PROJECT = "51072000-0000-0000-0000-00000000b002"
TASK_ID = "6c6d8a52-0d55-4f2e-9e43-3c1f0a7a8a10"

IDENTITY = {"org_id": ORG, "research_intern_id": INTERN, "is_default": True}
PROVISION_RECEIPT = {
    "schema_version": "intern.identity_provision_receipt.v1",
    "org_id": ORG,
    "command_id": "cmd-1",
    "input_digest": "a" * 64,
    "identity": IDENTITY,
}
CATALOG = {
    "schema_version": "intern.identity_catalog_page.v1",
    "org_id": ORG,
    "identities": [IDENTITY],
    "after_intern_id": None,
}
SELECTION = {"schema_version": "intern.identity_selection.v1", "identity": IDENTITY}
SUBLINEAR_TASK = {
    "id": TASK_ID,
    "identifier": "INT-7",
    "title": "Intern Task",
    "url": f"https://sublinear.local/issue/{TASK_ID}",
    "description": None,
    "state": {"id": "state-1", "name": "In Progress", "type": "started"},
    "project_id": "sublinear-project-6072c124-b40e-4637-a526-1997b8a8a881",
    "team_id": "team-1",
}

EXPECTED_OPERATIONS = {
    "provision_intern_identity": ("POST", "/smr/research-intern/identities"),
    "list_intern_identities": ("GET", "/smr/research-intern/identities"),
    "select_intern_identity": ("GET", "/smr/research-intern/identities/{intern_id}"),
    "declare_intern_task_grant": ("POST", "/smr/research-intern/task-grants"),
    "revoke_intern_task_grant": ("POST", "/smr/research-intern/task-grants/{grant_id}/revoke"),
    "declare_backend_context_bind_grant": (
        "POST",
        "/smr/research-intern/backend-context-bind-grants",
    ),
    "list_project_sublinear_tasks": ("GET", "/smr/projects/{project_id}/sublinear/tasks"),
    "get_project_sublinear_task": ("GET", "/smr/projects/{project_id}/sublinear/tasks/{task_id}"),
    "list_project_sublinear_task_comments": (
        "GET",
        "/smr/projects/{project_id}/sublinear/tasks/{task_id}/comments",
    ),
}


def _active(grant: dict[str, Any]) -> dict[str, Any]:
    return {**grant, "status": "active", "revocation_epoch": 0}


def _response(request: HttpRequest) -> Any:
    op = str(request.operation.operation_id)
    if op == "provision_intern_identity":
        return PROVISION_RECEIPT
    if op == "list_intern_identities":
        return CATALOG
    if op == "select_intern_identity":
        return SELECTION
    if op == "declare_intern_task_grant":
        return _active(TASK_GRANT)
    if op == "declare_backend_context_bind_grant":
        return _active(BIND_GRANT)
    if op == "revoke_intern_task_grant":
        return TASK_GRANT
    if op == "list_project_sublinear_tasks":
        return {
            "project_id": SMR_PROJECT,
            "sublinear_project_id": SUBLINEAR_TASK["project_id"],
            "tasks": [SUBLINEAR_TASK],
        }
    if op == "get_project_sublinear_task":
        return {"project_id": SMR_PROJECT, "task": SUBLINEAR_TASK}
    if op == "list_project_sublinear_task_comments":
        return {
            "project_id": SMR_PROJECT,
            "task_id": TASK_ID,
            "comments": [{"id": "c1", "body": "attempt retained", "url": "u#comment-c1"}],
        }
    raise AssertionError(op)


class _Transport:
    def __init__(self, override: dict[str, Any] | None = None) -> None:
        self.requests: list[HttpRequest] = []
        self.override = override or {}

    def execute(self, request: HttpRequest) -> Any:
        self.requests.append(request)
        op = str(request.operation.operation_id)
        return copy.deepcopy(self.override[op] if op in self.override else _response(request))


class _AsyncTransport(_Transport):
    async def execute(self, request: HttpRequest) -> Any:  # type: ignore[override]
        return _Transport.execute(self, request)


def _grant_request() -> InternTaskGrantDeclaration:
    return InternTaskGrantDeclaration(
        operation_id="op-grant-1",
        intern_id=INTERN,
        smr_project_ids=[SMR_PROJECT],
        task_operations=list(TASK_GRANT["task_operations"]),
        policy_revision=TASK_GRANT["policy_revision"],
        expires_unix_ms=TASK_GRANT["expires_unix_ms"],
    )


def _bind_request() -> BackendContextBindGrantDeclaration:
    return BackendContextBindGrantDeclaration(
        operation_id="op-bind-1",
        intern_id=INTERN,
        smr_project_ids=[SMR_PROJECT],
        policy_revision=BIND_GRANT["policy_revision"],
        expires_unix_ms=BIND_GRANT["expires_unix_ms"],
    )


def _drive_sync(api: ResearchInternAPI) -> list[Any]:
    return [
        api.identities.provision(
            InternIdentityProvisionRequest(
                command_id="cmd-1",
                kind=InternIdentityProvisionKind.ENSURE_DEFAULT,
                display_name="Intern",
            )
        ),
        api.identities.list(operation_id="op-list-1"),
        api.identities.select(INTERN, operation_id="op-sel-1"),
        api.task_grants.declare(_grant_request()),
        api.task_grants.declare_backend_context_bind(_bind_request()),
        api.task_grants.revoke(TASK_GRANT["grant_id"], expected_revocation_epoch=0),
        api.sublinear_tasks.list(SMR_PROJECT, limit=10),
        api.sublinear_tasks.get(SMR_PROJECT, TASK_ID),
        api.sublinear_tasks.comments(SMR_PROJECT, TASK_ID),
    ]


async def _drive_async(api: AsyncResearchInternAPI) -> list[Any]:
    return [
        await api.identities.provision(
            InternIdentityProvisionRequest(
                command_id="cmd-1",
                kind=InternIdentityProvisionKind.ENSURE_DEFAULT,
                display_name="Intern",
            )
        ),
        await api.identities.list(operation_id="op-list-1"),
        await api.identities.select(INTERN, operation_id="op-sel-1"),
        await api.task_grants.declare(_grant_request()),
        await api.task_grants.declare_backend_context_bind(_bind_request()),
        await api.task_grants.revoke(TASK_GRANT["grant_id"], expected_revocation_epoch=0),
        await api.sublinear_tasks.list(SMR_PROJECT, limit=10),
        await api.sublinear_tasks.get(SMR_PROJECT, TASK_ID),
        await api.sublinear_tasks.comments(SMR_PROJECT, TASK_ID),
    ]


def _sent(transport: _Transport) -> list[tuple[str, str, dict, Any]]:
    return [
        (str(r.operation.operation_id), r.path, dict(r.query), r.body) for r in transport.requests
    ]


def test_operations_match_backend_routes() -> None:
    registered = {str(k): v for k, v in RESEARCH_OPERATIONS.items()}
    for op, (method, path) in EXPECTED_OPERATIONS.items():
        assert str(registered[op].method.value).upper() == method
        assert registered[op].path_template == path
        assert registered[op].idempotent


def test_recorded_slot2_grants_decode_and_round_trip_exactly() -> None:
    for wire in (TASK_GRANT, BIND_GRANT):
        grant = InternTaskGrant.from_wire(wire)
        assert grant.status is InternTaskGrantStatus.REVOKED
        assert not grant.active
        assert grant.to_wire() == {
            **wire,
            "task_operations": wire["task_operations"],
            "catalog_projects": wire["catalog_projects"],
        }


def test_sync_and_async_arms_send_identical_requests() -> None:
    sync_transport = _Transport()
    sync_results = _drive_sync(ResearchInternAPI(sync_transport))  # type: ignore[arg-type]
    async_transport = _AsyncTransport()
    async_results = asyncio.run(_drive_async(AsyncResearchInternAPI(async_transport)))  # type: ignore[arg-type]
    assert _sent(sync_transport) == _sent(async_transport)
    assert [r.model_dump() for r in sync_results] == [r.model_dump() for r in async_results]

    sent = _sent(sync_transport)
    assert sent[0][1:] == (
        "/smr/research-intern/identities",
        {},
        {
            "schema_version": "intern.identity_provision_request.v1",
            "command_id": "cmd-1",
            "kind": "ensure_default",
            "display_name": "Intern",
        },
    )
    assert sent[1][1:3] == ("/smr/research-intern/identities", {"operation_id": "op-list-1"})
    assert sent[2][1:3] == (
        f"/smr/research-intern/identities/{INTERN}",
        {"operation_id": "op-sel-1"},
    )
    assert sent[3][3] == {
        "schema_version": "intern.task_grant_declaration.v1",
        "operation_id": "op-grant-1",
        "intern_id": INTERN,
        "smr_project_ids": [SMR_PROJECT],
        "task_operations": TASK_GRANT["task_operations"],
        "policy_revision": TASK_GRANT["policy_revision"],
        "expires_unix_ms": TASK_GRANT["expires_unix_ms"],
    }
    assert set(sent[4][3]) == {
        "schema_version",
        "operation_id",
        "intern_id",
        "smr_project_ids",
        "policy_revision",
        "expires_unix_ms",
    }
    assert sent[5][1:] == (
        f"/smr/research-intern/task-grants/{TASK_GRANT['grant_id']}/revoke",
        {},
        {"schema_version": "intern.task_grant_revocation.v1", "expected_revocation_epoch": 0},
    )
    assert sent[6][1:3] == (f"/smr/projects/{SMR_PROJECT}/sublinear/tasks", {"limit": 10})
    assert sent[7][1] == f"/smr/projects/{SMR_PROJECT}/sublinear/tasks/{TASK_ID}"
    assert sent[8][1] == f"/smr/projects/{SMR_PROJECT}/sublinear/tasks/{TASK_ID}/comments"

    declared = sync_results[3]
    assert declared.active and declared.grant_id == TASK_GRANT["grant_id"]
    assert sync_results[7].task.state.name == "In Progress"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_operations", ["task.history.read", "task.catalog.read"]),  # unsorted
        ("task_operations", ["task.context.bind"]),  # not owner-declarable
        ("smr_project_ids", ["00000000-0000-0000-0000-000000000000"]),  # nil
        ("intern_id", "D44B13C3-41FA-469B-A2B5-0482ABE3B014"),  # non-canonical
        ("operation_id", "has space"),
        ("expires_unix_ms", 0),
    ],
)
def test_grant_declaration_refuses_locally_like_backend(field: str, value: Any) -> None:
    wire = _grant_request().to_wire()
    wire[field] = value
    with pytest.raises(ValidationError):
        InternTaskGrantDeclaration.model_validate(wire)


def test_unknown_response_fields_are_contract_changes() -> None:
    transport = _Transport({"declare_intern_task_grant": {**_active(TASK_GRANT), "extra": 1}})
    with pytest.raises(ValidationError):
        ResearchInternAPI(transport).task_grants.declare(_grant_request())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("op", "payload", "call", "message"),
    [
        (
            "select_intern_identity",
            {
                **SELECTION,
                "identity": {
                    **IDENTITY,
                    "research_intern_id": "11111111-1111-1111-1111-111111111111",
                },
            },
            lambda api: api.identities.select(INTERN, operation_id="op"),
            "selection drifted",
        ),
        (
            "declare_intern_task_grant",
            {**_active(TASK_GRANT), "policy_revision": "other"},
            lambda api: api.task_grants.declare(_grant_request()),
            "identity drifted",
        ),
        (
            "declare_backend_context_bind_grant",
            _active(TASK_GRANT),
            lambda api: api.task_grants.declare_backend_context_bind(_bind_request()),
            "operations drifted",
        ),
        (
            "revoke_intern_task_grant",
            BIND_GRANT,
            lambda api: api.task_grants.revoke(TASK_GRANT["grant_id"], expected_revocation_epoch=0),
            "revocation identity drifted",
        ),
        (
            "get_project_sublinear_task",
            {"project_id": SMR_PROJECT, "task": {**SUBLINEAR_TASK, "id": "other"}},
            lambda api: api.sublinear_tasks.get(SMR_PROJECT, TASK_ID),
            "Task identity drifted",
        ),
    ],
)
def test_identity_drift_fails_closed(op: str, payload: Any, call: Any, message: str) -> None:
    api = ResearchInternAPI(_Transport({op: payload}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        call(api)


def test_sublinear_task_list_limit_is_bounded_like_backend() -> None:
    api = ResearchInternAPI(_Transport())  # type: ignore[arg-type]
    for bad in (0, 201):
        with pytest.raises(ValueError):
            api.sublinear_tasks.list(SMR_PROJECT, limit=bad)
    with pytest.raises(TypeError):
        api.sublinear_tasks.list(SMR_PROJECT, limit=True)


def test_path_segments_are_escaped() -> None:
    transport = _Transport(
        {
            "get_project_sublinear_task": {
                "project_id": SMR_PROJECT,
                "task": {**SUBLINEAR_TASK, "id": "a/b"},
            }
        }
    )
    ResearchInternAPI(transport).sublinear_tasks.get(SMR_PROJECT, "a/b")  # type: ignore[arg-type]
    assert transport.requests[0].path.endswith("/sublinear/tasks/a%2Fb")
