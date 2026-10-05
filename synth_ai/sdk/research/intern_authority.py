"""Intern identity, Task grant and project Sublinear Task operations.

Installed-SDK coverage for A07 (Intern create/select) and T03 (read/control the
same Task through backend product APIs). Each call is one backend route; see
:mod:`synth_ai.sdk.research.contracts.intern_authority` for the wire contracts.
Sync and async clients share request builders so both arms send identical
requests.
"""

from __future__ import annotations

from typing import cast
from urllib.parse import quote

from synth_ai.core.contracts.json_value import JsonObject
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts.intern_authority import (
    BackendContextBindGrantDeclaration,
    InternIdentityCatalogPage,
    InternIdentityProvisionReceipt,
    InternIdentityProvisionRequest,
    InternIdentitySelection,
    InternTaskGrant,
    InternTaskGrantDeclaration,
    InternTaskGrantRevocation,
    ProjectSublinearTask,
    ProjectSublinearTaskComments,
    ProjectSublinearTaskList,
)
from synth_ai.sdk.research.operations import research_operation

_IDENTITIES = "/smr/research-intern/identities"
_TASK_GRANTS = "/smr/research-intern/task-grants"
_CONTEXT_BIND_GRANTS = "/smr/research-intern/backend-context-bind-grants"
_SUBLINEAR_TASK_LIST_MAX = 200


def _segment(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return quote(value, safe="")


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(research_operation(operation_id), path, query=query or {}, body=body)


# -- request builders (shared by sync/async) --------------------------------


def _provision_identity(request: InternIdentityProvisionRequest) -> HttpRequest:
    return _request(
        "provision_intern_identity", _IDENTITIES, body=cast(JsonObject, request.to_wire())
    )


def _list_identities(operation_id: str, after_intern_id: str | None) -> HttpRequest:
    query: JsonObject = {"operation_id": operation_id}
    if after_intern_id is not None:
        query["after_intern_id"] = after_intern_id
    return _request("list_intern_identities", _IDENTITIES, query=query)


def _select_identity(intern_id: str, operation_id: str) -> HttpRequest:
    return _request(
        "select_intern_identity",
        f"{_IDENTITIES}/{_segment(intern_id, name='intern_id')}",
        query={"operation_id": operation_id},
    )


def _declare_task_grant(request: InternTaskGrantDeclaration) -> HttpRequest:
    return _request(
        "declare_intern_task_grant", _TASK_GRANTS, body=cast(JsonObject, request.to_wire())
    )


def _revoke_task_grant(grant_id: str, expected_revocation_epoch: int) -> HttpRequest:
    body = InternTaskGrantRevocation(expected_revocation_epoch=expected_revocation_epoch)
    return _request(
        "revoke_intern_task_grant",
        f"{_TASK_GRANTS}/{_segment(grant_id, name='grant_id')}/revoke",
        body=cast(JsonObject, body.to_wire()),
    )


def _declare_context_bind_grant(request: BackendContextBindGrantDeclaration) -> HttpRequest:
    return _request(
        "declare_backend_context_bind_grant",
        _CONTEXT_BIND_GRANTS,
        body=cast(JsonObject, request.to_wire()),
    )


def _sublinear_tasks_path(project_id: str) -> str:
    return f"/smr/projects/{_segment(project_id, name='project_id')}/sublinear/tasks"


def _list_sublinear_tasks(project_id: str, limit: int) -> HttpRequest:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if not 1 <= limit <= _SUBLINEAR_TASK_LIST_MAX:
        raise ValueError(f"limit must be between 1 and {_SUBLINEAR_TASK_LIST_MAX}")
    return _request(
        "list_project_sublinear_tasks", _sublinear_tasks_path(project_id), query={"limit": limit}
    )


def _get_sublinear_task(project_id: str, task_id: str) -> HttpRequest:
    return _request(
        "get_project_sublinear_task",
        f"{_sublinear_tasks_path(project_id)}/{_segment(task_id, name='task_id')}",
    )


def _list_sublinear_task_comments(project_id: str, task_id: str) -> HttpRequest:
    return _request(
        "list_project_sublinear_task_comments",
        f"{_sublinear_tasks_path(project_id)}/{_segment(task_id, name='task_id')}/comments",
    )


# -- response identity checks ----------------------------------------------


def _checked_provision(
    request: InternIdentityProvisionRequest, receipt: InternIdentityProvisionReceipt
) -> InternIdentityProvisionReceipt:
    if receipt.command_id != request.command_id:
        raise ValueError("Intern identity provision receipt command drifted")
    if receipt.identity.org_id != receipt.org_id:
        raise ValueError("Intern identity provision receipt org drifted")
    return receipt


def _checked_selection(
    intern_id: str, selection: InternIdentitySelection
) -> InternIdentitySelection:
    if selection.identity.research_intern_id != intern_id:
        raise ValueError("Intern identity selection drifted")
    return selection


def _checked_grant(
    grant: InternTaskGrant,
    *,
    intern_id: str,
    policy_revision: str,
    project_count: int,
    task_operations: tuple[str, ...] | None,
) -> InternTaskGrant:
    if grant.intern_id != intern_id or grant.policy_revision != policy_revision:
        raise ValueError("Intern Task grant identity drifted")
    if len(grant.catalog_projects.project_ids) != project_count:
        raise ValueError("Intern Task grant project scope drifted")
    if task_operations is not None and grant.task_operations != task_operations:
        raise ValueError("Intern Task grant operations drifted")
    return grant


def _checked_declared(
    request: InternTaskGrantDeclaration, grant: InternTaskGrant
) -> InternTaskGrant:
    return _checked_grant(
        grant,
        intern_id=request.intern_id,
        policy_revision=request.policy_revision,
        project_count=len(request.smr_project_ids),
        task_operations=tuple(request.task_operations),
    )


def _checked_context_bind(
    request: BackendContextBindGrantDeclaration, grant: InternTaskGrant
) -> InternTaskGrant:
    return _checked_grant(
        grant,
        intern_id=request.intern_id,
        policy_revision=request.policy_revision,
        project_count=len(request.smr_project_ids),
        task_operations=("task.command.bind_task_context", "task.context.bind"),
    )


def _checked_revoked(grant_id: str, grant: InternTaskGrant) -> InternTaskGrant:
    if grant.grant_id != grant_id:
        raise ValueError("Intern Task grant revocation identity drifted")
    return grant


def _checked_task(
    project_id: str, task_id: str, view: ProjectSublinearTask
) -> ProjectSublinearTask:
    if view.project_id != project_id or view.task.id != task_id:
        raise ValueError("project Sublinear Task identity drifted")
    return view


def _checked_task_list(project_id: str, page: ProjectSublinearTaskList) -> ProjectSublinearTaskList:
    if page.project_id != project_id:
        raise ValueError("project Sublinear Task list identity drifted")
    return page


def _checked_comments(
    project_id: str, task_id: str, page: ProjectSublinearTaskComments
) -> ProjectSublinearTaskComments:
    if page.project_id != project_id or page.task_id != task_id:
        raise ValueError("project Sublinear Task comments identity drifted")
    return page


# -- sync -------------------------------------------------------------------


class ResearchInternIdentitiesAPI:
    """Org-scoped canonical Intern identities: create (provision), list, select."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def provision(self, request: InternIdentityProvisionRequest) -> InternIdentityProvisionReceipt:
        receipt = InternIdentityProvisionReceipt.from_wire(
            self._transport.execute(_provision_identity(request))
        )
        return _checked_provision(request, receipt)

    def list(
        self, *, operation_id: str, after_intern_id: str | None = None
    ) -> InternIdentityCatalogPage:
        return InternIdentityCatalogPage.from_wire(
            self._transport.execute(_list_identities(operation_id, after_intern_id))
        )

    def select(self, intern_id: str, *, operation_id: str) -> InternIdentitySelection:
        selection = InternIdentitySelection.from_wire(
            self._transport.execute(_select_identity(intern_id, operation_id))
        )
        return _checked_selection(intern_id, selection)


class ResearchInternTaskGrantsAPI:
    """Owner-declared Sublinear Task grants for one Intern (control surface)."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def declare(self, request: InternTaskGrantDeclaration) -> InternTaskGrant:
        grant = InternTaskGrant.from_wire(self._transport.execute(_declare_task_grant(request)))
        return _checked_declared(request, grant)

    def revoke(self, grant_id: str, *, expected_revocation_epoch: int) -> InternTaskGrant:
        grant = InternTaskGrant.from_wire(
            self._transport.execute(_revoke_task_grant(grant_id, expected_revocation_epoch))
        )
        return _checked_revoked(grant_id, grant)

    def declare_backend_context_bind(
        self, request: BackendContextBindGrantDeclaration
    ) -> InternTaskGrant:
        grant = InternTaskGrant.from_wire(
            self._transport.execute(_declare_context_bind_grant(request))
        )
        return _checked_context_bind(request, grant)


class ProjectSublinearTasksAPI:
    """Read-only project Sublinear Task views served by the backend."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self, project_id: str, *, limit: int = 50) -> ProjectSublinearTaskList:
        page = ProjectSublinearTaskList.from_wire(
            self._transport.execute(_list_sublinear_tasks(project_id, limit))
        )
        return _checked_task_list(project_id, page)

    def get(self, project_id: str, task_id: str) -> ProjectSublinearTask:
        view = ProjectSublinearTask.from_wire(
            self._transport.execute(_get_sublinear_task(project_id, task_id))
        )
        return _checked_task(project_id, task_id, view)

    def comments(self, project_id: str, task_id: str) -> ProjectSublinearTaskComments:
        page = ProjectSublinearTaskComments.from_wire(
            self._transport.execute(_list_sublinear_task_comments(project_id, task_id))
        )
        return _checked_comments(project_id, task_id, page)


# -- async ------------------------------------------------------------------


class AsyncResearchInternIdentitiesAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def provision(
        self, request: InternIdentityProvisionRequest
    ) -> InternIdentityProvisionReceipt:
        receipt = InternIdentityProvisionReceipt.from_wire(
            await self._transport.execute(_provision_identity(request))
        )
        return _checked_provision(request, receipt)

    async def list(
        self, *, operation_id: str, after_intern_id: str | None = None
    ) -> InternIdentityCatalogPage:
        return InternIdentityCatalogPage.from_wire(
            await self._transport.execute(_list_identities(operation_id, after_intern_id))
        )

    async def select(self, intern_id: str, *, operation_id: str) -> InternIdentitySelection:
        selection = InternIdentitySelection.from_wire(
            await self._transport.execute(_select_identity(intern_id, operation_id))
        )
        return _checked_selection(intern_id, selection)


class AsyncResearchInternTaskGrantsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def declare(self, request: InternTaskGrantDeclaration) -> InternTaskGrant:
        grant = InternTaskGrant.from_wire(
            await self._transport.execute(_declare_task_grant(request))
        )
        return _checked_declared(request, grant)

    async def revoke(self, grant_id: str, *, expected_revocation_epoch: int) -> InternTaskGrant:
        grant = InternTaskGrant.from_wire(
            await self._transport.execute(_revoke_task_grant(grant_id, expected_revocation_epoch))
        )
        return _checked_revoked(grant_id, grant)

    async def declare_backend_context_bind(
        self, request: BackendContextBindGrantDeclaration
    ) -> InternTaskGrant:
        grant = InternTaskGrant.from_wire(
            await self._transport.execute(_declare_context_bind_grant(request))
        )
        return _checked_context_bind(request, grant)


class AsyncProjectSublinearTasksAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self, project_id: str, *, limit: int = 50) -> ProjectSublinearTaskList:
        page = ProjectSublinearTaskList.from_wire(
            await self._transport.execute(_list_sublinear_tasks(project_id, limit))
        )
        return _checked_task_list(project_id, page)

    async def get(self, project_id: str, task_id: str) -> ProjectSublinearTask:
        view = ProjectSublinearTask.from_wire(
            await self._transport.execute(_get_sublinear_task(project_id, task_id))
        )
        return _checked_task(project_id, task_id, view)

    async def comments(self, project_id: str, task_id: str) -> ProjectSublinearTaskComments:
        page = ProjectSublinearTaskComments.from_wire(
            await self._transport.execute(_list_sublinear_task_comments(project_id, task_id))
        )
        return _checked_comments(project_id, task_id, page)


__all__ = [
    "AsyncProjectSublinearTasksAPI",
    "AsyncResearchInternIdentitiesAPI",
    "AsyncResearchInternTaskGrantsAPI",
    "ProjectSublinearTasksAPI",
    "ResearchInternIdentitiesAPI",
    "ResearchInternTaskGrantsAPI",
]
