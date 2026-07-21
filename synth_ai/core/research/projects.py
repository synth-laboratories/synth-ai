"""Synchronous and asynchronous project operations."""

from __future__ import annotations

from synth_ai.core.contracts.json_value import JsonObject
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._wire import array_value
from synth_ai.core.research.contracts.common import ProjectId
from synth_ai.core.research.contracts.projects import (
    ResearchProject,
    ResearchProjectCreateRequest,
    ResearchProjectPatchRequest,
    ResearchProjectSetup,
)
from synth_ai.core.research.operations import research_operation


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
    )


def _projects(value: object) -> tuple[ResearchProject, ...]:
    return tuple(
        ResearchProject.from_wire(item)
        for item in array_value(value, operation_id="list_projects")
    )


class ResearchProjectSetupAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def retrieve(self, project_id: ProjectId) -> ResearchProjectSetup:
        value = self._transport.execute(
            _request(
                "retrieve_project_setup",
                f"/smr/projects/{project_id}/setup",
            )
        )
        return ResearchProjectSetup.from_wire(value)

    def prepare(self, project_id: ProjectId) -> ResearchProjectSetup:
        value = self._transport.execute(
            _request(
                "prepare_project_setup",
                f"/smr/projects/{project_id}/setup/prepare",
            )
        )
        return ResearchProjectSetup.from_wire(value)


class ResearchProjectsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.setup = ResearchProjectSetupAPI(transport)

    def create(self, request: ResearchProjectCreateRequest) -> ResearchProject:
        value = self._transport.execute(
            _request("create_project", "/smr/projects:runnable", body=request.to_wire())
        )
        return ResearchProject.from_wire(value)

    def list(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[ResearchProject, ...]:
        query: JsonObject = {"include_archived": include_archived, "limit": limit}
        if cursor is not None:
            query["cursor"] = cursor
        value = self._transport.execute(_request("list_projects", "/smr/projects", query=query))
        return _projects(value)

    def retrieve(self, project_id: ProjectId) -> ResearchProject:
        value = self._transport.execute(
            _request("retrieve_project", f"/smr/projects/{project_id}")
        )
        return ResearchProject.from_wire(value)

    def update(
        self,
        project_id: ProjectId,
        request: ResearchProjectPatchRequest,
    ) -> ResearchProject:
        value = self._transport.execute(
            _request(
                "update_project",
                f"/smr/projects/{project_id}",
                body=request.to_wire(),
            )
        )
        return ResearchProject.from_wire(value)

    def archive(self, project_id: ProjectId) -> ResearchProject:
        value = self._transport.execute(
            _request("archive_project", f"/smr/projects/{project_id}/archive")
        )
        return ResearchProject.from_wire(value)

    def unarchive(self, project_id: ProjectId) -> ResearchProject:
        value = self._transport.execute(
            _request("unarchive_project", f"/smr/projects/{project_id}/unarchive")
        )
        return ResearchProject.from_wire(value)


class AsyncResearchProjectSetupAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def retrieve(self, project_id: ProjectId) -> ResearchProjectSetup:
        value = await self._transport.execute(
            _request("retrieve_project_setup", f"/smr/projects/{project_id}/setup")
        )
        return ResearchProjectSetup.from_wire(value)

    async def prepare(self, project_id: ProjectId) -> ResearchProjectSetup:
        value = await self._transport.execute(
            _request("prepare_project_setup", f"/smr/projects/{project_id}/setup/prepare")
        )
        return ResearchProjectSetup.from_wire(value)


class AsyncResearchProjectsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.setup = AsyncResearchProjectSetupAPI(transport)

    async def create(self, request: ResearchProjectCreateRequest) -> ResearchProject:
        value = await self._transport.execute(
            _request("create_project", "/smr/projects:runnable", body=request.to_wire())
        )
        return ResearchProject.from_wire(value)

    async def list(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[ResearchProject, ...]:
        query: JsonObject = {"include_archived": include_archived, "limit": limit}
        if cursor is not None:
            query["cursor"] = cursor
        value = await self._transport.execute(
            _request("list_projects", "/smr/projects", query=query)
        )
        return _projects(value)

    async def retrieve(self, project_id: ProjectId) -> ResearchProject:
        value = await self._transport.execute(
            _request("retrieve_project", f"/smr/projects/{project_id}")
        )
        return ResearchProject.from_wire(value)

    async def update(
        self,
        project_id: ProjectId,
        request: ResearchProjectPatchRequest,
    ) -> ResearchProject:
        value = await self._transport.execute(
            _request("update_project", f"/smr/projects/{project_id}", body=request.to_wire())
        )
        return ResearchProject.from_wire(value)

    async def archive(self, project_id: ProjectId) -> ResearchProject:
        value = await self._transport.execute(
            _request("archive_project", f"/smr/projects/{project_id}/archive")
        )
        return ResearchProject.from_wire(value)

    async def unarchive(self, project_id: ProjectId) -> ResearchProject:
        value = await self._transport.execute(
            _request("unarchive_project", f"/smr/projects/{project_id}/unarchive")
        )
        return ResearchProject.from_wire(value)


__all__ = [
    "AsyncResearchProjectSetupAPI",
    "AsyncResearchProjectsAPI",
    "ResearchProjectSetupAPI",
    "ResearchProjectsAPI",
]
