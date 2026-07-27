"""Synchronous and asynchronous project operations."""

from __future__ import annotations

from typing import cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._wire import array_value
from synth_ai.core.research.contracts.common import ProjectId
from synth_ai.core.research.contracts.projects import (
    Project,
    ProjectPatch,
    ProjectSetup,
    ProjectSpec,
)
from synth_ai.core.research.operations import research_operation
from synth_ai.core.research.project_data import (
    AsyncProjectDatasetsAPI,
    AsyncProjectRepositoriesAPI,
    ProjectDatasetsAPI,
    ProjectRepositoriesAPI,
)
from synth_ai.core.research.project_workspaces import (
    AsyncProjectWorkspaceAPI,
    ProjectWorkspaceAPI,
)
from synth_ai.core.research.research_intern import (
    AsyncProjectComputerAPI,
    AsyncProjectDataBindingsAPI,
    ProjectComputerAPI,
    ProjectDataBindingsAPI,
)


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
    headers: dict[str, str] | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
        headers=headers or {},
    )


def _projects(value: object) -> tuple[Project, ...]:
    return tuple(
        Project.from_wire(item)
        for item in array_value(cast(JsonValue, value), operation_id="list_projects")
    )


class ProjectSetupAPI:
    """Project setup state retrieval and preparation."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def retrieve(self, project_id: ProjectId) -> ProjectSetup:
        """Retrieve the setup state for a Project.

        Args:
            project_id: Project to inspect.

        Returns:
            The current Project setup state and blockers.
        """
        value = self._transport.execute(
            _request(
                "retrieve_project_setup",
                f"/smr/projects/{project_id}/setup",
            )
        )
        return ProjectSetup.from_wire(value)

    def prepare(self, project_id: ProjectId) -> ProjectSetup:
        """Prepare setup for a Project and return its setup state.

        Args:
            project_id: Project to prepare.

        Returns:
            The Project setup state after preparation.
        """
        value = self._transport.execute(
            _request(
                "prepare_project_setup",
                f"/smr/projects/{project_id}/setup/prepare",
            )
        )
        return ProjectSetup.from_wire(value)


class ProjectsAPI:
    """Project lifecycle operations and nested data, setup, and workspace namespaces."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.datasets = ProjectDatasetsAPI(transport)
        self.computer = ProjectComputerAPI(transport)
        self.data_bindings = ProjectDataBindingsAPI(transport)
        self.repositories = ProjectRepositoriesAPI(transport)
        self.setup = ProjectSetupAPI(transport)
        self.workspace = ProjectWorkspaceAPI(transport)

    def create(self, request: ProjectSpec) -> Project:
        """Create a runnable Project from a typed Project specification.

        Args:
            request: Project specification to serialize into the create request body.

        Returns:
            The created Project.
        """
        value = self._transport.execute(
            _request("create_project", "/smr/projects:runnable", body=request.to_wire())
        )
        return Project.from_wire(value)

    def list(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[Project, ...]:
        """List Projects visible to the authenticated organization.

        Args:
            include_archived: Whether to include archived Projects in the result.
            limit: Maximum number of Projects to request.
            cursor: Optional pagination cursor returned by the backend.

        Returns:
            The Projects returned by the backend.
        """
        query: JsonObject = {"include_archived": include_archived, "limit": limit}
        if cursor is not None:
            query["cursor"] = cursor
        value = self._transport.execute(_request("list_projects", "/smr/projects", query=query))
        return _projects(value)

    def retrieve(self, project_id: ProjectId) -> Project:
        """Retrieve a Project.

        Args:
            project_id: Project to retrieve.

        Returns:
            The requested Project.
        """
        value = self._transport.execute(_request("retrieve_project", f"/smr/projects/{project_id}"))
        return Project.from_wire(value)

    def update(
        self,
        project_id: ProjectId,
        request: ProjectPatch,
    ) -> Project:
        """Update mutable Project fields from a typed Project patch.

        Args:
            project_id: Project to update.
            request: Project patch to serialize into the update request body.

        Returns:
            The updated Project.
        """
        value = self._transport.execute(
            _request(
                "update_project",
                f"/smr/projects/{project_id}",
                body=request.to_wire(),
            )
        )
        return Project.from_wire(value)

    def archive(self, project_id: ProjectId) -> Project:
        """Archive a Project with a deterministic idempotency key.

        Args:
            project_id: Project to archive.

        Returns:
            The archived Project.
        """
        value = self._transport.execute(
            _request(
                "archive_project",
                f"/smr/projects/{project_id}/archive",
                headers={"Idempotency-Key": f"archive-project:{project_id}"},
            )
        )
        return Project.from_wire(value)

    def unarchive(self, project_id: ProjectId) -> Project:
        """Unarchive a Project.

        Args:
            project_id: Project to unarchive.

        Returns:
            The unarchived Project.
        """
        value = self._transport.execute(
            _request("unarchive_project", f"/smr/projects/{project_id}/unarchive")
        )
        return Project.from_wire(value)


class AsyncProjectSetupAPI:
    """Project setup state retrieval and preparation."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def retrieve(self, project_id: ProjectId) -> ProjectSetup:
        """Retrieve the setup state for a Project.

        Args:
            project_id: Project to inspect.

        Returns:
            The current Project setup state and blockers.
        """
        value = await self._transport.execute(
            _request("retrieve_project_setup", f"/smr/projects/{project_id}/setup")
        )
        return ProjectSetup.from_wire(value)

    async def prepare(self, project_id: ProjectId) -> ProjectSetup:
        """Prepare setup for a Project and return its setup state.

        Args:
            project_id: Project to prepare.

        Returns:
            The Project setup state after preparation.
        """
        value = await self._transport.execute(
            _request("prepare_project_setup", f"/smr/projects/{project_id}/setup/prepare")
        )
        return ProjectSetup.from_wire(value)


class AsyncProjectsAPI:
    """Project lifecycle operations and nested data, setup, and workspace namespaces."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.computer = AsyncProjectComputerAPI(transport)
        self.data_bindings = AsyncProjectDataBindingsAPI(transport)
        self.datasets = AsyncProjectDatasetsAPI(transport)
        self.repositories = AsyncProjectRepositoriesAPI(transport)
        self.setup = AsyncProjectSetupAPI(transport)
        self.workspace = AsyncProjectWorkspaceAPI(transport)

    async def create(self, request: ProjectSpec) -> Project:
        """Create a runnable Project from a typed Project specification.

        Args:
            request: Project specification to serialize into the create request body.

        Returns:
            The created Project.
        """
        value = await self._transport.execute(
            _request("create_project", "/smr/projects:runnable", body=request.to_wire())
        )
        return Project.from_wire(value)

    async def list(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[Project, ...]:
        """List Projects visible to the authenticated organization.

        Args:
            include_archived: Whether to include archived Projects in the result.
            limit: Maximum number of Projects to request.
            cursor: Optional pagination cursor returned by the backend.

        Returns:
            The Projects returned by the backend.
        """
        query: JsonObject = {"include_archived": include_archived, "limit": limit}
        if cursor is not None:
            query["cursor"] = cursor
        value = await self._transport.execute(
            _request("list_projects", "/smr/projects", query=query)
        )
        return _projects(value)

    async def retrieve(self, project_id: ProjectId) -> Project:
        """Retrieve a Project.

        Args:
            project_id: Project to retrieve.

        Returns:
            The requested Project.
        """
        value = await self._transport.execute(
            _request("retrieve_project", f"/smr/projects/{project_id}")
        )
        return Project.from_wire(value)

    async def update(
        self,
        project_id: ProjectId,
        request: ProjectPatch,
    ) -> Project:
        """Update mutable Project fields from a typed Project patch.

        Args:
            project_id: Project to update.
            request: Project patch to serialize into the update request body.

        Returns:
            The updated Project.
        """
        value = await self._transport.execute(
            _request("update_project", f"/smr/projects/{project_id}", body=request.to_wire())
        )
        return Project.from_wire(value)

    async def archive(self, project_id: ProjectId) -> Project:
        """Archive a Project with a deterministic idempotency key.

        Args:
            project_id: Project to archive.

        Returns:
            The archived Project.
        """
        value = await self._transport.execute(
            _request(
                "archive_project",
                f"/smr/projects/{project_id}/archive",
                headers={"Idempotency-Key": f"archive-project:{project_id}"},
            )
        )
        return Project.from_wire(value)

    async def unarchive(self, project_id: ProjectId) -> Project:
        """Unarchive a Project.

        Args:
            project_id: Project to unarchive.

        Returns:
            The unarchived Project.
        """
        value = await self._transport.execute(
            _request("unarchive_project", f"/smr/projects/{project_id}/unarchive")
        )
        return Project.from_wire(value)


ResearchProjectSetupAPI = ProjectSetupAPI
ResearchProjectsAPI = ProjectsAPI
AsyncResearchProjectSetupAPI = AsyncProjectSetupAPI
AsyncResearchProjectsAPI = AsyncProjectsAPI


__all__ = [
    "AsyncProjectSetupAPI",
    "AsyncProjectWorkspaceAPI",
    "AsyncProjectDatasetsAPI",
    "AsyncProjectRepositoriesAPI",
    "AsyncProjectsAPI",
    "ProjectDatasetsAPI",
    "ProjectRepositoriesAPI",
    "ProjectSetupAPI",
    "ProjectWorkspaceAPI",
    "ProjectsAPI",
    "AsyncResearchProjectSetupAPI",
    "AsyncResearchProjectsAPI",
    "ResearchProjectSetupAPI",
    "ResearchProjectsAPI",
]
