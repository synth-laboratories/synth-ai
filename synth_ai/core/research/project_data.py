"""Project repository and dataset APIs over the shared core transport.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from typing import cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._wire import array_value
from synth_ai.core.research.contracts.common import (
    ProjectDatasetId,
    ProjectId,
    ProjectRepositoryId,
)
from synth_ai.core.research.contracts.project_data import (
    ProjectDataset,
    ProjectDatasetUpload,
    ProjectRepository,
    ProjectRepositoryDeletion,
    ProjectRepositoryPatch,
    ProjectRepositorySpec,
)
from synth_ai.core.research.operations import research_operation


def _request(
    operation_id: str,
    path: str,
    *,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        body=body,
    )


def _repositories(value: object, *, project_id: ProjectId) -> tuple[ProjectRepository, ...]:
    repositories = tuple(
        ProjectRepository.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_project_external_repositories",
        )
    )
    if any(repository.project_id != project_id for repository in repositories):
        raise ValueError("project repository list crossed its requested project boundary")
    return repositories


def _datasets(value: object, *, project_id: ProjectId) -> tuple[ProjectDataset, ...]:
    datasets = tuple(
        ProjectDataset.from_wire(item)
        for item in array_value(cast(JsonValue, value), operation_id="list_project_datasets")
    )
    if any(dataset.project_id != project_id for dataset in datasets):
        raise ValueError("project dataset list crossed its requested project boundary")
    return datasets


def _repository(
    value: object,
    *,
    project_id: ProjectId,
    repository_id: ProjectRepositoryId | None = None,
) -> ProjectRepository:
    repository = ProjectRepository.from_wire(cast(JsonValue, value))
    if repository.project_id != project_id:
        raise ValueError("project repository response crossed its requested project boundary")
    if repository_id is not None and repository.repository_id != repository_id:
        raise ValueError("project repository response identity drifted")
    return repository


def _dataset(value: object, *, project_id: ProjectId) -> ProjectDataset:
    dataset = ProjectDataset.from_wire(cast(JsonValue, value))
    if dataset.project_id != project_id:
        raise ValueError("project dataset response crossed its requested project boundary")
    return dataset


class ProjectRepositoriesAPI:
    """External source repositories attached to Research projects."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self, project_id: ProjectId) -> tuple[ProjectRepository, ...]:
        value = self._transport.execute(
            _request(
                "list_project_external_repositories",
                f"/smr/projects/{project_id}/external-repositories",
            )
        )
        return _repositories(value, project_id=project_id)

    def create(
        self,
        project_id: ProjectId,
        request: ProjectRepositorySpec,
    ) -> ProjectRepository:
        value = self._transport.execute(
            _request(
                "create_project_external_repository",
                f"/smr/projects/{project_id}/external-repositories",
                body=request.to_wire(),
            )
        )
        return _repository(value, project_id=project_id)

    def update(
        self,
        project_id: ProjectId,
        repository_id: ProjectRepositoryId,
        request: ProjectRepositoryPatch,
    ) -> ProjectRepository:
        value = self._transport.execute(
            _request(
                "update_project_external_repository",
                f"/smr/projects/{project_id}/external-repositories/{repository_id}",
                body=request.to_wire(),
            )
        )
        return _repository(
            value,
            project_id=project_id,
            repository_id=repository_id,
        )

    def delete(
        self,
        project_id: ProjectId,
        repository_id: ProjectRepositoryId,
    ) -> ProjectRepositoryDeletion:
        value = self._transport.execute(
            _request(
                "delete_project_external_repository",
                f"/smr/projects/{project_id}/external-repositories/{repository_id}",
            )
        )
        receipt = ProjectRepositoryDeletion.from_wire(value)
        if receipt.repository_id != repository_id:
            raise ValueError("project repository deletion identity drifted")
        return receipt


class ProjectDatasetsAPI:
    """Project-scoped datasets and their raw content."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self, project_id: ProjectId) -> tuple[ProjectDataset, ...]:
        value = self._transport.execute(
            _request(
                "list_project_datasets",
                f"/smr/projects/{project_id}/datasets",
            )
        )
        return _datasets(value, project_id=project_id)

    def upload(
        self,
        project_id: ProjectId,
        request: ProjectDatasetUpload,
    ) -> ProjectDataset:
        value = self._transport.execute(
            _request(
                "create_project_dataset",
                f"/smr/projects/{project_id}/datasets",
                body=request.to_wire(),
            )
        )
        return _dataset(value, project_id=project_id)

    def download(
        self,
        project_id: ProjectId,
        dataset_id: ProjectDatasetId,
    ) -> bytes:
        operation = research_operation("retrieve_project_dataset_content")
        return self._transport.request_bytes(
            operation.method.value,
            f"/smr/projects/{project_id}/datasets/{dataset_id}/download",
            operation_id=str(operation.operation_id),
        )


class AsyncProjectRepositoriesAPI:
    """Native-async peer of :class:`ProjectRepositoriesAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self, project_id: ProjectId) -> tuple[ProjectRepository, ...]:
        value = await self._transport.execute(
            _request(
                "list_project_external_repositories",
                f"/smr/projects/{project_id}/external-repositories",
            )
        )
        return _repositories(value, project_id=project_id)

    async def create(
        self,
        project_id: ProjectId,
        request: ProjectRepositorySpec,
    ) -> ProjectRepository:
        value = await self._transport.execute(
            _request(
                "create_project_external_repository",
                f"/smr/projects/{project_id}/external-repositories",
                body=request.to_wire(),
            )
        )
        return _repository(value, project_id=project_id)

    async def update(
        self,
        project_id: ProjectId,
        repository_id: ProjectRepositoryId,
        request: ProjectRepositoryPatch,
    ) -> ProjectRepository:
        value = await self._transport.execute(
            _request(
                "update_project_external_repository",
                f"/smr/projects/{project_id}/external-repositories/{repository_id}",
                body=request.to_wire(),
            )
        )
        return _repository(
            value,
            project_id=project_id,
            repository_id=repository_id,
        )

    async def delete(
        self,
        project_id: ProjectId,
        repository_id: ProjectRepositoryId,
    ) -> ProjectRepositoryDeletion:
        value = await self._transport.execute(
            _request(
                "delete_project_external_repository",
                f"/smr/projects/{project_id}/external-repositories/{repository_id}",
            )
        )
        receipt = ProjectRepositoryDeletion.from_wire(value)
        if receipt.repository_id != repository_id:
            raise ValueError("project repository deletion identity drifted")
        return receipt


class AsyncProjectDatasetsAPI:
    """Native-async peer of :class:`ProjectDatasetsAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self, project_id: ProjectId) -> tuple[ProjectDataset, ...]:
        value = await self._transport.execute(
            _request(
                "list_project_datasets",
                f"/smr/projects/{project_id}/datasets",
            )
        )
        return _datasets(value, project_id=project_id)

    async def upload(
        self,
        project_id: ProjectId,
        request: ProjectDatasetUpload,
    ) -> ProjectDataset:
        value = await self._transport.execute(
            _request(
                "create_project_dataset",
                f"/smr/projects/{project_id}/datasets",
                body=request.to_wire(),
            )
        )
        return _dataset(value, project_id=project_id)

    async def download(
        self,
        project_id: ProjectId,
        dataset_id: ProjectDatasetId,
    ) -> bytes:
        operation = research_operation("retrieve_project_dataset_content")
        return await self._transport.request_bytes(
            operation.method.value,
            f"/smr/projects/{project_id}/datasets/{dataset_id}/download",
            operation_id=str(operation.operation_id),
        )


__all__ = [
    "AsyncProjectDatasetsAPI",
    "AsyncProjectRepositoriesAPI",
    "ProjectDatasetsAPI",
    "ProjectRepositoriesAPI",
]
