"""Stable MCP adapters for typed project repositories and datasets."""

from __future__ import annotations

import base64
from collections.abc import Callable

from synth_ai.core.research.client import ResearchClient
from synth_ai.core.research.contracts.common import (
    ProjectDatasetId,
    ProjectId,
    ProjectRepositoryId,
)
from synth_ai.core.research.contracts.project_data import (
    ProjectDatasetEncoding,
    ProjectDatasetUpload,
    ProjectRepositoryPatch,
    ProjectRepositoryRole,
    ProjectRepositorySpec,
    ResourceMetadata,
)
from synth_ai.mcp.research.registry import (
    JSONDict,
    READ_SCOPES,
    WRITE_SCOPES,
    ToolDefinition,
    tool_schema,
)
from synth_ai.mcp.research.request_models import (
    optional_int,
    optional_string,
    require_string,
)


CoreClientFactory = Callable[[JSONDict], ResearchClient]


def _metadata(args: JSONDict, *, required: bool) -> ResourceMetadata | None:
    if "metadata" not in args:
        return ResourceMetadata() if required else None
    value = args["metadata"]
    if not isinstance(value, dict):
        raise ValueError("'metadata' must be an object")
    return ResourceMetadata(value)


def _dataset_content(args: JSONDict) -> str:
    value = args.get("content")
    if not isinstance(value, str):
        raise ValueError("'content' is required and must be a string")
    return value


def build_project_data_tools(
    client_from_args: CoreClientFactory,
) -> list[ToolDefinition]:
    """Build seven stable adapters backed by the bounded operation registry."""

    def list_repositories(args: JSONDict) -> list[JSONDict]:
        project_id = ProjectId(require_string(args, "project_id"))
        with client_from_args(args) as client:
            return [
                repository.to_wire()
                for repository in client.projects.repositories.list(project_id)
            ]

    def create_repository(args: JSONDict) -> JSONDict:
        project_id = ProjectId(require_string(args, "project_id"))
        role = optional_string(args, "role")
        request = ProjectRepositorySpec(
            name=require_string(args, "name"),
            url=require_string(args, "url"),
            default_branch=optional_string(args, "default_branch"),
            role=(
                ProjectRepositoryRole(role)
                if role is not None
                else ProjectRepositoryRole.DEPENDENCY
            ),
            metadata=_metadata(args, required=True) or ResourceMetadata(),
        )
        with client_from_args(args) as client:
            return client.projects.repositories.create(project_id, request).to_wire()

    def update_repository(args: JSONDict) -> JSONDict:
        project_id = ProjectId(require_string(args, "project_id"))
        repository_id = ProjectRepositoryId(
            require_string(args, "repository_id")
        )
        role = optional_string(args, "role")
        request = ProjectRepositoryPatch(
            url=optional_string(args, "url"),
            default_branch=optional_string(args, "default_branch"),
            role=ProjectRepositoryRole(role) if role is not None else None,
            metadata=_metadata(args, required=False),
        )
        with client_from_args(args) as client:
            return client.projects.repositories.update(
                project_id,
                repository_id,
                request,
            ).to_wire()

    def delete_repository(args: JSONDict) -> JSONDict:
        project_id = ProjectId(require_string(args, "project_id"))
        repository_id = ProjectRepositoryId(
            require_string(args, "repository_id")
        )
        with client_from_args(args) as client:
            return client.projects.repositories.delete(
                project_id,
                repository_id,
            ).to_wire()

    def list_datasets(args: JSONDict) -> list[JSONDict]:
        project_id = ProjectId(require_string(args, "project_id"))
        with client_from_args(args) as client:
            return [
                dataset.to_wire()
                for dataset in client.projects.datasets.list(project_id)
            ]

    def upload_dataset(args: JSONDict) -> JSONDict:
        project_id = ProjectId(require_string(args, "project_id"))
        encoding = optional_string(args, "encoding")
        request = ProjectDatasetUpload(
            name=require_string(args, "name"),
            content=_dataset_content(args),
            encoding=(
                ProjectDatasetEncoding(encoding)
                if encoding is not None
                else ProjectDatasetEncoding.TEXT
            ),
            content_type=optional_string(args, "content_type"),
            format=optional_string(args, "format"),
            row_count=optional_int(args, "row_count"),
            metadata=_metadata(args, required=True) or ResourceMetadata(),
        )
        with client_from_args(args) as client:
            return client.projects.datasets.upload(project_id, request).to_wire()

    def get_dataset_content(args: JSONDict) -> JSONDict:
        project_id = ProjectId(require_string(args, "project_id"))
        dataset_id = ProjectDatasetId(require_string(args, "dataset_id"))
        with client_from_args(args) as client:
            content = client.projects.datasets.download(project_id, dataset_id)
        return {
            "dataset_id": dataset_id,
            "encoding": "base64",
            "content": base64.b64encode(content).decode("ascii"),
            "size_bytes": len(content),
        }

    repository_role_schema = {
        "type": "string",
        "enum": [role.value for role in ProjectRepositoryRole],
    }
    dataset_encoding_schema = {
        "type": "string",
        "enum": [encoding.value for encoding in ProjectDatasetEncoding],
    }
    return [
        ToolDefinition(
            name="smr_list_project_repositories",
            description="List typed external repositories attached to a project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string"}},
                required=["project_id"],
            ),
            handler=list_repositories,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_create_project_repository",
            description="Attach one typed external repository to a project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "name": {"type": "string"},
                    "url": {"type": "string"},
                    "default_branch": {"type": "string"},
                    "role": repository_role_schema,
                    "metadata": {"type": "object"},
                },
                required=["project_id", "name", "url"],
            ),
            handler=create_repository,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_update_project_repository",
            description="Update one typed external project repository.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "repository_id": {"type": "string"},
                    "url": {"type": "string"},
                    "default_branch": {"type": "string"},
                    "role": repository_role_schema,
                    "metadata": {"type": "object"},
                },
                required=["project_id", "repository_id"],
            ),
            handler=update_repository,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_delete_project_repository",
            description="Delete one typed external project repository.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "repository_id": {"type": "string"},
                },
                required=["project_id", "repository_id"],
            ),
            handler=delete_repository,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_project_datasets",
            description="List typed datasets attached to a project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string"}},
                required=["project_id"],
            ),
            handler=list_datasets,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_upload_project_dataset",
            description="Upload one binary-safe dataset to a project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                    "encoding": dataset_encoding_schema,
                    "content_type": {"type": "string"},
                    "format": {"type": "string"},
                    "row_count": {"type": "integer", "minimum": 0},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "name", "content"],
            ),
            handler=upload_dataset,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_project_dataset_content",
            description="Fetch raw dataset bytes as an explicit base64 MCP payload.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                },
                required=["project_id", "dataset_id"],
            ),
            handler=get_dataset_content,
            required_scopes=READ_SCOPES,
        ),
    ]


__all__ = ["build_project_data_tools"]
