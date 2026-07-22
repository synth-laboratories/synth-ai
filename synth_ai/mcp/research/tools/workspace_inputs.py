"""Stable MCP adapters for project-owned workspace inputs."""

from __future__ import annotations

from collections.abc import Callable

from synth_ai.core.research.client import ResearchClient
from synth_ai.core.research.contracts.common import ProjectId
from synth_ai.core.research.contracts.workspaces import (
    WORKSPACE_BATCH_UPLOAD_FILE_LIMIT,
    WorkspaceFileEncoding,
    WorkspaceFileKind,
    WorkspaceFilesBatchUploadRequest,
    WorkspaceFileUpload,
    WorkspaceMetadata,
    WorkspaceSourceRepositorySpec,
)
from synth_ai.mcp.research.registry import (
    JSONDict,
    READ_SCOPES,
    WRITE_SCOPES,
    ToolDefinition,
    tool_schema,
)
from synth_ai.mcp.research.request_models import optional_string, require_string


CoreClientFactory = Callable[[JSONDict], ResearchClient]

_FILE_FIELDS = frozenset(
    {"path", "content", "content_type", "encoding", "kind", "metadata"}
)


def _file_upload(value: object, *, index: int) -> WorkspaceFileUpload:
    if not isinstance(value, dict):
        raise ValueError(f"files[{index}] must be an object")
    unknown = set(value).difference(_FILE_FIELDS)
    if unknown:
        raise ValueError(
            f"files[{index}] contains unsupported fields: {', '.join(sorted(unknown))}"
        )
    path = value.get("path")
    content = value.get("content")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"files[{index}].path must be a non-empty string")
    if not isinstance(content, str):
        raise ValueError(f"files[{index}].content must be a string")
    content_type = value.get("content_type")
    if content_type is not None and not isinstance(content_type, str):
        raise ValueError(f"files[{index}].content_type must be a string")
    encoding = value.get("encoding")
    if encoding is not None and not isinstance(encoding, str):
        raise ValueError(f"files[{index}].encoding must be a string")
    kind = value.get("kind")
    if kind is not None and not isinstance(kind, str):
        raise ValueError(f"files[{index}].kind must be a string")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"files[{index}].metadata must be an object")
    return WorkspaceFileUpload(
        path=path,
        content=content,
        content_type=content_type,
        encoding=WorkspaceFileEncoding(encoding) if encoding is not None else None,
        kind=WorkspaceFileKind(kind) if kind is not None else None,
        metadata=WorkspaceMetadata(metadata),
    )


def _batch_request(args: JSONDict) -> WorkspaceFilesBatchUploadRequest:
    raw_files = args.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("'files' must be a non-empty array")
    return WorkspaceFilesBatchUploadRequest(
        files=tuple(
            _file_upload(value, index=index)
            for index, value in enumerate(raw_files)
        )
    )


def build_workspace_input_tools(
    client_from_args: CoreClientFactory,
) -> list[ToolDefinition]:
    """Build stable adapters over the same typed SDK used by Python callers."""

    def attach_source_repository(args: JSONDict) -> JSONDict:
        request = WorkspaceSourceRepositorySpec(
            url=require_string(args, "url"),
            default_branch=optional_string(args, "default_branch"),
            commit_sha=optional_string(args, "commit_sha"),
        )
        with client_from_args(args) as client:
            return client.projects.workspace.set_source_repository(
                ProjectId(require_string(args, "project_id")),
                request,
            ).to_wire()

    def get_workspace_inputs(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.projects.workspace.retrieve(
                ProjectId(require_string(args, "project_id"))
            ).to_wire()

    def upload_workspace_files(args: JSONDict) -> JSONDict:
        request = _batch_request(args)
        with client_from_args(args) as client:
            return client.projects.workspace.upload_batches(
                ProjectId(require_string(args, "project_id")),
                request,
            ).to_wire()

    return [
        ToolDefinition(
            name="smr_attach_source_repo",
            description=(
                "Set the public source repository used to bootstrap one Research "
                "project workspace."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Research project ID.",
                    },
                    "url": {
                        "type": "string",
                        "description": "Public source repository URL.",
                    },
                    "default_branch": {
                        "type": "string",
                        "description": "Optional default branch override.",
                    },
                    "commit_sha": {
                        "type": "string",
                        "pattern": "^[0-9a-fA-F]{7,64}$",
                        "description": "Optional immutable source commit.",
                    },
                },
                required=["project_id", "url"],
            ),
            handler=attach_source_repository,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_workspace_inputs",
            description="Fetch the typed workspace bootstrap inputs for a project.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Research project ID.",
                    }
                },
                required=["project_id"],
            ),
            handler=get_workspace_inputs,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_upload_workspace_files",
            description=(
                "Upload up to 10,000 project workspace files in deterministic, "
                "retry-safe batches and return the complete ordered receipt."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Research project ID.",
                    },
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": WORKSPACE_BATCH_UPLOAD_FILE_LIMIT,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                                "content_type": {"type": "string"},
                                "encoding": {
                                    "type": "string",
                                    "enum": [
                                        item.value for item in WorkspaceFileEncoding
                                    ],
                                },
                                "kind": {
                                    "type": "string",
                                    "enum": [item.value for item in WorkspaceFileKind],
                                },
                                "metadata": {"type": "object"},
                            },
                            "required": ["path", "content"],
                            "additionalProperties": False,
                        },
                    },
                },
                required=["project_id", "files"],
            ),
            handler=upload_workspace_files,
            required_scopes=WRITE_SCOPES,
        ),
    ]


__all__ = ["build_workspace_input_tools"]
