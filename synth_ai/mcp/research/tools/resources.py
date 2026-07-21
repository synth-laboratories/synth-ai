"""Phase 3 resource MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import ToolDefinition, tool_schema

_FILE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "content_type": {"type": "string"},
        "encoding": {"type": "string"},
        "kind": {
            "type": "string",
            "enum": ["file", "source_bundle"],
        },
        "visibility": {"type": "string"},
        "metadata": {"type": "object"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def build_resource_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_list_project_files",
            description="List project-scoped Phase 3 stored files.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "visibility": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["project_id"],
            ),
            handler=server._tool_list_project_files,
        ),
        ToolDefinition(
            name="smr_create_project_files",
            description="Create project-scoped Phase 3 stored files.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "items": _FILE_ITEM_SCHEMA,
                    },
                },
                required=["project_id", "files"],
            ),
            handler=server._tool_create_project_files,
        ),
        ToolDefinition(
            name="smr_get_project_file",
            description="Fetch one project-scoped stored file record.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "file_id": {"type": "string"},
                },
                required=["project_id", "file_id"],
            ),
            handler=server._tool_get_project_file,
        ),
        ToolDefinition(
            name="smr_get_file_content",
            description="Fetch stored file content as utf-8 text or base64 payload.",
            input_schema=tool_schema(
                {"file_id": {"type": "string"}},
                required=["file_id"],
            ),
            handler=server._tool_get_file_content,
        ),
        ToolDefinition(
            name="smr_list_run_file_mounts",
            description="List run file mounts, including auto-bound project files.",
            input_schema=tool_schema(
                {"run_id": {"type": "string"}},
                required=["run_id"],
            ),
            handler=server._tool_list_run_file_mounts,
        ),
        ToolDefinition(
            name="smr_upload_run_files",
            description="Upload run-scoped files and mount model-visible ones into the active run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string"},
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "items": _FILE_ITEM_SCHEMA,
                    },
                },
                required=["run_id", "files"],
            ),
            handler=server._tool_upload_run_files,
        ),
        ToolDefinition(
            name="smr_list_project_external_repositories",
            description="List project-scoped external repository resources.",
            input_schema=tool_schema(
                {"project_id": {"type": "string"}},
                required=["project_id"],
            ),
            handler=server._tool_list_project_external_repositories,
        ),
        ToolDefinition(
            name="smr_create_project_external_repository",
            description="Create or upsert a project-scoped external repository resource.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "name": {"type": "string"},
                    "url": {"type": "string"},
                    "default_branch": {"type": "string"},
                    "role": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "name", "url"],
            ),
            handler=server._tool_create_project_external_repository,
        ),
        ToolDefinition(
            name="smr_patch_project_external_repository",
            description="Patch an existing project-scoped external repository resource.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "repository_id": {"type": "string"},
                    "url": {"type": "string"},
                    "default_branch": {"type": "string"},
                    "role": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "repository_id"],
            ),
            handler=server._tool_patch_project_external_repository,
        ),
        ToolDefinition(
            name="smr_list_run_repository_mounts",
            description="List external repositories mounted onto a run.",
            input_schema=tool_schema(
                {"run_id": {"type": "string"}},
                required=["run_id"],
            ),
            handler=server._tool_list_run_repository_mounts,
        ),
        ToolDefinition(
            name="smr_create_run_repository_mount",
            description="Bind one existing external repository resource onto a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string"},
                    "repository_id": {"type": "string"},
                    "mount_name": {"type": "string"},
                },
                required=["run_id", "repository_id"],
            ),
            handler=server._tool_create_run_repository_mount,
        ),
        ToolDefinition(
            name="smr_list_project_credential_refs",
            description="List project-scoped credential refs.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "kind": {"type": "string"},
                },
                required=["project_id"],
            ),
            handler=server._tool_list_project_credential_refs,
        ),
        ToolDefinition(
            name="smr_create_project_credential_ref",
            description="Create or upsert a project-scoped credential ref.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "label": {"type": "string"},
                    "provider": {"type": "string"},
                    "funding_source": {"type": "string"},
                    "credential_name": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "kind", "label"],
            ),
            handler=server._tool_create_project_credential_ref,
        ),
        ToolDefinition(
            name="smr_patch_project_credential_ref",
            description="Patch an existing project-scoped credential ref.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "credential_ref_id": {"type": "string"},
                    "provider": {"type": "string"},
                    "funding_source": {"type": "string"},
                    "credential_name": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "credential_ref_id"],
            ),
            handler=server._tool_patch_project_credential_ref,
        ),
        ToolDefinition(
            name="smr_list_run_credential_bindings",
            description="List credential refs bound onto a run.",
            input_schema=tool_schema(
                {"run_id": {"type": "string"}},
                required=["run_id"],
            ),
            handler=server._tool_list_run_credential_bindings,
        ),
        ToolDefinition(
            name="smr_create_run_credential_binding",
            description="Bind one existing credential ref onto a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string"},
                    "credential_ref_id": {"type": "string"},
                },
                required=["run_id", "credential_ref_id"],
            ),
            handler=server._tool_create_run_credential_binding,
        ),
    ]


__all__ = ["build_resource_tools"]
