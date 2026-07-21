"""Workspace-input MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import ToolDefinition, tool_schema


def build_workspace_input_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_attach_source_repo",
            description="Attach a source repository and refresh the canonical project repo bootstrap state.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "url": {"type": "string", "description": "Public source repository URL."},
                    "default_branch": {
                        "type": "string",
                        "description": "Optional default branch override.",
                    },
                },
                required=["project_id", "url"],
            ),
            handler=server._tool_attach_source_repo,
        ),
        ToolDefinition(
            name="smr_get_workspace_inputs",
            description="Fetch the current workspace bootstrap inputs for a project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_get_workspace_inputs,
        ),
        ToolDefinition(
            name="smr_upload_workspace_files",
            description="Upload project-scoped workspace files that will be added to worker workspaces.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
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
                                "metadata": {"type": "object"},
                            },
                            "required": ["path", "content"],
                            "additionalProperties": False,
                        },
                    },
                },
                required=["project_id", "files"],
            ),
            handler=server._tool_upload_workspace_files,
        ),
    ]


__all__ = ["build_workspace_input_tools"]
