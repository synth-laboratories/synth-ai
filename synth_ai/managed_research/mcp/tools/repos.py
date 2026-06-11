"""Work-stage repo tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema


def build_repo_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_work_repos_list",
            description="List project repo bindings.",
            input_schema=tool_schema(
                {"project_id": {"type": "string"}},
                required=["project_id"],
            ),
            handler=server._tool_work_repos_list,
        ),
        ToolDefinition(
            name="smr_work_repos_attach",
            description="Attach a GitHub repo to a project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "github_repo": {"type": "string"},
                    "pr_write_enabled": {"type": "boolean"},
                },
                required=["project_id", "github_repo"],
            ),
            handler=server._tool_work_repos_attach,
        ),
        ToolDefinition(
            name="smr_work_repos_detach",
            description="Detach a GitHub repo from a project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "github_repo": {"type": "string"},
                },
                required=["project_id", "github_repo"],
            ),
            handler=server._tool_work_repos_detach,
        ),
    ]


__all__ = ["build_repo_tools"]
