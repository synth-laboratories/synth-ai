"""Results-stage PR tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema


def build_pr_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_results_prs_list",
            description="List project pull requests published by Managed Research.",
            input_schema=tool_schema(
                {"project_id": {"type": "string"}},
                required=["project_id"],
            ),
            handler=server._tool_results_prs_list,
        ),
        ToolDefinition(
            name="smr_results_prs_get",
            description="Fetch one project pull request result.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "pr_id": {"type": "string"},
                },
                required=["project_id", "pr_id"],
            ),
            handler=server._tool_results_prs_get,
        ),
    ]


__all__ = ["build_pr_tools"]
