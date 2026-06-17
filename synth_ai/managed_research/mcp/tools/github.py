"""Setup-stage GitHub tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema


def build_github_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_setup_github_status",
            description="Fetch org-level GitHub connection status.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_setup_github_status,
        ),
        ToolDefinition(
            name="smr_setup_github_start_oauth",
            description="Start the org-level GitHub connect flow.",
            input_schema=tool_schema(
                {
                    "redirect_uri": {"type": "string"},
                },
                required=[],
            ),
            handler=server._tool_setup_github_start_oauth,
        ),
        ToolDefinition(
            name="smr_setup_github_list_repos",
            description="List repos available through the org-level GitHub integration.",
            input_schema=tool_schema(
                {
                    "page": {"type": "integer"},
                    "per_page": {"type": "integer"},
                },
                required=[],
            ),
            handler=server._tool_setup_github_list_repos,
        ),
        ToolDefinition(
            name="smr_setup_github_disconnect",
            description="Disconnect the org-level GitHub integration.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_setup_github_disconnect,
        ),
    ]


__all__ = ["build_github_tools"]
