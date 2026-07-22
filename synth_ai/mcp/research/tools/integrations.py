"""Integration MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import (
    READ_SCOPES,
    WRITE_SCOPES,
    ToolDefinition,
    tool_schema,
)


def build_integration_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_setup_github_status",
            description="Get the authenticated org's GitHub integration status.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_setup_github_status,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_setup_github_start_oauth",
            description="Start GitHub OAuth setup for the authenticated org.",
            input_schema=tool_schema(
                {"redirect_uri": {"type": "string"}},
                required=[],
            ),
            handler=server._tool_setup_github_start_oauth,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_setup_github_list_repos",
            description="List GitHub repositories available to the authenticated org.",
            input_schema=tool_schema(
                {
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                required=[],
            ),
            handler=server._tool_setup_github_list_repos,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_setup_github_disconnect",
            description="Disconnect the authenticated org's GitHub integration.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_setup_github_disconnect,
            required_scopes=WRITE_SCOPES,
        ),
    ]


__all__ = ["build_integration_tools"]
