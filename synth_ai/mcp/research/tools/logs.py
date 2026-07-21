"""Log-related MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import ToolDefinition, tool_schema


def build_log_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_list_run_log_archives",
            description="List archived log bundles for a run.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_list_run_log_archives,
        )
    ]


__all__ = ["build_log_tools"]
