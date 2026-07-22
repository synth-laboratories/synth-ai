"""Status-stage readiness tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import ToolDefinition, tool_schema


def build_readiness_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_status_readiness",
            description="Get the derived readiness checklist for a project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string"}},
                required=["project_id"],
            ),
            handler=server._tool_status_readiness,
        )
    ]


__all__ = ["build_readiness_tools"]
