"""Results-stage model tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import ToolDefinition, tool_schema


def build_model_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_results_models_list",
            description="List project models.",
            input_schema=tool_schema(
                {"project_id": {"type": "string"}},
                required=["project_id"],
            ),
            handler=server._tool_results_models_list,
        ),
        ToolDefinition(
            name="smr_results_models_get",
            description="Fetch one project model.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "model_id": {"type": "string"},
                },
                required=["project_id", "model_id"],
            ),
            handler=server._tool_results_models_get,
        ),
        ToolDefinition(
            name="smr_results_models_download",
            description="Download one project model.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "model_id": {"type": "string"},
                },
                required=["project_id", "model_id"],
            ),
            handler=server._tool_results_models_download,
        ),
        ToolDefinition(
            name="smr_results_models_export",
            description="Export one project model to the bound target.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "model_id": {"type": "string"},
                },
                required=["project_id", "model_id"],
            ),
            handler=server._tool_results_models_export,
        ),
    ]


__all__ = ["build_model_tools"]
