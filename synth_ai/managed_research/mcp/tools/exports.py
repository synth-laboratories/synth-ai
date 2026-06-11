"""Setup-stage export target tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema


def build_export_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_setup_exports_list_targets",
            description="List org export targets.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_setup_exports_list_targets,
        ),
        ToolDefinition(
            name="smr_setup_exports_create_target",
            description="Create an org export target.",
            input_schema=tool_schema(
                {
                    "label": {"type": "string"},
                    "bucket": {"type": "string"},
                    "prefix": {"type": "string"},
                    "region": {"type": "string"},
                    "endpoint_url": {"type": "string"},
                    "access_key_id": {"type": "string"},
                    "secret_access_key": {"type": "string"},
                },
                required=["label", "bucket", "access_key_id", "secret_access_key"],
            ),
            handler=server._tool_setup_exports_create_target,
        ),
    ]


__all__ = ["build_export_tools"]
