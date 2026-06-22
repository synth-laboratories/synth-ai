"""Work-stage dataset tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema


def build_dataset_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_work_datasets_list",
            description="List project datasets.",
            input_schema=tool_schema(
                {"project_id": {"type": "string"}},
                required=["project_id"],
            ),
            handler=server._tool_work_datasets_list,
        ),
        ToolDefinition(
            name="smr_work_datasets_upload",
            description="Upload a dataset to a project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                    "encoding": {"type": "string"},
                    "content_type": {"type": "string"},
                    "format": {"type": "string"},
                    "row_count": {"type": "integer"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "name", "content"],
            ),
            handler=server._tool_work_datasets_upload,
        ),
        ToolDefinition(
            name="smr_work_datasets_download",
            description="Download a project dataset.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                },
                required=["project_id", "dataset_id"],
            ),
            handler=server._tool_work_datasets_download,
        ),
    ]


__all__ = ["build_dataset_tools"]
