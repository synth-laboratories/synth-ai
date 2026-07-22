"""Artifact MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import ToolDefinition, tool_schema


def build_artifact_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_list_run_artifacts",
            description="List artifacts for a run from the stable API artifact contract.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "artifact_type": {"type": "string"},
                    "limit": {"type": "integer"},
                    "cursor": {"type": "string"},
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_artifacts,
        ),
        ToolDefinition(
            name="smr_get_run_artifact_manifest",
            description="Fetch the stable manifest of run artifacts, outputs, reports, models, datasets, and archive links.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string"},
                    "project_id": {"type": "string"},
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_artifact_manifest,
        ),
        ToolDefinition(
            name="smr_get_artifact",
            description="Fetch one artifact metadata record by artifact id.",
            input_schema=tool_schema(
                {"artifact_id": {"type": "string"}},
                required=["artifact_id"],
            ),
            handler=server._tool_get_artifact,
        ),
        ToolDefinition(
            name="smr_get_artifact_content",
            description="Fetch artifact content as utf-8 text or base64 payload.",
            input_schema=tool_schema(
                {
                    "artifact_id": {"type": "string"},
                    "disposition": {"type": "string"},
                },
                required=["artifact_id"],
            ),
            handler=server._tool_get_artifact_content,
        ),
        ToolDefinition(
            name="smr_download_artifact",
            description="Download artifact content to a path on the machine running this MCP server.",
            input_schema=tool_schema(
                {
                    "artifact_id": {"type": "string"},
                    "output_path": {"type": "string"},
                    "disposition": {"type": "string"},
                },
                required=["artifact_id", "output_path"],
            ),
            handler=server._tool_download_artifact,
        ),
        ToolDefinition(
            name="smr_list_run_models",
            description="List model resources produced by a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string"},
                    "project_id": {"type": "string"},
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_models,
        ),
        ToolDefinition(
            name="smr_list_run_datasets",
            description="List dataset resources mounted or produced by a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string"},
                    "project_id": {"type": "string"},
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_datasets,
        ),
    ]


__all__ = ["build_artifact_tools"]
