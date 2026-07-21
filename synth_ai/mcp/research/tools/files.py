"""Work-stage file tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import ToolDefinition, tool_schema
from synth_ai.mcp.research.tools.resources import _FILE_ITEM_SCHEMA


def build_file_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_list_run_output_files",
            description=(
                "List output files produced by a run. Returns file metadata including "
                "artifact_type, path, content_type, and output_file_id for use with "
                "smr_get_run_output_file_content."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "artifact_type": {
                        "type": "string",
                        "description": "Optional filter by artifact type.",
                    },
                    "limit": {"type": "integer", "description": "Maximum files to return."},
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_output_files,
        ),
        ToolDefinition(
            name="smr_get_run_output_file_content",
            description=(
                "Fetch the content of a run output file by output_file_id. "
                "For S3-backed files this follows a redirect to the presigned download URL "
                "and returns the content."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "output_file_id": {
                        "type": "string",
                        "description": "Output file id from smr_list_run_output_files.",
                    },
                    "disposition": {
                        "type": "string",
                        "enum": ["inline", "attachment"],
                        "description": "Content-Disposition hint. Defaults to inline.",
                    },
                },
                required=["run_id", "output_file_id"],
            ),
            handler=server._tool_get_run_output_file_content,
        ),
        ToolDefinition(
            name="smr_work_files_list",
            description="List project files for the work stage.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "visibility": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["project_id"],
            ),
            handler=server._tool_list_project_files,
        ),
        ToolDefinition(
            name="smr_work_files_upload",
            description="Upload project files for the work stage.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "items": _FILE_ITEM_SCHEMA,
                    },
                },
                required=["project_id", "files"],
            ),
            handler=server._tool_create_project_files,
        ),
    ]


__all__ = ["build_file_tools"]
