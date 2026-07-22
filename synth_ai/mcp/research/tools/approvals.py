"""Approval MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.mcp.research.registry import ToolDefinition, tool_schema


def build_approval_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_list_run_approvals",
            description="List approvals for a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "status_filter": {
                        "type": "string",
                        "description": "Optional approval status filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "Maximum approvals to return.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_approvals,
        ),
        ToolDefinition(
            name="smr_approve_run_approval",
            description="Approve a pending run approval request.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "approval_id": {"type": "string", "description": "Approval id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "comment": {
                        "type": "string",
                        "description": "Optional operator comment stored with the decision.",
                    },
                },
                required=["run_id", "approval_id"],
            ),
            handler=server._tool_approve_run_approval,
        ),
        ToolDefinition(
            name="smr_deny_run_approval",
            description="Deny a pending run approval request.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "approval_id": {"type": "string", "description": "Approval id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "comment": {
                        "type": "string",
                        "description": "Optional operator comment stored with the decision.",
                    },
                },
                required=["run_id", "approval_id"],
            ),
            handler=server._tool_deny_run_approval,
        ),
        ToolDefinition(
            name="smr_respond_to_run_question",
            description="Respond to a pending operator question for a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "question_id": {"type": "string", "description": "Question id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "response_text": {
                        "type": "string",
                        "description": "Required operator response text.",
                    },
                },
                required=["run_id", "question_id", "response_text"],
            ),
            handler=server._tool_respond_to_run_question,
        ),
    ]


__all__ = ["build_approval_tools"]
