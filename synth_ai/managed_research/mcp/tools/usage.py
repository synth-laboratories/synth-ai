"""Usage MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema


def build_usage_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_get_billing_entitlements",
            description="Fetch the canonical org-level entitlement snapshot.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_get_billing_entitlements,
        ),
        ToolDefinition(
            name="smr_get_run_usage",
            description="Fetch canonical nominal, billed, internal, token, and breakdown usage for a run.",
            input_schema=tool_schema(
                {
                    "run_id": {
                        "type": "string",
                        "description": "Run id.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_usage,
        ),
        ToolDefinition(
            name="smr_get_run_resource_limits",
            description="Fetch configured resource limits for a run.",
            input_schema=tool_schema(
                {
                    "run_id": {
                        "type": "string",
                        "description": "Run id.",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project id for project-scoped run lookup.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_resource_limits,
        ),
        ToolDefinition(
            name="smr_get_run_progress_toward_resource_limits",
            description="Fetch current run progress toward resource limits, active blockers, and extension posture.",
            input_schema=tool_schema(
                {
                    "run_id": {
                        "type": "string",
                        "description": "Run id.",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project id for project-scoped run lookup.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_progress_toward_resource_limits,
        ),
        ToolDefinition(
            name="smr_request_resource_limit_extension",
            description="Request a USD spend limit extension for a run or project, optionally resolving blockers and resuming work.",
            input_schema=tool_schema(
                {
                    "scope": {
                        "type": "string",
                        "enum": ["run", "project"],
                        "description": "Limit scope to extend.",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Run id when scope is run.",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project id. Required for project scope; optional for project-scoped run lookup.",
                    },
                    "limit_value": {
                        "type": "number",
                        "description": "New absolute USD limit.",
                    },
                    "additional_value": {
                        "type": "number",
                        "description": "USD amount to add to the current limit.",
                    },
                    "resource_limit_id": {
                        "type": "string",
                        "description": "Optional resource limit id being extended.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Human-readable reason for the extension request.",
                    },
                    "resolve_blockers": {
                        "type": "boolean",
                        "description": "Resolve matching active budget blockers after extending.",
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Resume the run or unpause the project after extending.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Optional caller-supplied idempotency key.",
                    },
                },
                required=["scope"],
            ),
            handler=server._tool_request_resource_limit_extension,
        ),
        ToolDefinition(
            name="smr_get_project_usage",
            description="Fetch canonical project usage rollups and budgets.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Project id.",
                    },
                },
                required=["project_id"],
            ),
            handler=server._tool_get_project_usage,
        ),
    ]


__all__ = ["build_usage_tools"]
