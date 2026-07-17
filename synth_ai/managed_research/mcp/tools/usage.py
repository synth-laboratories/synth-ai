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
            name="smr_preview_admin_promotion_discount",
            description=(
                "Preview backend-authored draft promotion economics as an admin. "
                "This read-scoped preview does not activate or enforce a campaign, "
                "consume caps, grant benefits, or debit usage."
            ),
            input_schema=tool_schema(
                {
                    "campaign_id": {
                        "type": "string",
                        "description": "Draft promotion campaign id.",
                    },
                    "nominal_customer_debit_microcents": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Nominal customer debit scenario in microcents.",
                    },
                    "provider_cost_pico_usd": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Provider cost scenario in pico-USD.",
                    },
                },
                required=[
                    "campaign_id",
                    "nominal_customer_debit_microcents",
                    "provider_cost_pico_usd",
                ],
            ),
            handler=server._tool_preview_admin_promotion_discount,
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
            description="Request a run resource-limit extension, optionally resolving blockers and resuming work.",
            input_schema=tool_schema(
                {
                    "scope": {
                        "type": "string",
                        "enum": ["run"],
                        "description": "Limit scope to extend. Project-scoped run lookup is selected by also passing project_id.",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Run id when scope is run.",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project id for project-scoped run lookup.",
                    },
                    "limit_value": {
                        "type": "number",
                        "description": "New absolute limit in the requested unit.",
                    },
                    "additional_value": {
                        "type": "number",
                        "description": "Amount to add to the current limit in the requested unit.",
                    },
                    "selector": {
                        "type": "object",
                        "description": "Optional resource selector, for example {'kind':'run','resource_id':'wallclock_seconds'}.",
                    },
                    "resource_limit_id": {
                        "type": "string",
                        "description": "Optional resource limit id being extended.",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Optional limit metric, for example spend_usd or wallclock_seconds. Defaults to spend_usd only when selector/resource_limit_id are omitted.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Optional unit. Must match the metric, for example usd for spend_usd or seconds for wallclock_seconds.",
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
        ToolDefinition(
            name="smr_get_project_economics",
            description="Fetch canonical project economics rollups.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Project id.",
                    },
                },
                required=["project_id"],
            ),
            handler=server._tool_get_project_economics,
        ),
    ]


__all__ = ["build_usage_tools"]
