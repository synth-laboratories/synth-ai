"""Live dev-environment MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema

_DEV_ENVIRONMENT_ID = {
    "dev_environment_id": {
        "type": "string",
        "description": "Live DevEnvironment id.",
    }
}

_METADATA = {
    "metadata": {
        "type": "object",
        "description": "Optional caller metadata for lifecycle receipts.",
    }
}


def _dev_environment_action_schema() -> dict[str, Any]:
    return tool_schema(
        {
            **_DEV_ENVIRONMENT_ID,
            **_METADATA,
        },
        required=["dev_environment_id"],
    )


def build_dev_environment_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_list_dev_environment_topologies",
            description="List available live DevEnvironment topology templates.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_list_dev_environment_topologies,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_topology",
            description="Fetch one live DevEnvironment topology template.",
            input_schema=tool_schema(
                {
                    "topology_id": {
                        "type": "string",
                        "description": "Topology id, for example synth-dev.",
                    },
                    "version": {
                        "type": "string",
                        "description": "Optional topology version.",
                    },
                },
                required=["topology_id"],
            ),
            handler=server._tool_get_dev_environment_topology,
        ),
        ToolDefinition(
            name="smr_seed_dev_environment_topology_manifest",
            description="Seed the immutable /smr/environments catalog manifest required by a DevEnvironment topology.",
            input_schema=tool_schema(
                {
                    "topology_id": {
                        "type": "string",
                        "description": "Topology id. Defaults to synth-dev.",
                    },
                    "version": {
                        "type": "string",
                        "description": "Optional topology version.",
                    },
                },
                required=[],
            ),
            handler=server._tool_seed_dev_environment_topology_manifest,
        ),
        ToolDefinition(
            name="smr_list_dev_environments",
            description="List live DevEnvironments visible to the authenticated org.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Optional project id filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum row count.",
                    },
                },
                required=[],
            ),
            handler=server._tool_list_dev_environments,
        ),
        ToolDefinition(
            name="smr_list_dev_environment_materialization_queue",
            description="List pending DevEnvironment materialization work for substrate workers.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Optional project id filter.",
                    },
                    "host_kind": {
                        "type": "string",
                        "description": "Optional substrate filter, for example daytona.",
                    },
                    "backend_target": {
                        "type": "string",
                        "description": "Optional backend target filter.",
                    },
                    "worker_id": {
                        "type": "string",
                        "description": "Optional worker id; includes that worker's active leases.",
                    },
                    "include_leased": {
                        "type": "boolean",
                        "description": "Whether to include actively leased work.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum row count.",
                    },
                },
                required=[],
            ),
            handler=server._tool_list_dev_environment_materialization_queue,
        ),
        ToolDefinition(
            name="smr_create_dev_environment",
            description="Create a live DevEnvironment bound to a project and immutable environment manifest.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed Research project id.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable DevEnvironment name.",
                    },
                    "environment_name": {
                        "type": "string",
                        "description": "Immutable /smr/environments catalog manifest name.",
                    },
                    "backend_target": {
                        "type": "string",
                        "description": "Backend target, for example dev, staging, or prod.",
                    },
                    "topology_id": {
                        "type": "string",
                        "description": "Topology id. Defaults to synth-dev.",
                    },
                    "topology_version": {
                        "type": "string",
                        "description": "Optional topology version.",
                    },
                    "environment_digest": {
                        "type": "string",
                        "description": "Optional immutable environment manifest digest.",
                    },
                    "host_kind": {
                        "type": "string",
                        "description": "Execution substrate kind, for example daytona.",
                    },
                    "quota_class": {
                        "type": "string",
                        "description": "Optional quota class.",
                    },
                    "uptime_rate_microcents_per_hour": {
                        "type": "integer",
                        "description": "Optional DevEnvironment uptime price in microcents per hour.",
                    },
                    "billing_model_class": {
                        "type": "string",
                        "description": "Optional billing model class, value or premium.",
                    },
                    **_METADATA,
                },
                required=["project_id", "name", "environment_name"],
            ),
            handler=server._tool_create_dev_environment,
        ),
        ToolDefinition(
            name="smr_create_dev_environment_from_topology",
            description="Seed the topology catalog manifest and create a live DevEnvironment bound to the returned digest.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed Research project id.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable DevEnvironment name.",
                    },
                    "backend_target": {
                        "type": "string",
                        "description": "Backend target, for example dev, staging, or prod.",
                    },
                    "topology_id": {
                        "type": "string",
                        "description": "Topology id. Defaults to synth-dev.",
                    },
                    "topology_version": {
                        "type": "string",
                        "description": "Optional topology version.",
                    },
                    "host_kind": {
                        "type": "string",
                        "description": "Execution substrate kind, for example daytona.",
                    },
                    "quota_class": {
                        "type": "string",
                        "description": "Optional quota class.",
                    },
                    "uptime_rate_microcents_per_hour": {
                        "type": "integer",
                        "description": "Optional DevEnvironment uptime price in microcents per hour.",
                    },
                    "billing_model_class": {
                        "type": "string",
                        "description": "Optional billing model class, value or premium.",
                    },
                    **_METADATA,
                },
                required=["project_id", "name"],
            ),
            handler=server._tool_create_dev_environment_from_topology,
        ),
        ToolDefinition(
            name="smr_get_dev_environment",
            description="Fetch one live DevEnvironment.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_get_dev_environment,
        ),
        ToolDefinition(
            name="smr_claim_dev_environment_materialization",
            description="Claim pending DevEnvironment materialization work with a bounded lease.",
            input_schema=tool_schema(
                {
                    **_DEV_ENVIRONMENT_ID,
                    "worker_id": {
                        "type": "string",
                        "description": "Materializer worker id.",
                    },
                    "lease_seconds": {
                        "type": "integer",
                        "description": "Optional lease duration in seconds.",
                    },
                    **_METADATA,
                },
                required=["dev_environment_id", "worker_id"],
            ),
            handler=server._tool_claim_dev_environment_materialization,
        ),
        ToolDefinition(
            name="smr_preflight_dev_environment",
            description="Preflight one live DevEnvironment before deploy or run binding.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_preflight_dev_environment,
        ),
        ToolDefinition(
            name="smr_deploy_dev_environment",
            description="Deploy a live DevEnvironment onto its configured substrate.",
            input_schema=_dev_environment_action_schema(),
            handler=server._tool_deploy_dev_environment,
        ),
        ToolDefinition(
            name="smr_start_dev_environment",
            description="Start a stopped live DevEnvironment.",
            input_schema=_dev_environment_action_schema(),
            handler=server._tool_start_dev_environment,
        ),
        ToolDefinition(
            name="smr_stop_dev_environment",
            description="Stop a live DevEnvironment and record whether substrate state is retained or destroyed.",
            input_schema=tool_schema(
                {
                    **_DEV_ENVIRONMENT_ID,
                    "decision": {
                        "type": "string",
                        "enum": ["retain", "destroy"],
                        "description": "Stop retention decision.",
                    },
                    **_METADATA,
                },
                required=["dev_environment_id"],
            ),
            handler=server._tool_stop_dev_environment,
        ),
        ToolDefinition(
            name="smr_snapshot_dev_environment",
            description="Record a snapshot marker for a live DevEnvironment.",
            input_schema=_dev_environment_action_schema(),
            handler=server._tool_snapshot_dev_environment,
        ),
        ToolDefinition(
            name="smr_report_dev_environment_materialization",
            description="Report substrate materialization state, service health, logs, and receipts for a live DevEnvironment.",
            input_schema=tool_schema(
                {
                    **_DEV_ENVIRONMENT_ID,
                    "result": {
                        "type": "string",
                        "enum": ["succeeded", "failed", "degraded"],
                        "description": "Materializer result.",
                    },
                    "lifecycle_state": {
                        "type": "string",
                        "enum": ["deployed", "running", "stopped"],
                        "description": "Materialized lifecycle state for successful reports.",
                    },
                    "service_summary": {
                        "type": "object",
                        "description": "Service health/endpoints reported by the substrate.",
                    },
                    "log_entries": {
                        "type": "array",
                        "description": "Materializer log entries.",
                        "items": {"type": "object"},
                    },
                    "receipt_refs": {
                        "type": "array",
                        "description": "Materializer receipt references.",
                        "items": {"type": "object"},
                    },
                    **_METADATA,
                    "error": {
                        "type": "object",
                        "description": "Typed materialization error payload.",
                    },
                },
                required=["dev_environment_id"],
            ),
            handler=server._tool_report_dev_environment_materialization,
        ),
        ToolDefinition(
            name="smr_destroy_dev_environment",
            description="Soft-delete a live DevEnvironment.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_destroy_dev_environment,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_services",
            description="Read live DevEnvironment service status projection.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_get_dev_environment_services,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_attach",
            description="Read operator attach projection for a live DevEnvironment.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_get_dev_environment_attach,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_logs",
            description="Read live DevEnvironment log projection.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_get_dev_environment_logs,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_runs",
            description="Read runs bound to a live DevEnvironment.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_get_dev_environment_runs,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_usage",
            description="Read canonical usage facts scoped to a live DevEnvironment.",
            input_schema=tool_schema(
                {
                    **_DEV_ENVIRONMENT_ID,
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum recent usage-fact count.",
                    },
                },
                required=["dev_environment_id"],
            ),
            handler=server._tool_get_dev_environment_usage,
        ),
        ToolDefinition(
            name="smr_preflight_dev_environment_billing",
            description="Preflight configured DevEnvironment uptime billing admission.",
            input_schema=tool_schema(
                {
                    **_DEV_ENVIRONMENT_ID,
                    "model_class": {
                        "type": "string",
                        "description": "Billing model class. Defaults to value.",
                    },
                    "estimated_customer_debit_microcents": {
                        "type": "integer",
                        "description": "Estimated customer debit in microcents.",
                    },
                },
                required=["dev_environment_id"],
            ),
            handler=server._tool_preflight_dev_environment_billing,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_billing_drawdown",
            description="Read billing debit drawdown scoped to a live DevEnvironment.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_get_dev_environment_billing_drawdown,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_receipts",
            description="Read source refs and bounded hydrated evidence for runs bound to a live DevEnvironment.",
            input_schema=tool_schema(_DEV_ENVIRONMENT_ID, required=["dev_environment_id"]),
            handler=server._tool_get_dev_environment_receipts,
        ),
        ToolDefinition(
            name="smr_get_dev_environment_evidence",
            description="Read a composed DevEnvironment proof snapshot over existing owner routes.",
            input_schema=tool_schema(
                {
                    **_DEV_ENVIRONMENT_ID,
                    "usage_limit": {
                        "type": "integer",
                        "description": "Optional maximum recent usage-fact count.",
                    },
                    "include_preflight": {
                        "type": "boolean",
                        "description": "Whether to include DevEnvironment preflight.",
                    },
                    "include_logs": {
                        "type": "boolean",
                        "description": "Whether to include bounded DevEnvironment logs.",
                    },
                    "include_billing": {
                        "type": "boolean",
                        "description": "Whether to include billing preflight and drawdown.",
                    },
                },
                required=["dev_environment_id"],
            ),
            handler=server._tool_get_dev_environment_evidence,
        ),
    ]


__all__ = ["build_dev_environment_tools"]
