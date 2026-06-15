"""Progress-oriented MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema
from synth_ai.managed_research.mcp.tools.smr_policy_schemas import run_policy_input_schema
from synth_ai.managed_research.models.smr_actor_models import (
    SMR_ACTOR_SUBTYPE_VALUES,
    SMR_ACTOR_TYPE_VALUES,
)
from synth_ai.managed_research.models.smr_agent_kinds import SMR_AGENT_KIND_VALUES
from synth_ai.managed_research.models.smr_agent_models import SMR_AGENT_MODEL_VALUES
from synth_ai.managed_research.models.smr_horizons import SMR_INTENDED_HORIZON_HOURS_VALUES
from synth_ai.managed_research.models.smr_host_kinds import SMR_HOST_KIND_VALUES
from synth_ai.managed_research.models.smr_providers import PROVIDER_VALUES
from synth_ai.managed_research.models.smr_work_modes import SMR_WORK_MODE_VALUES


def _actor_model_assignment_schema(*, field_label: str) -> dict[str, Any]:
    return {
        "type": "array",
        "description": field_label,
        "items": {
            "type": "object",
            "properties": {
                "actor_type": {
                    "type": "string",
                    "enum": list(SMR_ACTOR_TYPE_VALUES),
                },
                "actor_subtype": {
                    "type": "string",
                    "enum": list(SMR_ACTOR_SUBTYPE_VALUES),
                },
                "agent_model": {
                    "type": "string",
                    "enum": list(SMR_AGENT_MODEL_VALUES),
                },
                "agent_model_params": {
                    "type": "object",
                },
            },
            "required": ["actor_type", "actor_subtype", "agent_model"],
        },
    }


def _usage_limit_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Optional run-scoped usage limit. Do not include provider secrets.",
        "properties": {
            "max_spend_usd": {"type": "number"},
            "max_wallclock_seconds": {"type": "integer"},
            "max_gpu_hours": {"type": "number"},
            "max_tokens": {"type": "integer"},
        },
    }


def _provider_bindings_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "description": (
            "Run-scoped provider bindings. Use resource_bindings only for external "
            "repos and credential refs."
        ),
        "items": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": list(PROVIDER_VALUES),
                    "description": "Authenticated API launch provider.",
                },
                "config": {
                    "type": "object",
                    "description": "Provider-specific non-secret launch config.",
                },
                "limit": _usage_limit_schema(),
            },
            "required": ["provider"],
        },
    }


def _objective_launch_properties() -> dict[str, Any]:
    return {
        "objective": {
            "type": "string",
            "description": (
                "Optional run objective. The SDK enqueues it as kickoff text and "
                "requires a final report unless require_report is false or a "
                "kickoff_contract is provided."
            ),
        },
        "open_ended_question": {
            "type": "object",
            "description": "Optional inline open-ended-question parent for this run.",
        },
        "directed_effort_outcome": {
            "type": "object",
            "description": "Optional inline directed-effort-outcome parent for this run.",
        },
        "required_work_products": {
            "type": "array",
            "description": "Optional required work products for the generated kickoff contract.",
            "items": {"type": "object"},
        },
        "require_report": {
            "type": "boolean",
            "description": "When objective is provided, require a default final report work product.",
        },
    }


def build_progress_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_get_project_setup",
            description=(
                "Fetch the canonical project setup authority for a managed research project."
            ),
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_get_project_setup,
        ),
        ToolDefinition(
            name="smr_prepare_project_setup",
            description=(
                "Run the explicit setup-authority preparation step before launch "
                "preflight and trigger."
            ),
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_prepare_project_setup,
        ),
        ToolDefinition(
            name="smr_get_launch_preflight",
            description=("Fetch the canonical launch preflight for a concrete run request."),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "host_kind": {
                        "type": "string",
                        "enum": list(SMR_HOST_KIND_VALUES),
                        "description": "Execution substrate for the launch.",
                    },
                    "work_mode": {
                        "type": "string",
                        "enum": list(SMR_WORK_MODE_VALUES),
                        "description": "Run work mode.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": list(SMR_WORK_MODE_VALUES),
                        "description": "Product work mode alias for work_mode.",
                    },
                    "intended_horizon_hours": {
                        "type": "integer",
                        "enum": list(SMR_INTENDED_HORIZON_HOURS_VALUES),
                        "description": "Customer-facing intended horizon. Allowed values: 1, 4, 8, 24, or 168.",
                    },
                    "providers": _provider_bindings_schema(),
                    "limit": _usage_limit_schema(),
                    **_objective_launch_properties(),
                    "worker_pool_id": {
                        "type": "string",
                        "description": "Optional worker pool override.",
                    },
                    "timebox_seconds": {"type": "integer", "description": "Optional run timebox."},
                    "agent_profile": {
                        "type": "string",
                        "description": "Optional shared agent profile override.",
                    },
                    "agent_model": {
                        "type": "string",
                        "enum": list(SMR_AGENT_MODEL_VALUES),
                        "description": "Optional shared agent model override.",
                    },
                    "agent_harness": {
                        "type": "string",
                        "enum": list(SMR_AGENT_KIND_VALUES),
                        "description": "Optional shared agent harness override.",
                    },
                    "agent_kind": {
                        "type": "string",
                        "enum": list(SMR_AGENT_KIND_VALUES),
                        "description": "Compatibility alias for agent_harness.",
                    },
                    "agent_model_params": {
                        "type": "object",
                        "description": "Optional model params override.",
                    },
                    "actor_model_overrides": _actor_model_assignment_schema(
                        field_label="Optional actor-scoped model overrides."
                    ),
                    "initial_runtime_messages": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional kickoff runtime messages.",
                    },
                    "workflow": {"type": "object", "description": "Optional workflow payload."},
                    "sandbox_override": {
                        "type": "object",
                        "description": "Optional sandbox override.",
                    },
                    "environment": {
                        "type": "object",
                        "description": "Optional Environment v1 reference, e.g. {'schema_version': '2026-05-14-environment-v1', 'name': 'symbolic-craftax-py311'}.",
                    },
                    "local_execution": {
                        "type": "object",
                        "description": "Explicit synth-dev local lane identity for slot-backed launches.",
                    },
                    "execution_profile": {
                        "type": "object",
                        "description": "Explicit product execution profile for local docker/daytona launches.",
                    },
                    "run_policy": run_policy_input_schema(),
                    "kickoff_contract": {
                        "type": "object",
                        "description": "Optional staged-run kickoff contract.",
                    },
                    "resource_bindings": {
                        "type": "object",
                        "description": "Optional Phase 3 run resource bindings for external repos and credential refs.",
                    },
                    "ai_cache": {
                        "type": "object",
                        "description": "Optional run-scoped local AI-cache request with mode and proxy base_url.",
                    },
                    "primary_objective_id": {
                        "type": "string",
                        "description": "Optional existing project objective id to bind as this run's primary objective.",
                    },
                    "primary_objective_kind": {
                        "type": "string",
                        "enum": ["open_ended_question", "directed_effort_outcome"],
                        "description": "Optional discriminator for primary_objective_id.",
                    },
                    "primary_parent_ref": {
                        "type": "object",
                        "description": "Compatibility object for existing project-scoped parent objective binding.",
                    },
                    "primary_parent": {
                        "type": "object",
                        "description": "Optional inline run-scoped parent objective creation payload.",
                    },
                    "idempotency_key_run_create": {
                        "type": "string",
                        "description": "Optional idempotency key.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Deprecated compatibility alias.",
                    },
                },
                required=["project_id"],
            ),
            handler=server._tool_get_launch_preflight,
        ),
    ]


__all__ = ["build_progress_tools"]
