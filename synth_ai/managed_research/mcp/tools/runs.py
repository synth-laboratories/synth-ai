"""Run-related MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.objective_tools import RunObjectiveScopeToolOperation
from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema
from synth_ai.managed_research.mcp.tools.smr_policy_schemas import run_policy_input_schema
from synth_ai.managed_research.models.run_control import ManagedResearchActorControlAction
from synth_ai.managed_research.models.runtime_intent import (
    RuntimeIntentKind,
    RuntimeIntentStatus,
    RuntimeMessageMode,
)
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


def _horizon_launch_properties() -> dict[str, Any]:
    return {
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
    }


def build_run_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_start_run",
            description=(
                "Start a Managed Research run with product launch fields. Prefer "
                "mode and intended_horizon_hours; backend runbooks remain internal."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    **_horizon_launch_properties(),
                    "work_mode": {
                        "type": "string",
                        "enum": list(SMR_WORK_MODE_VALUES),
                        "description": "Compatibility alias for mode.",
                    },
                    "objective": {
                        "type": "string",
                        "description": "Optional kickoff objective text; sent as an initial queued message.",
                    },
                    "initial_runtime_messages": {
                        "type": "array",
                        "description": "Optional kickoff runtime messages to enqueue durably before the run starts.",
                        "items": {"type": "object"},
                    },
                    "host_kind": {
                        "type": "string",
                        "enum": list(SMR_HOST_KIND_VALUES),
                        "description": "Authenticated API execution host kind for this run.",
                    },
                    "providers": _provider_bindings_schema(),
                    "limit": _usage_limit_schema(),
                    "timebox_seconds": {
                        "type": "integer",
                        "description": "Compatibility hard timebox. Prefer intended_horizon_hours.",
                    },
                },
                "required": ["project_id"],
                "additionalProperties": True,
            },
            handler=server._tool_start_run,
        ),
        ToolDefinition(
            name="smr_trigger_run",
            description=(
                "Trigger a managed research run. Follow the canonical setup -> "
                "launch-preflight -> trigger flow, and always branch on "
                "result.error in MCP clients."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "host_kind": {
                        "type": "string",
                        "enum": list(SMR_HOST_KIND_VALUES),
                        "description": "Authenticated API execution host kind for this run.",
                    },
                    "work_mode": {
                        "type": "string",
                        "enum": list(SMR_WORK_MODE_VALUES),
                        "description": "Run work mode.",
                    },
                    **_horizon_launch_properties(),
                    "providers": _provider_bindings_schema(),
                    "limit": _usage_limit_schema(),
                    "worker_pool_id": {
                        "type": "string",
                        "description": "Optional worker pool override.",
                    },
                    "timebox_seconds": {"type": "integer", "description": "Optional run timebox."},
                    "agent_profile": {
                        "type": "string",
                        "description": "Optional agent profile override. Prefer this when you want an exact backend-managed preset.",
                    },
                    "agent_model": {
                        "type": "string",
                        "enum": list(SMR_AGENT_MODEL_VALUES),
                        "description": "Optional run-level agent model override using a backend catalog model id such as gpt-5.4, gpt-5.4-nano, or gpt-oss-120b.",
                    },
                    "agent_harness": {
                        "type": "string",
                        "enum": list(SMR_AGENT_KIND_VALUES),
                        "description": (
                            "Optional run-level agent harness override. "
                            "Supported values are codex and opencode_sdk."
                        ),
                    },
                    "agent_kind": {
                        "type": "string",
                        "enum": list(SMR_AGENT_KIND_VALUES),
                        "description": "Compatibility alias for agent_harness.",
                    },
                    "agent_model_params": {
                        "type": "object",
                        "description": "Optional agent model params override (for example reasoning_effort).",
                    },
                    "actor_model_overrides": _actor_model_assignment_schema(
                        field_label=(
                            "Optional actor-scoped model overrides keyed by actor_type "
                            "and actor_subtype."
                        )
                    ),
                    "initial_runtime_messages": {
                        "type": "array",
                        "description": "Optional kickoff runtime messages to enqueue durably before the run starts. Use this instead of the removed prompt field.",
                        "items": {"type": "object"},
                    },
                    "workflow": {"type": "object", "description": "Optional workflow override."},
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
                        "description": "Optional staged-run kickoff contract. This becomes the authoritative staged contract persisted on the run.",
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
                    "effort_id": {
                        "type": "string",
                        "description": "Optional Factory Effort ID to link this run to.",
                    },
                    "idempotency_key_run_create": {
                        "type": "string",
                        "description": "Optional idempotency key for the launch request.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Deprecated compatibility alias for idempotency_key_run_create.",
                    },
                },
                required=["project_id"],
            ),
            handler=server._tool_trigger_run,
        ),
        ToolDefinition(
            name="smr_start_one_off_run",
            description=(
                "Trigger a one-off managed research run using the caller's default "
                "Miscellaneous project."
            ),
            input_schema=tool_schema(
                {
                    "host_kind": {
                        "type": "string",
                        "enum": list(SMR_HOST_KIND_VALUES),
                        "description": "Public execution host kind for this run.",
                    },
                    "work_mode": {
                        "type": "string",
                        "enum": list(SMR_WORK_MODE_VALUES),
                        "description": "Run work mode.",
                    },
                    **_horizon_launch_properties(),
                    "providers": _provider_bindings_schema(),
                    "limit": _usage_limit_schema(),
                    "worker_pool_id": {
                        "type": "string",
                        "description": "Optional worker pool override.",
                    },
                    "timebox_seconds": {"type": "integer", "description": "Optional run timebox."},
                    "agent_profile": {
                        "type": "string",
                        "description": "Optional agent profile override.",
                    },
                    "agent_model": {
                        "type": "string",
                        "enum": list(SMR_AGENT_MODEL_VALUES),
                        "description": "Optional run-level agent model override.",
                    },
                    "agent_harness": {
                        "type": "string",
                        "enum": list(SMR_AGENT_KIND_VALUES),
                        "description": "Optional run-level agent harness override.",
                    },
                    "agent_kind": {
                        "type": "string",
                        "enum": list(SMR_AGENT_KIND_VALUES),
                        "description": "Compatibility alias for agent_harness.",
                    },
                    "agent_model_params": {
                        "type": "object",
                        "description": "Optional agent model params override.",
                    },
                    "actor_model_overrides": _actor_model_assignment_schema(
                        field_label="Optional actor-scoped model overrides."
                    ),
                    "initial_runtime_messages": {
                        "type": "array",
                        "description": "Optional kickoff runtime messages to enqueue durably before the run starts.",
                        "items": {"type": "object"},
                    },
                    "workflow": {"type": "object", "description": "Optional workflow override."},
                    "sandbox_override": {
                        "type": "object",
                        "description": "Optional sandbox override.",
                    },
                    "environment": {
                        "type": "object",
                        "description": "Optional Environment v1 reference.",
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
                    "effort_id": {
                        "type": "string",
                        "description": "Optional Factory Effort ID to link this run to.",
                    },
                    "idempotency_key_run_create": {
                        "type": "string",
                        "description": "Optional idempotency key for the launch request.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Deprecated compatibility alias for idempotency_key_run_create.",
                    },
                },
                required=[],
            ),
            handler=server._tool_start_one_off_run,
        ),
        ToolDefinition(
            name="smr_list_runs",
            description="List runs for a project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "active_only": {"type": "boolean", "description": "Only return active runs."},
                    "public_state": {
                        "type": "string",
                        "description": "Optional public run-state filter.",
                    },
                    "limit": {"type": "integer", "description": "Maximum runs to return."},
                },
                required=["project_id"],
            ),
            handler=server._tool_list_runs,
        ),
        ToolDefinition(
            name="smr_get_run",
            description="Fetch a run by id.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run,
        ),
        ToolDefinition(
            name="smr_get_run_contract",
            description=(
                "Fetch the strict run_contract for a run. Use this for terminality, "
                "finalization, recovery, incident, artifact, and lifecycle-invariant status."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_get_run_contract,
        ),
        ToolDefinition(
            name="smr_get_run_execution",
            description=(
                "Read the high-level execution projection for a run: actors, "
                "tasks/objectives, participant messages, timeline events, and output refs. "
                "Use this for normal run inspection before falling back to raw "
                "timeline, transcript, or operator evidence."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "view": {
                        "type": "string",
                        "enum": ["summary", "detail"],
                        "description": "Projection detail level. Defaults to summary.",
                    },
                    "event_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "actor_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                    },
                    "task_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 250,
                    },
                    "message_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                    },
                    "work_product_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_get_run_execution,
        ),
        ToolDefinition(
            name="smr_list_run_task_events",
            description=(
                "List backend-owned task lifecycle events for a run. Use this for "
                "normal task inspection before falling back to raw timelines or "
                "operator evidence."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "cursor": {"type": "string", "description": "Optional pagination cursor."},
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_list_run_task_events,
        ),
        ToolDefinition(
            name="smr_list_run_objective_events",
            description=(
                "List backend-owned objective lifecycle events for a run. Use this "
                "for OEQ/DEO progress and review inspection before raw evidence."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "cursor": {"type": "string", "description": "Optional pagination cursor."},
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_list_run_objective_events,
        ),
        ToolDefinition(
            name="smr_get_run_work_graph",
            description=(
                "Fetch the run work graph bundle: execution summary, task events, "
                "and objective events."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_get_run_work_graph,
        ),
        ToolDefinition(
            name="smr_list_tasks",
            description="List task views for a run or objective.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id for run task events."},
                    "objective_id": {
                        "type": "string",
                        "description": "Objective id for objective task links.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["open_ended_question", "directed_effort_outcome"],
                        "description": "Optional objective kind discriminator.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                required=["project_id"],
            ),
            handler=server._tool_list_tasks,
        ),
        ToolDefinition(
            name="smr_create_task",
            description="Create or plan a task through the product task wrapper.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "payload": {"type": "object", "description": "Task payload."},
                    "mode": {"type": "string", "description": "Runtime intent mode."},
                    "body": {"type": "string", "description": "Optional operator-facing body."},
                },
                required=["project_id", "run_id", "payload"],
            ),
            handler=server._tool_create_task,
        ),
        ToolDefinition(
            name="smr_update_task",
            description="Update a task through the product task wrapper.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "task_id": {"type": "string", "description": "Task id."},
                    "payload": {"type": "object", "description": "Task patch payload."},
                    "mode": {"type": "string", "description": "Runtime intent mode."},
                    "body": {"type": "string", "description": "Optional operator-facing body."},
                },
                required=["project_id", "run_id", "task_id", "payload"],
            ),
            handler=server._tool_update_task,
        ),
        ToolDefinition(
            name="smr_cancel_task",
            description="Stop a task through the product task wrapper.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "task_id": {"type": "string", "description": "Task id."},
                    "reason": {"type": "string", "description": "Optional stop reason."},
                    "mode": {"type": "string", "description": "Runtime intent mode."},
                    "body": {"type": "string", "description": "Optional operator-facing body."},
                },
                required=["project_id", "run_id", "task_id"],
            ),
            handler=server._tool_cancel_task,
        ),
        ToolDefinition(
            name="smr_reassign_task",
            description="Reassign a task through the product task wrapper.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "task_id": {"type": "string", "description": "Task id."},
                    "assignee": {"type": "string", "description": "New assignee."},
                    "mode": {"type": "string", "description": "Runtime intent mode."},
                    "body": {"type": "string", "description": "Optional operator-facing body."},
                },
                required=["project_id", "run_id", "task_id", "assignee"],
            ),
            handler=server._tool_reassign_task,
        ),
        ToolDefinition(
            name="smr_get_run_logical_timeline",
            description=(
                "Read the operator-facing logical timeline for a run. "
                "Use this for actors, checkpoints, branch provenance, and queue "
                "chronology instead of the low-level runtime timeline."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_get_run_logical_timeline,
        ),
        ToolDefinition(
            name="smr_get_run_event_log",
            description=(
                "Read the typed run event log for a project-scoped run, with optional "
                "source, kind, status, and limit filters."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional event source filters.",
                    },
                    "event_kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional event kind filters.",
                    },
                    "statuses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional event status filters.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                    },
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_get_run_event_log,
        ),
        ToolDefinition(
            name="smr_get_run_authority_readouts",
            description=(
                "Read backend-owned authority/readout projections for a run. "
                "Set include_runtime_authority only when privileged runtime-authority detail is needed."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "include_runtime_authority": {
                        "type": "boolean",
                        "description": "Include privileged runtime-authority detail when allowed.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_authority_readouts,
        ),
        ToolDefinition(
            name="smr_get_run_operator_evidence",
            description=(
                "Read the bundled operator evidence for a project-scoped run, "
                "including runtime/logical timelines, transcript slices, and reconciliation evidence."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "runtime_timeline_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "logical_timeline_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "transcript_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "reconciliation_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_get_run_operator_evidence,
        ),
        ToolDefinition(
            name="smr_get_run_traces",
            description=(
                "Read persisted run traces for a run. "
                "Use this to inspect or download session-backed Codex traces and other persisted operator-facing trace artifacts."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_traces,
        ),
        ToolDefinition(
            name="smr_get_run_actor_trace",
            description=(
                "Read privileged actor-scoped trace activity for one run actor. "
                "Returns persisted transcript events, optional live transcript events "
                "after live_cursor, and completed raw-session trace artifact references "
                "when available."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                    "actor_key": {
                        "type": "string",
                        "description": "Stable actor key, for example orchestrator-1.",
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Persisted transcript pagination cursor.",
                    },
                    "live_cursor": {
                        "type": "string",
                        "description": "Redis live cursor returned by a previous response.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum transcript events to inspect per source.",
                    },
                    "include_live": {
                        "type": "boolean",
                        "description": "Whether to include live events after live_cursor.",
                    },
                    "include_traces": {
                        "type": "boolean",
                        "description": "Whether to include completed raw trace artifact references.",
                    },
                },
                required=["project_id", "run_id", "actor_key"],
            ),
            handler=server._tool_get_run_actor_trace,
        ),
        ToolDefinition(
            name="smr_list_run_actor_traces",
            description=(
                "List privileged actor trace subjects for a run, or list raw trace "
                "artifacts for one actor when actor_key is supplied. This is the "
                "entrypoint for browsing actor-level debug traces."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                    "actor_key": {
                        "type": "string",
                        "description": "Optional actor key to return only that actor's raw trace artifact refs.",
                    },
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_list_run_actor_traces,
        ),
        ToolDefinition(
            name="smr_get_raw_trace_events",
            description=(
                "Page raw trace events for a privileged run trace artifact. Defaults "
                "to summary redaction; use safe for short excerpts or raw only when "
                "explicitly needed for internal debugging."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                    "artifact_id": {
                        "type": "string",
                        "description": "raw_session_events artifact id.",
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Pagination cursor returned by the previous page.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum raw events to return.",
                    },
                    "redaction_mode": {
                        "type": "string",
                        "enum": ["summary", "safe", "raw"],
                        "description": "Raw event redaction mode.",
                    },
                    "reconstruct": {
                        "type": "boolean",
                        "description": "Attach normalized input/output/tool/shell reconstruction hints.",
                    },
                    "category": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "Optional category filter, string or list.",
                    },
                    "method": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "Optional method/tool filter, string or list.",
                    },
                },
                required=["project_id", "run_id", "artifact_id"],
            ),
            handler=server._tool_get_raw_trace_events,
        ),
        ToolDefinition(
            name="smr_download_raw_trace",
            description=(
                "Create a short-lived privileged raw trace download URL, or download "
                "it to destination. Requires confirm_raw_download=true so large or "
                "sensitive blobs are not fetched accidentally."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                    "artifact_id": {
                        "type": "string",
                        "description": "raw_session_events artifact id.",
                    },
                    "confirm_raw_download": {
                        "type": "boolean",
                        "description": "Must be true to create or fetch a raw download.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Optional local destination path. Omit to return only a presigned URL.",
                    },
                    "expires_in": {
                        "type": "integer",
                        "description": "URL TTL in seconds, between 60 and 3600.",
                    },
                },
                required=[
                    "project_id",
                    "run_id",
                    "artifact_id",
                    "confirm_raw_download",
                ],
            ),
            handler=server._tool_download_raw_trace,
        ),
        ToolDefinition(
            name="smr_get_run_actor_usage",
            description=(
                "Read actor-centric usage for a run. "
                "Use this for truthful per-actor spend, provider, and model activity rather than guessing from worker config."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_actor_usage,
        ),
        ToolDefinition(
            name="smr_control_project_run_actor",
            description=(
                "Pause or resume one actor inside a project-scoped run. "
                "This is operator control, not project-truth promotion."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                    "actor_id": {"type": "string", "description": "Actor id to control."},
                    "action": {
                        "type": "string",
                        "enum": [item.value for item in ManagedResearchActorControlAction],
                        "description": "Actor control action.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional operator reason recorded in audit metadata.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Optional idempotency key for replay correlation.",
                    },
                },
                required=["project_id", "run_id", "actor_id", "action"],
            ),
            handler=server._tool_control_project_run_actor,
        ),
        ToolDefinition(
            name="smr_list_run_participants",
            description=(
                "List participant sessions for a run from actor/session records, including whether usage recording is present or missing."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_participants,
        ),
        ToolDefinition(
            name="smr_get_run_artifact_progress",
            description="Read live required/optional artifact progress for a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_artifact_progress,
        ),
        ToolDefinition(
            name="smr_list_run_actor_logs",
            description="List redacted exec stdout/stderr actor log events for a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "actor_id": {"type": "string", "description": "Optional actor id filter."},
                    "turn_id": {"type": "string", "description": "Optional turn id filter."},
                    "kind": {
                        "type": "string",
                        "description": "Optional kind filter: exec.stdout, exec.stderr, stdout, or stderr.",
                    },
                    "since": {"type": "string", "description": "Optional ISO-8601 lower bound."},
                    "cursor": {"type": "string", "description": "Optional pagination cursor."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_actor_logs,
        ),
        ToolDefinition(
            name="smr_get_run_primary_parent",
            description="Fetch the bound primary parent objective for a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_primary_parent,
        ),
        ToolDefinition(
            name="smr_run_objective_scopes",
            description=(
                "List or register which project objectives are in scope, primary, "
                "supporting, reviewer, blocker, or out of scope for a run."
            ),
            input_schema=tool_schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": [operation.value for operation in RunObjectiveScopeToolOperation],
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                    "payload": {
                        "type": "object",
                        "description": "Scope registration payload.",
                    },
                },
                required=["operation", "run_id"],
            ),
            handler=server._tool_run_objective_scopes,
        ),
        ToolDefinition(
            name="smr_stop_run",
            description=(
                "Stop a queued or running run. Response includes "
                "`control_intent_id` and `control_intent_ack_at` so a replay "
                "of the same control correlates with the original intent."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_stop_run,
        ),
        ToolDefinition(
            name="smr_pause_run",
            description=(
                "Pause a live run without stopping it. Response includes "
                "`control_intent_id` and `control_intent_ack_at` for "
                "idempotent replay correlation."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_pause_run,
        ),
        ToolDefinition(
            name="smr_resume_run",
            description=(
                "Resume a paused run. Response includes `control_intent_id` "
                "and `control_intent_ack_at` for idempotent replay correlation."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_resume_run,
        ),
        ToolDefinition(
            name="smr_branch_run_from_checkpoint",
            description=(
                "Create a child run from a checkpoint. "
                "Use mode=exact for a pure branch and mode=with_message to seed the child with one bootstrap message."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement. Requires run_id when present.",
                    },
                    "run_id": {"type": "string", "description": "Optional run id scope."},
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Checkpoint id reference.",
                    },
                    "checkpoint_record_id": {
                        "type": "string",
                        "description": "Checkpoint record id reference.",
                    },
                    "checkpoint_uri": {
                        "type": "string",
                        "description": "Checkpoint URI reference.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["exact", "with_message"],
                        "description": "Whether to create an exact branch or bootstrap the child with a new message.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Required when mode=with_message.",
                    },
                    "reason": {"type": "string", "description": "Optional operator reason."},
                    "title": {"type": "string", "description": "Optional branch label."},
                    "source_node_id": {
                        "type": "string",
                        "description": "Optional logical timeline node provenance.",
                    },
                },
                required=[],
            ),
            handler=server._tool_branch_run_from_checkpoint,
        ),
        ToolDefinition(
            name="smr_runtime_message_queue",
            description=(
                "List or enqueue durable runtime messages for a run. "
                "Use operation=list for inspection and operation=enqueue for live operator steering. "
                "Do not use this tool to branch from checkpoints."
            ),
            input_schema=tool_schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": ["list", "enqueue"],
                        "description": "List the queue or enqueue a new runtime message.",
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                    "status": {
                        "type": "string",
                        "description": "Optional message status filter when listing.",
                    },
                    "viewer_role": {
                        "type": "string",
                        "description": "Optional viewer role filter when listing.",
                    },
                    "viewer_target": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "Optional viewer target filter when listing.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "Maximum messages to return when listing.",
                    },
                    "topic": {"type": "string", "description": "Optional runtime topic."},
                    "causation_id": {
                        "type": "string",
                        "description": "Optional causation id for correlation.",
                    },
                    "mode": {"type": "string", "description": "Optional message mode."},
                    "spawn_policy": {
                        "type": "string",
                        "enum": ["live_only", "request_template"],
                        "description": "Optional spawn policy when enqueueing.",
                    },
                    "sender": {"type": "string", "description": "Optional sender identity."},
                    "target": {
                        "type": "string",
                        "description": "Optional runtime target when enqueueing.",
                    },
                    "participant_session_id": {
                        "type": "string",
                        "description": "Optional participant session target when enqueueing.",
                    },
                    "action": {"type": "string", "description": "Optional action label."},
                    "body": {"type": "string", "description": "Optional message body text."},
                    "payload": {
                        "type": "object",
                        "description": "Optional JSON payload when enqueueing.",
                    },
                },
                required=["operation", "run_id"],
            ),
            handler=server._tool_runtime_message_queue,
        ),
        ToolDefinition(
            name="smr_list_messages",
            description="List product-level message queue messages for a run.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "thread_id": {"type": "string", "description": "Optional thread filter."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_list_messages,
        ),
        ToolDefinition(
            name="smr_send_message",
            description="Send a product-level message queue message to a run.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "body": {"type": "string", "description": "Message body."},
                    "intent": {
                        "type": "string",
                        "enum": ["queue", "steer", "interrupt", "note"],
                        "description": "Message intent. Defaults to queue.",
                    },
                    "audience": {"type": "object", "description": "Optional audience selector."},
                    "payload": {"type": "object", "description": "Optional message payload."},
                    "message_kind": {"type": "string", "description": "Optional message kind."},
                    "thread_id": {"type": "string", "description": "Optional thread id."},
                    "parent_message_id": {
                        "type": "string",
                        "description": "Optional parent message id.",
                    },
                    "fallback_policy": {
                        "type": "string",
                        "description": "Optional backend fallback policy.",
                    },
                    "idempotency_key": {"type": "string"},
                    "correlation_id": {"type": "string"},
                    "causation_id": {"type": "string"},
                },
                required=["project_id", "run_id"],
            ),
            handler=server._tool_send_message,
        ),
        ToolDefinition(
            name="smr_edit_message",
            description="Edit a product-level message queue message for a run.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "message_id": {"type": "string", "description": "Message id."},
                    "body": {"type": "string", "description": "Replacement body."},
                    "payload": {"type": "object", "description": "Replacement payload."},
                },
                required=["project_id", "run_id", "message_id"],
            ),
            handler=server._tool_edit_message,
        ),
        ToolDefinition(
            name="smr_retract_message",
            description="Retract a product-level message queue message for a run.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {"type": "string", "description": "Run id."},
                    "message_id": {"type": "string", "description": "Message id."},
                },
                required=["project_id", "run_id", "message_id"],
            ),
            handler=server._tool_retract_message,
        ),
        ToolDefinition(
            name="smr_runtime_intents",
            description=(
                "Submit, list, or get typed runtime intents for operator steering. "
                "Use this for approvals, questions, task/run state changes, spend records, "
                "task plans, and milestone writes when you want durable ack/resolution state."
            ),
            input_schema=tool_schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": ["submit", "list", "get"],
                        "description": "Submit a new intent, list intents, or fetch one intent.",
                    },
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "runtime_intent_id": {
                        "type": "string",
                        "description": "Required for operation=get.",
                    },
                    "status": {
                        "type": "string",
                        "enum": [item.value for item in RuntimeIntentStatus],
                        "description": "Optional status filter for operation=list.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "Maximum intents to return for operation=list.",
                    },
                    "intent": {
                        "type": "object",
                        "description": "Required for operation=submit. Shape: {kind, payload}.",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": [item.value for item in RuntimeIntentKind],
                            },
                            "payload": {"type": "object"},
                        },
                        "required": ["kind", "payload"],
                    },
                    "mode": {
                        "type": "string",
                        "enum": [item.value for item in RuntimeMessageMode],
                        "description": "Intent delivery mode for operation=submit.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional operator-facing body for operation=submit.",
                    },
                    "causation_id": {
                        "type": "string",
                        "description": "Optional source message id for operation=submit.",
                    },
                },
                required=["operation", "run_id"],
            ),
            handler=server._tool_runtime_intents,
        ),
        ToolDefinition(
            name="smr_list_active_runs",
            description="List active runs for a project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_list_active_runs,
        ),
        ToolDefinition(
            name="smr_get_run_transcript",
            description=(
                "Fetch transcript events for a run — what the agent said, did, and "
                "produced, in chronological order. Returns one page of events with "
                "next_cursor for paging and live_resume_cursor for live polling. "
                "Call repeatedly with cursor=next_cursor to page forward. For live "
                "runs, pass cursor=live_resume_cursor on each poll to receive only "
                "new events since the last call."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "cursor": {
                        "type": "string",
                        "description": "Pagination cursor. Omit for the first page.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Events per page (max 200, default 100).",
                    },
                    "participant_session_id": {
                        "type": "string",
                        "description": "Optional filter to a single participant session.",
                    },
                    "view": {
                        "type": "string",
                        "enum": ["operator", "debug", "public"],
                        "description": "Backend redaction view. Defaults to operator.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_get_run_transcript,
        ),
        ToolDefinition(
            name="smr_watch_run_events",
            description=(
                "Read a bounded batch from the live run SSE stream. Returns typed "
                "snapshot/transcript events, including backend-redacted reasoning "
                "summary and tool-call lifecycle events. Use max_events and "
                "timeout_seconds to keep MCP calls finite."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "transcript_cursor": {
                        "type": "string",
                        "description": "Optional live transcript cursor to resume from.",
                    },
                    "last_event_id": {
                        "type": "string",
                        "description": "Optional SSE Last-Event-ID resume token.",
                    },
                    "view": {
                        "type": "string",
                        "enum": ["operator", "debug", "public"],
                        "description": "Backend redaction view. Defaults to operator.",
                    },
                    "max_events": {
                        "type": "integer",
                        "description": "Maximum events to return, capped at 50.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Stream timeout for this bounded MCP call.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_watch_run_events,
        ),
        ToolDefinition(
            name="smr_list_run_questions",
            description="List pending or historical questions for a run.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "status_filter": {
                        "type": "string",
                        "description": "Optional question status filter.",
                    },
                    "limit": {"type": "integer", "description": "Maximum questions to return."},
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_questions,
        ),
        ToolDefinition(
            name="smr_create_run_checkpoint",
            description="Request a run checkpoint.",
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Optional checkpoint id override.",
                    },
                    "reason": {"type": "string", "description": "Optional checkpoint reason."},
                },
                required=["run_id"],
            ),
            handler=server._tool_create_run_checkpoint,
        ),
        ToolDefinition(
            name="smr_list_run_checkpoints",
            description=(
                "List checkpoints for a run, including restorable/branchable flags "
                "and any recoverable checkpoint quota failure details."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_list_run_checkpoints,
        ),
        ToolDefinition(
            name="smr_restore_run_checkpoint",
            description=(
                "Restore a run to a restorable checkpoint. "
                "Use smr_branch_run_from_checkpoint for child-run branching; "
                "checkpoint quota failures return a structured error with operator_action."
            ),
            input_schema=tool_schema(
                {
                    "run_id": {"type": "string", "description": "Run id."},
                    "project_id": {
                        "type": "string",
                        "description": "Optional project-scoped route enforcement.",
                    },
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Optional checkpoint id override.",
                    },
                    "checkpoint_record_id": {
                        "type": "string",
                        "description": "Optional checkpoint record id reference.",
                    },
                    "checkpoint_uri": {
                        "type": "string",
                        "description": "Optional checkpoint URI reference.",
                    },
                    "reason": {"type": "string", "description": "Optional restore reason."},
                    "mode": {
                        "type": "string",
                        "enum": ["in_place", "branch"],
                        "description": (
                            "Restore mode. Prefer in_place; branch is a compatibility alias."
                        ),
                    },
                },
                required=["run_id"],
            ),
            handler=server._tool_restore_run_checkpoint,
        ),
    ]


__all__ = ["build_run_tools"]
