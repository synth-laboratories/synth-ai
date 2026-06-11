"""Project-related MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.objective_tools import (
    CompatObjectiveToolOperation,
    ObjectiveToolOperation,
)
from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema
from synth_ai.managed_research.models.smr_actor_models import (
    SMR_ACTOR_SUBTYPE_VALUES,
    SMR_ACTOR_TYPE_VALUES,
)
from synth_ai.managed_research.models.smr_agent_models import SMR_AGENT_MODEL_VALUES
from synth_ai.managed_research.models.smr_credential_providers import (
    SMR_CREDENTIAL_PROVIDER_VALUES,
)
from synth_ai.managed_research.models.smr_environment_kinds import (
    SMR_ENVIRONMENT_KIND_VALUES,
)
from synth_ai.managed_research.models.smr_funding_sources import SMR_FUNDING_SOURCE_VALUES
from synth_ai.managed_research.models.smr_runtime_kinds import SMR_RUNTIME_KIND_VALUES


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


def build_project_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="smr_health_check",
            description="Return a connectivity and setup report for the managed-research MCP server.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Optional project id to verify access.",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=[],
            ),
            handler=server._tool_health_check,
        ),
        ToolDefinition(
            name="smr_create_runnable_project",
            description=(
                "Create a managed research project with the full runnable launch "
                "contract required for SDK, MCP, and eval flows."
            ),
            input_schema=tool_schema(
                {
                    "name": {"type": "string", "description": "Human-readable project name."},
                    "timezone": {
                        "type": "string",
                        "description": "Project timezone. Defaults to UTC when omitted.",
                    },
                    "pool_id": {
                        "type": "string",
                        "description": "Required execution pool id.",
                    },
                    "runtime_kind": {
                        "type": "string",
                        "enum": list(SMR_RUNTIME_KIND_VALUES),
                        "description": "Required project runtime kind.",
                    },
                    "environment_kind": {
                        "type": "string",
                        "enum": list(SMR_ENVIRONMENT_KIND_VALUES),
                        "description": "Required project environment kind.",
                    },
                    "orchestrator_profile_id": {
                        "type": "string",
                        "description": "Required shared orchestrator profile id.",
                    },
                    "default_worker_profile_id": {
                        "type": "string",
                        "description": "Required default worker profile id.",
                    },
                    "worker_profile_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional shared worker profile bindings.",
                    },
                    "actor_model_assignments": _actor_model_assignment_schema(
                        field_label=(
                            "Optional actor-scoped worker model overrides. "
                            "Use this instead of shared top-level model selection."
                        )
                    ),
                    "budgets": {
                        "type": "object",
                        "description": "Optional project budgets payload.",
                    },
                    "key_policy": {
                        "type": "object",
                        "description": "Optional project key-policy payload.",
                    },
                    "execution_policy": {
                        "type": "object",
                        "description": "Optional execution-policy payload.",
                    },
                    "scenario": {
                        "type": "string",
                        "description": "Optional project research scenario.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional project notes / launch context.",
                    },
                    "metered_infra": {
                        "type": "object",
                        "description": "Optional metered-infra metadata payload.",
                    },
                    "schedule": {
                        "type": "object",
                        "description": "Optional project schedule payload.",
                    },
                    "integrations": {
                        "type": "object",
                        "description": "Optional project integrations payload.",
                    },
                    "synth_ai": {
                        "type": "object",
                        "description": "Optional project synth_ai payload.",
                    },
                    "policy": {
                        "type": "object",
                        "description": "Optional project policy payload.",
                    },
                    "trial_matrix": {
                        "type": "object",
                        "description": "Optional project trial-matrix payload.",
                    },
                },
                required=[
                    "name",
                    "pool_id",
                    "runtime_kind",
                    "environment_kind",
                    "orchestrator_profile_id",
                    "default_worker_profile_id",
                ],
            ),
            handler=server._tool_create_runnable_project,
        ),
        ToolDefinition(
            name="smr_list_projects",
            description="List managed research projects.",
            input_schema=tool_schema(
                {
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived projects.",
                    },
                    "limit": {"type": "integer", "description": "Maximum projects to return."},
                },
                required=[],
            ),
            handler=server._tool_list_projects,
        ),
        ToolDefinition(
            name="smr_get_project",
            description="Fetch a managed research project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_get_project,
        ),
        ToolDefinition(
            name="smr_get_default_project",
            description=(
                "Fetch the authenticated user's default Miscellaneous managed research project."
            ),
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_get_default_project,
        ),
        ToolDefinition(
            name="smr_rename_project",
            description="Rename a managed research project without changing runs, repos, tasks, or artifacts.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "name": {"type": "string", "description": "New human-readable project name."},
                },
                required=["project_id", "name"],
            ),
            handler=server._tool_rename_project,
        ),
        ToolDefinition(
            name="smr_patch_project",
            description="Patch a managed research project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "config": {
                        "type": "object",
                        "description": "Partial project fields to update.",
                    },
                    "actor_model_assignments": _actor_model_assignment_schema(
                        field_label=(
                            "Optional durable actor-scoped model assignments stored under "
                            "execution.actor_model_assignments."
                        )
                    ),
                },
                required=["project_id", "config"],
            ),
            handler=server._tool_patch_project,
        ),
        ToolDefinition(
            name="smr_get_project_status",
            description="Fetch a polling-friendly status snapshot for a project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_get_project_status,
        ),
        ToolDefinition(
            name="smr_get_project_workspace",
            description=(
                "Fetch the backend-owned project workspace projection: objectives, "
                "runs, experiments, curated knowledge, review queue, reports, and "
                "launch risks. Runs propose material; review or policy promotion "
                "owns durable project truth."
            ),
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_get_project_workspace,
        ),
        ToolDefinition(
            name="smr_list_project_experiments",
            description="List first-class SMR project experiments, optionally scoped to a run.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["project_id"],
            ),
            handler=server._tool_list_project_experiments,
        ),
        ToolDefinition(
            name="smr_create_project_experiment",
            description=(
                "Create a first-class SMR experiment/trial for a candidate attempt, "
                "policy, prompt, verifier, or other comparable treatment."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "parent_experiment_id": {"type": "string"},
                    "milestone_id": {"type": "string"},
                    "title": {"type": "string"},
                    "kind": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "intervention": {"type": "string"},
                    "comparison": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "status": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                    "proposal_correlation_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "baseline_snapshot": {
                        "type": "object",
                        "description": "Structured baseline candidate/source summary.",
                    },
                    "candidate_snapshot": {
                        "type": "object",
                        "description": "Structured candidate code/artifact summary.",
                    },
                    "protocol_snapshot": {
                        "type": "object",
                        "description": "Structured metric, split, seed-set, scorer, and acceptance protocol.",
                    },
                    "result_summary": {
                        "type": "object",
                        "description": "Optional headline result summary, usually filled after measurement.",
                    },
                    "decision_summary": {
                        "type": "object",
                        "description": "Optional verdict/decision summary.",
                    },
                    "artifact_refs": {
                        "type": "object",
                        "description": "Reviewable artifact references for candidate source and result JSON.",
                    },
                    "metadata": {"type": "object"},
                },
                required=["project_id", "title", "hypothesis"],
            ),
            handler=server._tool_create_project_experiment,
        ),
        ToolDefinition(
            name="smr_get_project_experiment",
            description="Fetch one first-class SMR project experiment.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                },
                required=["project_id", "experiment_id"],
            ),
            handler=server._tool_get_project_experiment,
        ),
        ToolDefinition(
            name="smr_patch_project_experiment",
            description="Patch experiment lifecycle fields such as status, verdict, summary, or metadata.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "payload": {"type": "object"},
                },
                required=["project_id", "experiment_id", "payload"],
            ),
            handler=server._tool_patch_project_experiment,
        ),
        ToolDefinition(
            name="smr_link_project_experiment_run",
            description="Link a first-class experiment to an SMR run with a lineage role.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "role": {"type": "string"},
                    "notes": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "experiment_id", "run_id"],
            ),
            handler=server._tool_link_project_experiment_run,
        ),
        ToolDefinition(
            name="smr_list_project_experiment_runs",
            description="List SMR run links for one first-class experiment.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["project_id", "experiment_id"],
            ),
            handler=server._tool_list_project_experiment_runs,
        ),
        ToolDefinition(
            name="smr_attach_project_experiment_container_run",
            description=(
                "Attach a concrete container/taskset/scorer execution receipt to "
                "an SMR experiment."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "container_run_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "experiment_run_id": {"type": "string"},
                    "role": {"type": "string"},
                    "container_name": {"type": "string"},
                    "container_version": {"type": "string"},
                    "container_digest": {"type": "string"},
                    "image_ref": {"type": "string"},
                    "taskset_id": {"type": "string"},
                    "taskset_version": {"type": "string"},
                    "taskset_seed": {"type": "integer"},
                    "eval_profile_id": {"type": "string"},
                    "eval_profile_version": {"type": "string"},
                    "verifier_or_scorer_id": {"type": "string"},
                    "verifier_or_scorer_version": {"type": "string"},
                    "status": {"type": "string"},
                },
                required=["project_id", "experiment_id", "container_run_id"],
            ),
            handler=server._tool_attach_project_experiment_container_run,
        ),
        ToolDefinition(
            name="smr_list_project_experiment_container_runs",
            description="List concrete execution receipts attached to one SMR experiment.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["project_id", "experiment_id"],
            ),
            handler=server._tool_list_project_experiment_container_runs,
        ),
        ToolDefinition(
            name="smr_attach_project_experiment_result",
            description=(
                "Attach a normalized measured result to an SMR experiment. "
                "Use one result row per candidate metric/split."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "candidate_id": {"type": "string"},
                    "candidate_kind": {"type": "string"},
                    "candidate_label": {"type": "string"},
                    "metric": {"type": "string"},
                    "metric_direction": {
                        "type": "string",
                        "enum": ["higher_is_better", "lower_is_better"],
                    },
                    "value": {"type": "number"},
                    "baseline_value": {"type": "number"},
                    "delta": {"type": "number"},
                    "dataset_or_task_set_id": {"type": "string"},
                    "sample_size": {"type": "integer"},
                    "seed_set": {"type": "array", "items": {"type": "integer"}},
                    "split_name": {"type": "string"},
                    "summary_artifact_id": {"type": "string"},
                    "per_example_artifact_id": {"type": "string"},
                    "summary_artifact_path": {"type": "string"},
                    "per_example_artifact_path": {"type": "string"},
                    "evidence_grade": {"type": "string"},
                    "truth_status": {"type": "string"},
                    "caveats": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "experiment_id", "metric", "value"],
            ),
            handler=server._tool_attach_project_experiment_result,
        ),
        ToolDefinition(
            name="smr_list_project_experiment_results",
            description="List normalized SMR experiment results for a project or one experiment.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "metric": {"type": "string"},
                    "taskset_id": {"type": "string"},
                    "taskset_seed": {"type": "integer"},
                    "comparison_cohort_key": {"type": "string"},
                    "truth_status": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["project_id"],
            ),
            handler=server._tool_list_project_experiment_results,
        ),
        ToolDefinition(
            name="smr_rank_project_experiment_results",
            description="Rank comparable SMR experiment results for one metric.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "metric": {"type": "string"},
                    "taskset_id": {"type": "string"},
                    "taskset_seed": {"type": "integer"},
                    "comparison_cohort_key": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["project_id", "metric"],
            ),
            handler=server._tool_rank_project_experiment_results,
        ),
        ToolDefinition(
            name="smr_list_project_changesets",
            description="List review-gated project ChangeSets.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "status": {
                        "type": "string",
                        "description": "Optional ChangeSet status filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum ChangeSets to return.",
                    },
                },
                required=["project_id"],
            ),
            handler=server._tool_list_project_changesets,
        ),
        ToolDefinition(
            name="smr_create_project_changeset",
            description=(
                "Create a proposed project ChangeSet. This stages project "
                "mutations for review and does not directly mutate canon."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "run_id": {"type": "string"},
                    "source": {"type": "string"},
                    "author_ref": {"type": "string"},
                    "review_policy": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "target_kind": {"type": "string"},
                                "target_id": {"type": "string"},
                                "operation": {"type": "string"},
                                "proposed_payload": {"type": "object"},
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["target_kind", "operation"],
                        },
                    },
                    "idempotency_key": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                required=["project_id", "title", "items"],
            ),
            handler=server._tool_create_project_changeset,
        ),
        ToolDefinition(
            name="smr_get_project_changeset",
            description="Fetch one review-gated project ChangeSet.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "changeset_id": {"type": "string"},
                },
                required=["project_id", "changeset_id"],
            ),
            handler=server._tool_get_project_changeset,
        ),
        ToolDefinition(
            name="smr_decide_project_changeset",
            description="Accept, promote, reject, supersede, or invalidate a proposed project ChangeSet.",
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id.",
                    },
                    "changeset_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": [
                            "accepted",
                            "promoted",
                            "rejected",
                            "superseded",
                            "invalidated",
                        ],
                    },
                    "decided_by_ref": {"type": "string"},
                    "decision_reason": {"type": "string"},
                },
                required=["project_id", "changeset_id", "decision", "decided_by_ref"],
            ),
            handler=server._tool_decide_project_changeset,
        ),
        ToolDefinition(
            name="smr_get_project_entitlement",
            description="Fetch the managed-research entitlement status for a project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_get_project_entitlement,
        ),
        ToolDefinition(
            name="smr_get_project_notes",
            description="Fetch the durable notebook text for a managed research project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_get_project_notes,
        ),
        ToolDefinition(
            name="smr_set_project_notes",
            description="Replace the durable notebook text for a managed research project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "notes": {"type": "string", "description": "Notebook text to store."},
                },
                required=["project_id", "notes"],
            ),
            handler=server._tool_set_project_notes,
        ),
        ToolDefinition(
            name="smr_append_project_notes",
            description="Append text to the durable notebook for a managed research project.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "notes": {"type": "string", "description": "Notebook text to append."},
                },
                required=["project_id", "notes"],
            ),
            handler=server._tool_append_project_notes,
        ),
        ToolDefinition(
            name="smr_get_org_knowledge",
            description="Fetch curated org-wide knowledge for the authenticated organization. This is distinct from project notes.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_get_org_knowledge,
        ),
        ToolDefinition(
            name="smr_set_org_knowledge",
            description="Replace curated org-wide knowledge for the authenticated organization. This is distinct from project notes.",
            input_schema=tool_schema(
                {
                    "content": {
                        "type": "string",
                        "description": "Curated org-wide knowledge content to store.",
                    },
                },
                required=["content"],
            ),
            handler=server._tool_set_org_knowledge,
        ),
        ToolDefinition(
            name="smr_get_project_knowledge",
            description="Fetch curated knowledge for a managed research project. This is distinct from the project notebook notes surface.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_get_project_knowledge,
        ),
        ToolDefinition(
            name="smr_set_project_knowledge",
            description="Replace curated knowledge for a managed research project. This is distinct from the project notebook notes surface.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "content": {
                        "type": "string",
                        "description": "Curated project knowledge content to store.",
                    },
                },
                required=["project_id", "content"],
            ),
            handler=server._tool_set_project_knowledge,
        ),
        ToolDefinition(
            name="smr_curated_knowledge",
            description=(
                "Get or set curated knowledge for the authenticated org or for one project. "
                "Use scope=org|project, and provide content when operation=set."
            ),
            input_schema=tool_schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": ["get", "set"],
                        "description": "Get the current durable knowledge blob or replace it.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["org", "project"],
                        "description": "Whether the knowledge is org-wide or project-scoped.",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Managed research project id when scope is project.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Replacement knowledge content when operation is set.",
                    },
                },
                required=["operation", "scope"],
            ),
            handler=server._tool_curated_knowledge,
        ),
        ToolDefinition(
            name="smr_pause_project",
            description="Pause a managed research project so new runs cannot start.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_pause_project,
        ),
        ToolDefinition(
            name="smr_resume_project",
            description="Resume a paused managed research project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_resume_project,
        ),
        ToolDefinition(
            name="smr_archive_project",
            description="Archive a managed research project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_archive_project,
        ),
        ToolDefinition(
            name="smr_unarchive_project",
            description="Unarchive a managed research project.",
            input_schema=tool_schema(
                {"project_id": {"type": "string", "description": "Managed research project id."}},
                required=["project_id"],
            ),
            handler=server._tool_unarchive_project,
        ),
        ToolDefinition(
            name="smr_get_capabilities",
            description=(
                "Fetch server capabilities for parity-safe client behavior. "
                "Run trigger maps backend ``error_code`` payloads (limits, routing, "
                "credits, project budget, managed inference) to structured MCP results."
            ),
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_get_capabilities,
        ),
        ToolDefinition(
            name="smr_get_limits",
            description="Fetch resource limits for the authenticated org's plan. This is informative only; setup authority and launch preflight remain authoritative.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_get_limits,
        ),
        ToolDefinition(
            name="smr_get_capacity_lane_preview",
            description=(
                "Preview the preferred/resolved capacity lane before launch. "
                "Call this before smr_get_launch_preflight or smr_trigger_run "
                "when you need a user-facing launch check."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["project_id"],
            ),
            handler=server._tool_get_capacity_lane_preview,
        ),
        ToolDefinition(
            name="smr_set_provider_key",
            description="Store or rotate a project-scoped provider key for a supported credential provider.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "provider": {
                        "type": "string",
                        "enum": list(SMR_CREDENTIAL_PROVIDER_VALUES),
                        "description": "Credential provider to configure.",
                    },
                    "funding_source": {
                        "type": "string",
                        "enum": list(SMR_FUNDING_SOURCE_VALUES),
                        "description": "Funding source bucket for this credential.",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Plaintext provider API key.",
                    },
                    "encrypted_key_b64": {
                        "type": "string",
                        "description": "Optional encrypted provider API key payload.",
                    },
                },
                required=["project_id", "provider", "funding_source"],
            ),
            handler=server._tool_set_provider_key,
        ),
        ToolDefinition(
            name="smr_get_provider_key_status",
            description="Check whether a project-scoped provider key is configured.",
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "provider": {
                        "type": "string",
                        "enum": list(SMR_CREDENTIAL_PROVIDER_VALUES),
                        "description": "Credential provider to inspect.",
                    },
                    "funding_source": {
                        "type": "string",
                        "enum": list(SMR_FUNDING_SOURCE_VALUES),
                        "description": "Funding source bucket for this credential.",
                    },
                },
                required=["project_id", "provider", "funding_source"],
            ),
            handler=server._tool_get_provider_key_status,
        ),
        ToolDefinition(
            name="smr_get_workspace_download_url",
            description=(
                "Return a short-lived presigned URL to download the project workspace as a tarball "
                "(git snapshot archived by the backend). Use smr_download_workspace_archive to save "
                "the file locally in one step, or fetch download_url with curl yourself."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["project_id"],
            ),
            handler=server._tool_get_workspace_download_url,
        ),
        ToolDefinition(
            name="smr_get_project_git",
            description=(
                "Read-only git metadata for the project workspace (commit, branch, remote-related fields). "
                "Pair with smr_get_workspace_download_url or smr_download_workspace_archive to retrieve files."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["project_id"],
            ),
            handler=server._tool_get_project_git,
        ),
        ToolDefinition(
            name="smr_download_workspace_archive",
            description=(
                "Download the project or run workspace tarball to a path on the machine running this MCP server. "
                "When run_id is provided, this resolves the run-specific archive. Parent directories are created. Large repos may take minutes; "
                "raise timeout_seconds if needed (default 600)."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "run_id": {
                        "type": "string",
                        "description": "Optional run id for immutable run-scoped archive resolution.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Absolute or home-relative path for the .tar.gz file (e.g. ~/smr-workspace.tar.gz).",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "HTTP timeout for the presigned download in seconds (default 600).",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["project_id", "output_path"],
            ),
            handler=server._tool_download_workspace_archive,
        ),
        ToolDefinition(
            name="smr_open_ended_questions",
            description="List, create, fetch, patch, or transition project-scoped open-ended questions.",
            input_schema=tool_schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": [operation.value for operation in CompatObjectiveToolOperation],
                    },
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "objective_id": {
                        "type": "string",
                        "description": "Parent objective id for get/patch/transition.",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional run filter when listing or creating.",
                    },
                    "payload": {"type": "object", "description": "Create or patch payload."},
                },
                required=["operation", "project_id"],
            ),
            handler=server._tool_open_ended_questions,
        ),
        ToolDefinition(
            name="smr_objectives",
            description=(
                "List, create, fetch, patch, pause, resume, withdraw, claim "
                "progress, request review, or inspect tasks/progress for project objectives."
            ),
            input_schema=tool_schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": [operation.value for operation in ObjectiveToolOperation],
                    },
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "objective_id": {
                        "type": "string",
                        "description": "Objective id for item operations.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["open_ended_question", "directed_effort_outcome"],
                        "description": "Optional objective kind discriminator.",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional run filter when listing.",
                    },
                    "payload": {
                        "type": "object",
                        "description": "Create, patch, claim, or review payload.",
                    },
                },
                required=["operation", "project_id"],
            ),
            handler=server._tool_objectives,
        ),
        ToolDefinition(
            name="smr_get_objective_status",
            description=(
                "Fetch the composed product status bundle for an objective: "
                "objective, progress, related tasks, milestones, claims, blockers, "
                "recent events, and run scopes."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "objective_id": {"type": "string", "description": "Objective id."},
                    "kind": {
                        "type": "string",
                        "enum": ["open_ended_question", "directed_effort_outcome"],
                        "description": "Optional objective kind discriminator.",
                    },
                    "task_limit": {"type": "integer", "description": "Optional task limit."},
                    "claim_limit": {"type": "integer", "description": "Optional claim limit."},
                    "event_limit": {"type": "integer", "description": "Optional event limit."},
                    "milestone_limit": {
                        "type": "integer",
                        "description": "Optional milestone limit.",
                    },
                },
                required=["project_id", "objective_id"],
            ),
            handler=server._tool_get_objective_status,
        ),
        ToolDefinition(
            name="smr_milestones",
            description="List, create, fetch, patch, or transition project milestones.",
            input_schema=tool_schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": ["list", "create", "get", "patch", "transition"],
                    },
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "milestone_id": {
                        "type": "string",
                        "description": "Milestone id for get/patch/transition.",
                    },
                    "run_id": {"type": "string", "description": "Optional run filter for list."},
                    "parent_kind": {
                        "type": "string",
                        "enum": ["open_ended_question", "directed_effort_outcome"],
                        "description": "Optional parent objective kind filter for list.",
                    },
                    "parent_id": {
                        "type": "string",
                        "description": "Optional parent objective id filter for list.",
                    },
                    "limit": {"type": "integer", "description": "Optional list limit."},
                    "payload": {
                        "type": "object",
                        "description": "Create, patch, or transition payload.",
                    },
                },
                required=["operation", "project_id"],
            ),
            handler=server._tool_milestones,
        ),
        ToolDefinition(
            name="smr_directed_effort_outcomes",
            description="List, create, fetch, patch, or transition project-scoped directed effort outcomes.",
            input_schema=tool_schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": [operation.value for operation in CompatObjectiveToolOperation],
                    },
                    "project_id": {"type": "string", "description": "Managed research project id."},
                    "objective_id": {
                        "type": "string",
                        "description": "Parent objective id for get/patch/transition.",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional run filter when listing or creating.",
                    },
                    "payload": {"type": "object", "description": "Create or patch payload."},
                },
                required=["operation", "project_id"],
            ),
            handler=server._tool_directed_effort_outcomes,
        ),
    ]


__all__ = ["build_project_tools"]
