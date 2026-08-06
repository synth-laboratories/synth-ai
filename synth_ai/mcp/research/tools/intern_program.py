"""Public MCP adapters for the Effort-primary Intern research program.

These are the operator's view of what the Intern is working on: the Effort
board, the rollup behind one Effort, and the Effort-bound objectives,
milestones, tasks, claims, and links beneath it.

Two boundaries are deliberate here:

* **Efforts themselves are not re-exposed.** ``research_list_factory_efforts``,
  ``research_get_effort``, and ``research_patch_effort`` already own Effort
  lifecycle on this server. ``intern_effort_board`` is the Intern's *rollup*
  over those Efforts, not a second Effort CRUD surface.
* **Memory reads keep one shape.** ``intern_memory_search`` / ``intern_memory_get``
  return the retrieval payload with no wrapper, field for field with the
  agent-facing memory tools, so an operator inspecting memory sees exactly what
  the Intern saw. Nulls are kept rather than stripped for the same reason.
* **No swarm planning.** The Intern kicks a run off and polls it; SMR owns the
  resulting task graph. Nothing here plans a swarm's tasks, and no tool on this
  server should. The Intern side of the loop is
  ``intern_progress_claim_create``, which folds a finished run's result back
  into the objective that motivated it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synth_ai.mcp.research.registry import (
    READ_SCOPES,
    WRITE_SCOPES,
    JSONDict,
    ToolDefinition,
    tool_schema,
)
from synth_ai.mcp.research.request_models import (
    optional_int,
    optional_string,
    require_string,
)
from synth_ai.sdk.research.client import Client as ResearchClient
from synth_ai.sdk.research.contracts.intern_program import (
    InternEffortTaskCreateRequest,
    InternEffortTaskPatchRequest,
    InternMilestoneCreateRequest,
    InternMilestoneTransitionRequest,
    InternObjectiveCreateRequest,
    InternObjectiveLinkCreateRequest,
    InternObjectivePatchRequest,
    InternProgressClaimCreateRequest,
)

CoreClientFactory = Callable[[JSONDict], ResearchClient]

_OBJECTIVE_KINDS = ["open_ended_question", "directed_effort_outcome"]
_MILESTONE_STATES = [
    "planned",
    "ready",
    "active",
    "validation_pending",
    "validating",
    "accepted",
    "blocked",
    "failed",
    "stopped",
]
_TASK_STATES = [
    "planned",
    "ready",
    "assigned",
    "in_progress",
    "review_required",
    "repair_required",
    "blocked",
    "done",
    "failed",
    "stopped",
    "superseded",
]
_OBJECTIVE_STATUSES = [
    "active",
    "paused",
    "blocked",
    "review_pending",
    "complete",
    "failed",
    "withdrawn",
]
_LINK_KINDS = [
    "smr_run",
    "smr_open_ended_question",
    "smr_directed_effort_outcome",
    "smr_work_product",
    "smr_report",
    "smr_experiment",
]
_LINK_ROLES = ["primary", "supporting", "reviewer", "blocker", "out_of_scope"]
_CRITERIA_MAX = 12

_EFFORT_ID = {"type": "string", "description": "Owning Effort ID."}
_OBJECTIVE_KIND_PROPERTY = {
    "type": "string",
    "enum": _OBJECTIVE_KINDS,
    "description": (
        "Which objective table to address. Open-ended questions and directed "
        "effort outcomes are stored separately."
    ),
}


def _string_list(args: JSONDict, key: str) -> tuple[str, ...] | None:
    """Read a bounded list of strings, or ``None`` when the key is absent."""
    if key not in args:
        return None
    value = args[key]
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be an array of strings")
    if len(value) > _CRITERIA_MAX:
        raise ValueError(f"'{key}' accepts at most {_CRITERIA_MAX} entries")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"'{key}' must be an array of strings")
    return tuple(str(item) for item in value)


def _string_list_or_empty(args: JSONDict, key: str) -> tuple[str, ...]:
    return _string_list(args, key) or ()


def _optional_percent(args: JSONDict) -> float | None:
    value = args.get("percent_complete")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("'percent_complete' must be a number when provided")
    return float(value)


def _criteria_property(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": _CRITERIA_MAX,
        "description": description,
    }


def build_intern_program_tools(
    client_from_args: CoreClientFactory,
) -> list[ToolDefinition]:
    """Build the Effort board, Effort rollup, and Effort-bound planner tools."""

    def intern_effort_board(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.program.list_efforts(
                factory_id=optional_string(args, "factory_id"),
                status=optional_string(args, "status"),
                limit=optional_int(args, "limit") or 25,
                cursor=optional_string(args, "cursor"),
            ).to_wire()

    def intern_effort_detail(args: JSONDict) -> JSONDict:
        effort_id = require_string(args, "effort_id")
        with client_from_args(args) as client:
            return client.intern.program.get_effort(effort_id).to_wire()

    def intern_objective_list(args: JSONDict) -> list[JSONDict]:
        effort_id = require_string(args, "effort_id")
        with client_from_args(args) as client:
            return [
                objective.to_wire()
                for objective in client.intern.program.list_objectives(
                    effort_id,
                    status=optional_string(args, "status"),
                    limit=optional_int(args, "limit") or 50,
                )
            ]

    def intern_objective_create(args: JSONDict) -> JSONDict:
        effort_id = require_string(args, "effort_id")
        request = InternObjectiveCreateRequest.model_validate(
            {
                "objective_kind": require_string(args, "objective_kind"),
                "title": require_string(args, "title"),
                "description": require_string(args, "description"),
                "scope": require_string(args, "scope"),
                "question_text": optional_string(args, "question_text"),
                "outcome_text": optional_string(args, "outcome_text"),
                "evidence_requirements": _string_list_or_empty(args, "evidence_requirements"),
                "resolution_criteria": _string_list_or_empty(args, "resolution_criteria"),
                "success_criteria": _string_list_or_empty(args, "success_criteria"),
                "deliverable_requirements": _string_list_or_empty(args, "deliverable_requirements"),
                "max_evaluation_iterations": optional_int(args, "max_evaluation_iterations") or 1,
                "idempotency_key": optional_string(args, "idempotency_key"),
            }
        )
        with client_from_args(args) as client:
            return client.intern.program.create_objective(effort_id, request).to_wire()

    def intern_objective_get(args: JSONDict) -> JSONDict:
        objective_id = require_string(args, "objective_id")
        objective_kind = require_string(args, "objective_kind")
        with client_from_args(args) as client:
            return client.intern.program.get_objective(
                objective_id,
                objective_kind=objective_kind,
            ).to_wire()

    def intern_objective_update(args: JSONDict) -> JSONDict:
        objective_id = require_string(args, "objective_id")
        patch = args.get("patch")
        if not isinstance(patch, dict) or not patch:
            raise ValueError("'patch' must be a non-empty object")
        request = InternObjectivePatchRequest.model_validate(
            {"objective_kind": require_string(args, "objective_kind"), **patch}
        )
        with client_from_args(args) as client:
            return client.intern.program.update_objective(objective_id, request).to_wire()

    def intern_milestone_list(args: JSONDict) -> list[JSONDict]:
        effort_id = require_string(args, "effort_id")
        with client_from_args(args) as client:
            return [
                milestone.to_wire()
                for milestone in client.intern.program.list_milestones(
                    effort_id,
                    state=optional_string(args, "state"),
                    limit=optional_int(args, "limit") or 50,
                )
            ]

    def intern_milestone_create(args: JSONDict) -> JSONDict:
        effort_id = require_string(args, "effort_id")
        request = InternMilestoneCreateRequest.model_validate(
            {
                "parent_kind": require_string(args, "parent_kind"),
                "parent_id": require_string(args, "parent_id"),
                "milestone_kind": require_string(args, "milestone_kind"),
                "title": require_string(args, "title"),
                "objective": require_string(args, "objective"),
                "acceptance_criteria": _string_list_or_empty(args, "acceptance_criteria"),
                "position": optional_int(args, "position") or 0,
                "idempotency_key": optional_string(args, "idempotency_key"),
            }
        )
        with client_from_args(args) as client:
            return client.intern.program.create_milestone(effort_id, request).to_wire()

    def intern_milestone_transition(args: JSONDict) -> JSONDict:
        milestone_id = require_string(args, "milestone_id")
        request = InternMilestoneTransitionRequest.model_validate(
            {
                "next_state": require_string(args, "next_state"),
                "reason": optional_string(args, "reason"),
            }
        )
        with client_from_args(args) as client:
            return client.intern.program.transition_milestone(
                milestone_id,
                request,
            ).to_wire()

    def intern_task_list(args: JSONDict) -> list[JSONDict]:
        effort_id = require_string(args, "effort_id")
        with client_from_args(args) as client:
            return [
                task.to_wire()
                for task in client.intern.program.list_tasks(
                    effort_id,
                    state=optional_string(args, "state"),
                    milestone_id=optional_string(args, "milestone_id"),
                    limit=optional_int(args, "limit") or 100,
                )
            ]

    def intern_task_create(args: JSONDict) -> JSONDict:
        effort_id = require_string(args, "effort_id")
        request = InternEffortTaskCreateRequest.model_validate(
            {
                "title": require_string(args, "title"),
                "body": optional_string(args, "body"),
                "milestone_id": optional_string(args, "milestone_id"),
                "objective_kind": optional_string(args, "objective_kind"),
                "objective_id": optional_string(args, "objective_id"),
                "acceptance_criteria": _string_list_or_empty(args, "acceptance_criteria"),
                "dependency_task_ids": _string_list_or_empty(args, "dependency_task_ids"),
                "position": optional_int(args, "position") or 0,
                "idempotency_key": optional_string(args, "idempotency_key"),
            }
        )
        with client_from_args(args) as client:
            return client.intern.program.create_task(effort_id, request).to_wire()

    def intern_task_update(args: JSONDict) -> JSONDict:
        intern_task_id = require_string(args, "intern_task_id")
        patch = args.get("patch")
        if not isinstance(patch, dict) or not patch:
            raise ValueError("'patch' must be a non-empty object")
        request = InternEffortTaskPatchRequest.model_validate(patch)
        with client_from_args(args) as client:
            return client.intern.program.update_task(intern_task_id, request).to_wire()

    def intern_progress_claim_list(args: JSONDict) -> list[JSONDict]:
        effort_id = require_string(args, "effort_id")
        with client_from_args(args) as client:
            return [
                claim.to_wire()
                for claim in client.intern.program.list_progress_claims(
                    effort_id,
                    limit=optional_int(args, "limit") or 50,
                )
            ]

    def intern_progress_claim_create(args: JSONDict) -> JSONDict:
        effort_id = require_string(args, "effort_id")
        request = InternProgressClaimCreateRequest.model_validate(
            {
                "objective_kind": require_string(args, "objective_kind"),
                "objective_id": require_string(args, "objective_id"),
                "summary": require_string(args, "summary"),
                "claim_kind": optional_string(args, "claim_kind") or "progress",
                "percent_complete": _optional_percent(args),
                "intern_task_id": optional_string(args, "intern_task_id"),
                "milestone_id": optional_string(args, "milestone_id"),
                "smr_run_id": optional_string(args, "smr_run_id"),
                "evidence_refs": _string_list_or_empty(args, "evidence_refs"),
                "expected_remaining_work": _string_list_or_empty(args, "expected_remaining_work"),
                "idempotency_key": optional_string(args, "idempotency_key"),
            }
        )
        with client_from_args(args) as client:
            return client.intern.program.create_progress_claim(
                effort_id,
                request,
            ).to_wire()

    def intern_objective_link_list(args: JSONDict) -> list[JSONDict]:
        effort_id = require_string(args, "effort_id")
        with client_from_args(args) as client:
            return [
                link.to_wire()
                for link in client.intern.program.list_objective_links(
                    effort_id,
                    limit=optional_int(args, "limit") or 50,
                )
            ]

    def intern_objective_link_create(args: JSONDict) -> JSONDict:
        objective_id = require_string(args, "objective_id")
        request = InternObjectiveLinkCreateRequest.model_validate(
            {
                "objective_kind": require_string(args, "objective_kind"),
                "link_kind": require_string(args, "link_kind"),
                "target_id": require_string(args, "target_id"),
                "link_role": optional_string(args, "link_role") or "primary",
                "note": optional_string(args, "note"),
            }
        )
        with client_from_args(args) as client:
            return client.intern.program.create_objective_link(
                objective_id,
                request,
            ).to_wire()

    def intern_memory_search(args: JSONDict) -> JSONDict:
        kinds = _string_list(args, "kinds")
        with client_from_args(args) as client:
            response = client.intern.program.search_memory(
                require_string(args, "query"),
                kinds=kinds,
                limit=optional_int(args, "limit") or 20,
            )
        # Dumped with nulls intact: this payload must stay byte-comparable with
        # the agent-facing memory tool result, which emits them.
        return response.model_dump(mode="json")

    def intern_memory_get(args: JSONDict) -> JSONDict:
        hit_id = require_string(args, "hit_id")
        with client_from_args(args) as client:
            return client.intern.program.get_memory_hit(hit_id).model_dump(mode="json")

    return [
        ToolDefinition(
            name="intern_effort_board",
            description=(
                "List the Intern's Efforts as board rows: each Effort plus its open "
                "objective/task/question counts. Rows carry counts only -- call "
                "intern_effort_detail for one Effort's bodies."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {
                        "type": "string",
                        "description": "Restrict to one Factory's Efforts.",
                    },
                    "status": {
                        "type": "string",
                        "enum": [
                            "active",
                            "paused",
                            "waiting",
                            "blocked",
                            "ready_for_review",
                            "archived_reference",
                        ],
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "cursor": {
                        "type": "string",
                        "description": "Opaque continuation from a previous next_cursor.",
                    },
                },
                required=[],
            ),
            handler=intern_effort_board,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_effort_detail",
            description=(
                "Roll up one Effort: progress (objectives/milestones/tasks/claims/open "
                "questions), results (work products, summaries, latest report), "
                "experiments (linked runs, read-only), and knowledge. The runtime block "
                "is secondary ops detail, not the product surface."
            ),
            input_schema=tool_schema(
                {"effort_id": _EFFORT_ID},
                required=["effort_id"],
            ),
            handler=intern_effort_detail,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_objective_list",
            description="List the Intern objectives opened under one Effort.",
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "status": {"type": "string", "enum": _OBJECTIVE_STATUSES},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                required=["effort_id"],
            ),
            handler=intern_objective_list,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_objective_create",
            description=(
                "Open one Intern objective under an Effort. The Effort is the binding: "
                "an objective cannot exist without one. Open-ended questions require "
                "question_text; directed effort outcomes require outcome_text."
            ),
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "objective_kind": _OBJECTIVE_KIND_PROPERTY,
                    "title": {"type": "string", "minLength": 1, "maxLength": 200},
                    "description": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "scope": {"type": "string", "minLength": 1, "maxLength": 4000},
                    "question_text": {
                        "type": "string",
                        "maxLength": 20000,
                        "description": "Required for open_ended_question.",
                    },
                    "evidence_requirements": _criteria_property("Open-ended question only."),
                    "resolution_criteria": _criteria_property("Open-ended question only."),
                    "outcome_text": {
                        "type": "string",
                        "maxLength": 20000,
                        "description": "Required for directed_effort_outcome.",
                    },
                    "success_criteria": _criteria_property("Directed effort outcome only."),
                    "deliverable_requirements": _criteria_property("Directed effort outcome only."),
                    "max_evaluation_iterations": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "idempotency_key": {"type": "string", "maxLength": 200},
                },
                required=[
                    "effort_id",
                    "objective_kind",
                    "title",
                    "description",
                    "scope",
                ],
            ),
            handler=intern_objective_create,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_objective_get",
            description="Fetch one Intern objective.",
            input_schema=tool_schema(
                {
                    "objective_id": {"type": "string"},
                    "objective_kind": _OBJECTIVE_KIND_PROPERTY,
                },
                required=["objective_id", "objective_kind"],
            ),
            handler=intern_objective_get,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_objective_update",
            description=(
                "Revise one Intern objective. The Effort binding is immutable -- an "
                "objective cannot be moved between Efforts."
            ),
            input_schema=tool_schema(
                {
                    "objective_id": {"type": "string"},
                    "objective_kind": _OBJECTIVE_KIND_PROPERTY,
                    "patch": {
                        "type": "object",
                        "description": (
                            "Any of title, description, scope, question_text, "
                            "outcome_text, status, evaluation_state, review_summary, "
                            "success_criteria, resolution_criteria, "
                            "evidence_requirements, deliverable_requirements."
                        ),
                    },
                },
                required=["objective_id", "objective_kind", "patch"],
            ),
            handler=intern_objective_update,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_milestone_list",
            description="List the milestones under one Effort.",
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "state": {"type": "string", "enum": _MILESTONE_STATES},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                required=["effort_id"],
            ),
            handler=intern_milestone_list,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_milestone_create",
            description=(
                "Break one Intern objective into a subquestion or suboutcome milestone "
                "under its Effort."
            ),
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "parent_kind": _OBJECTIVE_KIND_PROPERTY,
                    "parent_id": {"type": "string"},
                    "milestone_kind": {
                        "type": "string",
                        "enum": ["subquestion", "suboutcome"],
                    },
                    "title": {"type": "string", "minLength": 1, "maxLength": 200},
                    "objective": {"type": "string", "minLength": 1, "maxLength": 8000},
                    "acceptance_criteria": _criteria_property(
                        "What acceptance of this milestone requires."
                    ),
                    "position": {"type": "integer", "minimum": 0},
                    "idempotency_key": {"type": "string", "maxLength": 200},
                },
                required=[
                    "effort_id",
                    "parent_kind",
                    "parent_id",
                    "milestone_kind",
                    "title",
                    "objective",
                ],
            ),
            handler=intern_milestone_create,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_milestone_transition",
            description=(
                "Move a milestone to its next lifecycle state. Legality is decided by "
                "the shared milestone state machine; an illegal move is refused."
            ),
            input_schema=tool_schema(
                {
                    "milestone_id": {"type": "string"},
                    "next_state": {"type": "string", "enum": _MILESTONE_STATES},
                    "reason": {"type": "string", "maxLength": 2000},
                },
                required=["milestone_id", "next_state"],
            ),
            handler=intern_milestone_transition,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_task_list",
            description=(
                "List the Intern planner checklist for one Effort. These are the "
                "Intern's own tasks; they are not swarm run tasks and are not nodes in "
                "any run's task graph."
            ),
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "state": {"type": "string", "enum": _TASK_STATES},
                    "milestone_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                required=["effort_id"],
            ),
            handler=intern_task_list,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_task_create",
            description=(
                "Add one task to the Intern planner checklist for an Effort. This does "
                "not create or plan a swarm task."
            ),
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "title": {"type": "string", "minLength": 1, "maxLength": 200},
                    "body": {"type": "string", "maxLength": 8000},
                    "milestone_id": {"type": "string"},
                    "objective_kind": _OBJECTIVE_KIND_PROPERTY,
                    "objective_id": {"type": "string"},
                    "acceptance_criteria": _criteria_property("What done means for this task."),
                    "dependency_task_ids": _criteria_property(
                        "Other Intern planner task IDs this one waits on."
                    ),
                    "position": {"type": "integer", "minimum": 0},
                    "idempotency_key": {"type": "string", "maxLength": 200},
                },
                required=["effort_id", "title"],
            ),
            handler=intern_task_create,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_task_update",
            description=(
                "Revise one Intern planner task. State moves validate against the task "
                "state machine."
            ),
            input_schema=tool_schema(
                {
                    "intern_task_id": {"type": "string"},
                    "patch": {
                        "type": "object",
                        "description": (
                            "Any of title, body, state, acceptance_criteria, "
                            "artifact_ids, assigned_actor_key, assignment_reason, "
                            "position."
                        ),
                    },
                },
                required=["intern_task_id", "patch"],
            ),
            handler=intern_task_update,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_progress_claim_list",
            description="List the append-only progress claims folded into one Effort.",
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                required=["effort_id"],
            ),
            handler=intern_progress_claim_list,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_progress_claim_create",
            description=(
                "Fold a result into an Intern objective. This is the Intern half of "
                "kickoff-and-poll: a run finishes on the SMR surfaces and the claim "
                "records what it proved. smr_run_id is a read-only reference to that "
                "run; recording a claim grants no authority over it."
            ),
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "objective_kind": _OBJECTIVE_KIND_PROPERTY,
                    "objective_id": {"type": "string"},
                    "summary": {"type": "string", "minLength": 1, "maxLength": 8000},
                    "claim_kind": {
                        "type": "string",
                        "enum": ["progress", "achievement"],
                    },
                    "percent_complete": {"type": "number", "minimum": 0, "maximum": 100},
                    "intern_task_id": {"type": "string"},
                    "milestone_id": {"type": "string"},
                    "smr_run_id": {
                        "type": "string",
                        "description": "Run whose results this claim folds in (reference only).",
                    },
                    "evidence_refs": _criteria_property("Evidence backing the claim."),
                    "expected_remaining_work": _criteria_property(
                        "What is still open after this claim."
                    ),
                    "idempotency_key": {"type": "string", "maxLength": 200},
                },
                required=["effort_id", "objective_kind", "objective_id", "summary"],
            ),
            handler=intern_progress_claim_create,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_objective_link_list",
            description=(
                "List the references from one Effort's Intern objectives to the "
                "SMR-owned runs, reports, work products, and experiments they relate to."
            ),
            input_schema=tool_schema(
                {
                    "effort_id": _EFFORT_ID,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                required=["effort_id"],
            ),
            handler=intern_objective_link_list,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_objective_link_create",
            description=(
                "Reference an SMR-owned row from an Intern objective. This writes one "
                "Intern row and never touches the target, so it grants no authority "
                "over the linked run, report, or work product."
            ),
            input_schema=tool_schema(
                {
                    "objective_id": {"type": "string"},
                    "objective_kind": _OBJECTIVE_KIND_PROPERTY,
                    "link_kind": {"type": "string", "enum": _LINK_KINDS},
                    "target_id": {"type": "string"},
                    "link_role": {"type": "string", "enum": _LINK_ROLES},
                    "note": {"type": "string", "maxLength": 2000},
                },
                required=["objective_id", "objective_kind", "link_kind", "target_id"],
            ),
            handler=intern_objective_link_create,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_memory_search",
            description=(
                "Search the Intern's own history by keyword -- events, handoffs, "
                "messages, and thread segments. Returns the same bounded retrieval "
                "payload the Intern's own memory tool returns, so an operator reads "
                "what the Intern read."
            ),
            input_schema=tool_schema(
                {
                    "query": {"type": "string", "minLength": 1, "maxLength": 256},
                    "kinds": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["event", "handoff", "message", "segment"],
                        },
                        "maxItems": 4,
                        "description": "Restrict to these record kinds.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                required=["query"],
            ),
            handler=intern_memory_search,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_memory_get",
            description=(
                "Fetch one memory hit by the stable kind:id handle that "
                "intern_memory_search returned."
            ),
            input_schema=tool_schema(
                {"hit_id": {"type": "string", "minLength": 1}},
                required=["hit_id"],
            ),
            handler=intern_memory_get,
            required_scopes=READ_SCOPES,
        ),
    ]


__all__ = ["build_intern_program_tools"]
