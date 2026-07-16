"""Factory and Effort MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import (
    READ_SCOPES,
    WRITE_SCOPES,
    ToolDefinition,
    tool_schema,
)
from synth_ai.managed_research.models.factories import (
    EFFORT_STATUS_VALUES,
    EFFORT_TYPE_VALUES,
    FACTORY_ACTOR_OUTPUT_KIND_VALUES,
    FACTORY_ACTOR_OUTPUT_STATUS_VALUES,
    FACTORY_ACTOR_ROLE_VALUES,
    FACTORY_IDEA_SOURCE_VALUES,
    FACTORY_IDEA_STATUS_VALUES,
    FACTORY_KIND_VALUES,
    FACTORY_LIFECYCLE_STATE_VALUES,
    FACTORY_PROJECT_ROLE_VALUES,
    FACTORY_PROJECT_STATUS_VALUES,
)


def _factory_mutation_properties() -> dict[str, Any]:
    return {
        "name": {"type": "string", "description": "Human-readable Factory name."},
        "description": {"type": "string", "description": "Optional Factory description."},
        "kind": {
            "type": "string",
            "enum": list(FACTORY_KIND_VALUES),
            "description": "Factory kind.",
        },
        "status": {
            "type": "string",
            "enum": list(FACTORY_LIFECYCLE_STATE_VALUES),
            "description": "Factory lifecycle status.",
        },
        "budget_policy": {"type": "object", "description": "Optional Factory budget policy."},
        "cap_policy": {"type": "object", "description": "Optional Factory cap policy."},
        "publication_policy": {
            "type": "object",
            "description": "Optional public/private publication policy.",
        },
        "authorization_policy": {
            "type": "object",
            "description": "Optional Factory authorization and workspace scope policy.",
        },
        "metadata": {"type": "object", "description": "Optional Factory metadata."},
    }


def _effort_mutation_properties() -> dict[str, Any]:
    return {
        "factory_id": {"type": "string", "description": "Factory ID."},
        "project_id": {"type": "string", "description": "Managed Research project ID."},
        "name": {"type": "string", "description": "Human-readable Effort name."},
        "allow_implicit_project_link": {
            "type": "boolean",
            "description": "Allow backend to link the Project to the Factory when missing.",
        },
        "hypothesis_or_topic": {
            "type": "string",
            "description": "Research hypothesis, topic, or operating context.",
        },
        "status": {
            "type": "string",
            "enum": list(EFFORT_STATUS_VALUES),
            "description": "Effort status.",
        },
        "effort_type": {
            "type": "string",
            "enum": list(EFFORT_TYPE_VALUES),
            "description": "Effort type.",
        },
        "recurrence_policy": {"type": "object", "description": "Optional recurrence policy."},
        "next_wake_at": {
            "type": "string",
            "description": "Optional ISO-8601 next wake timestamp.",
        },
        "latest_run_id": {"type": "string", "description": "Optional latest linked Run ID."},
        "latest_report_id": {
            "type": "string",
            "description": "Optional latest linked Report ID.",
        },
        "latest_work_product_id": {
            "type": "string",
            "description": "Optional latest linked WorkProduct ID.",
        },
        "decision_needed": {
            "type": "boolean",
            "description": "Whether the Effort needs operator input.",
        },
        "decision_note": {"type": "string", "description": "Optional decision note."},
        "budget_policy": {"type": "object", "description": "Optional Effort budget policy."},
        "publication_policy": {
            "type": "object",
            "description": "Optional public/private publication policy.",
        },
        "authorization_policy": {
            "type": "object",
            "description": "Optional Effort authorization and workspace scope policy.",
        },
        "actor_notes": {"type": "object", "description": "Optional role/runbook notes."},
        "metadata": {"type": "object", "description": "Optional Effort metadata."},
    }


def _factory_project_mutation_properties() -> dict[str, Any]:
    return {
        "project_id": {"type": "string", "description": "Managed Research project ID."},
        "role": {
            "type": "string",
            "enum": list(FACTORY_PROJECT_ROLE_VALUES),
            "description": (
                "Factory Project role. Use canonical for the primary workspace, auxiliary "
                "for additional active workspaces, and archived_reference for history."
            ),
        },
        "status": {
            "type": "string",
            "enum": list(FACTORY_PROJECT_STATUS_VALUES),
            "description": "Factory workspace Project link status.",
        },
        "display_name": {
            "type": "string",
            "description": "Optional Factory-local Project display name.",
        },
        "description": {
            "type": "string",
            "description": "Optional Factory-local Project description.",
        },
        "workspace_policy": {
            "type": "object",
            "description": "Workspace policy for this Project under the Factory.",
        },
        "resource_bindings": {
            "type": "object",
            "description": "Long-lived workspace resource bindings.",
        },
        "feed_health": {
            "type": "object",
            "description": "Feed and ingestion health metadata.",
        },
        "default_launch_profile": {
            "type": "object",
            "description": "Default launch profile layered into due Effort launches.",
        },
        "metadata": {"type": "object", "description": "Optional link metadata."},
    }


def _factory_workspace_link_properties() -> dict[str, Any]:
    return {
        key: value
        for key, value in _factory_project_mutation_properties().items()
        if key not in {"role", "status"}
    }


def _factory_idea_mutation_properties() -> dict[str, Any]:
    return {
        "title": {"type": "string", "description": "Short Garden idea title."},
        "body": {
            "type": "string",
            "description": "Optional detail, hypothesis, evidence, or question.",
        },
        "status": {
            "type": "string",
            "enum": list(FACTORY_IDEA_STATUS_VALUES),
            "description": "Garden idea lifecycle status.",
        },
        "source": {
            "type": "string",
            "enum": list(FACTORY_IDEA_SOURCE_VALUES),
            "description": "Who or what contributed the idea.",
        },
        "project_id": {"type": "string", "description": "Optional linked Project ID."},
        "effort_id": {"type": "string", "description": "Optional linked Effort ID."},
        "run_id": {"type": "string", "description": "Optional linked Run ID."},
        "priority": {"type": "string", "description": "Optional priority label."},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags for retrieval and triage.",
        },
        "promotion_target": {
            "type": "object",
            "description": "Optional target to create or update when the idea is promoted.",
        },
        "metadata": {"type": "object", "description": "Optional idea metadata."},
    }


def _factory_actor_output_mutation_properties() -> dict[str, Any]:
    return {
        "actor_role": {
            "type": "string",
            "enum": list(FACTORY_ACTOR_ROLE_VALUES),
            "description": "Research Factory actor role.",
        },
        "kind": {
            "type": "string",
            "enum": list(FACTORY_ACTOR_OUTPUT_KIND_VALUES),
            "description": "Typed actor output kind.",
        },
        "title": {"type": "string", "description": "Short output title."},
        "summary": {"type": "string", "description": "Optional concise summary."},
        "status": {
            "type": "string",
            "enum": list(FACTORY_ACTOR_OUTPUT_STATUS_VALUES),
            "description": "Actor output lifecycle status.",
        },
        "project_id": {"type": "string", "description": "Optional linked Project ID."},
        "effort_id": {"type": "string", "description": "Optional linked Effort ID."},
        "run_id": {"type": "string", "description": "Optional linked Run ID."},
        "report_id": {"type": "string", "description": "Optional linked Report ID."},
        "work_product_id": {
            "type": "string",
            "description": "Optional linked WorkProduct ID.",
        },
        "payload": {
            "type": "object",
            "description": "Structured actor output payload.",
        },
        "metadata": {"type": "object", "description": "Optional output metadata."},
    }


def build_factory_tools(server: Any) -> list[ToolDefinition]:
    effort_properties = _effort_mutation_properties()
    effort_patch_properties = {
        key: value
        for key, value in effort_properties.items()
        if key not in {"factory_id", "project_id"}
    }
    factory_project_properties = _factory_project_mutation_properties()
    factory_project_patch_properties = {
        key: value for key, value in factory_project_properties.items() if key != "project_id"
    }
    workspace_link_properties = _factory_workspace_link_properties()
    idea_properties = _factory_idea_mutation_properties()
    idea_patch_properties = {key: value for key, value in idea_properties.items() if key != "title"}
    actor_output_properties = _factory_actor_output_mutation_properties()
    actor_output_patch_properties = {
        key: value
        for key, value in actor_output_properties.items()
        if key not in {"actor_role", "kind", "title"}
    }

    def preview_factory_wake(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            return client.factories.wake_due(
                str(args["factory_id"]),
                launch_request=args.get("launch_request"),
                limit=int(args.get("limit") or 10),
                allow_overlap=bool(args.get("allow_overlap") or False),
                dry_run=True,
                continue_on_error=bool(args.get("continue_on_error", True)),
            ).raw

    return [
        ToolDefinition(
            name="smr_create_factory",
            description=(
                "Create a persistent Research Factory: an R&D organization/workspace "
                "for proving hypotheses, building prototypes, and improving applied AI systems."
            ),
            input_schema=tool_schema(_factory_mutation_properties(), required=["name"]),
            handler=server._tool_create_factory,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_factories",
            description="List Research Factories for the authenticated organization.",
            input_schema=tool_schema({}, required=[]),
            handler=server._tool_list_factories,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_factory",
            description="Fetch one persistent Research Factory workspace by ID.",
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "description": "Factory ID."}},
                required=["factory_id"],
            ),
            handler=server._tool_get_factory,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_patch_factory",
            description="Update Factory status, policies, description, or metadata.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **{
                        key: value
                        for key, value in _factory_mutation_properties().items()
                        if key != "kind"
                    },
                },
                required=["factory_id"],
            ),
            handler=server._tool_patch_factory,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_pause_factory",
            description="Pause a Research Factory so operators see it as not active.",
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "description": "Factory ID."}},
                required=["factory_id"],
            ),
            handler=server._tool_pause_factory,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_resume_factory",
            description="Resume a paused or archived Research Factory.",
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "description": "Factory ID."}},
                required=["factory_id"],
            ),
            handler=server._tool_resume_factory,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_archive_factory",
            description="Archive a Research Factory.",
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "description": "Factory ID."}},
                required=["factory_id"],
            ),
            handler=server._tool_archive_factory,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_factory_status",
            description=(
                "Read the Factory status projection: canonical/auxiliary workspace Projects, Efforts, "
                "Garden ideas, typed actor outputs, linked Projects, latest Runs, "
                "reports/work products, decisions, pauses, wake metadata, and "
                "publication/cost summaries."
            ),
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "description": "Factory ID."}},
                required=["factory_id"],
            ),
            handler=server._tool_get_factory_status,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_experiment_history",
            description=(
                "Read owner-assembled experiment bundles with missing-evidence "
                "alerts and accepted/incomplete cycle counts."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                },
                required=["project_id"],
            ),
            handler=lambda args: server._client_from_args(args).factories.experiment_history(
                str(args["project_id"]), limit=int(args.get("limit") or 50)
            ),
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_compare_experiments",
            description=(
                "Compare accepted experiments only when backend-owned cohort, "
                "seed, scorer, taskset, and metric dimensions match."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {"type": "string"},
                    "experiment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                required=["project_id", "experiment_ids"],
            ),
            handler=lambda args: server._client_from_args(args).factories.compare_experiments(
                str(args["project_id"]),
                [str(item) for item in args["experiment_ids"]],
            ),
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_link_factory_project",
            description=(
                "Link a Project to the Factory. Use canonical for the primary workspace "
                "or auxiliary for additional active workspaces."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **factory_project_properties,
                },
                required=["factory_id", "project_id"],
            ),
            handler=server._tool_link_factory_project,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_link_factory_workspace_project",
            description="Link or replace the Factory's primary canonical workspace Project.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **workspace_link_properties,
                },
                required=["factory_id", "project_id"],
            ),
            handler=server._tool_link_factory_workspace_project,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_link_factory_auxiliary_project",
            description="Link an additional active auxiliary Project workspace to a Factory.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **workspace_link_properties,
                },
                required=["factory_id", "project_id"],
            ),
            handler=server._tool_link_factory_auxiliary_project,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_factory_projects",
            description=(
                "List Factory Project links, including canonical, auxiliary, and archived history."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived Factory Project links.",
                    },
                },
                required=["factory_id"],
            ),
            handler=server._tool_list_factory_projects,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_factory_project",
            description="Fetch one Factory workspace Project link by Factory and Project ID.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "project_id": {"type": "string", "description": "Managed Research project ID."},
                },
                required=["factory_id", "project_id"],
            ),
            handler=server._tool_get_factory_project,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_patch_factory_project",
            description=(
                "Update a Factory workspace Project link status, workspace policy, "
                "resource bindings, or launch defaults. Archive before replacement."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "project_id": {"type": "string", "description": "Managed Research project ID."},
                    **factory_project_patch_properties,
                },
                required=["factory_id", "project_id"],
            ),
            handler=server._tool_patch_factory_project,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_factory_workspace",
            description="Read the Factory workspace projection across canonical and auxiliary Projects.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived Factory Project links.",
                    },
                },
                required=["factory_id"],
            ),
            handler=server._tool_get_factory_workspace,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_create_factory_idea",
            description=(
                "Add a Garden idea to a Research Factory. Ideas preserve optional "
                "hypotheses, experiment leads, failures, or future topics before "
                "they become Efforts."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **idea_properties,
                },
                required=["factory_id", "title"],
            ),
            handler=server._tool_create_factory_idea,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_factory_ideas",
            description="List Garden ideas for a Research Factory.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "status": {
                        "type": "string",
                        "enum": list(FACTORY_IDEA_STATUS_VALUES),
                        "description": "Optional idea status filter.",
                    },
                    "source": {
                        "type": "string",
                        "enum": list(FACTORY_IDEA_SOURCE_VALUES),
                        "description": "Optional idea source filter.",
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived ideas.",
                    },
                    "limit": {"type": "integer", "description": "Maximum ideas to return."},
                },
                required=["factory_id"],
            ),
            handler=server._tool_list_factory_ideas,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_factory_idea",
            description="Fetch one Garden idea by Factory and idea ID.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "idea_id": {"type": "string", "description": "Factory idea ID."},
                },
                required=["factory_id", "idea_id"],
            ),
            handler=server._tool_get_factory_idea,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_patch_factory_idea",
            description="Update, pause, reject, promote, or archive a Garden idea.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "idea_id": {"type": "string", "description": "Factory idea ID."},
                    **idea_patch_properties,
                },
                required=["factory_id", "idea_id"],
            ),
            handler=server._tool_patch_factory_idea,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_record_factory_actor_output",
            description=(
                "Record a typed Research Factory actor output such as a Seraph brief, "
                "Gardener digest, Architect feed-health note, finding report, or "
                "success-measurement card."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **actor_output_properties,
                },
                required=["factory_id", "actor_role", "kind", "title"],
            ),
            handler=server._tool_record_factory_actor_output,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_record_seraph_brief",
            description="Record a Seraph brief: priority, scope, escalation, or decision guidance.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **actor_output_patch_properties,
                    "title": {"type": "string", "description": "Short brief title."},
                },
                required=["factory_id", "title"],
            ),
            handler=server._tool_record_seraph_brief,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_record_gardener_digest",
            description="Record a Gardener digest: preserved memory, cleanup, archival, or reusable cases.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **actor_output_patch_properties,
                    "title": {"type": "string", "description": "Short digest title."},
                },
                required=["factory_id", "title"],
            ),
            handler=server._tool_record_gardener_digest,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_record_architect_feed_health",
            description="Record an Architect feed-health note for connectors, resources, or evaluation feeds.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    **actor_output_patch_properties,
                    "title": {"type": "string", "description": "Short feed-health title."},
                },
                required=["factory_id", "title"],
            ),
            handler=server._tool_record_architect_feed_health,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_factory_actor_outputs",
            description="List typed actor outputs for a Research Factory.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "actor_role": {
                        "type": "string",
                        "enum": list(FACTORY_ACTOR_ROLE_VALUES),
                        "description": "Optional actor role filter.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": list(FACTORY_ACTOR_OUTPUT_KIND_VALUES),
                        "description": "Optional output kind filter.",
                    },
                    "status": {
                        "type": "string",
                        "enum": list(FACTORY_ACTOR_OUTPUT_STATUS_VALUES),
                        "description": "Optional output status filter.",
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived actor outputs.",
                    },
                    "limit": {"type": "integer", "description": "Maximum outputs to return."},
                },
                required=["factory_id"],
            ),
            handler=server._tool_list_factory_actor_outputs,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_factory_actor_output",
            description="Fetch one typed actor output by Factory and output ID.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "actor_output_id": {
                        "type": "string",
                        "description": "Factory actor output ID.",
                    },
                },
                required=["factory_id", "actor_output_id"],
            ),
            handler=server._tool_get_factory_actor_output,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_patch_factory_actor_output",
            description="Update status, links, payload, or metadata for a typed actor output.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "actor_output_id": {
                        "type": "string",
                        "description": "Factory actor output ID.",
                    },
                    **actor_output_patch_properties,
                },
                required=["factory_id", "actor_output_id"],
            ),
            handler=server._tool_patch_factory_actor_output,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_factory_open_decisions",
            description="List Efforts in a Factory that need operator decisions.",
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "description": "Factory ID."}},
                required=["factory_id"],
            ),
            handler=server._tool_list_factory_open_decisions,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_preview_factory_wake",
            description="Preview due Factory experiments without starting runs.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "launch_request": {
                        "type": "object",
                        "description": "Optional default launch request.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum due Efforts to evaluate.",
                    },
                    "allow_overlap": {
                        "type": "boolean",
                        "description": "Preview overlap decisions for active Efforts.",
                    },
                    "continue_on_error": {
                        "type": "boolean",
                        "description": "Continue after a preview failure.",
                    },
                },
                required=["factory_id"],
            ),
            handler=preview_factory_wake,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_wake_due_factory_efforts",
            description=(
                "Evaluate due persistent Factory Efforts and launch cloud research "
                "engineering Runs through the managed run-start boundary."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "launch_request": {
                        "type": "object",
                        "description": "Default run launch request for due Efforts.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum due Efforts to evaluate.",
                    },
                    "allow_overlap": {
                        "type": "boolean",
                        "description": "Launch even when an Effort already has a nonterminal run.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Return launch decisions without starting runs.",
                    },
                    "continue_on_error": {
                        "type": "boolean",
                        "description": "Continue evaluating later Efforts after a launch failure.",
                    },
                },
                required=["factory_id"],
            ),
            handler=server._tool_wake_due_factory_efforts,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_factory_efforts",
            description="List Efforts under one Research Factory.",
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "description": "Factory ID."}},
                required=["factory_id"],
            ),
            handler=server._tool_list_factory_efforts,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_create_effort",
            description=(
                "Create a persistent R&D Effort under a Research Factory. The Effort "
                "tracks a hypothesis, experiment, or topic across time and must use "
                "the Factory workspace Project once one exists."
            ),
            input_schema=tool_schema(
                effort_properties,
                required=["factory_id", "project_id", "name"],
            ),
            handler=server._tool_create_effort,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_effort",
            description="Fetch one Research Factory Effort by ID.",
            input_schema=tool_schema(
                {"effort_id": {"type": "string", "description": "Effort ID."}},
                required=["effort_id"],
            ),
            handler=server._tool_get_effort,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_patch_effort",
            description="Update an Effort status, wake metadata, links, or notes.",
            input_schema=tool_schema(
                {
                    "effort_id": {"type": "string", "description": "Effort ID."},
                    **effort_patch_properties,
                },
                required=["effort_id"],
            ),
            handler=server._tool_patch_effort,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_pause_effort",
            description="Pause a Factory Effort.",
            input_schema=tool_schema(
                {"effort_id": {"type": "string", "description": "Effort ID."}},
                required=["effort_id"],
            ),
            handler=server._tool_pause_effort,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_resume_effort",
            description="Resume a paused, waiting, blocked, or review-ready Effort.",
            input_schema=tool_schema(
                {"effort_id": {"type": "string", "description": "Effort ID."}},
                required=["effort_id"],
            ),
            handler=server._tool_resume_effort,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_mark_effort_waiting",
            description="Mark an Effort waiting, optionally with the next wake time.",
            input_schema=tool_schema(
                {
                    "effort_id": {"type": "string", "description": "Effort ID."},
                    "next_wake_at": {
                        "type": "string",
                        "description": "Optional ISO-8601 next wake timestamp.",
                    },
                    "note": {"type": "string", "description": "Optional decision note."},
                },
                required=["effort_id"],
            ),
            handler=server._tool_mark_effort_waiting,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_schedule_effort",
            description=(
                "Schedule an Effort wake with recurrence policy and an optional "
                "stored launch request."
            ),
            input_schema=tool_schema(
                {
                    "effort_id": {"type": "string", "description": "Effort ID."},
                    "next_wake_at": {
                        "type": "string",
                        "description": "ISO-8601 next wake timestamp.",
                    },
                    "recurrence_policy": {
                        "type": "object",
                        "description": "Optional recurrence policy for future wakes.",
                    },
                    "launch_request": {
                        "type": "object",
                        "description": "Optional run launch request stored with the Effort.",
                    },
                },
                required=["effort_id", "next_wake_at"],
            ),
            handler=server._tool_schedule_effort,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_mark_effort_ready_for_review",
            description="Mark an Effort ready for operator review.",
            input_schema=tool_schema(
                {
                    "effort_id": {"type": "string", "description": "Effort ID."},
                    "note": {"type": "string", "description": "Optional decision note."},
                },
                required=["effort_id"],
            ),
            handler=server._tool_mark_effort_ready_for_review,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_resolve_effort_decision",
            description="Resolve an Effort decision flag and optionally record the note.",
            input_schema=tool_schema(
                {
                    "effort_id": {"type": "string", "description": "Effort ID."},
                    "note": {"type": "string", "description": "Decision resolution note."},
                },
                required=["effort_id"],
            ),
            handler=server._tool_resolve_effort_decision,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_launch_effort",
            description=(
                "Launch a cloud research engineering Run from a persistent Effort. "
                "The SDK resolves the Effort Project and links the Run with effort_id."
            ),
            input_schema=tool_schema(
                {
                    "effort_id": {"type": "string", "description": "Effort ID."},
                    "objective": {
                        "type": "string",
                        "description": "Optional user-facing objective to queue as the initial runtime message.",
                    },
                    "runbook": {"type": "string", "description": "Optional runbook kind."},
                    "runbook_preset": {
                        "type": "string",
                        "description": "Optional runbook preset.",
                    },
                    "worker_pool_id": {
                        "type": "string",
                        "description": "Optional worker pool ID.",
                    },
                    "timebox_seconds": {
                        "type": "integer",
                        "description": "Optional run timebox.",
                    },
                },
                required=["effort_id"],
            ),
            handler=server._tool_launch_effort,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_graduation_proposals",
            description=(
                "List Gardener-authored graduation proposals for a Managed Research "
                "project. Each proposal suggests promoting a set of related Runs into "
                "a persistent Effort."
            ),
            input_schema=tool_schema(
                {
                    "project_id": {
                        "type": "string",
                        "description": "Managed Research project ID.",
                    }
                },
                required=["project_id"],
            ),
            handler=server._tool_list_graduation_proposals,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_graduate_runs_to_effort",
            description=(
                "Graduate a set of Runs into a persistent Research Factory Effort. "
                "Creates the Effort under the runs' project and links the runs to it."
            ),
            input_schema=tool_schema(
                {
                    "run_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Run IDs to graduate into the new Effort.",
                    },
                    "name": {"type": "string", "description": "Human-readable Effort name."},
                    "project_id": {
                        "type": "string",
                        "description": "Managed Research project ID.",
                    },
                    "factory_id": {
                        "type": "string",
                        "description": "Optional Factory ID to own the Effort.",
                    },
                },
                required=["run_ids", "name", "project_id"],
            ),
            handler=server._tool_graduate_runs_to_effort,
            required_scopes=WRITE_SCOPES,
        ),
    ]


__all__ = ["build_factory_tools"]
