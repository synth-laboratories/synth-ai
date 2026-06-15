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
    FACTORY_KIND_VALUES,
    FACTORY_LIFECYCLE_STATE_VALUES,
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
        "metadata": {"type": "object", "description": "Optional Factory metadata."},
    }


def _effort_mutation_properties() -> dict[str, Any]:
    return {
        "factory_id": {"type": "string", "description": "Factory ID."},
        "project_id": {"type": "string", "description": "Managed Research project ID."},
        "name": {"type": "string", "description": "Human-readable Effort name."},
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
        "actor_notes": {"type": "object", "description": "Optional role/runbook notes."},
        "metadata": {"type": "object", "description": "Optional Effort metadata."},
    }


def build_factory_tools(server: Any) -> list[ToolDefinition]:
    effort_properties = _effort_mutation_properties()
    effort_patch_properties = {
        key: value
        for key, value in effort_properties.items()
        if key not in {"factory_id", "project_id"}
    }
    return [
        ToolDefinition(
            name="smr_create_factory",
            description="Create a Research Factory for the authenticated organization.",
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
            description="Fetch one Research Factory by ID.",
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
                "Read the Factory status projection: projects, efforts, latest runs, "
                "latest reports/work products, decisions, pauses, wake metadata, and "
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
            description="Create an Effort under a Research Factory and project.",
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
                "Launch a Managed Research run from an Effort. The SDK resolves the "
                "Effort project and links the run with effort_id."
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
    ]


__all__ = ["build_factory_tools"]
