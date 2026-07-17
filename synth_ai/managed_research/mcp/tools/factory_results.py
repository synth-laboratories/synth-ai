"""Factory Result MCP tool definitions.

Results are the public product noun a Factory produces. These tools expose the
Result surface with public nouns; evaluation and current-best selection are
optional. They resolve to the same backend authority the legacy candidate/
champion storage uses — never a second source of truth.
"""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import (
    READ_SCOPES,
    WRITE_SCOPES,
    ToolDefinition,
    tool_schema,
)


def build_factory_result_tools(server: Any) -> list[ToolDefinition]:
    def list_factory_results(args: dict[str, Any]) -> list[dict[str, Any]]:
        with server._client_from_args(args) as client:
            return [
                item.raw
                for item in client.factories.results.list(
                    str(args["factory_id"]),
                    effort_id=args.get("effort_id"),
                    run_id=args.get("run_id"),
                    kind=args.get("kind"),
                    readiness=args.get("readiness"),
                    evaluation_status=args.get("evaluation_status"),
                    current_best=args.get("current_best"),
                    limit=int(args.get("limit") or 100),
                )
            ]

    def get_factory_result(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            return client.factories.results.get(
                str(args["factory_id"]), str(args["result_id"])
            ).raw

    def evaluate_factory_result(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            return client.factories.results.evaluate(
                str(args["factory_id"]),
                str(args["result_id"]),
                evaluation=dict(args["evaluation"]),
            ).raw

    def select_factory_result_current_best(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            return client.factories.results.select_current_best(
                str(args["factory_id"]),
                result_id=str(args["result_id"]),
                reason=str(args["reason"]),
                scope=args.get("scope"),
                effort_id=args.get("effort_id"),
            ).raw

    def restore_factory_result_current_best(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            return client.factories.results.restore_current_best(
                str(args["factory_id"]),
                result_id=str(args["result_id"]),
                reason=str(args["reason"]),
                scope=args.get("scope"),
                effort_id=args.get("effort_id"),
            ).raw

    def list_factory_result_selection_events(args: dict[str, Any]) -> list[dict[str, Any]]:
        with server._client_from_args(args) as client:
            return [
                item.raw
                for item in client.factories.results.selection_events(
                    str(args["factory_id"]),
                    limit=int(args.get("limit") or 100),
                )
            ]

    return [
        ToolDefinition(
            name="smr_list_factory_results",
            description=(
                "List Results a Research Factory has produced (reports, prompts, "
                "policies, datasets, models, artifacts, code changes). Filter by "
                "effort, run, kind, readiness, evaluation status, or current-best "
                "state. Ordinary Results carry no evaluation or selection."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "effort_id": {"type": "string", "description": "Filter by Effort."},
                    "run_id": {"type": "string", "description": "Filter by Run."},
                    "kind": {"type": "string", "description": "Filter by Result kind."},
                    "readiness": {
                        "type": "string",
                        "description": "Filter by readiness.",
                    },
                    "evaluation_status": {
                        "type": "string",
                        "description": "Filter by evaluation status.",
                    },
                    "current_best": {
                        "type": "boolean",
                        "description": "Filter to (or exclude) current-best Results.",
                    },
                    "limit": {"type": "integer", "description": "Maximum Results."},
                },
                required=["factory_id"],
            ),
            handler=list_factory_results,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_factory_result",
            description="Fetch one Factory Result by its result id (WorkProduct id).",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "result_id": {"type": "string", "description": "Result ID."},
                },
                required=["factory_id", "result_id"],
            ),
            handler=get_factory_result,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_evaluate_factory_result",
            description=(
                "Attach a benchmark-owned grading record to a Result. Only "
                "candidate-backed Results accept evaluation; the backend stores "
                "exactly what the grader proved and never grades itself."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "result_id": {"type": "string", "description": "Result ID."},
                    "evaluation": {
                        "type": "object",
                        "description": "Benchmark-owned grading record.",
                    },
                },
                required=["factory_id", "result_id", "evaluation"],
            ),
            handler=evaluate_factory_result,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_select_factory_result_current_best",
            description=(
                "Select a passing Result as current best for a named objective/scope. "
                "Idempotent and historical: it appends a selection event and never "
                "deletes or rewrites the prior Result."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "result_id": {"type": "string", "description": "Result ID."},
                    "reason": {"type": "string", "description": "Selection reason."},
                    "scope": {
                        "type": "string",
                        "description": "Objective/scope of the current-best selection.",
                    },
                    "effort_id": {"type": "string", "description": "Optional Effort."},
                },
                required=["factory_id", "result_id", "reason"],
            ),
            handler=select_factory_result_current_best,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_restore_factory_result_current_best",
            description=(
                "Restore a prior Result as current best for a named objective/scope. "
                "Idempotent and historical; appends a selection event."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "result_id": {"type": "string", "description": "Result ID."},
                    "reason": {"type": "string", "description": "Restore reason."},
                    "scope": {
                        "type": "string",
                        "description": "Objective/scope of the current-best selection.",
                    },
                    "effort_id": {"type": "string", "description": "Optional Effort."},
                },
                required=["factory_id", "result_id", "reason"],
            ),
            handler=restore_factory_result_current_best,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_factory_result_selection_events",
            description=(
                "List the append-only current-best selection history for a Factory."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "limit": {"type": "integer", "description": "Maximum events."},
                },
                required=["factory_id"],
            ),
            handler=list_factory_result_selection_events,
            required_scopes=READ_SCOPES,
        ),
    ]


__all__ = ["build_factory_result_tools"]
