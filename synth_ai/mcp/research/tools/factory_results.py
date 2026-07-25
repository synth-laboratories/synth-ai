"""Factory Result MCP tool definitions.

Results are the public product noun a Factory produces. These tools expose the
Result surface with public nouns; evaluation and current-best selection are
optional. They resolve to the same backend authority the legacy candidate/
champion storage uses — never a second source of truth.
"""

from __future__ import annotations

from typing import Any

from synth_ai.core.research.contracts.factory_lenses import (
    FactoryEvaluationStatus,
    FactoryLensDirection,
    FactoryLensMissingPolicy,
    FactoryLensSpec,
    FactoryLensTieBreak,
    FactoryPreferenceAction,
    FactoryPreferenceRequest,
    FactoryResultEvaluationRequest,
    FactoryResultKind,
)
from synth_ai.mcp.research.registry import (
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
            return client.factories.results.get(str(args["factory_id"]), str(args["result_id"])).raw

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

    def define_factory_evaluation_lens(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            return client.factories.lenses.define(
                str(args["factory_id"]),
                FactoryLensSpec(
                    lens_key=str(args["lens_key"]),
                    direction=FactoryLensDirection(str(args["direction"])),
                    objective=args.get("objective"),
                    missing_policy=FactoryLensMissingPolicy(
                        str(args.get("missing_policy") or "ineligible")
                    ),
                    tie_break=FactoryLensTieBreak(str(args.get("tie_break") or "earliest_result")),
                    eligible_result_kinds=tuple(
                        FactoryResultKind(str(kind))
                        for kind in (args.get("eligible_result_kinds") or [])
                    ),
                ),
            ).__dict__

    def list_factory_evaluation_lenses(args: dict[str, Any]) -> list[dict[str, Any]]:
        with server._client_from_args(args) as client:
            return [
                dict(item.__dict__)
                for item in client.factories.lenses.list(
                    str(args["factory_id"]),
                    include_superseded=bool(args.get("include_superseded") or False),
                    limit=int(args.get("limit") or 100),
                )
            ]

    def get_factory_best_results(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            best = client.factories.lenses.best_so_far(str(args["factory_id"]))
            return {
                "factory_id": best.factory_id,
                "optimizes": best.optimizes,
                "lenses": [dict(lens.__dict__) for lens in best.lenses],
            }

    def record_factory_result_evaluation(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            return dict(
                client.factories.lenses.record_evaluation(
                    str(args["factory_id"]),
                    str(args["result_id"]),
                    FactoryResultEvaluationRequest(
                        lens_key=str(args["lens_key"]),
                        attempt_key=str(args["attempt_key"]),
                        status=FactoryEvaluationStatus(str(args["status"])),
                        score=args.get("score"),
                        baseline_score=args.get("baseline_score"),
                        evaluator=args.get("evaluator"),
                        record=dict(args.get("record") or {}),
                    ),
                ).__dict__
            )

    def record_factory_result_preference(args: dict[str, Any]) -> dict[str, Any]:
        with server._client_from_args(args) as client:
            return dict(
                client.factories.lenses.prefer(
                    str(args["factory_id"]),
                    FactoryPreferenceRequest(
                        idempotency_key=str(args["idempotency_key"]),
                        reason=str(args["reason"]),
                        action=FactoryPreferenceAction(str(args.get("action") or "prefer")),
                        result_id=args.get("result_id"),
                        lens_key=args.get("lens_key"),
                    ),
                ).__dict__
            )

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
            name="smr_define_factory_evaluation_lens",
            description=(
                "Declare how a Factory compares its Results, for the minority of "
                "Factories that optimize something. Re-declaring an existing "
                "lens_key appends a new immutable version rather than editing the "
                "old one, so earlier best-so-far answers stay reproducible. "
                "Requires the Factory to be on the Result authority."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "lens_key": {
                        "type": "string",
                        "description": "Stable name for this objective.",
                    },
                    "direction": {
                        "type": "string",
                        "description": "maximize or minimize.",
                    },
                    "objective": {
                        "type": "string",
                        "description": "Human description of what is optimized.",
                    },
                    "missing_policy": {
                        "type": "string",
                        "description": (
                            "ineligible (default) drops unscored Results; worst "
                            "ranks them behind every scored one."
                        ),
                    },
                    "tie_break": {
                        "type": "string",
                        "description": (
                            "earliest_result (default), latest_result, or lowest_result_id."
                        ),
                    },
                    "eligible_result_kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Result kinds this lens scores; empty means all.",
                    },
                },
                required=["factory_id", "lens_key", "direction"],
            ),
            handler=define_factory_evaluation_lens,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_list_factory_evaluation_lenses",
            description=(
                "List a Factory's evaluation lens versions. An empty list means "
                "the Factory optimizes nothing, which is the ordinary case."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "include_superseded": {
                        "type": "boolean",
                        "description": (
                            "Include older versions, needed to audit a best-so-far "
                            "answer computed under a definition since replaced."
                        ),
                    },
                    "limit": {"type": "integer", "description": "Maximum rows."},
                },
                required=["factory_id"],
            ),
            handler=list_factory_evaluation_lenses,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_get_factory_best_results",
            description=(
                "Derived best-so-far for every lens a Factory declares. Read "
                "outcome before result_id: 'no_eligible_results' means the lens "
                "does not apply, 'no_scored_results' means nothing is evaluated "
                "yet, and optimizes=false means the Factory hillclimbs nothing. "
                "None of those mean the Factory produced no work."
            ),
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "description": "Factory ID."}},
                required=["factory_id"],
            ),
            handler=get_factory_best_results,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="smr_record_factory_result_evaluation",
            description=(
                "Store one externally owned verdict for a Result under a lens. "
                "The backend never grades. Idempotent under attempt_key; a "
                "correction uses a NEW attempt_key so the earlier belief survives."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "result_id": {
                        "type": "string",
                        "description": "Result envelope id or the WorkProduct id it pins.",
                    },
                    "lens_key": {"type": "string", "description": "Lens to score under."},
                    "attempt_key": {
                        "type": "string",
                        "description": "Caller-stable key making retries idempotent.",
                    },
                    "status": {
                        "type": "string",
                        "description": "pending, evaluated, failed, or timeout.",
                    },
                    "score": {
                        "type": "number",
                        "description": "Required when status is 'evaluated'.",
                    },
                    "baseline_score": {"type": "number", "description": "Optional baseline."},
                    "evaluator": {"type": "string", "description": "Who evaluated."},
                    "record": {
                        "type": "object",
                        "description": "The evaluator's full typed record.",
                    },
                },
                required=["factory_id", "result_id", "lens_key", "attempt_key", "status"],
            ),
            handler=record_factory_result_evaluation,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="smr_record_factory_result_preference",
            description=(
                "Record a human preference for a Result. Preference sits BESIDE "
                "the derived best and never overwrites it: a reviewer disagreeing "
                "with the lens is real information, and collapsing the two into "
                "one pointer destroys it."
            ),
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "description": "Factory ID."},
                    "action": {
                        "type": "string",
                        "description": "prefer (default) or retract.",
                    },
                    "result_id": {
                        "type": "string",
                        "description": "Required for 'prefer'.",
                    },
                    "lens_key": {
                        "type": "string",
                        "description": "Optional lens the preference is scoped to.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Caller-stable key making retries idempotent.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why. An unexplained override is unauditable later.",
                    },
                },
                required=["factory_id", "idempotency_key", "reason"],
            ),
            handler=record_factory_result_preference,
            required_scopes=WRITE_SCOPES,
        ),
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
            description=("List the append-only current-best selection history for a Factory."),
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
