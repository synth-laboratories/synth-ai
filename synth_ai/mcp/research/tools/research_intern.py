"""Bounded MCP adapters for the typed Research Intern control plane."""

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
    require_int,
    require_string,
)
from synth_ai.sdk.research.client import Client as ResearchClient
from synth_ai.sdk.research.contracts.research_intern import (
    InternAsyncCommandKind,
    InternAsyncCommandRequest,
    InternAsyncEnsureRequest,
    InternAsyncInstructionKind,
    InternAsyncInstructionRequest,
    InternRuntimeOutcome,
    InternSyncCommandKind,
    InternSyncCommandRequest,
    InternSyncSessionCreateRequest,
    MagiDecisionRequest,
    MagiMode,
    ResearchInternAcceptanceReceiptPublicationRequest,
    ResearchInternEventAppendRequest,
    ResearchInternPatchRequest,
    ResearchInternProvisionRequest,
    ResearchInternSessionCloseRequest,
    ResearchInternSessionCreateRequest,
    ResearchInternTracePublicationRequest,
    ResearchInternTurnRequest,
)
from synth_ai.sdk.research.research_intern import (
    ResearchInternEventCursor,
    legacy_intern_sessions_enabled,
)

CoreClientFactory = Callable[[JSONDict], ResearchClient]
_RESEARCH_INTERN_EVENT_SEQUENCE_MAX = 2**31 - 1

#: MCP tools that drive the legacy ``/smr/research-intern/sessions`` plane.
#: They are excluded from registration unless the environment explicitly opts
#: in via ``SYNTH_ALLOW_LEGACY_INTERN_SESSIONS=1``; QA treats any legacy
#: ``/sessions`` hit as a hard fail, so the default surface is sync-sessions
#: only (``intern_sync_*`` / ``intern_async_*``).
LEGACY_INTERN_SESSION_TOOL_NAMES = frozenset(
    {
        "research_append_research_intern_event",
        "research_close_research_intern_session",
        "research_create_research_intern_session",
        "research_exchange_research_intern_turn",
        "research_get_research_intern_session",
        "research_list_research_intern_events",
        "research_list_research_intern_sessions",
        "research_publish_research_intern_session_trace",
        "research_run_research_intern_turn",
        "research_sync_research_intern_session",
        "research_watch_research_intern_events",
    }
)


def _request_payload(args: JSONDict, names: tuple[str, ...]) -> JSONDict:
    return {name: args[name] for name in names if name in args}


def _wire_list(values: tuple[Any, ...]) -> list[JSONDict]:
    return [value.to_wire() for value in values]


def _optional_number_default(args: JSONDict, name: str, default: float) -> float:
    value = args.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number when provided")
    return float(value)


def _event_cursor(args: JSONDict) -> ResearchInternEventCursor:
    session_id = require_string(args, "session_id")
    after_sequence = optional_int(args, "after_sequence") or 0
    state_generation = optional_int(args, "state_generation") or 0
    event_id = optional_string(args, "event_id")
    if after_sequence == 0:
        if state_generation != 0 or event_id is not None:
            raise ValueError("the zero event cursor cannot include generation or event_id")
        return ResearchInternEventCursor(session_id=session_id)
    if state_generation < 1 or event_id is None:
        raise ValueError(
            "a nonzero event cursor requires state_generation and content-addressed event_id"
        )
    return ResearchInternEventCursor(
        session_id=session_id,
        after_sequence=after_sequence,
        state_generation=state_generation,
        event_id=event_id,
        last_event_id=str(after_sequence),
    )


def build_research_intern_tools(
    client_from_args: CoreClientFactory,
) -> list[ToolDefinition]:
    """Build the stable, typed Intern tools over the shared core client."""

    def provision(args: JSONDict) -> JSONDict:
        request = ResearchInternProvisionRequest.model_validate(
            _request_payload(
                args,
                ("display_name", "policies", "attribution_team_id", "metadata"),
            )
        )
        with client_from_args(args) as client:
            return client.intern.provision(request).to_wire()

    def retrieve(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.retrieve().to_wire()

    def intern_sync_create(args: JSONDict) -> JSONDict:
        request = InternSyncSessionCreateRequest.model_validate(
            _request_payload(
                args,
                ("objective", "idempotency_key", "binding", "metadata"),
            )
        )
        with client_from_args(args) as client:
            return client.intern.sync_.create(request).to_wire()

    def intern_sync_list(args: JSONDict) -> list[JSONDict]:
        with client_from_args(args) as client:
            return _wire_list(client.intern.sync_.list(limit=optional_int(args, "limit") or 100))

    def intern_sync_get(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.sync_.get(require_string(args, "sync_session_id")).to_wire()

    def intern_sync_command(args: JSONDict) -> JSONDict:
        request = InternSyncCommandRequest.model_validate(
            _request_payload(
                args,
                (
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "command_kind",
                    "payload",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.sync_.command(
                require_string(args, "sync_session_id"),
                request,
            ).to_wire()

    def intern_sync_send(args: JSONDict) -> JSONDict:
        context = args.get("context")
        if context is not None and not isinstance(context, dict):
            raise ValueError("'context' must be an object when provided")
        with client_from_args(args) as client:
            return client.intern.sync_.send_message(
                require_string(args, "sync_session_id"),
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                body=require_string(args, "body"),
                turn_id=optional_string(args, "turn_id"),
                context=context,
            ).to_wire()

    def intern_sync_intervene(args: JSONDict) -> JSONDict:
        context = args.get("context")
        state_patch = args.get("state_patch")
        if context is not None and not isinstance(context, dict):
            raise ValueError("'context' must be an object when provided")
        if state_patch is not None and not isinstance(state_patch, dict):
            raise ValueError("'state_patch' must be an object when provided")
        with client_from_args(args) as client:
            return client.intern.sync_.intervene(
                require_string(args, "sync_session_id"),
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                body=require_string(args, "body"),
                turn_id=optional_string(args, "turn_id"),
                state_patch=state_patch,
                context=context,
            ).to_wire()

    def intern_sync_answer(args: JSONDict) -> JSONDict:
        context = args.get("context")
        if context is not None and not isinstance(context, dict):
            raise ValueError("'context' must be an object when provided")
        with client_from_args(args) as client:
            return client.intern.sync_.answer_interaction(
                require_string(args, "sync_session_id"),
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                interaction_id=require_string(args, "interaction_id"),
                answer=require_string(args, "answer"),
                context=context,
            ).to_wire()

    def intern_sync_pause(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.sync_.pause(
                require_string(args, "sync_session_id"),
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                rationale=require_string(args, "rationale"),
            ).to_wire()

    def intern_sync_resume(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.sync_.resume(
                require_string(args, "sync_session_id"),
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
            ).to_wire()

    def intern_sync_close(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.sync_.close(
                require_string(args, "sync_session_id"),
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                rationale=require_string(args, "rationale"),
                outcome=InternRuntimeOutcome(
                    optional_string(args, "outcome") or InternRuntimeOutcome.COMPLETED.value
                ),
            ).to_wire()

    def intern_sync_events(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.sync_.events(
                require_string(args, "sync_session_id"),
                after_sequence=optional_int(args, "after_sequence") or 0,
                limit=optional_int(args, "limit") or 100,
            ).to_wire()

    def intern_sync_tail(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.sync_.tail(
                require_string(args, "sync_session_id"),
                after_sequence=optional_int(args, "after_sequence") or 0,
                event_count_max=optional_int(args, "event_count_max") or 1,
                timeout_seconds=_optional_number_default(
                    args,
                    "timeout_seconds",
                    30.0,
                ),
            ).to_wire()

    def intern_async_ensure(args: JSONDict) -> JSONDict:
        request = InternAsyncEnsureRequest.model_validate(
            _request_payload(
                args,
                (
                    "objective",
                    "idempotency_key",
                    "binding",
                    "budget",
                    "metadata",
                    "factory_ready_wait_seconds",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.async_.ensure(request).to_wire()

    def intern_async_get(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.async_.get().to_wire()

    def intern_async_command(args: JSONDict) -> JSONDict:
        request = InternAsyncCommandRequest.model_validate(
            _request_payload(
                args,
                (
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "command_kind",
                    "payload",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.async_.command(request).to_wire()

    def intern_async_send(args: JSONDict) -> JSONDict:
        request = InternAsyncInstructionRequest.model_validate(
            {
                **_request_payload(
                    args,
                    (
                        "command_id",
                        "idempotency_key",
                        "expected_generation",
                        "body",
                        "context",
                    ),
                ),
                "instruction_kind": optional_string(args, "instruction_kind")
                or InternAsyncInstructionKind.MESSAGE.value,
            }
        )
        with client_from_args(args) as client:
            return client.intern.async_.send(request).to_wire()

    def intern_async_events(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.async_.events(
                after_sequence=optional_int(args, "after_sequence") or 0,
                limit=optional_int(args, "limit") or 100,
            ).to_wire()

    def intern_async_tail(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.async_.tail(
                after_sequence=optional_int(args, "after_sequence") or 0,
                event_count_max=optional_int(args, "event_count_max") or 1,
                timeout_seconds=_optional_number_default(
                    args,
                    "timeout_seconds",
                    30.0,
                ),
            ).to_wire()

    def intern_async_pause(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.async_.pause(
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                reason=require_string(args, "reason"),
            ).to_wire()

    def intern_async_resume(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.async_.resume(
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
            ).to_wire()

    def intern_async_cancel(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.async_.cancel(
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                reason=require_string(args, "reason"),
            ).to_wire()

    def intern_async_provide_input(args: JSONDict) -> JSONDict:
        context = args.get("context")
        if context is not None and not isinstance(context, dict):
            raise ValueError("'context' must be an object when provided")
        with client_from_args(args) as client:
            return client.intern.async_.provide_input(
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                interaction_id=require_string(args, "interaction_id"),
                body=require_string(args, "body"),
                context=context,
            ).to_wire()

    def intern_async_intervene(args: JSONDict) -> JSONDict:
        context = args.get("context")
        if context is not None and not isinstance(context, dict):
            raise ValueError("'context' must be an object when provided")
        with client_from_args(args) as client:
            return client.intern.async_.intervene(
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                body=require_string(args, "body"),
                context=context,
            ).to_wire()

    def intern_async_redirect_objective(args: JSONDict) -> JSONDict:
        context = args.get("context")
        if context is not None and not isinstance(context, dict):
            raise ValueError("'context' must be an object when provided")
        with client_from_args(args) as client:
            return client.intern.async_.redirect_objective(
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                objective=require_string(args, "objective"),
                context=context,
            ).to_wire()

    def intern_async_request_checkpoint(args: JSONDict) -> JSONDict:
        context = args.get("context")
        if context is not None and not isinstance(context, dict):
            raise ValueError("'context' must be an object when provided")
        with client_from_args(args) as client:
            return client.intern.async_.request_checkpoint(
                command_id=require_string(args, "command_id"),
                idempotency_key=require_string(args, "idempotency_key"),
                expected_generation=require_int(args, "expected_generation"),
                context=context,
            ).to_wire()

    def update(args: JSONDict) -> JSONDict:
        request = ResearchInternPatchRequest.model_validate(
            _request_payload(
                args,
                ("display_name", "status", "policies", "attribution_team_id", "metadata"),
            )
        )
        with client_from_args(args) as client:
            return client.intern.update(request).to_wire()

    def attach_factory(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.factories.attach(require_string(args, "factory_id")).to_wire()

    def list_factories(args: JSONDict) -> list[JSONDict]:
        with client_from_args(args) as client:
            return _wire_list(client.intern.factories.list())

    def create_session(args: JSONDict) -> JSONDict:
        request = ResearchInternSessionCreateRequest.model_validate(
            _request_payload(
                args,
                (
                    "factory_id",
                    "project_id",
                    "effort_id",
                    "run_id",
                    "objective",
                    "objective_bounds",
                    "idempotency_key",
                    "metadata",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.sessions.create(request).to_wire()

    def list_sessions(args: JSONDict) -> list[JSONDict]:
        limit = optional_int(args, "limit")
        with client_from_args(args) as client:
            return _wire_list(client.intern.sessions.list(limit=100 if limit is None else limit))

    def get_session(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.sessions.retrieve(require_string(args, "session_id")).to_wire()

    def append_event(args: JSONDict) -> JSONDict:
        request = ResearchInternEventAppendRequest.model_validate(
            _request_payload(
                args,
                (
                    "event_kind",
                    "mode",
                    "idempotency_key",
                    "expected_state_generation",
                    "body",
                    "payload",
                    "evidence_refs",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.sessions.append_event(
                require_string(args, "session_id"),
                request,
            ).to_wire()

    def list_events(args: JSONDict) -> list[JSONDict]:
        after_sequence = optional_int(args, "after_sequence")
        limit = optional_int(args, "limit")
        with client_from_args(args) as client:
            return _wire_list(
                client.intern.sessions.list_events(
                    require_string(args, "session_id"),
                    after_sequence=after_sequence or 0,
                    limit=100 if limit is None else limit,
                )
            )

    def watch_events(args: JSONDict) -> JSONDict:
        cursor = _event_cursor(args)
        with client_from_args(args) as client:
            return client.intern.sessions.watch(
                cursor.session_id,
                cursor=cursor,
                event_count_max=optional_int(args, "event_count_max") or 1,
                frame_count_max=optional_int(args, "frame_count_max") or 100,
                reconnect_count_max=optional_int(args, "reconnect_count_max") or 3,
                timeout_seconds=_optional_number_default(
                    args,
                    "timeout_seconds",
                    30.0,
                ),
            ).to_wire()

    def sync_session(args: JSONDict) -> JSONDict:
        limit = optional_int(args, "limit")
        with client_from_args(args) as client:
            return client.intern.sessions.sync(
                require_string(args, "session_id"),
                limit=200 if limit is None else limit,
            ).to_wire()

    def run_turn(args: JSONDict) -> JSONDict:
        request = ResearchInternTurnRequest.model_validate(
            _request_payload(
                args,
                (
                    "body",
                    "mode",
                    "idempotency_key",
                    "expected_session_state_generation",
                    "expected_intern_state_generation",
                    "control",
                    "rationale",
                    "state_patch",
                    "evidence_refs",
                    "wait_timeout_seconds",
                    "poll_interval_ms",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.sessions.turn(
                require_string(args, "session_id"),
                request,
            ).to_wire()

    def exchange_turn(args: JSONDict) -> JSONDict:
        cursor = _event_cursor(args)
        state_patch = args.get("state_patch")
        if state_patch is not None and not isinstance(state_patch, dict):
            raise ValueError("'state_patch' must be an object when provided")
        evidence_refs = args.get("evidence_refs")
        if evidence_refs is not None and (
            not isinstance(evidence_refs, list)
            or not all(isinstance(value, str) for value in evidence_refs)
        ):
            raise ValueError("'evidence_refs' must be an array of strings")
        with client_from_args(args) as client:
            reactive = client.intern.sessions.connect(
                cursor.session_id,
                cursor=cursor,
            )
            return reactive.exchange(
                require_string(args, "body"),
                idempotency_key=require_string(args, "idempotency_key"),
                mode=MagiMode(optional_string(args, "mode") or MagiMode.SYNC.value),
                control=optional_string(args, "control"),
                rationale=optional_string(args, "rationale"),
                state_patch=state_patch,
                evidence_refs=evidence_refs,
                wait_timeout_seconds=_optional_number_default(
                    args,
                    "wait_timeout_seconds",
                    15.0,
                ),
                poll_interval_ms=optional_int(args, "poll_interval_ms") or 250,
                recovery_timeout_seconds=_optional_number_default(
                    args,
                    "recovery_timeout_seconds",
                    60.0,
                ),
            ).to_wire()

    def close_session(args: JSONDict) -> JSONDict:
        request = ResearchInternSessionCloseRequest.model_validate(
            _request_payload(
                args,
                (
                    "idempotency_key",
                    "expected_state_generation",
                    "status",
                    "rationale",
                    "evidence_refs",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.sessions.close(
                require_string(args, "session_id"),
                request,
            ).to_wire()

    def publish_trace(args: JSONDict) -> JSONDict:
        request = ResearchInternTracePublicationRequest.model_validate(
            _request_payload(
                args,
                ("idempotency_key", "expected_state_generation"),
            )
        )
        with client_from_args(args) as client:
            return client.intern.sessions.publish_trace(
                require_string(args, "session_id"),
                request,
            ).to_wire()

    def record_decision(args: JSONDict) -> JSONDict:
        request = MagiDecisionRequest.model_validate(
            _request_payload(
                args,
                (
                    "mode",
                    "decision_kind",
                    "idempotency_key",
                    "factory_id",
                    "project_id",
                    "effort_id",
                    "run_id",
                    "session_id",
                    "expected_state_generation",
                    "experiment_id",
                    "evidence_refs",
                    "state_patch",
                    "rationale",
                    "verdict",
                    "uncertainty",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.decisions.record(request).to_wire()

    def list_decisions(args: JSONDict) -> list[JSONDict]:
        limit = optional_int(args, "limit")
        with client_from_args(args) as client:
            return _wire_list(client.intern.decisions.list(limit=100 if limit is None else limit))

    def get_decision(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.decisions.retrieve(require_string(args, "receipt_id")).to_wire()

    def publish_acceptance_receipt(args: JSONDict) -> JSONDict:
        request = ResearchInternAcceptanceReceiptPublicationRequest.model_validate(
            _request_payload(
                args,
                ("schema_version", "receipt_id", "lane", "candidate_id", "receipt"),
            )
        )
        with client_from_args(args) as client:
            return client.intern.acceptance_receipts.publish(request).to_wire()

    def get_acceptance_receipt(args: JSONDict) -> JSONDict:
        with client_from_args(args) as client:
            return client.intern.acceptance_receipts.retrieve(
                require_string(args, "receipt_id")
            ).to_wire()

    def list_acceptance_receipts(args: JSONDict) -> list[JSONDict]:
        limit = optional_int(args, "limit")
        with client_from_args(args) as client:
            return _wire_list(
                client.intern.acceptance_receipts.list(
                    candidate_id=optional_string(args, "candidate_id"),
                    lane=optional_string(args, "lane"),
                    limit=100 if limit is None else limit,
                )
            )

    policies_schema: JSONDict = {
        "type": "object",
        "properties": {
            "organization": {"type": "object"},
            "team": {"type": "object"},
            "user": {"type": "object"},
            "organization_policy_ref": {"type": "string"},
            "team_policy_ref": {"type": "string"},
            "user_policy_ref": {"type": "string"},
        },
        "additionalProperties": False,
    }
    session_selector = {"session_id": {"type": "string", "minLength": 1}}
    event_cursor_properties: JSONDict = {
        **session_selector,
        "after_sequence": {
            "type": "integer",
            "minimum": 0,
            "maximum": _RESEARCH_INTERN_EVENT_SEQUENCE_MAX,
        },
        "state_generation": {"type": "integer", "minimum": 0},
        "event_id": {
            "type": "string",
            "pattern": "^sha256:[0-9a-f]{64}$",
        },
    }
    event_properties: JSONDict = {
        **session_selector,
        "event_kind": {
            "type": "string",
            "enum": ["operator_message", "progress", "state_snapshot", "error"],
        },
        "mode": {"type": "string", "enum": ["sync", "async", "seraph"]},
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 512},
        "expected_state_generation": {"type": "integer", "minimum": 0},
        "body": {"type": "string", "maxLength": 20000},
        "payload": {"type": "object"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    }
    decision_properties: JSONDict = {
        "mode": {"type": "string", "enum": ["sync", "async", "seraph"]},
        "decision_kind": {
            "type": "string",
            "enum": ["delegate", "inspect", "pause", "intervene", "resume", "verdict", "revise"],
        },
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 512},
        "factory_id": {"type": "string"},
        "project_id": {"type": "string"},
        "effort_id": {"type": "string"},
        "run_id": {"type": "string"},
        "session_id": {"type": "string"},
        "expected_state_generation": {"type": "integer", "minimum": 0},
        "experiment_id": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "state_patch": {"type": "object"},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 20000},
        "verdict": {"type": "string"},
        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
    }
    receipt_selector = {
        "receipt_id": {
            "type": "string",
            "pattern": "^sha256:[0-9a-f]{64}$",
        }
    }
    async_command_identity: JSONDict = {
        "command_id": {"type": "string", "minLength": 1, "maxLength": 512},
        "idempotency_key": {
            "type": "string",
            "minLength": 1,
            "maxLength": 512,
        },
        "expected_generation": {"type": "integer", "minimum": 0},
    }
    async_context = {"context": {"type": "object"}}
    async_event_cursor: JSONDict = {
        "after_sequence": {"type": "integer", "minimum": 0},
    }
    sync_selector: JSONDict = {
        "sync_session_id": {"type": "string", "minLength": 1},
    }
    tools = [
        ToolDefinition(
            name="intern_sync_create",
            description="Create or replay one durable operator-present Sync session.",
            input_schema=tool_schema(
                {
                    "objective": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "binding": {"type": "object"},
                    "metadata": {"type": "object"},
                },
                required=["objective", "idempotency_key"],
            ),
            handler=intern_sync_create,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_list",
            description="List a bounded page of durable Sync Intern sessions.",
            input_schema=tool_schema(
                {"limit": {"type": "integer", "minimum": 1, "maximum": 500}},
                required=[],
            ),
            handler=intern_sync_list,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_get",
            description="Get one authoritative Sync Intern session projection.",
            input_schema=tool_schema(sync_selector, required=["sync_session_id"]),
            handler=intern_sync_get,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_command",
            description="Submit one exact durable Sync command envelope.",
            input_schema=tool_schema(
                {
                    **sync_selector,
                    **async_command_identity,
                    "command_kind": {
                        "type": "string",
                        "enum": [value.value for value in InternSyncCommandKind],
                    },
                    "payload": {"type": "object"},
                },
                required=[
                    "sync_session_id",
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "command_kind",
                ],
            ),
            handler=intern_sync_command,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_send",
            description="Send an operator message to one Sync Intern session.",
            input_schema=tool_schema(
                {
                    **sync_selector,
                    **async_command_identity,
                    "turn_id": {"type": "string", "minLength": 1},
                    "body": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "context": {"type": "object"},
                },
                required=[
                    "sync_session_id",
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "body",
                ],
            ),
            handler=intern_sync_send,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_intervene",
            description="Fence pending Sync work and replace it with operator direction.",
            input_schema=tool_schema(
                {
                    **sync_selector,
                    **async_command_identity,
                    "turn_id": {"type": "string", "minLength": 1},
                    "body": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "state_patch": {"type": "object"},
                    "context": {"type": "object"},
                },
                required=[
                    "sync_session_id",
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "body",
                ],
            ),
            handler=intern_sync_intervene,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_answer",
            description="Answer the exact pending Sync Intern interaction.",
            input_schema=tool_schema(
                {
                    **sync_selector,
                    **async_command_identity,
                    "interaction_id": {"type": "string", "minLength": 1},
                    "answer": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "context": {"type": "object"},
                },
                required=[
                    "sync_session_id",
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "interaction_id",
                    "answer",
                ],
            ),
            handler=intern_sync_answer,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_pause",
            description="Pause a Sync session and fence its pending effects.",
            input_schema=tool_schema(
                {
                    **sync_selector,
                    **async_command_identity,
                    "rationale": {"type": "string", "minLength": 1},
                },
                required=[
                    "sync_session_id",
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "rationale",
                ],
            ),
            handler=intern_sync_pause,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_resume",
            description="Resume a paused Sync Intern session.",
            input_schema=tool_schema(
                {**sync_selector, **async_command_identity},
                required=[
                    "sync_session_id",
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                ],
            ),
            handler=intern_sync_resume,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_close",
            description="Close a Sync session while retaining its durable ledger.",
            input_schema=tool_schema(
                {
                    **sync_selector,
                    **async_command_identity,
                    "outcome": {
                        "type": "string",
                        "enum": [value.value for value in InternRuntimeOutcome],
                    },
                    "rationale": {"type": "string", "minLength": 1},
                },
                required=[
                    "sync_session_id",
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "rationale",
                ],
            ),
            handler=intern_sync_close,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_events",
            description="Replay a bounded contiguous Sync Intern event page.",
            input_schema=tool_schema(
                {
                    **sync_selector,
                    **async_event_cursor,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                required=["sync_session_id"],
            ),
            handler=intern_sync_events,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_tail",
            description="Wait for a bounded number of Sync Intern SSE events.",
            input_schema=tool_schema(
                {
                    **sync_selector,
                    **async_event_cursor,
                    "event_count_max": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 300,
                    },
                },
                required=["sync_session_id"],
            ),
            handler=intern_sync_tail,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_ensure",
            description=("Ensure and return the organization's one durable Async Intern runtime."),
            input_schema=tool_schema(
                {
                    "objective": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 20000,
                    },
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "binding": {"type": "object"},
                    "budget": {"type": "object"},
                    "metadata": {"type": "object"},
                    "factory_ready_wait_seconds": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 60,
                        "description": (
                            "Bounded wait for the explicit Factory-ready "
                            "condition before binding; omit to refuse "
                            "immediately with a typed readiness report."
                        ),
                    },
                },
                required=["objective", "idempotency_key"],
            ),
            handler=intern_async_ensure,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_get",
            description="Get the authoritative projection of the org Async Intern.",
            input_schema=tool_schema({}, required=[]),
            handler=intern_async_get,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_command",
            description="Submit one exact durable command envelope and return its receipt.",
            input_schema=tool_schema(
                {
                    **async_command_identity,
                    "command_kind": {
                        "type": "string",
                        "enum": [value.value for value in InternAsyncCommandKind],
                    },
                    "payload": {"type": "object"},
                },
                required=[
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "command_kind",
                ],
            ),
            handler=intern_async_command,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_send",
            description="Send a durable message or typed instruction to the Async Intern.",
            input_schema=tool_schema(
                {
                    **async_command_identity,
                    "instruction_kind": {
                        "type": "string",
                        "enum": [value.value for value in InternAsyncInstructionKind],
                    },
                    "body": {"type": "string", "minLength": 1, "maxLength": 20000},
                    **async_context,
                },
                required=[
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "body",
                ],
            ),
            handler=intern_async_send,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_events",
            description="Replay a bounded contiguous Async Intern event page.",
            input_schema=tool_schema(
                {
                    **async_event_cursor,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                required=[],
            ),
            handler=intern_async_events,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_tail",
            description="Wait for a bounded number of Async Intern SSE events.",
            input_schema=tool_schema(
                {
                    **async_event_cursor,
                    "event_count_max": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 300,
                    },
                },
                required=[],
            ),
            handler=intern_async_tail,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_pause",
            description="Pause the Async Intern and fence older pending effects.",
            input_schema=tool_schema(
                {
                    **async_command_identity,
                    "reason": {"type": "string", "minLength": 1},
                },
                required=[
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "reason",
                ],
            ),
            handler=intern_async_pause,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_resume",
            description="Resume the paused Async Intern from its durable state.",
            input_schema=tool_schema(
                async_command_identity,
                required=["command_id", "idempotency_key", "expected_generation"],
            ),
            handler=intern_async_resume,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_cancel",
            description="Cancel the Async Intern's current work durably.",
            input_schema=tool_schema(
                {
                    **async_command_identity,
                    "reason": {"type": "string", "minLength": 1},
                },
                required=[
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "reason",
                ],
            ),
            handler=intern_async_cancel,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_provide_input",
            description="Answer the exact pending Async Intern interaction.",
            input_schema=tool_schema(
                {
                    **async_command_identity,
                    "interaction_id": {"type": "string", "minLength": 1},
                    "body": {"type": "string", "minLength": 1, "maxLength": 20000},
                    **async_context,
                },
                required=[
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "interaction_id",
                    "body",
                ],
            ),
            handler=intern_async_provide_input,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_intervene",
            description="Fence older work and steer one replacement Async cycle.",
            input_schema=tool_schema(
                {
                    **async_command_identity,
                    "body": {"type": "string", "minLength": 1, "maxLength": 20000},
                    **async_context,
                },
                required=[
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "body",
                ],
            ),
            handler=intern_async_intervene,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_redirect_objective",
            description="Replace the Async Intern objective and start replanning.",
            input_schema=tool_schema(
                {
                    **async_command_identity,
                    "objective": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 20000,
                    },
                    **async_context,
                },
                required=[
                    "command_id",
                    "idempotency_key",
                    "expected_generation",
                    "objective",
                ],
            ),
            handler=intern_async_redirect_objective,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="intern_async_request_checkpoint",
            description="Ask the next bounded Async cycle to yield a checkpoint.",
            input_schema=tool_schema(
                {**async_command_identity, **async_context},
                required=["command_id", "idempotency_key", "expected_generation"],
            ),
            handler=intern_async_request_checkpoint,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_provision_research_intern",
            description="Provision or replay the organization Research Intern.",
            input_schema=tool_schema(
                {
                    "display_name": {"type": "string", "minLength": 1},
                    "policies": policies_schema,
                    "attribution_team_id": {"type": ["string", "null"]},
                    "metadata": {"type": "object"},
                },
                required=[],
            ),
            handler=provision,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_get_research_intern",
            description="Retrieve the organization Research Intern and current generation.",
            input_schema=tool_schema({}, required=[]),
            handler=retrieve,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_update_research_intern",
            description="Update explicit mutable Research Intern fields.",
            input_schema=tool_schema(
                {
                    "display_name": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "enum": ["active", "paused", "archived"]},
                    "policies": policies_schema,
                    "attribution_team_id": {"type": ["string", "null"]},
                    "metadata": {"type": ["object", "null"]},
                },
                required=[],
            ),
            handler=update,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_attach_research_intern_factory",
            description="Attach the organization Research Intern to one Factory.",
            input_schema=tool_schema(
                {"factory_id": {"type": "string", "minLength": 1}},
                required=["factory_id"],
            ),
            handler=attach_factory,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_list_research_intern_factories",
            description="List Factory memberships for the organization Research Intern.",
            input_schema=tool_schema({}, required=[]),
            handler=list_factories,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_create_research_intern_session",
            description="DEPRECATED: prefer intern_sync_* / intern_async_*. Create or replay one durable Research Intern session.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string", "minLength": 1},
                    "project_id": {"type": "string", "minLength": 1},
                    "effort_id": {"type": "string", "minLength": 1},
                    "run_id": {"type": "string"},
                    "objective": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "objective_bounds": {"type": "object"},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "metadata": {"type": "object"},
                },
                required=[
                    "factory_id",
                    "project_id",
                    "effort_id",
                    "objective",
                    "idempotency_key",
                ],
            ),
            handler=create_session,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_list_research_intern_sessions",
            description="DEPRECATED: prefer intern_sync_* / intern_async_*. List a bounded page of Research Intern sessions.",
            input_schema=tool_schema(
                {"limit": {"type": "integer", "minimum": 1, "maximum": 500}},
                required=[],
            ),
            handler=list_sessions,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_get_research_intern_session",
            description="DEPRECATED: prefer intern_sync_* / intern_async_*. Retrieve one durable Research Intern session.",
            input_schema=tool_schema(session_selector, required=["session_id"]),
            handler=get_session,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_append_research_intern_event",
            description="DEPRECATED: prefer intern_sync_* / intern_async_*. Append one generation-fenced session event.",
            input_schema=tool_schema(
                event_properties,
                required=[
                    "session_id",
                    "event_kind",
                    "idempotency_key",
                    "expected_state_generation",
                ],
            ),
            handler=append_event,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_list_research_intern_events",
            description="DEPRECATED: prefer intern_sync_* / intern_async_*. Read a bounded, reconnectable page of ordered Intern events.",
            input_schema=tool_schema(
                {
                    **session_selector,
                    "after_sequence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": _RESEARCH_INTERN_EVENT_SEQUENCE_MAX,
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                required=["session_id"],
            ),
            handler=list_events,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_watch_research_intern_events",
            description=(
                "DEPRECATED: prefer intern_sync_* / intern_async_*. Wait for typed Intern SSE frames with explicit event, frame, "
                "reconnect, and time bounds."
            ),
            input_schema=tool_schema(
                {
                    **event_cursor_properties,
                    "event_count_max": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "frame_count_max": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5000,
                    },
                    "reconnect_count_max": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 300,
                    },
                },
                required=["session_id"],
            ),
            handler=watch_events,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_sync_research_intern_session",
            description="DEPRECATED: prefer intern_sync_* / intern_async_*. Project a bounded page of real runtime transcript events.",
            input_schema=tool_schema(
                {
                    **session_selector,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                required=["session_id"],
            ),
            handler=sync_session,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_run_research_intern_turn",
            description=(
                "DEPRECATED: prefer intern_sync_* / intern_async_*. Submit one operator turn to the bound real runtime and return "
                "the canonical bounded reply projection."
            ),
            input_schema=tool_schema(
                {
                    **session_selector,
                    "body": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "mode": {"type": "string", "enum": ["sync", "async", "seraph"]},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "expected_session_state_generation": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "expected_intern_state_generation": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "control": {
                        "type": "string",
                        "enum": ["pause", "intervene", "resume"],
                    },
                    "rationale": {"type": "string", "maxLength": 20000},
                    "state_patch": {"type": "object"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "wait_timeout_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 30,
                    },
                    "poll_interval_ms": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 2000,
                    },
                },
                required=[
                    "session_id",
                    "body",
                    "idempotency_key",
                    "expected_session_state_generation",
                    "expected_intern_state_generation",
                ],
            ),
            handler=run_turn,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_exchange_research_intern_turn",
            description=(
                "DEPRECATED: prefer intern_sync_* / intern_async_*. Submit one Intern turn and recover only its exact terminal "
                "agent/error event over the canonical stream."
            ),
            input_schema=tool_schema(
                {
                    **event_cursor_properties,
                    "body": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 20000,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["sync", "async", "seraph"],
                    },
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "control": {
                        "type": "string",
                        "enum": ["pause", "intervene", "resume"],
                    },
                    "rationale": {"type": "string", "maxLength": 20000},
                    "state_patch": {"type": "object"},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "wait_timeout_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 30,
                    },
                    "poll_interval_ms": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 2000,
                    },
                    "recovery_timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 300,
                    },
                },
                required=["session_id", "body", "idempotency_key"],
            ),
            handler=exchange_turn,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_close_research_intern_session",
            description="DEPRECATED: prefer intern_sync_* / intern_async_*. Close one exact Intern session generation without deleting evidence.",
            input_schema=tool_schema(
                {
                    **session_selector,
                    "idempotency_key": {"type": "string", "minLength": 1},
                    "expected_state_generation": {"type": "integer", "minimum": 0},
                    "status": {
                        "type": "string",
                        "enum": [
                            "completed",
                            "partial",
                            "failed",
                            "stopped",
                            "canceled",
                            "archived",
                        ],
                    },
                    "rationale": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                required=[
                    "session_id",
                    "idempotency_key",
                    "expected_state_generation",
                    "status",
                    "rationale",
                ],
            ),
            handler=close_session,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_publish_research_intern_session_trace",
            description=(
                "DEPRECATED: prefer intern_sync_* / intern_async_*. Publish one terminal Intern event chain through backend-owned "
                "Factory Trace V5 authority."
            ),
            input_schema=tool_schema(
                {
                    **session_selector,
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "expected_state_generation": {
                        "type": "integer",
                        "minimum": 1,
                    },
                },
                required=[
                    "session_id",
                    "idempotency_key",
                    "expected_state_generation",
                ],
            ),
            handler=publish_trace,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_record_research_intern_decision",
            description="Record one typed Magi decision with runtime actuation receipts.",
            input_schema=tool_schema(
                decision_properties,
                required=["mode", "decision_kind", "idempotency_key", "rationale"],
            ),
            handler=record_decision,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_list_research_intern_decisions",
            description="List a bounded page of content-addressed Magi decisions.",
            input_schema=tool_schema(
                {"limit": {"type": "integer", "minimum": 1, "maximum": 500}},
                required=[],
            ),
            handler=list_decisions,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_get_research_intern_decision",
            description="Retrieve one content-addressed Magi decision receipt.",
            input_schema=tool_schema(receipt_selector, required=["receipt_id"]),
            handler=get_decision,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_publish_research_intern_acceptance_receipt",
            description="Publish or replay one deterministic FactoryBench acceptance receipt.",
            input_schema=tool_schema(
                {
                    "schema_version": {"type": "string"},
                    **receipt_selector,
                    "lane": {
                        "type": "string",
                        "pattern": "^[a-z0-9][a-z0-9._-]{0,127}$",
                    },
                    "candidate_id": {"type": "string", "minLength": 1},
                    "receipt": {"type": "object"},
                },
                required=["receipt_id", "lane", "candidate_id", "receipt"],
            ),
            handler=publish_acceptance_receipt,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_get_research_intern_acceptance_receipt",
            description="Retrieve one public content-addressed acceptance receipt.",
            input_schema=tool_schema(receipt_selector, required=["receipt_id"]),
            handler=get_acceptance_receipt,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="research_list_research_intern_acceptance_receipts",
            description="List bounded acceptance receipts with candidate and lane filters.",
            input_schema=tool_schema(
                {
                    "candidate_id": {"type": "string"},
                    "lane": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                required=[],
            ),
            handler=list_acceptance_receipts,
            required_scopes=READ_SCOPES,
        ),
    ]
    if not legacy_intern_sessions_enabled():
        tools = [tool for tool in tools if tool.name not in LEGACY_INTERN_SESSION_TOOL_NAMES]
    return tools


__all__ = ["LEGACY_INTERN_SESSION_TOOL_NAMES", "build_research_intern_tools"]
