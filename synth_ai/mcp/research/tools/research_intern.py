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
    InternCrossMetaThreadMessageCreateRequest,
    InternRuntimeOutcome,
    InternSyncCommandKind,
    InternSyncCommandRequest,
    InternSyncSessionCreateRequest,
    MagiDecisionRequest,
    ResearchInternAcceptanceReceiptPublicationRequest,
    ResearchInternPatchRequest,
    ResearchInternProvisionRequest,
)

CoreClientFactory = Callable[[JSONDict], ResearchClient]
_RESEARCH_INTERN_EVENT_SEQUENCE_MAX = 2**31 - 1

# Retired MCP tools for the removed `/smr/research-intern/sessions` plane.
# Kept as a named set so CI can assert they stay unregistered even if someone
# reintroduces the opt-in flag. They are never added to the tool list.
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
            return _wire_list(client.intern.sync_.branches())

    def intern_sync_branches(args: JSONDict) -> list[JSONDict]:
        with client_from_args(args) as client:
            return _wire_list(client.intern.sync_.branches())

    def intern_meta_threads(args: JSONDict) -> list[JSONDict]:
        with client_from_args(args) as client:
            return _wire_list(client.intern.meta_threads.list())

    def intern_meta_segments(args: JSONDict) -> list[JSONDict]:
        with client_from_args(args) as client:
            return _wire_list(
                client.intern.meta_threads.segments(require_string(args, "meta_thread_id"))
            )

    def intern_meta_messages(args: JSONDict) -> list[JSONDict]:
        with client_from_args(args) as client:
            return _wire_list(
                client.intern.meta_threads.messages(
                    require_string(args, "meta_thread_id"),
                    limit=optional_int(args, "limit") or 200,
                )
            )

    def intern_meta_send(args: JSONDict) -> JSONDict:
        request = InternCrossMetaThreadMessageCreateRequest.model_validate(
            _request_payload(
                args,
                (
                    "message_id",
                    "source_meta_thread_id",
                    "destination_meta_thread_id",
                    "kind",
                    "idempotency_key",
                    "payload",
                    "linked_message_id",
                    "sync_session_id",
                    "segment_id",
                    "resolution",
                    "summary",
                ),
            )
        )
        with client_from_args(args) as client:
            return client.intern.meta_threads.send(request).to_wire()

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
            return client.intern.async_.ensure(
                request,
                maximum_daily_cost_cents=optional_int(args, "maximum_daily_cost_cents"),
                maximum_monthly_cost_cents=optional_int(args, "maximum_monthly_cost_cents"),
            ).to_wire()

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
            name="intern_meta_threads",
            description="List the Intern's one Sync and one Async meta-thread.",
            input_schema=tool_schema({}, required=[]),
            handler=intern_meta_threads,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_meta_segments",
            description="List head, live, and sealed segments for one meta-thread.",
            input_schema=tool_schema(
                {"meta_thread_id": {"type": "string", "minLength": 1}},
                required=["meta_thread_id"],
            ),
            handler=intern_meta_segments,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_meta_messages",
            description="List durable cross-lane messages visible to one meta-thread.",
            input_schema=tool_schema(
                {
                    "meta_thread_id": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                required=["meta_thread_id"],
            ),
            handler=intern_meta_messages,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_meta_send",
            description="Send one exact durable Sync-to-Async protocol object.",
            input_schema=tool_schema(
                {
                    "message_id": {"type": "string", "minLength": 1},
                    "source_meta_thread_id": {"type": "string", "minLength": 1},
                    "destination_meta_thread_id": {"type": "string", "minLength": 1},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "request_decision",
                            "open_branch_ack",
                            "decision_resolved",
                            "steer",
                            "note",
                        ],
                    },
                    "idempotency_key": {"type": "string", "minLength": 1},
                    "payload": {"type": "object"},
                    "linked_message_id": {"type": "string"},
                    "sync_session_id": {"type": "string"},
                    "segment_id": {"type": "string"},
                    "resolution": {
                        "type": "string",
                        "enum": ["completed", "denied", "superseded"],
                    },
                    "summary": {"type": "string", "maxLength": 4000},
                },
                required=[
                    "message_id",
                    "source_meta_thread_id",
                    "destination_meta_thread_id",
                    "kind",
                    "idempotency_key",
                ],
            ),
            handler=intern_meta_send,
            required_scopes=WRITE_SCOPES,
        ),
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
            name="intern_sync_branches",
            description="List the Sync head plus every live or sealed branch.",
            input_schema=tool_schema({}, required=[]),
            handler=intern_sync_branches,
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="intern_sync_list",
            description="Compatibility alias of intern_sync_branches.",
            input_schema=tool_schema({}, required=[]),
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
            description=(
                "Ensure and return the organization's one durable Async Intern runtime. "
                "Optional maximum_daily_cost_cents / maximum_monthly_cost_cents set spend "
                "ceilings (also accepted under budget); omitted values get server defaults."
            ),
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
                    "budget": {
                        "type": "object",
                        "description": (
                            "AsyncRuntimeBudget fields including maximum_cost_cents, "
                            "maximum_daily_cost_cents, maximum_monthly_cost_cents, "
                            "maximum_cycles, maximum_concurrent_runs."
                        ),
                    },
                    "maximum_daily_cost_cents": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Day spend ceiling in cents; overrides budget.",
                    },
                    "maximum_monthly_cost_cents": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Month spend ceiling in cents; overrides budget.",
                    },
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
            description=(
                "Pause the Async Intern, fence older pending effects, and free the "
                "sticky exe.dev host lease. Resume reacquires the lease."
            ),
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
    return tools


__all__ = ["LEGACY_INTERN_SESSION_TOOL_NAMES", "build_research_intern_tools"]
