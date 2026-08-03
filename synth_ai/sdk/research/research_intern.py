"""Research Intern, Magi decision, and project resource operations."""

from __future__ import annotations

import asyncio
import builtins
import os
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Literal, cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.errors import TimeoutError as SynthTimeoutError
from synth_ai.core.errors import TransientServiceError
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.streaming import SseEvent
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts._wire import array_value
from synth_ai.sdk.research.contracts.common import FactoryId, ProjectId
from synth_ai.sdk.research.contracts.dataset_revisions import (
    DatasetRevisionFinalizeRequest,
    DatasetRevisionFinalizeResponse,
    DatasetRevisionPreparationResponse,
    DatasetRevisionPrepareRequest,
)
from synth_ai.sdk.research.contracts.factory_role_receipts import (
    FactoryRoleReceiptMintRequest,
    FactoryRoleReceiptResponse,
)
from synth_ai.sdk.research.contracts.project_runtime import (
    ProjectComputerExecuteRequest,
    ProjectComputerInspectRequest,
    ProjectComputerLeaseAcquireRequest,
    ProjectComputerLeaseReleaseRequest,
    ProjectComputerLeaseRenewRequest,
    ProjectComputerLeaseResponse,
    ProjectComputerOperationReconcileRequest,
    ProjectRuntimeOperation,
    ProjectRuntimeOperationReceipt,
)
from synth_ai.sdk.research.contracts.research_intern import (
    DataBindingCreateRequest,
    DataBindingResponse,
    DatasetRevisionCreateRequest,
    DatasetRevisionLifecycleRequest,
    DatasetRevisionResponse,
    InternAsyncCommandKind,
    InternAsyncCommandReceipt,
    InternAsyncCommandRequest,
    InternAsyncEnsureRequest,
    InternAsyncEvent,
    InternAsyncEventPage,
    InternAsyncEventStreamEnvelope,
    InternAsyncInstructionKind,
    InternAsyncInstructionRequest,
    InternAsyncRuntime,
    InternRuntimeOutcome,
    InternSyncCommandKind,
    InternSyncCommandReceipt,
    InternSyncCommandRequest,
    InternSyncEvent,
    InternSyncEventPage,
    InternSyncEventStreamEnvelope,
    InternSyncSession,
    InternSyncSessionCreateRequest,
    MagiDecisionKind,
    MagiDecisionReceiptResponse,
    MagiDecisionRequest,
    MagiMode,
    ProjectComputerCleanupReceiptResponse,
    ProjectComputerCleanupRequest,
    ProjectComputerProvisionRequest,
    ProjectComputerReplaceRequest,
    ProjectComputerResponse,
    ResearchInternAcceptanceReceiptPublicationRequest,
    ResearchInternAcceptanceReceiptPublicationResponse,
    ResearchInternEventAppendRequest,
    ResearchInternEventKind,
    ResearchInternEventResponse,
    ResearchInternEventStreamEnvelope,
    ResearchInternEventStreamEvent,
    ResearchInternEventStreamHeartbeat,
    ResearchInternEventStreamPayload,
    ResearchInternFactoryMembershipResponse,
    ResearchInternPatchRequest,
    ResearchInternProvisionRequest,
    ResearchInternResponse,
    ResearchInternSessionCloseRequest,
    ResearchInternSessionCreateRequest,
    ResearchInternSessionResponse,
    ResearchInternSessionSyncResponse,
    ResearchInternTracePublicationRequest,
    ResearchInternTracePublicationResponse,
    ResearchInternTurnControl,
    ResearchInternTurnRequest,
    ResearchInternTurnResponse,
    ResearchInternTurnStatus,
)
from synth_ai.sdk.research.operations import (
    dataset_revision_publication_operation,
    research_operation,
)

_MONOTONIC = time.monotonic
_RESEARCH_INTERN_EVENT_SEQUENCE_MAX = 2**31 - 1


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
    )


def _dataset_publication_request(
    operation_id: str,
    path: str,
    *,
    body: JsonObject,
) -> HttpRequest:
    return HttpRequest(
        dataset_revision_publication_operation(operation_id),
        path,
        body=body,
    )


def _validate_operation_receipt(
    receipt: ProjectRuntimeOperationReceipt,
    request: (
        ProjectComputerInspectRequest
        | ProjectComputerExecuteRequest
        | ProjectComputerOperationReconcileRequest
    ),
    *,
    operation: ProjectRuntimeOperation | None,
) -> ProjectRuntimeOperationReceipt:
    if (
        receipt.operation_id != request.operation_id
        or receipt.idempotency_key != request.idempotency_key
        or receipt.resource_generation != request.expected_generation
    ):
        raise ValueError("Project Computer operation receipt identity drifted")
    if operation is not None and receipt.operation is not operation:
        raise ValueError("Project Computer operation receipt kind drifted")
    return receipt


def _memberships(value: object) -> tuple[ResearchInternFactoryMembershipResponse, ...]:
    return tuple(
        ResearchInternFactoryMembershipResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_research_intern_factories",
        )
    )


def _decisions(value: object) -> tuple[MagiDecisionReceiptResponse, ...]:
    return tuple(
        MagiDecisionReceiptResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_magi_decisions",
        )
    )


def _intern_async_event_page(
    value: object,
    *,
    after_sequence: int,
) -> InternAsyncEventPage:
    events = tuple(
        InternAsyncEvent.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_intern_async_runtime_events",
        )
    )
    expected_sequence = after_sequence + 1
    runtime_id: str | None = None
    for event in events:
        if event.sequence != expected_sequence:
            raise ValueError("Async Intern event page has a cursor gap")
        if runtime_id is not None and event.runtime_id != runtime_id:
            raise ValueError("Async Intern event page crossed runtime identity")
        runtime_id = event.runtime_id
        expected_sequence += 1
    return InternAsyncEventPage(
        events=events,
        next_sequence=events[-1].sequence if events else after_sequence,
    )


def _intern_sync_event_page(
    value: object,
    *,
    after_sequence: int,
    runtime_id: str,
) -> InternSyncEventPage:
    events = tuple(
        InternSyncEvent.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_intern_runtime_events",
        )
    )
    expected_sequence = after_sequence + 1
    for event in events:
        if event.sequence != expected_sequence:
            raise ValueError("Sync Intern event page has a cursor gap")
        if event.runtime_id != runtime_id:
            raise ValueError("Sync Intern event page crossed runtime identity")
        expected_sequence += 1
    return InternSyncEventPage(
        events=events,
        next_sequence=events[-1].sequence if events else after_sequence,
    )


def _intern_async_stream_event(
    frame: SseEvent,
    *,
    expected_sequence: int,
    runtime_id: str | None,
) -> InternAsyncEvent:
    envelope = InternAsyncEventStreamEnvelope.from_wire(frame.json_data())
    event = envelope.event
    if (
        event.sequence != expected_sequence
        or frame.event_id != str(event.sequence)
        or frame.event != event.event_kind
    ):
        raise ValueError("Async Intern SSE cursor or event identity drifted")
    if runtime_id is not None and event.runtime_id != runtime_id:
        raise ValueError("Async Intern SSE crossed runtime identity")
    return event


def _intern_sync_stream_event(
    frame: SseEvent,
    *,
    expected_sequence: int,
    runtime_id: str,
) -> InternSyncEvent:
    envelope = InternSyncEventStreamEnvelope.from_wire(frame.json_data())
    event = envelope.event
    if (
        event.sequence != expected_sequence
        or frame.event_id != str(event.sequence)
        or frame.event != event.event_kind
    ):
        raise ValueError("Sync Intern SSE cursor or event identity drifted")
    if event.runtime_id != runtime_id:
        raise ValueError("Sync Intern SSE crossed runtime identity")
    return event


def _sessions(value: object) -> tuple[ResearchInternSessionResponse, ...]:
    return tuple(
        ResearchInternSessionResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_research_intern_sessions",
        )
    )


def _events(value: object) -> tuple[ResearchInternEventResponse, ...]:
    return tuple(
        ResearchInternEventResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_research_intern_events",
        )
    )


def _acceptance_receipts(
    value: object,
) -> tuple[ResearchInternAcceptanceReceiptPublicationResponse, ...]:
    return tuple(
        ResearchInternAcceptanceReceiptPublicationResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_research_intern_acceptance_receipts",
        )
    )


def _bounded_limit(limit: int, *, maximum: int = 500) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return limit


def _digest_hex(value: str) -> str:
    normalized = value.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("receipt digest must be a sha256 digest")
    return normalized


def _role_receipts(value: object) -> tuple[FactoryRoleReceiptResponse, ...]:
    return tuple(
        FactoryRoleReceiptResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_factory_role_receipts",
        )
    )


def _data_bindings(value: object) -> tuple[DataBindingResponse, ...]:
    return tuple(
        DataBindingResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_data_bindings",
        )
    )


def _dataset_revisions(value: object) -> tuple[DatasetRevisionResponse, ...]:
    return tuple(
        DatasetRevisionResponse.from_wire(item)
        for item in array_value(
            cast(JsonValue, value),
            operation_id="list_dataset_revisions",
        )
    )


class ResearchInternFactoriesAPI:
    """Factory memberships owned by the organization Research Intern."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def attach(self, factory_id: FactoryId) -> ResearchInternFactoryMembershipResponse:
        """Attach one Factory to the organization Research Intern."""
        value = self._transport.execute(
            _request(
                "attach_research_intern_factory",
                f"/smr/research-intern/factories/{factory_id}",
            )
        )
        membership = ResearchInternFactoryMembershipResponse.from_wire(value)
        if membership.factory_id != str(factory_id):
            raise ValueError("Research Intern Factory membership identity drifted")
        return membership

    def list(self) -> tuple[ResearchInternFactoryMembershipResponse, ...]:
        """List Factory memberships for the organization Research Intern."""
        return _memberships(
            self._transport.execute(
                _request(
                    "list_research_intern_factories",
                    "/smr/research-intern/factories",
                )
            )
        )

    def mint_role_receipt(
        self,
        factory_id: FactoryId,
        request: FactoryRoleReceiptMintRequest,
    ) -> FactoryRoleReceiptResponse:
        """Mint one idempotent Factory role receipt from owner runtime evidence."""
        receipt = FactoryRoleReceiptResponse.from_wire(
            self._transport.execute(
                _request(
                    "mint_factory_role_receipt",
                    f"/smr/research-intern/factories/{factory_id}/role-receipts",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.factory_id != str(factory_id)
            or receipt.idempotency_key != request.idempotency_key
            or receipt.candidate_id != request.candidate_id
            or receipt.role != request.role
            or receipt.runtime_evidence.run_id != request.run_id
        ):
            raise ValueError("Factory role receipt identity drifted")
        return receipt

    def list_role_receipts(
        self,
        factory_id: FactoryId,
        candidate_id: str | None = None,
    ) -> tuple[FactoryRoleReceiptResponse, ...]:
        """List durable Factory role receipts, optionally for one candidate."""
        query: JsonObject = {}
        if candidate_id is not None:
            query["candidate_id"] = candidate_id
        receipts = _role_receipts(
            self._transport.execute(
                _request(
                    "list_factory_role_receipts",
                    f"/smr/research-intern/factories/{factory_id}/role-receipts",
                    query=query,
                )
            )
        )
        if any(
            receipt.factory_id != str(factory_id)
            or (candidate_id is not None and receipt.candidate_id != candidate_id)
            for receipt in receipts
        ):
            raise ValueError("Factory role receipt list crossed its requested boundary")
        return receipts


class ResearchInternDecisionsAPI:
    """Durable Casper, Melchior, and Balthasar decision receipts."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def record(self, request: MagiDecisionRequest) -> MagiDecisionReceiptResponse:
        """Record one idempotent, evidence-linked Magi decision."""
        value = self._transport.execute(
            _request(
                "record_magi_decision",
                "/smr/research-intern/decisions",
                body=cast(JsonObject, request.to_wire()),
            )
        )
        receipt = MagiDecisionReceiptResponse.from_wire(value)
        if (
            receipt.idempotency_key != request.idempotency_key
            or receipt.factory_id != request.factory_id
            or receipt.project_id != request.project_id
            or receipt.effort_id != request.effort_id
            or receipt.run_id != request.run_id
            or receipt.session_id != request.session_id
            or receipt.mode is not request.mode
            or receipt.decision_kind is not request.decision_kind
        ):
            raise ValueError("Magi decision response crossed its requested boundary")
        if (
            request.expected_state_generation is not None
            and receipt.previous_state_generation != request.expected_state_generation
        ):
            raise ValueError("Magi decision response changed its expected state generation")
        return receipt

    def retrieve(self, receipt_id: str) -> MagiDecisionReceiptResponse:
        """Retrieve one content-addressed Magi decision receipt."""
        receipt = MagiDecisionReceiptResponse.from_wire(
            self._transport.execute(
                _request(
                    "get_magi_decision",
                    f"/smr/research-intern/decisions/{receipt_id}",
                )
            )
        )
        if receipt.receipt_id != receipt_id:
            raise ValueError("Magi decision receipt identity drifted")
        return receipt

    def list(self, *, limit: int = 100) -> tuple[MagiDecisionReceiptResponse, ...]:
        """List a bounded page of durable Magi decisions."""
        return _decisions(
            self._transport.execute(
                _request(
                    "list_magi_decisions",
                    "/smr/research-intern/decisions",
                    query={"limit": _bounded_limit(limit)},
                )
            )
        )

    def delegate(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        expected_state_generation: int,
        idempotency_key: str,
        rationale: str,
        session_id: str | None = None,
        evidence_refs: builtins.list[str] | None = None,
        state_patch: JsonObject | None = None,
        mode: MagiMode = MagiMode.SYNC,
    ) -> MagiDecisionReceiptResponse:
        """Delegate one exact Factory/run target with Casper by default."""
        return self.record(
            MagiDecisionRequest(
                mode=mode,
                decision_kind=MagiDecisionKind.DELEGATE,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs or [],
                state_patch=state_patch or {},
                rationale=rationale,
            )
        )

    def pause(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        expected_state_generation: int,
        idempotency_key: str,
        rationale: str,
        session_id: str | None = None,
        evidence_refs: builtins.list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Pause one exact Factory/run target through runtime authority."""
        return self.record(
            MagiDecisionRequest(
                mode=mode,
                decision_kind=MagiDecisionKind.PAUSE,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs or [],
                rationale=rationale,
            )
        )

    def intervene(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        expected_state_generation: int,
        idempotency_key: str,
        rationale: str,
        state_patch: JsonObject,
        session_id: str | None = None,
        evidence_refs: builtins.list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Steer one paused target with an exact durable state patch."""
        return self.record(
            MagiDecisionRequest(
                mode=mode,
                decision_kind=MagiDecisionKind.INTERVENE,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs or [],
                state_patch=state_patch,
                rationale=rationale,
            )
        )

    def resume(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        expected_state_generation: int,
        idempotency_key: str,
        rationale: str,
        session_id: str | None = None,
        evidence_refs: builtins.list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Resume one exact Factory/run target through runtime authority."""
        return self.record(
            MagiDecisionRequest(
                mode=mode,
                decision_kind=MagiDecisionKind.RESUME,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs or [],
                rationale=rationale,
            )
        )

    def verdict(
        self,
        *,
        idempotency_key: str,
        expected_state_generation: int,
        rationale: str,
        verdict: str,
        uncertainty: float,
        evidence_refs: builtins.list[str],
        factory_id: str | None = None,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        state_patch: JsonObject | None = None,
    ) -> MagiDecisionReceiptResponse:
        """Record an evidence-linked final Balthasar verdict."""
        return self.record(
            MagiDecisionRequest(
                mode=MagiMode.SERAPH,
                decision_kind=MagiDecisionKind.VERDICT,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs,
                state_patch=state_patch or {},
                rationale=rationale,
                verdict=verdict,
                uncertainty=uncertainty,
            )
        )


@dataclass(frozen=True, slots=True)
class ResearchInternEventCursor:
    """Reconnect position for one ordered Research Intern session event log.

    ``last_event_id`` is the numeric SSE id. ``event_id`` is the separate
    content-addressed identity carried by the backend payload.
    """

    session_id: str
    after_sequence: int = 0
    state_generation: int = 0
    event_id: str | None = None
    last_event_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if (
            self.after_sequence < 0
            or self.after_sequence > _RESEARCH_INTERN_EVENT_SEQUENCE_MAX
            or self.state_generation < 0
        ):
            raise ValueError(
                "event cursor sequence must be between 0 and "
                f"{_RESEARCH_INTERN_EVENT_SEQUENCE_MAX}; "
                "state generation must be non-negative"
            )
        if self.after_sequence == 0 and (
            self.event_id is not None or self.last_event_id is not None
        ):
            raise ValueError("the zero event cursor cannot carry a durable event identity")
        if self.after_sequence > 0 and (
            self.state_generation < 1 or self.event_id is None or self.last_event_id is None
        ):
            raise ValueError(
                "a nonzero event cursor requires generation, event_id, and Last-Event-ID"
            )
        if self.last_event_id is not None and self.last_event_id != str(self.after_sequence):
            raise ValueError("Last-Event-ID must equal the cursor's numeric sequence")
        if self.event_id is not None and (
            len(self.event_id) != 71
            or not self.event_id.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in self.event_id[7:])
        ):
            raise ValueError("event_id must be a lowercase sha256 digest")


@dataclass(frozen=True, slots=True)
class ResearchInternEventStreamObservation:
    """One bounded stream watch with its exact durable reconnect cursor."""

    session_id: str
    frames: tuple[ResearchInternEventStreamPayload, ...]
    events: tuple[ResearchInternEventResponse, ...]
    cursor: ResearchInternEventCursor | None
    timed_out: bool = False

    def to_wire(self) -> JsonObject:
        """Serialize a bounded watch result for MCP and other JSON consumers."""
        return {
            "session_id": self.session_id,
            "frames": [cast(JsonObject, frame.model_dump(mode="json")) for frame in self.frames],
            "events": [cast(JsonObject, event.to_wire()) for event in self.events],
            "cursor": (
                {
                    "session_id": self.cursor.session_id,
                    "after_sequence": self.cursor.after_sequence,
                    "state_generation": self.cursor.state_generation,
                    "event_id": self.cursor.event_id,
                    "last_event_id": self.cursor.last_event_id,
                }
                if self.cursor is not None
                else None
            ),
            "timed_out": self.timed_out,
        }


_RESEARCH_INTERN_EVENT_STREAM_EVENT_NAME = "research_intern_event"
_RESEARCH_INTERN_EVENT_STREAM_HEARTBEAT_NAME = "research_intern_heartbeat"


def _stream_timeout_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a number")
    normalized = float(value)
    if not 0 < normalized <= 300:
        raise ValueError("timeout_seconds must be greater than zero and at most 300")
    return normalized


def _stream_bound(value: int, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


class _ResearchInternEventStreamValidator:
    """Validate SSE framing and the durable event/cursor chain without synthesis."""

    def __init__(
        self,
        session_id: str,
        *,
        cursor: ResearchInternEventCursor | None,
        after_sequence: int,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
            or after_sequence > _RESEARCH_INTERN_EVENT_SEQUENCE_MAX
        ):
            raise ValueError(
                f"after_sequence must be between 0 and {_RESEARCH_INTERN_EVENT_SEQUENCE_MAX}"
            )
        if cursor is not None and cursor.session_id != session_id:
            raise ValueError("stream cursor belongs to another Intern session")
        if cursor is not None and after_sequence not in {0, cursor.after_sequence}:
            raise ValueError("after_sequence conflicts with the supplied stream cursor")
        self.session_id = session_id
        self.after_sequence = cursor.after_sequence if cursor is not None else after_sequence
        self.state_generation: int | None = (
            cursor.state_generation if cursor is not None else (0 if after_sequence == 0 else None)
        )
        self.event_id = cursor.event_id if cursor is not None else None

    @property
    def cursor(self) -> ResearchInternEventCursor:
        return ResearchInternEventCursor(
            session_id=self.session_id,
            after_sequence=self.after_sequence,
            state_generation=self.state_generation or 0,
            event_id=self.event_id,
            last_event_id=(str(self.after_sequence) if self.after_sequence > 0 else None),
        )

    def decode(self, frame: SseEvent) -> ResearchInternEventStreamPayload:
        try:
            payload = ResearchInternEventStreamEnvelope.model_validate(frame.json_data()).root
        except Exception as error:
            raise ValueError("Research Intern SSE data violated its typed envelope") from error
        if payload.session_id != self.session_id:
            raise ValueError("Research Intern SSE frame crossed its requested session")
        if isinstance(payload, ResearchInternEventStreamEvent):
            self._accept_event_frame(frame, payload)
        elif isinstance(payload, ResearchInternEventStreamHeartbeat):
            self._accept_heartbeat_frame(frame, payload)
        else:  # pragma: no cover - the discriminated union is exhaustive.
            raise ValueError("Research Intern SSE envelope kind is unsupported")
        return payload

    def _accept_event_frame(
        self,
        frame: SseEvent,
        payload: ResearchInternEventStreamEvent,
    ) -> None:
        event = payload.event
        if frame.event != _RESEARCH_INTERN_EVENT_STREAM_EVENT_NAME:
            raise ValueError("durable Intern event used the wrong SSE event name")
        if frame.event_id != str(event.sequence):
            raise ValueError("durable Intern SSE id must equal its numeric event sequence")
        if event.sequence != self.after_sequence + 1:
            raise ValueError("Research Intern SSE event sequence is not contiguous")
        if (
            self.state_generation is not None
            and event.previous_state_generation != self.state_generation
        ):
            raise ValueError("Research Intern SSE event broke the state-generation chain")
        if event.state_generation != event.previous_state_generation + 1:
            raise ValueError("Research Intern SSE event broke the state-generation transition")
        self.after_sequence = event.sequence
        self.state_generation = event.state_generation
        self.event_id = event.event_id

    def _accept_heartbeat_frame(
        self,
        frame: SseEvent,
        payload: ResearchInternEventStreamHeartbeat,
    ) -> None:
        if frame.event != _RESEARCH_INTERN_EVENT_STREAM_HEARTBEAT_NAME:
            raise ValueError("Intern heartbeat used the wrong SSE event name")
        if frame.event_id not in {None, str(self.after_sequence)}:
            raise ValueError("Intern heartbeat attempted to advance the durable SSE id")
        if (
            frame.retry_milliseconds is not None
            and frame.retry_milliseconds != payload.reconnect_after_ms
        ):
            raise ValueError("Intern heartbeat retry interval drifted from its payload")
        cursor = payload.cursor
        if self.after_sequence == 0:
            if cursor is not None:
                raise ValueError("zero-sequence Intern heartbeat carried a durable cursor")
            return
        if cursor is None or cursor.after_sequence != self.after_sequence:
            raise ValueError("Intern heartbeat did not preserve the reconnect sequence")
        if self.state_generation is not None and (cursor.state_generation != self.state_generation):
            raise ValueError("Intern heartbeat changed the state-generation cursor")
        if self.event_id is not None and cursor.event_id != self.event_id:
            raise ValueError("Intern heartbeat changed the content-addressed event cursor")
        self.state_generation = cursor.state_generation
        self.event_id = cursor.event_id


@dataclass(frozen=True, slots=True)
class ResearchInternEventObservation:
    """One bounded event page plus the exact cursor needed to reconnect."""

    session: ResearchInternSessionResponse
    events: tuple[ResearchInternEventResponse, ...]
    cursor: ResearchInternEventCursor


@dataclass(frozen=True, slots=True)
class ResearchInternOperatorTurn:
    """One backend-owned operator turn plus its ordered observed events."""

    response: ResearchInternTurnResponse
    observed_events: tuple[ResearchInternEventResponse, ...]
    cursor: ResearchInternEventCursor

    @property
    def operator_event(self) -> ResearchInternEventResponse:
        """Return the durable operator event."""
        return self.response.operator_event

    @property
    def decision_receipt(self) -> MagiDecisionReceiptResponse | None:
        """Return the optional runtime-control decision receipt."""
        return self.response.decision_receipt

    def to_wire(self) -> JsonObject:
        """Serialize the exact response, observed chain, and reconnect cursor."""
        return {
            "response": cast(JsonObject, self.response.to_wire()),
            "observed_events": [
                cast(JsonObject, event.to_wire()) for event in self.observed_events
            ],
            "cursor": {
                "session_id": self.cursor.session_id,
                "after_sequence": self.cursor.after_sequence,
                "state_generation": self.cursor.state_generation,
                "event_id": self.cursor.event_id,
                "last_event_id": self.cursor.last_event_id,
            },
        }


@dataclass(frozen=True, slots=True)
class ResearchInternExchange:
    """One ordered operator-to-Intern exchange from backend authority."""

    operator_turn: ResearchInternOperatorTurn
    agent_event: ResearchInternEventResponse
    observed_events: tuple[ResearchInternEventResponse, ...]
    cursor: ResearchInternEventCursor

    def to_wire(self) -> JsonObject:
        """Serialize one exact-turn exchange for MCP and other JSON consumers."""
        return {
            "operator_turn": self.operator_turn.to_wire(),
            "agent_event": cast(JsonObject, self.agent_event.to_wire()),
            "observed_events": [
                cast(JsonObject, event.to_wire()) for event in self.observed_events
            ],
            "cursor": {
                "session_id": self.cursor.session_id,
                "after_sequence": self.cursor.after_sequence,
                "state_generation": self.cursor.state_generation,
                "event_id": self.cursor.event_id,
                "last_event_id": self.cursor.last_event_id,
            },
        }


class ResearchInternPollingTimeoutError(TimeoutError):
    """A bounded Intern event wait ended before the requested event arrived."""

    def __init__(
        self,
        message: str,
        *,
        response: ResearchInternTurnResponse | None = None,
    ) -> None:
        super().__init__(message)
        self.response = response


class ResearchInternTurnFailedError(RuntimeError):
    """A canonical turn completed with a typed runtime failure payload."""

    def __init__(
        self,
        response: ResearchInternTurnResponse,
        *,
        terminal_event: ResearchInternEventResponse | None = None,
    ) -> None:
        message = (
            terminal_event.body
            if terminal_event is not None and terminal_event.body
            else (
                response.error.message
                if response.error is not None
                else f"Research Intern turn {response.turn_id} failed"
            )
        )
        super().__init__(message)
        self.response = response
        self.terminal_event = terminal_event


LEGACY_INTERN_SESSIONS_ENV = "SYNTH_ALLOW_LEGACY_INTERN_SESSIONS"


def legacy_intern_sessions_enabled() -> bool:
    """Whether the environment explicitly opted in to the legacy sessions plane."""
    return str(os.getenv(LEGACY_INTERN_SESSIONS_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class LegacyInternSessionsDisabledError(RuntimeError):
    """The legacy ``/smr/research-intern/sessions`` plane is disabled by default.

    QA for the Sync Intern release treats any legacy ``/sessions`` hit as a hard
    fail, so the plane requires an explicit opt-in instead of being callable by
    default. Use the ``/smr/research-intern/sync-sessions`` plane (``intern.sync_``
    on the SDK, ``intern_sync_*`` / ``intern_async_*`` MCP tools) instead.
    """

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"Legacy Research Intern sessions operation {operation!r} is disabled. "
            "Use the /smr/research-intern/sync-sessions plane instead "
            "(client.intern.sync_ / client.intern.async_, or the intern_sync_* / "
            "intern_async_* MCP tools). To explicitly opt back in, construct the "
            "client with allow_legacy_intern_sessions=True or set "
            f"{LEGACY_INTERN_SESSIONS_ENV}=1."
        )
        self.operation = operation


class ResearchInternSessionsAPI:
    """Durable reactive sessions and their ordered event logs.

    Legacy plane: every method requires the explicit
    ``allow_legacy_intern_sessions`` / ``SYNTH_ALLOW_LEGACY_INTERN_SESSIONS``
    opt-in and raises :class:`LegacyInternSessionsDisabledError` otherwise.
    """

    def __init__(
        self,
        transport: HttpTransport,
        owner: ResearchInternAPI,
        *,
        allow_legacy: bool = False,
    ) -> None:
        self._transport = transport
        self._owner = owner
        self._allow_legacy = allow_legacy

    def _require_legacy_enabled(self, operation: str) -> None:
        if self._allow_legacy or legacy_intern_sessions_enabled():
            return
        raise LegacyInternSessionsDisabledError(operation)

    def create(self, request: ResearchInternSessionCreateRequest) -> ResearchInternSessionResponse:
        """Create or replay one durable Intern session."""
        self._require_legacy_enabled("create")
        session = ResearchInternSessionResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_research_intern_session",
                    "/smr/research-intern/sessions",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            session.factory_id != request.factory_id
            or session.project_id != request.project_id
            or session.effort_id != request.effort_id
            or session.run_id != request.run_id
            or session.objective != request.objective
        ):
            raise ValueError("Research Intern session crossed its requested boundary")
        return session

    def list(self, *, limit: int = 100) -> tuple[ResearchInternSessionResponse, ...]:
        """List a bounded page of Intern sessions."""
        self._require_legacy_enabled("list")
        return _sessions(
            self._transport.execute(
                _request(
                    "list_research_intern_sessions",
                    "/smr/research-intern/sessions",
                    query={"limit": _bounded_limit(limit)},
                )
            )
        )

    def retrieve(self, session_id: str) -> ResearchInternSessionResponse:
        """Retrieve one Intern session by stable identity."""
        self._require_legacy_enabled("retrieve")
        session = ResearchInternSessionResponse.from_wire(
            self._transport.execute(
                _request(
                    "get_research_intern_session",
                    f"/smr/research-intern/sessions/{session_id}",
                )
            )
        )
        if session.session_id != session_id:
            raise ValueError("Research Intern session identity drifted")
        return session

    def append_event(
        self,
        session_id: str,
        request: ResearchInternEventAppendRequest,
    ) -> ResearchInternEventResponse:
        """Append one optimistic-concurrency-fenced event."""
        self._require_legacy_enabled("append_event")
        event = ResearchInternEventResponse.from_wire(
            self._transport.execute(
                _request(
                    "append_research_intern_event",
                    f"/smr/research-intern/sessions/{session_id}/events",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            event.session_id != session_id
            or event.idempotency_key != request.idempotency_key
            or event.event_kind is not request.event_kind
            or event.previous_state_generation != request.expected_state_generation
        ):
            raise ValueError("Research Intern event identity drifted")
        return event

    def list_events(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[ResearchInternEventResponse, ...]:
        """Observe an ordered, bounded page after one reconnect sequence."""
        self._require_legacy_enabled("list_events")
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
            or after_sequence > _RESEARCH_INTERN_EVENT_SEQUENCE_MAX
        ):
            raise ValueError(
                f"after_sequence must be between 0 and {_RESEARCH_INTERN_EVENT_SEQUENCE_MAX}"
            )
        events = _events(
            self._transport.execute(
                _request(
                    "list_research_intern_events",
                    f"/smr/research-intern/sessions/{session_id}/events",
                    query={
                        "after_sequence": after_sequence,
                        "limit": _bounded_limit(limit),
                    },
                )
            )
        )
        previous_sequence = after_sequence
        previous_generation: int | None = None
        for event in events:
            if (
                event.session_id != session_id
                or event.sequence != previous_sequence + 1
                or event.state_generation != event.previous_state_generation + 1
                or (
                    previous_generation is not None
                    and event.previous_state_generation != previous_generation
                )
            ):
                raise ValueError("Research Intern event page is not contiguous for its session")
            previous_sequence = event.sequence
            previous_generation = event.state_generation
        return events

    def stream_events(
        self,
        session_id: str,
        *,
        cursor: ResearchInternEventCursor | None = None,
        after_sequence: int = 0,
        timeout_seconds: float = 30.0,
    ) -> Iterator[ResearchInternEventStreamPayload]:
        """Stream typed backend-owned frames from one exact reconnect cursor."""
        self._require_legacy_enabled("stream_events")
        validator = _ResearchInternEventStreamValidator(
            session_id,
            cursor=cursor,
            after_sequence=after_sequence,
        )
        resume_sequence = validator.after_sequence
        for frame in self._transport.stream_sse(
            f"/smr/research-intern/sessions/{session_id}/events/stream",
            params={"after_sequence": resume_sequence},
            last_event_id=(str(resume_sequence) if resume_sequence > 0 else None),
            timeout_seconds=_stream_timeout_seconds(timeout_seconds),
            operation_id="stream_research_intern_session_events",
        ):
            yield validator.decode(frame)

    def watch(
        self,
        session_id: str,
        *,
        cursor: ResearchInternEventCursor | None = None,
        after_sequence: int = 0,
        event_count_max: int = 1,
        frame_count_max: int = 100,
        reconnect_count_max: int = 3,
        timeout_seconds: float = 30.0,
    ) -> ResearchInternEventStreamObservation:
        """Wait with explicit bounds, reconnecting only from durable SSE cursors."""
        self._require_legacy_enabled("watch")
        _stream_bound(event_count_max, name="event_count_max", maximum=500)
        _stream_bound(frame_count_max, name="frame_count_max", maximum=5_000)
        _stream_bound(reconnect_count_max, name="reconnect_count_max", maximum=20)
        timeout = _stream_timeout_seconds(timeout_seconds)
        validator = _ResearchInternEventStreamValidator(
            session_id,
            cursor=cursor,
            after_sequence=after_sequence,
        )
        current_cursor = cursor
        if current_cursor is None and after_sequence == 0:
            current_cursor = validator.cursor
        frames: list[ResearchInternEventStreamPayload] = []
        events: list[ResearchInternEventResponse] = []
        deadline = _MONOTONIC() + timeout
        timed_out = False
        transient_error: TransientServiceError | None = None
        for _ in range(reconnect_count_max):
            remaining = deadline - _MONOTONIC()
            if remaining <= 0:
                timed_out = True
                break
            try:
                for payload in self.stream_events(
                    session_id,
                    cursor=current_cursor,
                    after_sequence=(after_sequence if current_cursor is None else 0),
                    timeout_seconds=remaining,
                ):
                    if _MONOTONIC() >= deadline:
                        timed_out = True
                        break
                    frames.append(payload)
                    if isinstance(payload, ResearchInternEventStreamEvent):
                        events.append(payload.event)
                        current_cursor = ResearchInternEventCursor(
                            session_id=session_id,
                            after_sequence=payload.cursor.after_sequence,
                            state_generation=payload.cursor.state_generation,
                            event_id=payload.cursor.event_id,
                            last_event_id=str(payload.cursor.after_sequence),
                        )
                    elif payload.cursor is not None:
                        current_cursor = ResearchInternEventCursor(
                            session_id=session_id,
                            after_sequence=payload.cursor.after_sequence,
                            state_generation=payload.cursor.state_generation,
                            event_id=payload.cursor.event_id,
                            last_event_id=str(payload.cursor.after_sequence),
                        )
                    if len(events) >= event_count_max or len(frames) >= frame_count_max:
                        return ResearchInternEventStreamObservation(
                            session_id=session_id,
                            frames=tuple(frames),
                            events=tuple(events),
                            cursor=current_cursor,
                        )
                transient_error = None
                if timed_out:
                    break
            except SynthTimeoutError:
                timed_out = True
                break
            except TransientServiceError as error:
                if not error.retryable:
                    raise
                transient_error = error
                continue
        if transient_error is not None and not timed_out:
            raise transient_error
        return ResearchInternEventStreamObservation(
            session_id=session_id,
            frames=tuple(frames),
            events=tuple(events),
            cursor=current_cursor,
            timed_out=timed_out,
        )

    def sync(
        self,
        session_id: str,
        *,
        limit: int = 200,
    ) -> ResearchInternSessionSyncResponse:
        """Project a bounded page of canonical runtime transcript events."""
        self._require_legacy_enabled("sync")
        response = ResearchInternSessionSyncResponse.from_wire(
            self._transport.execute(
                _request(
                    "sync_research_intern_session",
                    f"/smr/research-intern/sessions/{session_id}/sync",
                    query={"limit": _bounded_limit(limit)},
                )
            )
        )
        if (
            response.session.session_id != session_id
            or response.session.run_id != response.source_run_id
            or response.projected_count != len(response.events)
            or any(event.session_id != session_id for event in response.events)
        ):
            raise ValueError("Research Intern runtime sync crossed its requested boundary")
        return response

    def turn(
        self,
        session_id: str,
        request: ResearchInternTurnRequest,
    ) -> ResearchInternTurnResponse:
        """Submit one canonical real-runtime operator turn."""
        self._require_legacy_enabled("turn")
        response = ResearchInternTurnResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_research_intern_session_turn",
                    f"/smr/research-intern/sessions/{session_id}/turns",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            response.session.session_id != session_id
            or response.operator_event.session_id != session_id
            or response.session.run_id != response.run_id
            or response.operator_event.body != request.body
            or response.reconnect_after_sequence != response.session.last_event_sequence
            or any(event.session_id != session_id for event in response.projected_events)
        ):
            raise ValueError("Research Intern turn crossed its requested boundary")
        return response

    def close(
        self,
        session_id: str,
        request: ResearchInternSessionCloseRequest,
    ) -> ResearchInternSessionResponse:
        """Close one exact session generation and retain its event history."""
        self._require_legacy_enabled("close")
        session = ResearchInternSessionResponse.from_wire(
            self._transport.execute(
                _request(
                    "close_research_intern_session",
                    f"/smr/research-intern/sessions/{session_id}/close",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            session.session_id != session_id
            or session.state_generation <= request.expected_state_generation
        ):
            raise ValueError("Research Intern close response identity drifted")
        return session

    def publish_trace(
        self,
        session_id: str,
        request: ResearchInternTracePublicationRequest,
    ) -> ResearchInternTracePublicationResponse:
        """Publish the terminal event chain through backend Trace V5 authority."""
        self._require_legacy_enabled("publish_trace")
        response = ResearchInternTracePublicationResponse.from_wire(
            self._transport.execute(
                _request(
                    "publish_research_intern_session_trace",
                    f"/smr/research-intern/sessions/{session_id}/trace:publish",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            response.session_id != session_id
            or response.idempotency_key != request.idempotency_key
            or response.state_generation != request.expected_state_generation
        ):
            raise ValueError("Research Intern trace publication identity drifted")
        return response

    def create_reactive(
        self,
        request: ResearchInternSessionCreateRequest,
    ) -> ResearchInternReactiveSession:
        """Create a reconnectable reactive view over one backend session."""
        self._require_legacy_enabled("create_reactive")
        session = self.create(request)
        return ResearchInternReactiveSession(
            self._owner,
            session,
            cursor=ResearchInternEventCursor(session_id=session.session_id),
        )

    def connect(
        self,
        session_id: str,
        *,
        cursor: ResearchInternEventCursor | None = None,
    ) -> ResearchInternReactiveSession:
        """Reconnect to a session from an exact previously returned cursor."""
        self._require_legacy_enabled("connect")
        session = self.retrieve(session_id)
        reconnect_cursor = cursor or ResearchInternEventCursor(session_id=session_id)
        if reconnect_cursor.session_id != session_id:
            raise ValueError("reconnect cursor belongs to another Intern session")
        if (
            reconnect_cursor.after_sequence > session.last_event_sequence
            or reconnect_cursor.state_generation > session.state_generation
        ):
            raise ValueError("reconnect cursor is ahead of the authoritative session")
        return ResearchInternReactiveSession(self._owner, session, cursor=reconnect_cursor)


class ResearchInternAcceptanceReceiptsAPI:
    """Content-addressed acceptance evidence publication and retrieval."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def publish(
        self,
        request: ResearchInternAcceptanceReceiptPublicationRequest,
    ) -> ResearchInternAcceptanceReceiptPublicationResponse:
        """Publish or replay one deterministic acceptance receipt."""
        digest_hex = _digest_hex(request.receipt_id)
        receipt = ResearchInternAcceptanceReceiptPublicationResponse.from_wire(
            self._transport.execute(
                _request(
                    "publish_research_intern_acceptance_receipt",
                    f"/smr/research-intern/acceptance-receipts/{digest_hex}",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.receipt_id != request.receipt_id
            or receipt.candidate_id != request.candidate_id
            or receipt.lane != request.lane
        ):
            raise ValueError("Acceptance receipt publication identity drifted")
        return receipt

    def retrieve(
        self,
        receipt_id: str,
    ) -> ResearchInternAcceptanceReceiptPublicationResponse:
        """Retrieve one public content-addressed acceptance receipt."""
        digest_hex = _digest_hex(receipt_id)
        receipt = ResearchInternAcceptanceReceiptPublicationResponse.from_wire(
            self._transport.execute(
                _request(
                    "get_research_intern_acceptance_receipt",
                    f"/smr/research-intern/acceptance-receipts/{digest_hex}",
                )
            )
        )
        if _digest_hex(receipt.receipt_id) != digest_hex:
            raise ValueError("Acceptance receipt retrieval identity drifted")
        return receipt

    def list(
        self,
        *,
        candidate_id: str | None = None,
        lane: str | None = None,
        limit: int = 100,
    ) -> tuple[ResearchInternAcceptanceReceiptPublicationResponse, ...]:
        """List bounded acceptance publications under organization scope."""
        query: JsonObject = {"limit": _bounded_limit(limit)}
        if candidate_id is not None:
            query["candidate_id"] = candidate_id
        if lane is not None:
            query["lane"] = lane
        receipts = _acceptance_receipts(
            self._transport.execute(
                _request(
                    "list_research_intern_acceptance_receipts",
                    "/smr/research-intern/acceptance-receipts",
                    query=query,
                )
            )
        )
        if any(
            (candidate_id is not None and receipt.candidate_id != candidate_id)
            or (lane is not None and receipt.lane != lane)
            for receipt in receipts
        ):
            raise ValueError("Acceptance receipt list crossed its requested boundary")
        return receipts


class ResearchInternReactiveSession:
    """Reconnectable operator loop over authoritative Intern session state."""

    def __init__(
        self,
        api: ResearchInternAPI,
        session: ResearchInternSessionResponse,
        *,
        cursor: ResearchInternEventCursor,
    ) -> None:
        self._api = api
        self.session = session
        self.cursor = cursor

    def _accept_event(self, event: ResearchInternEventResponse) -> None:
        if (
            event.session_id != self.session.session_id
            or event.sequence != self.cursor.after_sequence + 1
            or event.previous_state_generation != self.cursor.state_generation
            or event.state_generation != event.previous_state_generation + 1
        ):
            raise ValueError("Research Intern event broke the reconnect cursor chain")
        self.cursor = ResearchInternEventCursor(
            session_id=event.session_id,
            after_sequence=event.sequence,
            state_generation=event.state_generation,
            event_id=event.event_id,
            last_event_id=str(event.sequence),
        )

    def observe(self, *, limit: int = 100) -> ResearchInternEventObservation:
        """Read one bounded event page and advance the reconnect cursor."""
        events = self._api.sessions.list_events(
            self.session.session_id,
            after_sequence=self.cursor.after_sequence,
            limit=limit,
        )
        for event in events:
            self._accept_event(event)
        self.session = self._api.sessions.retrieve(self.session.session_id)
        return ResearchInternEventObservation(
            session=self.session,
            events=events,
            cursor=self.cursor,
        )

    def synchronize(
        self,
        *,
        page_limit: int = 100,
        page_count_max: int = 20,
    ) -> tuple[ResearchInternEventResponse, ...]:
        """Drain unseen session events with explicit page bounds."""
        _bounded_limit(page_limit)
        if not 1 <= page_count_max <= 100:
            raise ValueError("page_count_max must be between 1 and 100")
        observed: list[ResearchInternEventResponse] = []
        for _ in range(page_count_max):
            observation = self.observe(limit=page_limit)
            observed.extend(observation.events)
            if self.cursor.after_sequence >= observation.session.last_event_sequence:
                return tuple(observed)
            if not observation.events:
                raise ValueError("Intern session reported unseen events but returned no event page")
        raise ValueError("Intern session synchronization exceeded page_count_max")

    def synchronize_runtime(
        self,
        *,
        page_limit: int = 200,
        page_count_max: int = 20,
    ) -> tuple[ResearchInternEventResponse, ...]:
        """Project real runtime transcript events and advance the session cursor."""
        _bounded_limit(page_limit)
        if not 1 <= page_count_max <= 100:
            raise ValueError("page_count_max must be between 1 and 100")
        self.synchronize(page_limit=page_limit, page_count_max=page_count_max)
        projected: list[ResearchInternEventResponse] = []
        for _ in range(page_count_max):
            response = self._api.sessions.sync(
                self.session.session_id,
                limit=page_limit,
            )
            projected.extend(response.events)
            observed = self.synchronize(
                page_limit=page_limit,
                page_count_max=page_count_max,
            )
            observed_ids = {event.event_id for event in observed}
            if any(event.event_id not in observed_ids for event in response.events):
                raise ValueError("runtime-projected events were absent from the session event log")
            if not response.has_more:
                return tuple(projected)
        raise ValueError("Research Intern runtime synchronization exceeded page_count_max")

    def watch(
        self,
        *,
        event_count_max: int = 1,
        frame_count_max: int = 100,
        reconnect_count_max: int = 3,
        timeout_seconds: float = 30.0,
    ) -> ResearchInternEventStreamObservation:
        """Consume the canonical event stream without invoking runtime sync."""
        observation = self._api.sessions.watch(
            self.session.session_id,
            cursor=self.cursor,
            event_count_max=event_count_max,
            frame_count_max=frame_count_max,
            reconnect_count_max=reconnect_count_max,
            timeout_seconds=timeout_seconds,
        )
        for event in observation.events:
            self._accept_event(event)
        if observation.cursor is not None and observation.cursor != self.cursor:
            raise ValueError("Intern stream observation cursor diverged from its event chain")
        if observation.events:
            self.session = self._api.sessions.retrieve(self.session.session_id)
        return ResearchInternEventStreamObservation(
            session_id=observation.session_id,
            frames=observation.frames,
            events=observation.events,
            cursor=self.cursor,
            timed_out=observation.timed_out,
        )

    @staticmethod
    def _event_turn_id(event: ResearchInternEventResponse) -> str | None:
        marker = event.payload.get("research_intern_turn")
        if not isinstance(marker, dict):
            return None
        value = marker.get("turn_id")
        return value if isinstance(value, str) and value else None

    def _watch_through_sequence(
        self,
        sequence: int,
        *,
        timeout_seconds: float,
    ) -> tuple[ResearchInternEventResponse, ...]:
        if sequence < self.cursor.after_sequence:
            return ()
        deadline = time.monotonic() + _stream_timeout_seconds(timeout_seconds)
        observed: list[ResearchInternEventResponse] = []
        while self.cursor.after_sequence < sequence:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResearchInternPollingTimeoutError(
                    f"event stream did not reach sequence {sequence} for "
                    f"session {self.session.session_id}"
                )
            observation = self.watch(
                event_count_max=min(sequence - self.cursor.after_sequence, 500),
                timeout_seconds=remaining,
            )
            observed.extend(observation.events)
            if not observation.events:
                raise ResearchInternPollingTimeoutError(
                    f"event stream did not reach sequence {sequence} for "
                    f"session {self.session.session_id}"
                )
        return tuple(observed)

    def _wait_for_turn_terminal(
        self,
        turn_id: str,
        *,
        timeout_seconds: float,
        already_observed: tuple[ResearchInternEventResponse, ...] = (),
    ) -> tuple[ResearchInternEventResponse, tuple[ResearchInternEventResponse, ...]]:
        deadline = time.monotonic() + _stream_timeout_seconds(timeout_seconds)
        observed = list(already_observed)
        while True:
            terminal = next(
                (
                    event
                    for event in observed
                    if event.event_kind
                    in {
                        ResearchInternEventKind.AGENT_MESSAGE,
                        ResearchInternEventKind.ERROR,
                    }
                    and self._event_turn_id(event) == turn_id
                ),
                None,
            )
            if terminal is not None:
                return terminal, tuple(observed)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResearchInternPollingTimeoutError(
                    f"no terminal event observed for turn {turn_id}"
                )
            observation = self.watch(
                event_count_max=1,
                timeout_seconds=remaining,
            )
            observed.extend(observation.events)
            if not observation.events:
                raise ResearchInternPollingTimeoutError(
                    f"no terminal event observed for turn {turn_id}"
                )

    def append_event(
        self,
        *,
        event_kind: ResearchInternEventKind,
        idempotency_key: str,
        body: str | None = None,
        payload: JsonObject | None = None,
        evidence_refs: list[str] | None = None,
        mode: MagiMode | None = None,
    ) -> ResearchInternEventResponse:
        """Append one event after synchronizing its optimistic generation."""
        self.synchronize()
        event = self._api.sessions.append_event(
            self.session.session_id,
            ResearchInternEventAppendRequest(
                event_kind=event_kind,
                mode=mode,
                idempotency_key=idempotency_key,
                expected_state_generation=self.cursor.state_generation,
                body=body,
                payload=payload or {},
                evidence_refs=evidence_refs or [],
            ),
        )
        self._accept_event(event)
        self.session = self._api.sessions.retrieve(self.session.session_id)
        return event

    def append_operator_message(
        self,
        body: str,
        *,
        idempotency_key: str,
        payload: JsonObject | None = None,
        evidence_refs: list[str] | None = None,
    ) -> ResearchInternEventResponse:
        """Persist one operator message without claiming runtime actuation."""
        return self.append_event(
            event_kind=ResearchInternEventKind.OPERATOR_MESSAGE,
            idempotency_key=idempotency_key,
            body=body,
            payload=payload,
            evidence_refs=evidence_refs,
        )

    def _run_id(self) -> str:
        if self.session.run_id is None:
            raise ValueError("Magi actuation requires a session bound to a live run_id")
        return self.session.run_id

    def _intern_generation(self) -> int:
        return self._api.retrieve().state_generation

    def delegate(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        evidence_refs: list[str] | None = None,
        state_patch: JsonObject | None = None,
        mode: MagiMode = MagiMode.SYNC,
    ) -> MagiDecisionReceiptResponse:
        """Delegate this session's exact Factory/run target through Casper."""
        self.synchronize()
        return self._api.decisions.delegate(
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self._run_id(),
            session_id=self.session.session_id,
            expected_state_generation=self._intern_generation(),
            idempotency_key=idempotency_key,
            rationale=rationale,
            evidence_refs=evidence_refs,
            state_patch=state_patch,
            mode=mode,
        )

    def pause(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Pause this session's exact Factory and run target."""
        self.synchronize()
        return self._api.decisions.pause(
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self._run_id(),
            session_id=self.session.session_id,
            expected_state_generation=self._intern_generation(),
            idempotency_key=idempotency_key,
            rationale=rationale,
            evidence_refs=evidence_refs,
            mode=mode,
        )

    def intervene(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        state_patch: JsonObject,
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Steer this session's exact paused Factory and run target."""
        self.synchronize()
        return self._api.decisions.intervene(
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self._run_id(),
            session_id=self.session.session_id,
            expected_state_generation=self._intern_generation(),
            idempotency_key=idempotency_key,
            rationale=rationale,
            state_patch=state_patch,
            evidence_refs=evidence_refs,
            mode=mode,
        )

    def send_operator_turn(
        self,
        body: str,
        *,
        idempotency_key: str,
        mode: MagiMode = MagiMode.SYNC,
        control: ResearchInternTurnControl | None = None,
        rationale: str | None = None,
        state_patch: JsonObject | None = None,
        evidence_refs: list[str] | None = None,
        wait_timeout_seconds: float = 15.0,
        poll_interval_ms: int = 250,
        stream_timeout_seconds: float = 30.0,
    ) -> ResearchInternOperatorTurn:
        """Submit one turn and observe every durable response event over SSE."""
        self._run_id()
        # /sync is reconciliation at the initial request boundary. Once the
        # turn is accepted, only the canonical event stream advances the loop.
        self.synchronize_runtime()
        response = self._api.sessions.turn(
            self.session.session_id,
            ResearchInternTurnRequest(
                body=body,
                mode=mode,
                idempotency_key=idempotency_key,
                expected_session_state_generation=self.cursor.state_generation,
                expected_intern_state_generation=self._intern_generation(),
                control=control,
                rationale=rationale,
                state_patch=state_patch or {},
                evidence_refs=evidence_refs or [],
                wait_timeout_seconds=wait_timeout_seconds,
                poll_interval_ms=poll_interval_ms,
            ),
        )
        observed = self._watch_through_sequence(
            response.reconnect_after_sequence,
            timeout_seconds=stream_timeout_seconds,
        )
        if self.cursor.after_sequence != response.reconnect_after_sequence:
            raise ValueError("Research Intern turn response skipped unseen event sequence")
        self.session = self._api.sessions.retrieve(self.session.session_id)
        return ResearchInternOperatorTurn(
            response=response,
            observed_events=tuple(observed),
            cursor=self.cursor,
        )

    def resume(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Resume this session's exact Factory and run target."""
        self.synchronize()
        return self._api.decisions.resume(
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self._run_id(),
            session_id=self.session.session_id,
            expected_state_generation=self._intern_generation(),
            idempotency_key=idempotency_key,
            rationale=rationale,
            evidence_refs=evidence_refs,
            mode=mode,
        )

    def poll(
        self,
        *,
        event_kinds: set[ResearchInternEventKind] | None = None,
        max_events: int = 20,
        max_polls: int = 20,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.5,
    ) -> tuple[ResearchInternEventResponse, ...]:
        """Explicit degraded fallback when the canonical SSE path is unavailable."""
        _bounded_limit(max_events)
        if not 1 <= max_polls <= 1000:
            raise ValueError("max_polls must be between 1 and 1000")
        if not 0 < timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be greater than zero and at most 300")
        if not 0 <= poll_interval_seconds <= 30:
            raise ValueError("poll_interval_seconds must be between zero and 30")
        deadline = time.monotonic() + timeout_seconds
        selected: list[ResearchInternEventResponse] = []
        for _ in range(max_polls):
            observation = self.observe(limit=min(max_events, 100))
            candidates: tuple[ResearchInternEventResponse, ...] = observation.events
            if not candidates and self.session.run_id is not None:
                candidates = self.synchronize_runtime(
                    page_limit=min(max_events, 100),
                    page_count_max=20,
                )
            selected.extend(
                event
                for event in candidates
                if event_kinds is None or event.event_kind in event_kinds
            )
            if len(selected) >= max_events:
                return tuple(selected[:max_events])
            if selected or observation.session.status.value != "active":
                return tuple(selected)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_seconds, remaining))
        return tuple(selected)

    def wait_for_agent_message(
        self,
        *,
        max_polls: int = 60,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
    ) -> ResearchInternEventResponse:
        """Use the explicit degraded polling fallback for any agent message."""
        events = self.poll(
            event_kinds={ResearchInternEventKind.AGENT_MESSAGE},
            max_events=1,
            max_polls=max_polls,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        if not events:
            raise ResearchInternPollingTimeoutError(
                f"no agent message observed for session {self.session.session_id}"
            )
        return events[0]

    def evidence(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> tuple[ResearchInternEventResponse, ...]:
        """Return bounded, runtime-synchronized session evidence."""
        self.synchronize_runtime(page_limit=min(limit, 500))
        return self._api.sessions.list_events(
            self.session.session_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    def exchange(
        self,
        body: str,
        *,
        idempotency_key: str,
        mode: MagiMode = MagiMode.SYNC,
        control: ResearchInternTurnControl | None = None,
        rationale: str | None = None,
        state_patch: JsonObject | None = None,
        evidence_refs: list[str] | None = None,
        wait_timeout_seconds: float = 15.0,
        poll_interval_ms: int = 250,
        recovery_timeout_seconds: float = 60.0,
    ) -> ResearchInternExchange:
        """Submit one turn and recover its exact terminal reply from SSE."""
        turn = self.send_operator_turn(
            body,
            idempotency_key=idempotency_key,
            mode=mode,
            control=control,
            rationale=rationale,
            state_patch=state_patch,
            evidence_refs=evidence_refs,
            wait_timeout_seconds=wait_timeout_seconds,
            poll_interval_ms=poll_interval_ms,
            stream_timeout_seconds=min(
                _stream_timeout_seconds(recovery_timeout_seconds),
                30.0,
            ),
        )
        response = turn.response
        if response.status is ResearchInternTurnStatus.FAILED:
            raise ResearchInternTurnFailedError(response)
        try:
            terminal, observed = self._wait_for_turn_terminal(
                response.turn_id,
                timeout_seconds=recovery_timeout_seconds,
                already_observed=turn.observed_events,
            )
        except ResearchInternPollingTimeoutError as error:
            error.response = response
            raise
        if terminal.event_kind is ResearchInternEventKind.ERROR:
            raise ResearchInternTurnFailedError(
                response,
                terminal_event=terminal,
            )
        if response.agent_event is not None and response.agent_event.event_id != terminal.event_id:
            raise ValueError("turn response agent event diverged from the canonical SSE event")
        return ResearchInternExchange(
            operator_turn=turn,
            agent_event=terminal,
            observed_events=observed,
            cursor=self.cursor,
        )

    def verdict(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        verdict: str,
        uncertainty: float,
        evidence_refs: list[str],
        state_patch: JsonObject | None = None,
    ) -> MagiDecisionReceiptResponse:
        """Record an evidence-linked final Balthasar verdict."""
        self.synchronize()
        return self._api.decisions.verdict(
            idempotency_key=idempotency_key,
            expected_state_generation=self._intern_generation(),
            rationale=rationale,
            verdict=verdict,
            uncertainty=uncertainty,
            evidence_refs=evidence_refs,
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self.session.run_id,
            session_id=self.session.session_id,
            state_patch=state_patch,
        )

    def close(
        self,
        *,
        idempotency_key: str,
        status: Literal[
            "completed",
            "partial",
            "failed",
            "stopped",
            "canceled",
            "archived",
        ],
        rationale: str,
        evidence_refs: list[str] | None = None,
    ) -> ResearchInternSessionResponse:
        """Close this session at its exact current generation."""
        self.synchronize()
        self.session = self._api.sessions.close(
            self.session.session_id,
            ResearchInternSessionCloseRequest(
                idempotency_key=idempotency_key,
                expected_state_generation=self.cursor.state_generation,
                status=status,
                rationale=rationale,
                evidence_refs=evidence_refs or [],
            ),
        )
        self.synchronize()
        return self.session

    def teardown(
        self,
        *,
        idempotency_key: str,
        status: Literal[
            "completed",
            "partial",
            "failed",
            "stopped",
            "canceled",
            "archived",
        ] = "completed",
        rationale: str = "Research Intern session teardown complete.",
        evidence_refs: list[str] | None = None,
    ) -> ResearchInternSessionResponse:
        """Close the session while preserving all durable evidence."""
        return self.close(
            idempotency_key=idempotency_key,
            status=status,
            rationale=rationale,
            evidence_refs=evidence_refs,
        )


class ResearchInternSyncRuntimeAPI:
    """Synchronous transport for durable operator-present Sync sessions."""

    _PATH = "/smr/research-intern/sync-sessions"

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(self, request: InternSyncSessionCreateRequest) -> InternSyncSession:
        return InternSyncSession.from_wire(
            self._transport.execute(
                _request(
                    "create_intern_sync_session",
                    self._PATH,
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    def list(self, *, limit: int = 100) -> tuple[InternSyncSession, ...]:
        return tuple(
            InternSyncSession.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    self._transport.execute(
                        _request(
                            "list_intern_sync_sessions",
                            self._PATH,
                            query={"limit": _bounded_limit(limit)},
                        )
                    ),
                ),
                operation_id="list_intern_sync_sessions",
            )
        )

    def get(self, sync_session_id: str) -> InternSyncSession:
        session = InternSyncSession.from_wire(
            self._transport.execute(
                _request(
                    "get_intern_sync_session",
                    f"{self._PATH}/{sync_session_id}",
                )
            )
        )
        if session.sync_session_id != sync_session_id:
            raise ValueError("Sync Intern session identity drifted")
        return session

    def command(
        self,
        sync_session_id: str,
        request: InternSyncCommandRequest,
    ) -> InternSyncCommandReceipt:
        receipt = InternSyncCommandReceipt.from_wire(
            self._transport.execute(
                _request(
                    "command_intern_sync_session",
                    f"{self._PATH}/{sync_session_id}/commands",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if receipt.command_id != request.command_id or receipt.runtime_id != sync_session_id:
            raise ValueError("Sync Intern command receipt identity drifted")
        return receipt

    def send_message(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        body: str,
        turn_id: str | None = None,
        context: dict[str, JsonValue] | None = None,
    ) -> InternSyncCommandReceipt:
        return self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.OPERATOR_MESSAGE,
                payload={
                    "turn_id": turn_id or command_id,
                    "body": body,
                    "context": context or {},
                },
            ),
        )

    def intervene(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        body: str,
        turn_id: str | None = None,
        state_patch: dict[str, JsonValue] | None = None,
        context: dict[str, JsonValue] | None = None,
    ) -> InternSyncCommandReceipt:
        return self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.INTERVENE,
                payload={
                    "turn_id": turn_id or command_id,
                    "body": body,
                    "state_patch": state_patch or {},
                    "context": context or {},
                },
            ),
        )

    def answer_interaction(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        interaction_id: str,
        answer: str,
        context: dict[str, JsonValue] | None = None,
    ) -> InternSyncCommandReceipt:
        return self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.ANSWER_INTERACTION,
                payload={
                    "interaction_id": interaction_id,
                    "answer": answer,
                    "context": context or {},
                },
            ),
        )

    def pause(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        rationale: str,
    ) -> InternSyncCommandReceipt:
        return self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.PAUSE,
                payload={"rationale": rationale},
            ),
        )

    def resume(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
    ) -> InternSyncCommandReceipt:
        return self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.RESUME,
            ),
        )

    def close(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        rationale: str,
        outcome: InternRuntimeOutcome = InternRuntimeOutcome.COMPLETED,
    ) -> InternSyncCommandReceipt:
        return self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.CLOSE,
                payload={"outcome": outcome.value, "rationale": rationale},
            ),
        )

    def events(
        self,
        sync_session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> InternSyncEventPage:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        return _intern_sync_event_page(
            self._transport.execute(
                _request(
                    "list_intern_runtime_events",
                    f"/smr/research-intern/runtimes/sync/{sync_session_id}/events",
                    query={
                        "after_sequence": after_sequence,
                        "limit": _bounded_limit(limit),
                    },
                )
            ),
            after_sequence=after_sequence,
            runtime_id=sync_session_id,
        )

    def stream_events(
        self,
        sync_session_id: str,
        *,
        after_sequence: int = 0,
        timeout_seconds: float = 30.0,
    ) -> Iterator[InternSyncEvent]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        expected_sequence = after_sequence + 1
        path = f"/smr/research-intern/runtimes/sync/{sync_session_id}/events/stream"
        for frame in self._transport.stream_sse(
            path,
            params={"after_sequence": after_sequence},
            last_event_id=str(after_sequence) if after_sequence else None,
            timeout_seconds=_stream_timeout_seconds(timeout_seconds),
            operation_id="stream_intern_runtime_events",
        ):
            event = _intern_sync_stream_event(
                frame,
                expected_sequence=expected_sequence,
                runtime_id=sync_session_id,
            )
            expected_sequence = event.sequence + 1
            yield event

    def tail(
        self,
        sync_session_id: str,
        *,
        after_sequence: int = 0,
        event_count_max: int = 1,
        timeout_seconds: float = 30.0,
    ) -> InternSyncEventPage:
        _stream_bound(event_count_max, name="event_count_max", maximum=500)
        events: list[InternSyncEvent] = []
        for event in self.stream_events(
            sync_session_id,
            after_sequence=after_sequence,
            timeout_seconds=timeout_seconds,
        ):
            events.append(event)
            if len(events) >= event_count_max:
                break
        return InternSyncEventPage(
            events=tuple(events),
            next_sequence=events[-1].sequence if events else after_sequence,
        )


class ResearchInternAsyncRuntimeAPI:
    """Synchronous transport for the organization's singleton Async Intern."""

    _PATH = "/smr/research-intern/async"

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def ensure(self, request: InternAsyncEnsureRequest) -> InternAsyncRuntime:
        return InternAsyncRuntime.from_wire(
            self._transport.execute(
                _request(
                    "ensure_intern_async_runtime",
                    self._PATH,
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    def get(self) -> InternAsyncRuntime:
        return InternAsyncRuntime.from_wire(
            self._transport.execute(_request("get_intern_async_runtime", self._PATH))
        )

    def command(self, request: InternAsyncCommandRequest) -> InternAsyncCommandReceipt:
        receipt = InternAsyncCommandReceipt.from_wire(
            self._transport.execute(
                _request(
                    "command_intern_async_runtime",
                    f"{self._PATH}/commands",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if receipt.command_id != request.command_id:
            raise ValueError("Async Intern command receipt identity drifted")
        return receipt

    def send(self, request: InternAsyncInstructionRequest) -> InternAsyncCommandReceipt:
        return self.command(request.to_command())

    def pause(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        reason: str,
    ) -> InternAsyncCommandReceipt:
        return self.command(
            InternAsyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternAsyncCommandKind.PAUSE,
                payload={"reason": reason},
            )
        )

    def resume(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
    ) -> InternAsyncCommandReceipt:
        return self.command(
            InternAsyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternAsyncCommandKind.RESUME,
            )
        )

    def cancel(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        reason: str,
    ) -> InternAsyncCommandReceipt:
        return self.command(
            InternAsyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternAsyncCommandKind.CANCEL,
                payload={"reason": reason},
            )
        )

    def provide_input(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        interaction_id: str,
        body: str,
        context: dict[str, JsonValue] | None = None,
    ) -> InternAsyncCommandReceipt:
        return self.command(
            InternAsyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternAsyncCommandKind.PROVIDE_INPUT,
                payload={
                    "interaction_id": interaction_id,
                    "body": body,
                    "context": context or {},
                },
            )
        )

    def intervene(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        body: str,
        context: dict[str, JsonValue] | None = None,
    ) -> InternAsyncCommandReceipt:
        return self.send(
            InternAsyncInstructionRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                instruction_kind=InternAsyncInstructionKind.INTERVENE,
                body=body,
                context=context or {},
            )
        )

    def redirect_objective(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        objective: str,
        context: dict[str, JsonValue] | None = None,
    ) -> InternAsyncCommandReceipt:
        return self.send(
            InternAsyncInstructionRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                instruction_kind=InternAsyncInstructionKind.REDIRECT_OBJECTIVE,
                body=objective,
                context=context or {},
            )
        )

    def request_checkpoint(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        context: dict[str, JsonValue] | None = None,
    ) -> InternAsyncCommandReceipt:
        return self.send(
            InternAsyncInstructionRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                instruction_kind=InternAsyncInstructionKind.REQUEST_CHECKPOINT,
                context=context or {},
            )
        )

    def events(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> InternAsyncEventPage:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        return _intern_async_event_page(
            self._transport.execute(
                _request(
                    "list_intern_async_runtime_events",
                    f"{self._PATH}/events",
                    query={
                        "after_sequence": after_sequence,
                        "limit": _bounded_limit(limit),
                    },
                )
            ),
            after_sequence=after_sequence,
        )

    def stream_events(
        self,
        *,
        after_sequence: int = 0,
        timeout_seconds: float = 30.0,
    ) -> Iterator[InternAsyncEvent]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        expected_sequence = after_sequence + 1
        runtime_id: str | None = None
        for frame in self._transport.stream_sse(
            f"{self._PATH}/events/stream",
            params={"after_sequence": after_sequence},
            last_event_id=str(after_sequence) if after_sequence else None,
            timeout_seconds=_stream_timeout_seconds(timeout_seconds),
            operation_id="stream_intern_async_runtime_events",
        ):
            event = _intern_async_stream_event(
                frame,
                expected_sequence=expected_sequence,
                runtime_id=runtime_id,
            )
            runtime_id = event.runtime_id
            expected_sequence = event.sequence + 1
            yield event

    def tail(
        self,
        *,
        after_sequence: int = 0,
        event_count_max: int = 1,
        timeout_seconds: float = 30.0,
    ) -> InternAsyncEventPage:
        _stream_bound(event_count_max, name="event_count_max", maximum=500)
        events: list[InternAsyncEvent] = []
        for event in self.stream_events(
            after_sequence=after_sequence,
            timeout_seconds=timeout_seconds,
        ):
            events.append(event)
            if len(events) >= event_count_max:
                break
        return InternAsyncEventPage(
            events=tuple(events),
            next_sequence=events[-1].sequence if events else after_sequence,
        )


class ResearchInternAPI:
    """One durable organization Intern and its many Factory memberships."""

    def __init__(
        self,
        transport: HttpTransport,
        *,
        allow_legacy_intern_sessions: bool = False,
    ) -> None:
        self._transport = transport
        self.sync_ = ResearchInternSyncRuntimeAPI(transport)
        self.async_ = ResearchInternAsyncRuntimeAPI(transport)
        self.factories = ResearchInternFactoriesAPI(transport)
        self.decisions = ResearchInternDecisionsAPI(transport)
        self.sessions = ResearchInternSessionsAPI(
            transport,
            self,
            allow_legacy=allow_legacy_intern_sessions,
        )
        self.acceptance_receipts = ResearchInternAcceptanceReceiptsAPI(transport)

    def provision(
        self,
        request: ResearchInternProvisionRequest | None = None,
    ) -> ResearchInternResponse:
        """Provision or retrieve the one Intern for the authenticated organization."""
        body = (request or ResearchInternProvisionRequest()).to_wire()
        return ResearchInternResponse.from_wire(
            self._transport.execute(
                _request(
                    "provision_research_intern",
                    "/smr/research-intern",
                    body=cast(JsonObject, body),
                )
            )
        )

    def retrieve(self) -> ResearchInternResponse:
        """Retrieve the authenticated organization's Research Intern."""
        return ResearchInternResponse.from_wire(
            self._transport.execute(_request("get_research_intern", "/smr/research-intern"))
        )

    def update(self, request: ResearchInternPatchRequest) -> ResearchInternResponse:
        """Update mutable Intern policy, attribution, state, or display fields."""
        return ResearchInternResponse.from_wire(
            self._transport.execute(
                _request(
                    "patch_research_intern",
                    "/smr/research-intern",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )


class ProjectComputerAPI:
    """Provider-neutral Project Computer lifecycle operations."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def provision(
        self,
        project_id: ProjectId,
        request: ProjectComputerProvisionRequest,
    ) -> ProjectComputerResponse:
        """Provision a Project Computer from an exact Git and snapshot identity."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "provision_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if computer.project_id != str(project_id) or computer.factory_id != request.factory_id:
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    def retrieve(
        self,
        project_id: ProjectId,
        factory_id: FactoryId,
    ) -> ProjectComputerResponse:
        """Retrieve the current Project Computer for a Project and Factory."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "get_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    query={"factory_id": str(factory_id)},
                )
            )
        )
        if computer.project_id != str(project_id) or computer.factory_id != str(factory_id):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    def replace(
        self,
        project_id: ProjectId,
        request: ProjectComputerReplaceRequest,
    ) -> ProjectComputerResponse:
        """Replace a Project Computer using an adapter-authored restore receipt."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "replace_project_computer",
                    f"/smr/projects/{project_id}/computer/replace",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if computer.project_id != str(project_id) or computer.factory_id != request.factory_id:
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    def retire(
        self,
        project_id: ProjectId,
        factory_id: FactoryId,
    ) -> ProjectComputerResponse:
        """Retire the Project and Factory Computer while retaining its receipt."""
        computer = ProjectComputerResponse.from_wire(
            self._transport.execute(
                _request(
                    "retire_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    query={"factory_id": str(factory_id)},
                )
            )
        )
        if computer.project_id != str(project_id) or computer.factory_id != str(factory_id):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    def inspect(
        self,
        project_id: ProjectId,
        request: ProjectComputerInspectRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Inspect one generation without crossing its Factory boundary."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            self._transport.execute(
                _request(
                    "inspect_project_computer",
                    f"/smr/projects/{project_id}/computer/inspect",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(
            receipt,
            request,
            operation=ProjectRuntimeOperation.INSPECT,
        )

    def execute(
        self,
        project_id: ProjectId,
        request: ProjectComputerExecuteRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Execute a bounded argv under an exact lease and fencing digest."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            self._transport.execute(
                _request(
                    "execute_project_computer",
                    f"/smr/projects/{project_id}/computer/execute",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(
            receipt,
            request,
            operation=ProjectRuntimeOperation.EXECUTE,
        )

    def reconcile(
        self,
        project_id: ProjectId,
        request: ProjectComputerOperationReconcileRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Observe an accepted operation with an unknown provider outcome."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            self._transport.execute(
                _request(
                    "reconcile_project_computer_operation",
                    f"/smr/projects/{project_id}/computer/operations/reconcile",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(receipt, request, operation=None)

    def acquire_lease(
        self,
        project_id: ProjectId,
        request: ProjectComputerLeaseAcquireRequest,
    ) -> ProjectComputerLeaseResponse:
        """Acquire one generation-fenced execution lease."""
        lease = ProjectComputerLeaseResponse.from_wire(
            self._transport.execute(
                _request(
                    "acquire_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.state != "active"
        ):
            raise ValueError("Project Computer lease acquisition identity drifted")
        return lease

    def renew_lease(
        self,
        project_id: ProjectId,
        lease_id: str,
        request: ProjectComputerLeaseRenewRequest,
    ) -> ProjectComputerLeaseResponse:
        """Renew one exact lease while preserving its fencing identity."""
        lease = ProjectComputerLeaseResponse.from_wire(
            self._transport.execute(
                _request(
                    "renew_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases/{lease_id}/renew",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.lease_id != lease_id
            or lease.fencing_token_digest != request.fencing_token_digest
            or lease.state != "active"
        ):
            raise ValueError("Project Computer lease renewal identity drifted")
        return lease

    def release_lease(
        self,
        project_id: ProjectId,
        lease_id: str,
        request: ProjectComputerLeaseReleaseRequest,
    ) -> ProjectComputerLeaseResponse:
        """Release one exact lease and retain the final fencing identity."""
        lease = ProjectComputerLeaseResponse.from_wire(
            self._transport.execute(
                _request(
                    "release_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases/{lease_id}/release",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.lease_id != lease_id
            or lease.fencing_token_digest != request.fencing_token_digest
            or lease.state != "released"
        ):
            raise ValueError("Project Computer lease release identity drifted")
        return lease

    def cleanup(
        self,
        factory_id: FactoryId,
        request: ProjectComputerCleanupRequest,
    ) -> ProjectComputerCleanupReceiptResponse:
        """Retire every Project Computer in one Factory with owner-authored proof."""
        receipt = ProjectComputerCleanupReceiptResponse.from_wire(
            self._transport.execute(
                _request(
                    "cleanup_factory_project_computers",
                    (f"/smr/research-intern/factories/{factory_id}/project-computers/cleanup"),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.factory_id != str(factory_id)
            or receipt.idempotency_key != request.idempotency_key
        ):
            raise ValueError("Project Computer cleanup receipt identity drifted")
        return receipt


class ProjectDataBindingsAPI:
    """Project-scoped data bindings and immutable dataset revisions."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(
        self,
        project_id: ProjectId,
        request: DataBindingCreateRequest,
    ) -> DataBindingResponse:
        """Create one Factory-scoped Project data binding."""
        binding = DataBindingResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_data_binding",
                    f"/smr/projects/{project_id}/data-bindings",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if binding.project_id != str(project_id) or binding.factory_id != request.factory_id:
            raise ValueError("Data Binding response crossed its requested boundary")
        return binding

    def list(self, project_id: ProjectId) -> tuple[DataBindingResponse, ...]:
        """List data bindings without crossing the requested Project."""
        bindings = _data_bindings(
            self._transport.execute(
                _request(
                    "list_data_bindings",
                    f"/smr/projects/{project_id}/data-bindings",
                )
            )
        )
        if any(binding.project_id != str(project_id) for binding in bindings):
            raise ValueError("Data Binding list crossed its project boundary")
        return bindings

    def create_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        request: DatasetRevisionCreateRequest,
    ) -> DatasetRevisionResponse:
        """Append an immutable revision to a Project data binding."""
        revision = DatasetRevisionResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_dataset_revision",
                    (f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/revisions"),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            revision.project_id != str(project_id)
            or str(revision.data_binding_id) != data_binding_id
        ):
            raise ValueError("Dataset Revision response identity drifted")
        return revision

    def list_revisions(
        self,
        project_id: ProjectId,
        data_binding_id: str,
    ) -> tuple[DatasetRevisionResponse, ...]:
        """List immutable revisions for one Project data binding."""
        revisions = _dataset_revisions(
            self._transport.execute(
                _request(
                    "list_dataset_revisions",
                    (f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/revisions"),
                )
            )
        )
        if any(
            revision.project_id != str(project_id)
            or str(revision.data_binding_id) != data_binding_id
            for revision in revisions
        ):
            raise ValueError("Dataset Revision list identity drifted")
        return revisions

    def transition_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        dataset_revision_id: str,
        request: DatasetRevisionLifecycleRequest,
    ) -> DatasetRevisionResponse:
        """Apply a fail-closed lifecycle transition to one exact revision."""
        revision = DatasetRevisionResponse.from_wire(
            self._transport.execute(
                _request(
                    "transition_dataset_revision_lifecycle",
                    (
                        f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/"
                        f"revisions/{dataset_revision_id}/lifecycle"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            revision.project_id != str(project_id)
            or str(revision.data_binding_id) != data_binding_id
            or str(revision.dataset_revision_id) != dataset_revision_id
            or revision.state is not request.target_state
        ):
            raise ValueError("DatasetRevision lifecycle response identity drifted")
        return revision

    def prepare_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        request: DatasetRevisionPrepareRequest,
    ) -> DatasetRevisionPreparationResponse:
        """Prepare immutable upload targets for one exact DatasetRevision draft."""
        prepared = DatasetRevisionPreparationResponse.from_wire(
            self._transport.execute(
                _dataset_publication_request(
                    "prepareDatasetRevisionPublication",
                    (
                        f"/smr/projects/{project_id}/data-bindings/"
                        f"{data_binding_id}/revisions:prepare"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        draft = request.draft
        if (
            prepared.idempotency_key != request.idempotency_key
            or prepared.project_id != str(project_id)
            or str(prepared.data_binding_id) != data_binding_id
            or prepared.dataset_revision_id != draft.dataset_revision_id
            or prepared.factory_id != draft.factory_id
            or prepared.org_id != draft.org_id
        ):
            raise ValueError("DatasetRevision preparation identity drifted")
        return prepared

    def finalize_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        preparation_id: str,
        request: DatasetRevisionFinalizeRequest,
    ) -> DatasetRevisionFinalizeResponse:
        """Finalize one prepared DatasetRevision and verify all owner receipts."""
        finalized = DatasetRevisionFinalizeResponse.from_wire(
            self._transport.execute(
                _dataset_publication_request(
                    "finalizeDatasetRevisionPublication",
                    (
                        f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/"
                        f"revision-preparations/{preparation_id}:finalize"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        sealed = finalized.sealed_revision
        if (
            str(finalized.preparation_id) != preparation_id
            or sealed.project_id != str(project_id)
            or str(sealed.binding_id) != data_binding_id
        ):
            raise ValueError("DatasetRevision finalization identity drifted")
        return finalized


class AsyncResearchInternFactoriesAPI:
    """Native asynchronous Factory membership operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def attach(
        self,
        factory_id: FactoryId,
    ) -> ResearchInternFactoryMembershipResponse:
        """Attach one Factory to the organization Research Intern."""
        membership = ResearchInternFactoryMembershipResponse.from_wire(
            await self._transport.execute(
                _request(
                    "attach_research_intern_factory",
                    f"/smr/research-intern/factories/{factory_id}",
                )
            )
        )
        if membership.factory_id != str(factory_id):
            raise ValueError("Research Intern Factory membership identity drifted")
        return membership

    async def list(self) -> tuple[ResearchInternFactoryMembershipResponse, ...]:
        """List Factory memberships for the organization Research Intern."""
        return _memberships(
            await self._transport.execute(
                _request(
                    "list_research_intern_factories",
                    "/smr/research-intern/factories",
                )
            )
        )

    async def mint_role_receipt(
        self,
        factory_id: FactoryId,
        request: FactoryRoleReceiptMintRequest,
    ) -> FactoryRoleReceiptResponse:
        """Mint one idempotent Factory role receipt from owner runtime evidence."""
        receipt = FactoryRoleReceiptResponse.from_wire(
            await self._transport.execute(
                _request(
                    "mint_factory_role_receipt",
                    f"/smr/research-intern/factories/{factory_id}/role-receipts",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.factory_id != str(factory_id)
            or receipt.idempotency_key != request.idempotency_key
            or receipt.candidate_id != request.candidate_id
            or receipt.role != request.role
            or receipt.runtime_evidence.run_id != request.run_id
        ):
            raise ValueError("Factory role receipt identity drifted")
        return receipt

    async def list_role_receipts(
        self,
        factory_id: FactoryId,
        candidate_id: str | None = None,
    ) -> tuple[FactoryRoleReceiptResponse, ...]:
        """List durable Factory role receipts, optionally for one candidate."""
        query: JsonObject = {}
        if candidate_id is not None:
            query["candidate_id"] = candidate_id
        receipts = _role_receipts(
            await self._transport.execute(
                _request(
                    "list_factory_role_receipts",
                    f"/smr/research-intern/factories/{factory_id}/role-receipts",
                    query=query,
                )
            )
        )
        if any(
            receipt.factory_id != str(factory_id)
            or (candidate_id is not None and receipt.candidate_id != candidate_id)
            for receipt in receipts
        ):
            raise ValueError("Factory role receipt list crossed its requested boundary")
        return receipts


class AsyncResearchInternDecisionsAPI:
    """Native asynchronous Magi decision receipt operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def record(
        self,
        request: MagiDecisionRequest,
    ) -> MagiDecisionReceiptResponse:
        """Record one idempotent, evidence-linked Magi decision."""
        receipt = MagiDecisionReceiptResponse.from_wire(
            await self._transport.execute(
                _request(
                    "record_magi_decision",
                    "/smr/research-intern/decisions",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.idempotency_key != request.idempotency_key
            or receipt.factory_id != request.factory_id
            or receipt.project_id != request.project_id
            or receipt.effort_id != request.effort_id
            or receipt.run_id != request.run_id
            or receipt.session_id != request.session_id
            or receipt.mode is not request.mode
            or receipt.decision_kind is not request.decision_kind
        ):
            raise ValueError("Magi decision response crossed its requested boundary")
        if (
            request.expected_state_generation is not None
            and receipt.previous_state_generation != request.expected_state_generation
        ):
            raise ValueError("Magi decision response changed its expected state generation")
        return receipt

    async def retrieve(self, receipt_id: str) -> MagiDecisionReceiptResponse:
        """Retrieve one content-addressed Magi decision receipt."""
        receipt = MagiDecisionReceiptResponse.from_wire(
            await self._transport.execute(
                _request(
                    "get_magi_decision",
                    f"/smr/research-intern/decisions/{receipt_id}",
                )
            )
        )
        if receipt.receipt_id != receipt_id:
            raise ValueError("Magi decision receipt identity drifted")
        return receipt

    async def list(self, *, limit: int = 100) -> tuple[MagiDecisionReceiptResponse, ...]:
        """List a bounded page of durable Magi decisions."""
        return _decisions(
            await self._transport.execute(
                _request(
                    "list_magi_decisions",
                    "/smr/research-intern/decisions",
                    query={"limit": _bounded_limit(limit)},
                )
            )
        )

    async def delegate(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        expected_state_generation: int,
        idempotency_key: str,
        rationale: str,
        session_id: str | None = None,
        evidence_refs: builtins.list[str] | None = None,
        state_patch: JsonObject | None = None,
        mode: MagiMode = MagiMode.SYNC,
    ) -> MagiDecisionReceiptResponse:
        """Delegate one exact Factory/run target with Casper by default."""
        return await self.record(
            MagiDecisionRequest(
                mode=mode,
                decision_kind=MagiDecisionKind.DELEGATE,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs or [],
                state_patch=state_patch or {},
                rationale=rationale,
            )
        )

    async def pause(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        expected_state_generation: int,
        idempotency_key: str,
        rationale: str,
        session_id: str | None = None,
        evidence_refs: builtins.list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Pause one exact Factory/run target through runtime authority."""
        return await self.record(
            MagiDecisionRequest(
                mode=mode,
                decision_kind=MagiDecisionKind.PAUSE,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs or [],
                rationale=rationale,
            )
        )

    async def intervene(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        expected_state_generation: int,
        idempotency_key: str,
        rationale: str,
        state_patch: JsonObject,
        session_id: str | None = None,
        evidence_refs: builtins.list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Steer one paused target with an exact durable state patch."""
        return await self.record(
            MagiDecisionRequest(
                mode=mode,
                decision_kind=MagiDecisionKind.INTERVENE,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs or [],
                state_patch=state_patch,
                rationale=rationale,
            )
        )

    async def resume(
        self,
        *,
        factory_id: str,
        project_id: str,
        effort_id: str,
        run_id: str,
        expected_state_generation: int,
        idempotency_key: str,
        rationale: str,
        session_id: str | None = None,
        evidence_refs: builtins.list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Resume one exact Factory/run target through runtime authority."""
        return await self.record(
            MagiDecisionRequest(
                mode=mode,
                decision_kind=MagiDecisionKind.RESUME,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs or [],
                rationale=rationale,
            )
        )

    async def verdict(
        self,
        *,
        idempotency_key: str,
        expected_state_generation: int,
        rationale: str,
        verdict: str,
        uncertainty: float,
        evidence_refs: builtins.list[str],
        factory_id: str | None = None,
        project_id: str | None = None,
        effort_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        state_patch: JsonObject | None = None,
    ) -> MagiDecisionReceiptResponse:
        """Record an evidence-linked final Balthasar verdict."""
        return await self.record(
            MagiDecisionRequest(
                mode=MagiMode.SERAPH,
                decision_kind=MagiDecisionKind.VERDICT,
                idempotency_key=idempotency_key,
                factory_id=factory_id,
                project_id=project_id,
                effort_id=effort_id,
                run_id=run_id,
                session_id=session_id,
                expected_state_generation=expected_state_generation,
                evidence_refs=evidence_refs,
                state_patch=state_patch or {},
                rationale=rationale,
                verdict=verdict,
                uncertainty=uncertainty,
            )
        )


class AsyncResearchInternSessionsAPI:
    """Native asynchronous Intern session and event-log operations.

    Legacy plane: every method requires the explicit
    ``allow_legacy_intern_sessions`` / ``SYNTH_ALLOW_LEGACY_INTERN_SESSIONS``
    opt-in and raises :class:`LegacyInternSessionsDisabledError` otherwise.
    """

    def __init__(
        self,
        transport: AsyncHttpTransport,
        owner: AsyncResearchInternAPI,
        *,
        allow_legacy: bool = False,
    ) -> None:
        self._transport = transport
        self._owner = owner
        self._allow_legacy = allow_legacy

    def _require_legacy_enabled(self, operation: str) -> None:
        if self._allow_legacy or legacy_intern_sessions_enabled():
            return
        raise LegacyInternSessionsDisabledError(operation)

    async def create(
        self,
        request: ResearchInternSessionCreateRequest,
    ) -> ResearchInternSessionResponse:
        """Create or replay one durable Intern session."""
        self._require_legacy_enabled("create")
        session = ResearchInternSessionResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_research_intern_session",
                    "/smr/research-intern/sessions",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            session.factory_id != request.factory_id
            or session.project_id != request.project_id
            or session.effort_id != request.effort_id
            or session.run_id != request.run_id
            or session.objective != request.objective
        ):
            raise ValueError("Research Intern session crossed its requested boundary")
        return session

    async def list(self, *, limit: int = 100) -> tuple[ResearchInternSessionResponse, ...]:
        """List a bounded page of Intern sessions."""
        self._require_legacy_enabled("list")
        return _sessions(
            await self._transport.execute(
                _request(
                    "list_research_intern_sessions",
                    "/smr/research-intern/sessions",
                    query={"limit": _bounded_limit(limit)},
                )
            )
        )

    async def retrieve(self, session_id: str) -> ResearchInternSessionResponse:
        """Retrieve one Intern session by stable identity."""
        self._require_legacy_enabled("retrieve")
        session = ResearchInternSessionResponse.from_wire(
            await self._transport.execute(
                _request(
                    "get_research_intern_session",
                    f"/smr/research-intern/sessions/{session_id}",
                )
            )
        )
        if session.session_id != session_id:
            raise ValueError("Research Intern session identity drifted")
        return session

    async def append_event(
        self,
        session_id: str,
        request: ResearchInternEventAppendRequest,
    ) -> ResearchInternEventResponse:
        """Append one optimistic-concurrency-fenced event."""
        self._require_legacy_enabled("append_event")
        event = ResearchInternEventResponse.from_wire(
            await self._transport.execute(
                _request(
                    "append_research_intern_event",
                    f"/smr/research-intern/sessions/{session_id}/events",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            event.session_id != session_id
            or event.idempotency_key != request.idempotency_key
            or event.event_kind is not request.event_kind
            or event.previous_state_generation != request.expected_state_generation
        ):
            raise ValueError("Research Intern event identity drifted")
        return event

    async def list_events(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[ResearchInternEventResponse, ...]:
        """Observe an ordered, bounded page after one reconnect sequence."""
        self._require_legacy_enabled("list_events")
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
            or after_sequence > _RESEARCH_INTERN_EVENT_SEQUENCE_MAX
        ):
            raise ValueError(
                f"after_sequence must be between 0 and {_RESEARCH_INTERN_EVENT_SEQUENCE_MAX}"
            )
        events = _events(
            await self._transport.execute(
                _request(
                    "list_research_intern_events",
                    f"/smr/research-intern/sessions/{session_id}/events",
                    query={
                        "after_sequence": after_sequence,
                        "limit": _bounded_limit(limit),
                    },
                )
            )
        )
        previous_sequence = after_sequence
        previous_generation: int | None = None
        for event in events:
            if (
                event.session_id != session_id
                or event.sequence != previous_sequence + 1
                or event.state_generation != event.previous_state_generation + 1
                or (
                    previous_generation is not None
                    and event.previous_state_generation != previous_generation
                )
            ):
                raise ValueError("Research Intern event page is not contiguous for its session")
            previous_sequence = event.sequence
            previous_generation = event.state_generation
        return events

    async def stream_events(
        self,
        session_id: str,
        *,
        cursor: ResearchInternEventCursor | None = None,
        after_sequence: int = 0,
        timeout_seconds: float = 30.0,
    ) -> AsyncIterator[ResearchInternEventStreamPayload]:
        """Stream typed backend-owned frames from one exact reconnect cursor."""
        self._require_legacy_enabled("stream_events")
        validator = _ResearchInternEventStreamValidator(
            session_id,
            cursor=cursor,
            after_sequence=after_sequence,
        )
        resume_sequence = validator.after_sequence
        async for frame in self._transport.stream_sse(
            f"/smr/research-intern/sessions/{session_id}/events/stream",
            params={"after_sequence": resume_sequence},
            last_event_id=(str(resume_sequence) if resume_sequence > 0 else None),
            timeout_seconds=_stream_timeout_seconds(timeout_seconds),
            operation_id="stream_research_intern_session_events",
        ):
            yield validator.decode(frame)

    async def watch(
        self,
        session_id: str,
        *,
        cursor: ResearchInternEventCursor | None = None,
        after_sequence: int = 0,
        event_count_max: int = 1,
        frame_count_max: int = 100,
        reconnect_count_max: int = 3,
        timeout_seconds: float = 30.0,
    ) -> ResearchInternEventStreamObservation:
        """Wait with explicit bounds, reconnecting only from durable SSE cursors."""
        self._require_legacy_enabled("watch")
        _stream_bound(event_count_max, name="event_count_max", maximum=500)
        _stream_bound(frame_count_max, name="frame_count_max", maximum=5_000)
        _stream_bound(reconnect_count_max, name="reconnect_count_max", maximum=20)
        timeout = _stream_timeout_seconds(timeout_seconds)
        validator = _ResearchInternEventStreamValidator(
            session_id,
            cursor=cursor,
            after_sequence=after_sequence,
        )
        current_cursor = cursor
        if current_cursor is None and after_sequence == 0:
            current_cursor = validator.cursor
        frames: list[ResearchInternEventStreamPayload] = []
        events: list[ResearchInternEventResponse] = []
        deadline = _MONOTONIC() + timeout
        timed_out = False
        transient_error: TransientServiceError | None = None
        for _ in range(reconnect_count_max):
            remaining = deadline - _MONOTONIC()
            if remaining <= 0:
                timed_out = True
                break
            try:
                async with asyncio.timeout(remaining):
                    async for payload in self.stream_events(
                        session_id,
                        cursor=current_cursor,
                        after_sequence=(after_sequence if current_cursor is None else 0),
                        timeout_seconds=remaining,
                    ):
                        if _MONOTONIC() >= deadline:
                            timed_out = True
                            break
                        frames.append(payload)
                        if isinstance(payload, ResearchInternEventStreamEvent):
                            events.append(payload.event)
                            current_cursor = ResearchInternEventCursor(
                                session_id=session_id,
                                after_sequence=payload.cursor.after_sequence,
                                state_generation=payload.cursor.state_generation,
                                event_id=payload.cursor.event_id,
                                last_event_id=str(payload.cursor.after_sequence),
                            )
                        elif payload.cursor is not None:
                            current_cursor = ResearchInternEventCursor(
                                session_id=session_id,
                                after_sequence=payload.cursor.after_sequence,
                                state_generation=payload.cursor.state_generation,
                                event_id=payload.cursor.event_id,
                                last_event_id=str(payload.cursor.after_sequence),
                            )
                        if len(events) >= event_count_max or len(frames) >= frame_count_max:
                            return ResearchInternEventStreamObservation(
                                session_id=session_id,
                                frames=tuple(frames),
                                events=tuple(events),
                                cursor=current_cursor,
                            )
                transient_error = None
                if timed_out:
                    break
            except TimeoutError:
                timed_out = True
                break
            except SynthTimeoutError:
                timed_out = True
                break
            except TransientServiceError as error:
                if not error.retryable:
                    raise
                transient_error = error
                continue
        if transient_error is not None and not timed_out:
            raise transient_error
        return ResearchInternEventStreamObservation(
            session_id=session_id,
            frames=tuple(frames),
            events=tuple(events),
            cursor=current_cursor,
            timed_out=timed_out,
        )

    async def sync(
        self,
        session_id: str,
        *,
        limit: int = 200,
    ) -> ResearchInternSessionSyncResponse:
        """Project a bounded page of canonical runtime transcript events."""
        self._require_legacy_enabled("sync")
        response = ResearchInternSessionSyncResponse.from_wire(
            await self._transport.execute(
                _request(
                    "sync_research_intern_session",
                    f"/smr/research-intern/sessions/{session_id}/sync",
                    query={"limit": _bounded_limit(limit)},
                )
            )
        )
        if (
            response.session.session_id != session_id
            or response.session.run_id != response.source_run_id
            or response.projected_count != len(response.events)
            or any(event.session_id != session_id for event in response.events)
        ):
            raise ValueError("Research Intern runtime sync crossed its requested boundary")
        return response

    async def turn(
        self,
        session_id: str,
        request: ResearchInternTurnRequest,
    ) -> ResearchInternTurnResponse:
        """Submit one canonical real-runtime operator turn."""
        self._require_legacy_enabled("turn")
        response = ResearchInternTurnResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_research_intern_session_turn",
                    f"/smr/research-intern/sessions/{session_id}/turns",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            response.session.session_id != session_id
            or response.operator_event.session_id != session_id
            or response.session.run_id != response.run_id
            or response.operator_event.body != request.body
            or response.reconnect_after_sequence != response.session.last_event_sequence
            or any(event.session_id != session_id for event in response.projected_events)
        ):
            raise ValueError("Research Intern turn crossed its requested boundary")
        return response

    async def close(
        self,
        session_id: str,
        request: ResearchInternSessionCloseRequest,
    ) -> ResearchInternSessionResponse:
        """Close one exact session generation and retain its event history."""
        self._require_legacy_enabled("close")
        session = ResearchInternSessionResponse.from_wire(
            await self._transport.execute(
                _request(
                    "close_research_intern_session",
                    f"/smr/research-intern/sessions/{session_id}/close",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            session.session_id != session_id
            or session.state_generation <= request.expected_state_generation
        ):
            raise ValueError("Research Intern close response identity drifted")
        return session

    async def publish_trace(
        self,
        session_id: str,
        request: ResearchInternTracePublicationRequest,
    ) -> ResearchInternTracePublicationResponse:
        """Publish the terminal event chain through backend Trace V5 authority."""
        self._require_legacy_enabled("publish_trace")
        response = ResearchInternTracePublicationResponse.from_wire(
            await self._transport.execute(
                _request(
                    "publish_research_intern_session_trace",
                    f"/smr/research-intern/sessions/{session_id}/trace:publish",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            response.session_id != session_id
            or response.idempotency_key != request.idempotency_key
            or response.state_generation != request.expected_state_generation
        ):
            raise ValueError("Research Intern trace publication identity drifted")
        return response

    async def create_reactive(
        self,
        request: ResearchInternSessionCreateRequest,
    ) -> AsyncResearchInternReactiveSession:
        """Create an asynchronously reconnectable view over one backend session."""
        self._require_legacy_enabled("create_reactive")
        session = await self.create(request)
        return AsyncResearchInternReactiveSession(
            self._owner,
            session,
            cursor=ResearchInternEventCursor(session_id=session.session_id),
        )

    async def connect(
        self,
        session_id: str,
        *,
        cursor: ResearchInternEventCursor | None = None,
    ) -> AsyncResearchInternReactiveSession:
        """Reconnect asynchronously from one exact previously returned cursor."""
        self._require_legacy_enabled("connect")
        session = await self.retrieve(session_id)
        reconnect_cursor = cursor or ResearchInternEventCursor(session_id=session_id)
        if reconnect_cursor.session_id != session_id:
            raise ValueError("reconnect cursor belongs to another Intern session")
        if (
            reconnect_cursor.after_sequence > session.last_event_sequence
            or reconnect_cursor.state_generation > session.state_generation
        ):
            raise ValueError("reconnect cursor is ahead of the authoritative session")
        return AsyncResearchInternReactiveSession(
            self._owner,
            session,
            cursor=reconnect_cursor,
        )


class AsyncResearchInternAcceptanceReceiptsAPI:
    """Native asynchronous acceptance receipt publication and retrieval."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def publish(
        self,
        request: ResearchInternAcceptanceReceiptPublicationRequest,
    ) -> ResearchInternAcceptanceReceiptPublicationResponse:
        """Publish or replay one deterministic acceptance receipt."""
        digest_hex = _digest_hex(request.receipt_id)
        receipt = ResearchInternAcceptanceReceiptPublicationResponse.from_wire(
            await self._transport.execute(
                _request(
                    "publish_research_intern_acceptance_receipt",
                    f"/smr/research-intern/acceptance-receipts/{digest_hex}",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.receipt_id != request.receipt_id
            or receipt.candidate_id != request.candidate_id
            or receipt.lane != request.lane
        ):
            raise ValueError("Acceptance receipt publication identity drifted")
        return receipt

    async def retrieve(
        self,
        receipt_id: str,
    ) -> ResearchInternAcceptanceReceiptPublicationResponse:
        """Retrieve one public content-addressed acceptance receipt."""
        digest_hex = _digest_hex(receipt_id)
        receipt = ResearchInternAcceptanceReceiptPublicationResponse.from_wire(
            await self._transport.execute(
                _request(
                    "get_research_intern_acceptance_receipt",
                    f"/smr/research-intern/acceptance-receipts/{digest_hex}",
                )
            )
        )
        if _digest_hex(receipt.receipt_id) != digest_hex:
            raise ValueError("Acceptance receipt retrieval identity drifted")
        return receipt

    async def list(
        self,
        *,
        candidate_id: str | None = None,
        lane: str | None = None,
        limit: int = 100,
    ) -> tuple[ResearchInternAcceptanceReceiptPublicationResponse, ...]:
        """List bounded acceptance publications under organization scope."""
        query: JsonObject = {"limit": _bounded_limit(limit)}
        if candidate_id is not None:
            query["candidate_id"] = candidate_id
        if lane is not None:
            query["lane"] = lane
        receipts = _acceptance_receipts(
            await self._transport.execute(
                _request(
                    "list_research_intern_acceptance_receipts",
                    "/smr/research-intern/acceptance-receipts",
                    query=query,
                )
            )
        )
        if any(
            (candidate_id is not None and receipt.candidate_id != candidate_id)
            or (lane is not None and receipt.lane != lane)
            for receipt in receipts
        ):
            raise ValueError("Acceptance receipt list crossed its requested boundary")
        return receipts


class AsyncResearchInternSyncRuntimeAPI:
    """Native async transport for operator-present Sync sessions."""

    _PATH = "/smr/research-intern/sync-sessions"

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(
        self,
        request: InternSyncSessionCreateRequest,
    ) -> InternSyncSession:
        return InternSyncSession.from_wire(
            await self._transport.execute(
                _request(
                    "create_intern_sync_session",
                    self._PATH,
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    async def list(self, *, limit: int = 100) -> tuple[InternSyncSession, ...]:
        return tuple(
            InternSyncSession.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    await self._transport.execute(
                        _request(
                            "list_intern_sync_sessions",
                            self._PATH,
                            query={"limit": _bounded_limit(limit)},
                        )
                    ),
                ),
                operation_id="list_intern_sync_sessions",
            )
        )

    async def get(self, sync_session_id: str) -> InternSyncSession:
        session = InternSyncSession.from_wire(
            await self._transport.execute(
                _request(
                    "get_intern_sync_session",
                    f"{self._PATH}/{sync_session_id}",
                )
            )
        )
        if session.sync_session_id != sync_session_id:
            raise ValueError("Sync Intern session identity drifted")
        return session

    async def command(
        self,
        sync_session_id: str,
        request: InternSyncCommandRequest,
    ) -> InternSyncCommandReceipt:
        receipt = InternSyncCommandReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "command_intern_sync_session",
                    f"{self._PATH}/{sync_session_id}/commands",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if receipt.command_id != request.command_id or receipt.runtime_id != sync_session_id:
            raise ValueError("Sync Intern command receipt identity drifted")
        return receipt

    async def send_message(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        body: str,
        turn_id: str | None = None,
        context: dict[str, JsonValue] | None = None,
    ) -> InternSyncCommandReceipt:
        return await self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.OPERATOR_MESSAGE,
                payload={
                    "turn_id": turn_id or command_id,
                    "body": body,
                    "context": context or {},
                },
            ),
        )

    async def intervene(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        body: str,
        turn_id: str | None = None,
        state_patch: dict[str, JsonValue] | None = None,
        context: dict[str, JsonValue] | None = None,
    ) -> InternSyncCommandReceipt:
        return await self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.INTERVENE,
                payload={
                    "turn_id": turn_id or command_id,
                    "body": body,
                    "state_patch": state_patch or {},
                    "context": context or {},
                },
            ),
        )

    async def answer_interaction(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        interaction_id: str,
        answer: str,
        context: dict[str, JsonValue] | None = None,
    ) -> InternSyncCommandReceipt:
        return await self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.ANSWER_INTERACTION,
                payload={
                    "interaction_id": interaction_id,
                    "answer": answer,
                    "context": context or {},
                },
            ),
        )

    async def pause(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        rationale: str,
    ) -> InternSyncCommandReceipt:
        return await self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.PAUSE,
                payload={"rationale": rationale},
            ),
        )

    async def resume(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
    ) -> InternSyncCommandReceipt:
        return await self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.RESUME,
            ),
        )

    async def close(
        self,
        sync_session_id: str,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        rationale: str,
        outcome: InternRuntimeOutcome = InternRuntimeOutcome.COMPLETED,
    ) -> InternSyncCommandReceipt:
        return await self.command(
            sync_session_id,
            InternSyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternSyncCommandKind.CLOSE,
                payload={"outcome": outcome.value, "rationale": rationale},
            ),
        )

    async def events(
        self,
        sync_session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> InternSyncEventPage:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        return _intern_sync_event_page(
            await self._transport.execute(
                _request(
                    "list_intern_runtime_events",
                    f"/smr/research-intern/runtimes/sync/{sync_session_id}/events",
                    query={
                        "after_sequence": after_sequence,
                        "limit": _bounded_limit(limit),
                    },
                )
            ),
            after_sequence=after_sequence,
            runtime_id=sync_session_id,
        )

    async def stream_events(
        self,
        sync_session_id: str,
        *,
        after_sequence: int = 0,
        timeout_seconds: float = 30.0,
    ) -> AsyncIterator[InternSyncEvent]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        expected_sequence = after_sequence + 1
        path = f"/smr/research-intern/runtimes/sync/{sync_session_id}/events/stream"
        async for frame in self._transport.stream_sse(
            path,
            params={"after_sequence": after_sequence},
            last_event_id=str(after_sequence) if after_sequence else None,
            timeout_seconds=_stream_timeout_seconds(timeout_seconds),
            operation_id="stream_intern_runtime_events",
        ):
            event = _intern_sync_stream_event(
                frame,
                expected_sequence=expected_sequence,
                runtime_id=sync_session_id,
            )
            expected_sequence = event.sequence + 1
            yield event

    async def tail(
        self,
        sync_session_id: str,
        *,
        after_sequence: int = 0,
        event_count_max: int = 1,
        timeout_seconds: float = 30.0,
    ) -> InternSyncEventPage:
        _stream_bound(event_count_max, name="event_count_max", maximum=500)
        events: list[InternSyncEvent] = []
        async for event in self.stream_events(
            sync_session_id,
            after_sequence=after_sequence,
            timeout_seconds=timeout_seconds,
        ):
            events.append(event)
            if len(events) >= event_count_max:
                break
        return InternSyncEventPage(
            events=tuple(events),
            next_sequence=events[-1].sequence if events else after_sequence,
        )


class AsyncResearchInternAsyncRuntimeAPI:
    """Native async transport for the organization's singleton Async Intern."""

    _PATH = "/smr/research-intern/async"

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def ensure(self, request: InternAsyncEnsureRequest) -> InternAsyncRuntime:
        return InternAsyncRuntime.from_wire(
            await self._transport.execute(
                _request(
                    "ensure_intern_async_runtime",
                    self._PATH,
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    async def get(self) -> InternAsyncRuntime:
        return InternAsyncRuntime.from_wire(
            await self._transport.execute(_request("get_intern_async_runtime", self._PATH))
        )

    async def command(self, request: InternAsyncCommandRequest) -> InternAsyncCommandReceipt:
        receipt = InternAsyncCommandReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "command_intern_async_runtime",
                    f"{self._PATH}/commands",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if receipt.command_id != request.command_id:
            raise ValueError("Async Intern command receipt identity drifted")
        return receipt

    async def send(self, request: InternAsyncInstructionRequest) -> InternAsyncCommandReceipt:
        return await self.command(request.to_command())

    async def pause(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        reason: str,
    ) -> InternAsyncCommandReceipt:
        return await self.command(
            InternAsyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternAsyncCommandKind.PAUSE,
                payload={"reason": reason},
            )
        )

    async def resume(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
    ) -> InternAsyncCommandReceipt:
        return await self.command(
            InternAsyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternAsyncCommandKind.RESUME,
            )
        )

    async def cancel(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        reason: str,
    ) -> InternAsyncCommandReceipt:
        return await self.command(
            InternAsyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternAsyncCommandKind.CANCEL,
                payload={"reason": reason},
            )
        )

    async def provide_input(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        interaction_id: str,
        body: str,
        context: dict[str, JsonValue] | None = None,
    ) -> InternAsyncCommandReceipt:
        return await self.command(
            InternAsyncCommandRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                command_kind=InternAsyncCommandKind.PROVIDE_INPUT,
                payload={
                    "interaction_id": interaction_id,
                    "body": body,
                    "context": context or {},
                },
            )
        )

    async def intervene(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        body: str,
        context: dict[str, JsonValue] | None = None,
    ) -> InternAsyncCommandReceipt:
        return await self.send(
            InternAsyncInstructionRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                instruction_kind=InternAsyncInstructionKind.INTERVENE,
                body=body,
                context=context or {},
            )
        )

    async def redirect_objective(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        objective: str,
        context: dict[str, JsonValue] | None = None,
    ) -> InternAsyncCommandReceipt:
        return await self.send(
            InternAsyncInstructionRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                instruction_kind=InternAsyncInstructionKind.REDIRECT_OBJECTIVE,
                body=objective,
                context=context or {},
            )
        )

    async def request_checkpoint(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        context: dict[str, JsonValue] | None = None,
    ) -> InternAsyncCommandReceipt:
        return await self.send(
            InternAsyncInstructionRequest(
                command_id=command_id,
                idempotency_key=idempotency_key,
                expected_generation=expected_generation,
                instruction_kind=InternAsyncInstructionKind.REQUEST_CHECKPOINT,
                context=context or {},
            )
        )

    async def events(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> InternAsyncEventPage:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        return _intern_async_event_page(
            await self._transport.execute(
                _request(
                    "list_intern_async_runtime_events",
                    f"{self._PATH}/events",
                    query={
                        "after_sequence": after_sequence,
                        "limit": _bounded_limit(limit),
                    },
                )
            ),
            after_sequence=after_sequence,
        )

    async def stream_events(
        self,
        *,
        after_sequence: int = 0,
        timeout_seconds: float = 30.0,
    ) -> AsyncIterator[InternAsyncEvent]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        expected_sequence = after_sequence + 1
        runtime_id: str | None = None
        async for frame in self._transport.stream_sse(
            f"{self._PATH}/events/stream",
            params={"after_sequence": after_sequence},
            last_event_id=str(after_sequence) if after_sequence else None,
            timeout_seconds=_stream_timeout_seconds(timeout_seconds),
            operation_id="stream_intern_async_runtime_events",
        ):
            event = _intern_async_stream_event(
                frame,
                expected_sequence=expected_sequence,
                runtime_id=runtime_id,
            )
            runtime_id = event.runtime_id
            expected_sequence = event.sequence + 1
            yield event

    async def tail(
        self,
        *,
        after_sequence: int = 0,
        event_count_max: int = 1,
        timeout_seconds: float = 30.0,
    ) -> InternAsyncEventPage:
        _stream_bound(event_count_max, name="event_count_max", maximum=500)
        events: list[InternAsyncEvent] = []
        async for event in self.stream_events(
            after_sequence=after_sequence,
            timeout_seconds=timeout_seconds,
        ):
            events.append(event)
            if len(events) >= event_count_max:
                break
        return InternAsyncEventPage(
            events=tuple(events),
            next_sequence=events[-1].sequence if events else after_sequence,
        )


class AsyncResearchInternAPI:
    """Native asynchronous organization Research Intern operations."""

    def __init__(
        self,
        transport: AsyncHttpTransport,
        *,
        allow_legacy_intern_sessions: bool = False,
    ) -> None:
        self._transport = transport
        self.sync_ = AsyncResearchInternSyncRuntimeAPI(transport)
        self.async_ = AsyncResearchInternAsyncRuntimeAPI(transport)
        self.factories = AsyncResearchInternFactoriesAPI(transport)
        self.decisions = AsyncResearchInternDecisionsAPI(transport)
        self.sessions = AsyncResearchInternSessionsAPI(
            transport,
            self,
            allow_legacy=allow_legacy_intern_sessions,
        )
        self.acceptance_receipts = AsyncResearchInternAcceptanceReceiptsAPI(transport)

    async def provision(
        self,
        request: ResearchInternProvisionRequest | None = None,
    ) -> ResearchInternResponse:
        """Provision or retrieve the one Intern for the authenticated organization."""
        return ResearchInternResponse.from_wire(
            await self._transport.execute(
                _request(
                    "provision_research_intern",
                    "/smr/research-intern",
                    body=cast(
                        JsonObject,
                        (request or ResearchInternProvisionRequest()).to_wire(),
                    ),
                )
            )
        )

    async def retrieve(self) -> ResearchInternResponse:
        """Retrieve the authenticated organization's Research Intern."""
        return ResearchInternResponse.from_wire(
            await self._transport.execute(_request("get_research_intern", "/smr/research-intern"))
        )

    async def update(
        self,
        request: ResearchInternPatchRequest,
    ) -> ResearchInternResponse:
        """Update mutable Intern policy, attribution, state, or display fields."""
        return ResearchInternResponse.from_wire(
            await self._transport.execute(
                _request(
                    "patch_research_intern",
                    "/smr/research-intern",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )


class AsyncResearchInternReactiveSession:
    """Native asynchronous peer of the event-driven Intern operator loop."""

    def __init__(
        self,
        api: AsyncResearchInternAPI,
        session: ResearchInternSessionResponse,
        *,
        cursor: ResearchInternEventCursor,
    ) -> None:
        self._api = api
        self.session = session
        self.cursor = cursor

    def _accept_event(self, event: ResearchInternEventResponse) -> None:
        if (
            event.session_id != self.session.session_id
            or event.sequence != self.cursor.after_sequence + 1
            or event.previous_state_generation != self.cursor.state_generation
            or event.state_generation != event.previous_state_generation + 1
        ):
            raise ValueError("Research Intern event broke the reconnect cursor chain")
        self.cursor = ResearchInternEventCursor(
            session_id=event.session_id,
            after_sequence=event.sequence,
            state_generation=event.state_generation,
            event_id=event.event_id,
            last_event_id=str(event.sequence),
        )

    @staticmethod
    def _event_turn_id(event: ResearchInternEventResponse) -> str | None:
        return ResearchInternReactiveSession._event_turn_id(event)

    async def observe(self, *, limit: int = 100) -> ResearchInternEventObservation:
        """Read one bounded page and advance the exact reconnect cursor."""
        events = await self._api.sessions.list_events(
            self.session.session_id,
            after_sequence=self.cursor.after_sequence,
            limit=limit,
        )
        for event in events:
            self._accept_event(event)
        self.session = await self._api.sessions.retrieve(self.session.session_id)
        return ResearchInternEventObservation(
            session=self.session,
            events=events,
            cursor=self.cursor,
        )

    async def synchronize(
        self,
        *,
        page_limit: int = 100,
        page_count_max: int = 20,
    ) -> tuple[ResearchInternEventResponse, ...]:
        """Drain unseen durable events with explicit page bounds."""
        _bounded_limit(page_limit)
        _stream_bound(page_count_max, name="page_count_max", maximum=100)
        observed: list[ResearchInternEventResponse] = []
        for _ in range(page_count_max):
            observation = await self.observe(limit=page_limit)
            observed.extend(observation.events)
            if self.cursor.after_sequence >= observation.session.last_event_sequence:
                return tuple(observed)
            if not observation.events:
                raise ValueError("Intern session reported unseen events without an event page")
        raise ValueError("Intern session synchronization exceeded page_count_max")

    async def synchronize_runtime(
        self,
        *,
        page_limit: int = 200,
        page_count_max: int = 20,
    ) -> tuple[ResearchInternEventResponse, ...]:
        """Reconcile runtime projection only at an initial/reconnect boundary."""
        _bounded_limit(page_limit)
        _stream_bound(page_count_max, name="page_count_max", maximum=100)
        await self.synchronize(
            page_limit=page_limit,
            page_count_max=page_count_max,
        )
        projected: list[ResearchInternEventResponse] = []
        for _ in range(page_count_max):
            response = await self._api.sessions.sync(
                self.session.session_id,
                limit=page_limit,
            )
            projected.extend(response.events)
            observed = await self.synchronize(
                page_limit=page_limit,
                page_count_max=page_count_max,
            )
            observed_ids = {event.event_id for event in observed}
            if any(event.event_id not in observed_ids for event in response.events):
                raise ValueError("runtime-projected events were absent from the event log")
            if not response.has_more:
                return tuple(projected)
        raise ValueError("Research Intern runtime synchronization exceeded page_count_max")

    async def watch(
        self,
        *,
        event_count_max: int = 1,
        frame_count_max: int = 100,
        reconnect_count_max: int = 3,
        timeout_seconds: float = 30.0,
    ) -> ResearchInternEventStreamObservation:
        """Consume the canonical SSE stream without runtime-sync polling."""
        observation = await self._api.sessions.watch(
            self.session.session_id,
            cursor=self.cursor,
            event_count_max=event_count_max,
            frame_count_max=frame_count_max,
            reconnect_count_max=reconnect_count_max,
            timeout_seconds=timeout_seconds,
        )
        for event in observation.events:
            self._accept_event(event)
        if observation.cursor is not None and observation.cursor != self.cursor:
            raise ValueError("Intern stream observation cursor diverged from its event chain")
        if observation.events:
            self.session = await self._api.sessions.retrieve(self.session.session_id)
        return ResearchInternEventStreamObservation(
            session_id=observation.session_id,
            frames=observation.frames,
            events=observation.events,
            cursor=self.cursor,
            timed_out=observation.timed_out,
        )

    async def _watch_through_sequence(
        self,
        sequence: int,
        *,
        timeout_seconds: float,
    ) -> tuple[ResearchInternEventResponse, ...]:
        if sequence < self.cursor.after_sequence:
            return ()
        deadline = time.monotonic() + _stream_timeout_seconds(timeout_seconds)
        observed: list[ResearchInternEventResponse] = []
        while self.cursor.after_sequence < sequence:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResearchInternPollingTimeoutError(
                    f"event stream did not reach sequence {sequence} for "
                    f"session {self.session.session_id}"
                )
            observation = await self.watch(
                event_count_max=min(sequence - self.cursor.after_sequence, 500),
                timeout_seconds=remaining,
            )
            observed.extend(observation.events)
            if not observation.events:
                raise ResearchInternPollingTimeoutError(
                    f"event stream did not reach sequence {sequence} for "
                    f"session {self.session.session_id}"
                )
        return tuple(observed)

    async def _wait_for_turn_terminal(
        self,
        turn_id: str,
        *,
        timeout_seconds: float,
        already_observed: tuple[ResearchInternEventResponse, ...] = (),
    ) -> tuple[ResearchInternEventResponse, tuple[ResearchInternEventResponse, ...]]:
        deadline = time.monotonic() + _stream_timeout_seconds(timeout_seconds)
        observed = list(already_observed)
        while True:
            terminal = next(
                (
                    event
                    for event in observed
                    if event.event_kind
                    in {
                        ResearchInternEventKind.AGENT_MESSAGE,
                        ResearchInternEventKind.ERROR,
                    }
                    and self._event_turn_id(event) == turn_id
                ),
                None,
            )
            if terminal is not None:
                return terminal, tuple(observed)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResearchInternPollingTimeoutError(
                    f"no terminal event observed for turn {turn_id}"
                )
            observation = await self.watch(
                event_count_max=1,
                timeout_seconds=remaining,
            )
            observed.extend(observation.events)
            if not observation.events:
                raise ResearchInternPollingTimeoutError(
                    f"no terminal event observed for turn {turn_id}"
                )

    def _run_id(self) -> str:
        if self.session.run_id is None:
            raise ValueError("Magi actuation requires a session bound to a live run_id")
        return self.session.run_id

    async def _intern_generation(self) -> int:
        return (await self._api.retrieve()).state_generation

    async def delegate(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        evidence_refs: list[str] | None = None,
        state_patch: JsonObject | None = None,
        mode: MagiMode = MagiMode.SYNC,
    ) -> MagiDecisionReceiptResponse:
        """Delegate this exact target through Casper."""
        await self.synchronize()
        return await self._api.decisions.delegate(
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self._run_id(),
            session_id=self.session.session_id,
            expected_state_generation=await self._intern_generation(),
            idempotency_key=idempotency_key,
            rationale=rationale,
            evidence_refs=evidence_refs,
            state_patch=state_patch,
            mode=mode,
        )

    async def pause(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Pause this exact target through Melchior by default."""
        await self.synchronize()
        return await self._api.decisions.pause(
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self._run_id(),
            session_id=self.session.session_id,
            expected_state_generation=await self._intern_generation(),
            idempotency_key=idempotency_key,
            rationale=rationale,
            evidence_refs=evidence_refs,
            mode=mode,
        )

    async def intervene(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        state_patch: JsonObject,
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Intervene on this paused target through Melchior by default."""
        await self.synchronize()
        return await self._api.decisions.intervene(
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self._run_id(),
            session_id=self.session.session_id,
            expected_state_generation=await self._intern_generation(),
            idempotency_key=idempotency_key,
            rationale=rationale,
            state_patch=state_patch,
            evidence_refs=evidence_refs,
            mode=mode,
        )

    async def resume(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.ASYNC,
    ) -> MagiDecisionReceiptResponse:
        """Resume this target through Melchior by default."""
        await self.synchronize()
        return await self._api.decisions.resume(
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self._run_id(),
            session_id=self.session.session_id,
            expected_state_generation=await self._intern_generation(),
            idempotency_key=idempotency_key,
            rationale=rationale,
            evidence_refs=evidence_refs,
            mode=mode,
        )

    async def send_operator_turn(
        self,
        body: str,
        *,
        idempotency_key: str,
        mode: MagiMode = MagiMode.SYNC,
        control: ResearchInternTurnControl | None = None,
        rationale: str | None = None,
        state_patch: JsonObject | None = None,
        evidence_refs: list[str] | None = None,
        wait_timeout_seconds: float = 15.0,
        poll_interval_ms: int = 250,
        stream_timeout_seconds: float = 30.0,
    ) -> ResearchInternOperatorTurn:
        """Submit one turn and observe its entire durable response chain."""
        self._run_id()
        await self.synchronize_runtime()
        response = await self._api.sessions.turn(
            self.session.session_id,
            ResearchInternTurnRequest(
                body=body,
                mode=mode,
                idempotency_key=idempotency_key,
                expected_session_state_generation=self.cursor.state_generation,
                expected_intern_state_generation=await self._intern_generation(),
                control=control,
                rationale=rationale,
                state_patch=state_patch or {},
                evidence_refs=evidence_refs or [],
                wait_timeout_seconds=wait_timeout_seconds,
                poll_interval_ms=poll_interval_ms,
            ),
        )
        observed = await self._watch_through_sequence(
            response.reconnect_after_sequence,
            timeout_seconds=stream_timeout_seconds,
        )
        if self.cursor.after_sequence != response.reconnect_after_sequence:
            raise ValueError("Research Intern turn response skipped unseen event sequence")
        self.session = await self._api.sessions.retrieve(self.session.session_id)
        return ResearchInternOperatorTurn(
            response=response,
            observed_events=observed,
            cursor=self.cursor,
        )

    async def exchange(
        self,
        body: str,
        *,
        idempotency_key: str,
        mode: MagiMode = MagiMode.SYNC,
        control: ResearchInternTurnControl | None = None,
        rationale: str | None = None,
        state_patch: JsonObject | None = None,
        evidence_refs: list[str] | None = None,
        wait_timeout_seconds: float = 15.0,
        poll_interval_ms: int = 250,
        recovery_timeout_seconds: float = 60.0,
    ) -> ResearchInternExchange:
        """Recover the exact turn terminal from the canonical async SSE stream."""
        turn = await self.send_operator_turn(
            body,
            idempotency_key=idempotency_key,
            mode=mode,
            control=control,
            rationale=rationale,
            state_patch=state_patch,
            evidence_refs=evidence_refs,
            wait_timeout_seconds=wait_timeout_seconds,
            poll_interval_ms=poll_interval_ms,
            stream_timeout_seconds=min(
                _stream_timeout_seconds(recovery_timeout_seconds),
                30.0,
            ),
        )
        response = turn.response
        if response.status is ResearchInternTurnStatus.FAILED:
            raise ResearchInternTurnFailedError(response)
        try:
            terminal, observed = await self._wait_for_turn_terminal(
                response.turn_id,
                timeout_seconds=recovery_timeout_seconds,
                already_observed=turn.observed_events,
            )
        except ResearchInternPollingTimeoutError as error:
            error.response = response
            raise
        if terminal.event_kind is ResearchInternEventKind.ERROR:
            raise ResearchInternTurnFailedError(response, terminal_event=terminal)
        if response.agent_event is not None and response.agent_event.event_id != terminal.event_id:
            raise ValueError("turn response agent event diverged from canonical SSE")
        return ResearchInternExchange(
            operator_turn=turn,
            agent_event=terminal,
            observed_events=observed,
            cursor=self.cursor,
        )

    async def verdict(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        verdict: str,
        uncertainty: float,
        evidence_refs: list[str],
        state_patch: JsonObject | None = None,
    ) -> MagiDecisionReceiptResponse:
        """Record the evidence-linked final Balthasar verdict."""
        await self.synchronize()
        return await self._api.decisions.verdict(
            idempotency_key=idempotency_key,
            expected_state_generation=await self._intern_generation(),
            rationale=rationale,
            verdict=verdict,
            uncertainty=uncertainty,
            evidence_refs=evidence_refs,
            factory_id=self.session.factory_id,
            project_id=self.session.project_id,
            effort_id=self.session.effort_id,
            run_id=self.session.run_id,
            session_id=self.session.session_id,
            state_patch=state_patch,
        )

    async def close(
        self,
        *,
        idempotency_key: str,
        status: Literal[
            "completed",
            "partial",
            "failed",
            "stopped",
            "canceled",
            "archived",
        ],
        rationale: str,
        evidence_refs: list[str] | None = None,
    ) -> ResearchInternSessionResponse:
        """Close this session at its exact durable generation."""
        await self.synchronize()
        self.session = await self._api.sessions.close(
            self.session.session_id,
            ResearchInternSessionCloseRequest(
                idempotency_key=idempotency_key,
                expected_state_generation=self.cursor.state_generation,
                status=status,
                rationale=rationale,
                evidence_refs=evidence_refs or [],
            ),
        )
        await self.synchronize()
        return self.session


class AsyncProjectComputerAPI:
    """Native asynchronous Project Computer lifecycle operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def provision(
        self,
        project_id: ProjectId,
        request: ProjectComputerProvisionRequest,
    ) -> ProjectComputerResponse:
        """Provision a Project Computer from an exact Git and snapshot identity."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "provision_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if computer.project_id != str(project_id) or computer.factory_id != request.factory_id:
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    async def retrieve(
        self,
        project_id: ProjectId,
        factory_id: FactoryId,
    ) -> ProjectComputerResponse:
        """Retrieve the current Project Computer for a Project and Factory."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "get_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    query={"factory_id": str(factory_id)},
                )
            )
        )
        if computer.project_id != str(project_id) or computer.factory_id != str(factory_id):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    async def replace(
        self,
        project_id: ProjectId,
        request: ProjectComputerReplaceRequest,
    ) -> ProjectComputerResponse:
        """Replace a Project Computer using an adapter-authored restore receipt."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "replace_project_computer",
                    f"/smr/projects/{project_id}/computer/replace",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if computer.project_id != str(project_id) or computer.factory_id != request.factory_id:
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    async def retire(
        self,
        project_id: ProjectId,
        factory_id: FactoryId,
    ) -> ProjectComputerResponse:
        """Retire the Project and Factory Computer while retaining its receipt."""
        computer = ProjectComputerResponse.from_wire(
            await self._transport.execute(
                _request(
                    "retire_project_computer",
                    f"/smr/projects/{project_id}/computer",
                    query={"factory_id": str(factory_id)},
                )
            )
        )
        if computer.project_id != str(project_id) or computer.factory_id != str(factory_id):
            raise ValueError("Project Computer response crossed its requested boundary")
        return computer

    async def inspect(
        self,
        project_id: ProjectId,
        request: ProjectComputerInspectRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Inspect one generation without crossing its Factory boundary."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "inspect_project_computer",
                    f"/smr/projects/{project_id}/computer/inspect",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(
            receipt,
            request,
            operation=ProjectRuntimeOperation.INSPECT,
        )

    async def execute(
        self,
        project_id: ProjectId,
        request: ProjectComputerExecuteRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Execute a bounded argv under an exact lease and fencing digest."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "execute_project_computer",
                    f"/smr/projects/{project_id}/computer/execute",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(
            receipt,
            request,
            operation=ProjectRuntimeOperation.EXECUTE,
        )

    async def reconcile(
        self,
        project_id: ProjectId,
        request: ProjectComputerOperationReconcileRequest,
    ) -> ProjectRuntimeOperationReceipt:
        """Observe an accepted operation with an unknown provider outcome."""
        receipt = ProjectRuntimeOperationReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "reconcile_project_computer_operation",
                    f"/smr/projects/{project_id}/computer/operations/reconcile",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_operation_receipt(receipt, request, operation=None)

    async def acquire_lease(
        self,
        project_id: ProjectId,
        request: ProjectComputerLeaseAcquireRequest,
    ) -> ProjectComputerLeaseResponse:
        """Acquire one generation-fenced execution lease."""
        lease = ProjectComputerLeaseResponse.from_wire(
            await self._transport.execute(
                _request(
                    "acquire_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.state != "active"
        ):
            raise ValueError("Project Computer lease acquisition identity drifted")
        return lease

    async def renew_lease(
        self,
        project_id: ProjectId,
        lease_id: str,
        request: ProjectComputerLeaseRenewRequest,
    ) -> ProjectComputerLeaseResponse:
        """Renew one exact lease while preserving its fencing identity."""
        lease = ProjectComputerLeaseResponse.from_wire(
            await self._transport.execute(
                _request(
                    "renew_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases/{lease_id}/renew",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.lease_id != lease_id
            or lease.fencing_token_digest != request.fencing_token_digest
            or lease.state != "active"
        ):
            raise ValueError("Project Computer lease renewal identity drifted")
        return lease

    async def release_lease(
        self,
        project_id: ProjectId,
        lease_id: str,
        request: ProjectComputerLeaseReleaseRequest,
    ) -> ProjectComputerLeaseResponse:
        """Release one exact lease and retain the final fencing identity."""
        lease = ProjectComputerLeaseResponse.from_wire(
            await self._transport.execute(
                _request(
                    "release_project_computer_lease",
                    f"/smr/projects/{project_id}/computer/leases/{lease_id}/release",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            lease.project_id != str(project_id)
            or lease.factory_id != request.factory_id
            or lease.generation != request.expected_generation
            or lease.lease_id != lease_id
            or lease.fencing_token_digest != request.fencing_token_digest
            or lease.state != "released"
        ):
            raise ValueError("Project Computer lease release identity drifted")
        return lease

    async def cleanup(
        self,
        factory_id: FactoryId,
        request: ProjectComputerCleanupRequest,
    ) -> ProjectComputerCleanupReceiptResponse:
        """Retire every Project Computer in one Factory with owner-authored proof."""
        receipt = ProjectComputerCleanupReceiptResponse.from_wire(
            await self._transport.execute(
                _request(
                    "cleanup_factory_project_computers",
                    (f"/smr/research-intern/factories/{factory_id}/project-computers/cleanup"),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            receipt.factory_id != str(factory_id)
            or receipt.idempotency_key != request.idempotency_key
        ):
            raise ValueError("Project Computer cleanup receipt identity drifted")
        return receipt


class AsyncProjectDataBindingsAPI:
    """Native asynchronous Project data-binding operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(
        self,
        project_id: ProjectId,
        request: DataBindingCreateRequest,
    ) -> DataBindingResponse:
        """Create one Factory-scoped Project data binding."""
        binding = DataBindingResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_data_binding",
                    f"/smr/projects/{project_id}/data-bindings",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if binding.project_id != str(project_id) or binding.factory_id != request.factory_id:
            raise ValueError("Data Binding response crossed its requested boundary")
        return binding

    async def list(
        self,
        project_id: ProjectId,
    ) -> tuple[DataBindingResponse, ...]:
        """List data bindings without crossing the requested Project."""
        bindings = _data_bindings(
            await self._transport.execute(
                _request(
                    "list_data_bindings",
                    f"/smr/projects/{project_id}/data-bindings",
                )
            )
        )
        if any(binding.project_id != str(project_id) for binding in bindings):
            raise ValueError("Data Binding list crossed its project boundary")
        return bindings

    async def create_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        request: DatasetRevisionCreateRequest,
    ) -> DatasetRevisionResponse:
        """Append an immutable revision to a Project data binding."""
        revision = DatasetRevisionResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_dataset_revision",
                    (f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/revisions"),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            revision.project_id != str(project_id)
            or str(revision.data_binding_id) != data_binding_id
        ):
            raise ValueError("Dataset Revision response identity drifted")
        return revision

    async def list_revisions(
        self,
        project_id: ProjectId,
        data_binding_id: str,
    ) -> tuple[DatasetRevisionResponse, ...]:
        """List immutable revisions for one Project data binding."""
        revisions = _dataset_revisions(
            await self._transport.execute(
                _request(
                    "list_dataset_revisions",
                    (f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/revisions"),
                )
            )
        )
        if any(
            revision.project_id != str(project_id)
            or str(revision.data_binding_id) != data_binding_id
            for revision in revisions
        ):
            raise ValueError("Dataset Revision list identity drifted")
        return revisions

    async def transition_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        dataset_revision_id: str,
        request: DatasetRevisionLifecycleRequest,
    ) -> DatasetRevisionResponse:
        """Apply a fail-closed lifecycle transition to one exact revision."""
        revision = DatasetRevisionResponse.from_wire(
            await self._transport.execute(
                _request(
                    "transition_dataset_revision_lifecycle",
                    (
                        f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/"
                        f"revisions/{dataset_revision_id}/lifecycle"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if (
            revision.project_id != str(project_id)
            or str(revision.data_binding_id) != data_binding_id
            or str(revision.dataset_revision_id) != dataset_revision_id
            or revision.state is not request.target_state
        ):
            raise ValueError("DatasetRevision lifecycle response identity drifted")
        return revision

    async def prepare_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        request: DatasetRevisionPrepareRequest,
    ) -> DatasetRevisionPreparationResponse:
        """Prepare immutable upload targets for one exact DatasetRevision draft."""
        prepared = DatasetRevisionPreparationResponse.from_wire(
            await self._transport.execute(
                _dataset_publication_request(
                    "prepareDatasetRevisionPublication",
                    (
                        f"/smr/projects/{project_id}/data-bindings/"
                        f"{data_binding_id}/revisions:prepare"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        draft = request.draft
        if (
            prepared.idempotency_key != request.idempotency_key
            or prepared.project_id != str(project_id)
            or str(prepared.data_binding_id) != data_binding_id
            or prepared.dataset_revision_id != draft.dataset_revision_id
            or prepared.factory_id != draft.factory_id
            or prepared.org_id != draft.org_id
        ):
            raise ValueError("DatasetRevision preparation identity drifted")
        return prepared

    async def finalize_revision(
        self,
        project_id: ProjectId,
        data_binding_id: str,
        preparation_id: str,
        request: DatasetRevisionFinalizeRequest,
    ) -> DatasetRevisionFinalizeResponse:
        """Finalize one prepared DatasetRevision and verify all owner receipts."""
        finalized = DatasetRevisionFinalizeResponse.from_wire(
            await self._transport.execute(
                _dataset_publication_request(
                    "finalizeDatasetRevisionPublication",
                    (
                        f"/smr/projects/{project_id}/data-bindings/{data_binding_id}/"
                        f"revision-preparations/{preparation_id}:finalize"
                    ),
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        sealed = finalized.sealed_revision
        if (
            str(finalized.preparation_id) != preparation_id
            or sealed.project_id != str(project_id)
            or str(sealed.binding_id) != data_binding_id
        ):
            raise ValueError("DatasetRevision finalization identity drifted")
        return finalized


__all__ = [
    "AsyncProjectComputerAPI",
    "AsyncProjectDataBindingsAPI",
    "AsyncResearchInternAsyncRuntimeAPI",
    "AsyncResearchInternAcceptanceReceiptsAPI",
    "AsyncResearchInternAPI",
    "AsyncResearchInternDecisionsAPI",
    "AsyncResearchInternFactoriesAPI",
    "AsyncResearchInternReactiveSession",
    "AsyncResearchInternSessionsAPI",
    "LEGACY_INTERN_SESSIONS_ENV",
    "LegacyInternSessionsDisabledError",
    "ProjectComputerAPI",
    "ProjectDataBindingsAPI",
    "ResearchInternAcceptanceReceiptsAPI",
    "ResearchInternAPI",
    "ResearchInternAsyncRuntimeAPI",
    "ResearchInternDecisionsAPI",
    "ResearchInternEventCursor",
    "ResearchInternEventObservation",
    "ResearchInternEventStreamObservation",
    "ResearchInternExchange",
    "ResearchInternFactoriesAPI",
    "ResearchInternOperatorTurn",
    "ResearchInternPollingTimeoutError",
    "ResearchInternReactiveSession",
    "ResearchInternSessionsAPI",
    "ResearchInternTurnFailedError",
    "legacy_intern_sessions_enabled",
]
