"""Research Intern, Magi decision, and project resource operations."""

from __future__ import annotations

import builtins
import time
from collections.abc import AsyncIterator, Iterator
from typing import Literal, cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
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
    InternAcceptanceFixtureReceipt,
    InternAcceptanceFixtureRequest,
    InternAsyncCommandKind,
    InternAsyncCommandReceipt,
    InternAsyncCommandRequest,
    InternAsyncEnsureRequest,
    InternAsyncEvent,
    InternAsyncEventPage,
    InternAsyncEventStreamEnvelope,
    InternAsyncHandoffModelRequest,
    InternAsyncHandoffReviewRequest,
    InternAsyncInstructionKind,
    InternAsyncInstructionRequest,
    InternAsyncRuntime,
    InternAsyncRuntimeBudget,
    InternCrossMetaThreadMessage,
    InternCrossMetaThreadMessageCreateRequest,
    InternMetaHandoff,
    InternMetaHandoffContinueRequest,
    InternMetaThread,
    InternMetaThreadKind,
    InternMetaThreadSegment,
    InternRuntimeOutcome,
    InternSyncApprovalCard,
    InternSyncCommandKind,
    InternSyncCommandReceipt,
    InternSyncCommandRequest,
    InternSyncDeployPacket,
    InternSyncEvent,
    InternSyncEventPage,
    InternSyncEventStreamEnvelope,
    InternSyncPresenceLease,
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
    ResearchInternFactoryMembershipResponse,
    ResearchInternPatchRequest,
    ResearchInternProvisionRequest,
    ResearchInternResponse,
)
from synth_ai.sdk.research.intern_program import (
    AsyncInternProgramAPI,
    InternProgramAPI,
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


def _async_ensure_request_with_budget_overrides(
    request: InternAsyncEnsureRequest,
    *,
    maximum_daily_cost_cents: int | None = None,
    maximum_monthly_cost_cents: int | None = None,
) -> InternAsyncEnsureRequest:
    """Merge optional day/month ceiling kwargs into the ensure budget."""

    if maximum_daily_cost_cents is None and maximum_monthly_cost_cents is None:
        return request
    budget = InternAsyncRuntimeBudget.model_validate(
        {
            **request.budget.model_dump(mode="python"),
            **(
                {"maximum_daily_cost_cents": maximum_daily_cost_cents}
                if maximum_daily_cost_cents is not None
                else {}
            ),
            **(
                {"maximum_monthly_cost_cents": maximum_monthly_cost_cents}
                if maximum_monthly_cost_cents is not None
                else {}
            ),
        }
    )
    return InternAsyncEnsureRequest.model_validate(
        {
            **request.model_dump(mode="python"),
            "budget": budget.model_dump(mode="python"),
        }
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
                    "get_public_research_intern_acceptance_receipt",
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


class ResearchInternMetaThreadsAPI:
    """Exact graph and cross-lane protocol projections for one Intern."""

    _PATH = "/smr/research-intern/meta-threads"

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self) -> tuple[InternMetaThread, ...]:
        """List backend-owned Sync and Async meta-thread projections."""
        return tuple(
            InternMetaThread.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    self._transport.execute(_request("list_intern_meta_threads", self._PATH)),
                ),
                operation_id="list_intern_meta_threads",
            )
        )

    def get(self, meta_thread_id: str) -> InternMetaThread:
        """Retrieve one meta-thread and reject a mismatched response identity."""
        thread = InternMetaThread.from_wire(
            self._transport.execute(
                _request(
                    "get_intern_meta_thread",
                    f"{self._PATH}/{meta_thread_id}",
                )
            )
        )
        if thread.meta_thread_id != meta_thread_id:
            raise ValueError("Intern meta-thread identity drifted")
        return thread

    def segments(self, meta_thread_id: str) -> tuple[InternMetaThreadSegment, ...]:
        """List the live and sealed segments of one meta-thread."""
        return tuple(
            InternMetaThreadSegment.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    self._transport.execute(
                        _request(
                            "list_intern_meta_thread_segments",
                            f"{self._PATH}/{meta_thread_id}/segments",
                        )
                    ),
                ),
                operation_id="list_intern_meta_thread_segments",
            )
        )

    def handoffs(self, meta_thread_id: str) -> tuple[InternMetaHandoff, ...]:
        """List durable cross-lane handoff records for one meta-thread."""
        return tuple(
            InternMetaHandoff.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    self._transport.execute(
                        _request(
                            "list_intern_meta_thread_handoffs",
                            f"{self._PATH}/{meta_thread_id}/handoffs",
                        )
                    ),
                ),
                operation_id="list_intern_meta_thread_handoffs",
            )
        )

    def messages(
        self, meta_thread_id: str, *, limit: int = 200
    ) -> tuple[InternCrossMetaThreadMessage, ...]:
        """Read a bounded page of cross-meta-thread messages."""
        return tuple(
            InternCrossMetaThreadMessage.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    self._transport.execute(
                        _request(
                            "list_intern_meta_thread_messages",
                            f"{self._PATH}/{meta_thread_id}/messages",
                            query={"limit": _bounded_limit(limit)},
                        )
                    ),
                ),
                operation_id="list_intern_meta_thread_messages",
            )
        )

    def send(
        self, request: InternCrossMetaThreadMessageCreateRequest
    ) -> InternCrossMetaThreadMessage:
        """Send a typed cross-lane message and verify its receipt identity."""
        message = InternCrossMetaThreadMessage.from_wire(
            self._transport.execute(
                _request(
                    "create_intern_meta_thread_message",
                    f"{self._PATH}/messages",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if message.message_id != request.message_id:
            raise ValueError("Cross meta-thread message identity drifted")
        return message


class ResearchInternSyncRuntimeAPI:
    """Synchronous transport for durable operator-present Sync sessions."""

    _PATH = "/smr/research-intern/sync-sessions"

    def __init__(
        self,
        transport: HttpTransport,
        meta_threads: ResearchInternMetaThreadsAPI | None = None,
    ) -> None:
        self._transport = transport
        self._meta_threads = meta_threads or ResearchInternMetaThreadsAPI(transport)

    def branches(self) -> tuple[InternMetaThreadSegment, ...]:
        """List the Sync head plus every live or sealed branch projection."""

        sync_threads = [
            thread
            for thread in self._meta_threads.list()
            if thread.kind is InternMetaThreadKind.SYNC
        ]
        if len(sync_threads) != 1:
            raise ValueError("Research Intern must expose exactly one Sync meta-thread")
        return self._meta_threads.segments(sync_threads[0].meta_thread_id)

    def create(self, request: InternSyncSessionCreateRequest) -> InternSyncSession:
        """Create a Sync session from a typed backend request."""
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
        """List a bounded page of typed Sync session projections."""
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
        """Retrieve a Sync session and reject response identity drift."""
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

    def deploy_packet(self, sync_session_id: str) -> InternSyncDeployPacket:
        """Retrieve the backend deploy packet for the exact Sync session."""
        packet = InternSyncDeployPacket.from_wire(
            self._transport.execute(
                _request(
                    "get_intern_sync_deploy_packet",
                    f"{self._PATH}/{sync_session_id}/deploy-packet",
                )
            )
        )
        if packet.sync_session_id != sync_session_id:
            raise ValueError("Sync Intern deploy packet identity drifted")
        return packet

    def presence(
        self,
        sync_session_id: str,
        *,
        connection_id: str,
        connection_generation: int = 1,
    ) -> InternSyncPresenceLease:
        """Acquire or renew this operator's presence lease on the session.

        Billed sync launches (rollouts, deploys) are gated on an active
        authenticated operator presence (``intern_sync_active_presence_required``).
        Hold and renew this lease (before ``expires_at``) for as long as the
        Intern is expected to launch work on your behalf.
        """

        lease = InternSyncPresenceLease.from_wire(
            self._transport.execute(
                _request(
                    "acquire_intern_sync_presence",
                    f"{self._PATH}/{sync_session_id}/presence",
                    body={
                        "connection_id": connection_id,
                        "connection_generation": connection_generation,
                    },
                )
            )
        )
        if lease.sync_session_id != sync_session_id:
            raise ValueError("Sync Intern presence lease identity drifted")
        return lease

    def release_presence(
        self,
        sync_session_id: str,
        *,
        connection_id: str,
        connection_generation: int = 1,
    ) -> InternSyncPresenceLease:
        """Release this operator's presence lease on the session."""

        lease = InternSyncPresenceLease.from_wire(
            self._transport.execute(
                _request(
                    "release_intern_sync_presence",
                    f"{self._PATH}/{sync_session_id}/presence/release",
                    body={
                        "connection_id": connection_id,
                        "connection_generation": connection_generation,
                    },
                )
            )
        )
        if lease.sync_session_id != sync_session_id:
            raise ValueError("Sync Intern presence lease identity drifted")
        return lease

    def approvals(self, sync_session_id: str) -> tuple[InternSyncApprovalCard, ...]:
        """List operator approval cards for this session.

        Billed MCP actions (rollout launches, deploys) park in
        ``awaiting_approval`` with ``capability_operator_approval_required``
        until the present operator decides them.
        """

        payload = self._transport.execute(
            _request(
                "list_intern_sync_approvals",
                f"{self._PATH}/{sync_session_id}/approvals",
            )
        )
        cards = tuple(
            InternSyncApprovalCard.from_wire(item)
            for item in array_value(
                cast(JsonValue, payload), operation_id="list_intern_sync_approvals"
            )
        )
        if any(card.sync_session_id != sync_session_id for card in cards):
            raise ValueError("Sync Intern approval card identity drifted")
        return cards

    def decide_approval(
        self,
        approval_id: str,
        *,
        decision: Literal["approve", "deny", "edit"],
        comment: str | None = None,
    ) -> InternSyncApprovalCard:
        """Decide one approval card: ``approve``, ``deny``, or ``edit``.

        Requires an active presence lease on the card's session
        (``intern_sync_active_presence_required`` otherwise).
        """

        body: JsonObject = {"decision": decision}
        if comment is not None:
            body["comment"] = comment
        card = InternSyncApprovalCard.from_wire(
            self._transport.execute(
                _request(
                    "decide_intern_sync_approval",
                    f"/smr/research-intern/sync-approvals/{approval_id}/decision",
                    body=body,
                )
            )
        )
        if card.approval_id != approval_id:
            raise ValueError("Sync Intern approval card identity drifted")
        return card

    def command(
        self,
        sync_session_id: str,
        request: InternSyncCommandRequest,
    ) -> InternSyncCommandReceipt:
        """Submit a fenced command and verify its command and runtime identities."""
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
        """Submit an operator message with an idempotency key and expected generation."""
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
        """Submit a fenced intervention with the requested state patch and context."""
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
        """Answer one interaction through a generation-fenced Sync command."""
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
        """Request a fenced pause and retain the operator rationale in the receipt.

        Pause and close act on the session, not the machine: the shared org
        exe.dev VM (one box for all of the org's Sync and Async work) is
        retained, so guest workspaces and Codex threads survive for resume.
        """

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
        """Request resumption at the expected Sync generation."""
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
        """Close a Sync session with an explicit outcome and rationale."""
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
        """Read a bounded event page after a non-negative reconnect sequence."""
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
        """Stream typed SSE events while validating runtime identity and contiguous sequence."""
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
        """Collect a bounded number of streamed events and return their reconnect cursor."""
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

    def ensure(
        self,
        request: InternAsyncEnsureRequest,
        *,
        maximum_daily_cost_cents: int | None = None,
        maximum_monthly_cost_cents: int | None = None,
    ) -> InternAsyncRuntime:
        """Ensure the org Async Intern. Day/month kwargs override ``request.budget``."""

        ensure_request = _async_ensure_request_with_budget_overrides(
            request,
            maximum_daily_cost_cents=maximum_daily_cost_cents,
            maximum_monthly_cost_cents=maximum_monthly_cost_cents,
        )
        return InternAsyncRuntime.from_wire(
            self._transport.execute(
                _request(
                    "ensure_intern_async_runtime",
                    self._PATH,
                    body=cast(JsonObject, ensure_request.to_wire()),
                )
            )
        )

    def get(self) -> InternAsyncRuntime:
        """Retrieve the organization Intern's backend-owned Async runtime."""
        return InternAsyncRuntime.from_wire(
            self._transport.execute(_request("get_intern_async_runtime", self._PATH))
        )

    def command(self, request: InternAsyncCommandRequest) -> InternAsyncCommandReceipt:
        """Submit a fenced Async command and reject command receipt identity drift."""
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

    def handoff_model(self, request: InternAsyncHandoffModelRequest) -> InternAsyncCommandReceipt:
        """Change Async model/effort via spine handoff (no meta-thread id)."""

        receipt = InternAsyncCommandReceipt.from_wire(
            self._transport.execute(
                _request(
                    "handoff_intern_async_model",
                    f"{self._PATH}/handoff-model",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if receipt.command_id != request.command_id:
            raise ValueError("Async Intern handoff-model receipt identity drifted")
        return receipt

    def seal_handoff_for_review(
        self, request: InternAsyncHandoffReviewRequest
    ) -> InternMetaHandoff:
        """Attended seal: park model/effort switch at needs_review."""

        return InternMetaHandoff.from_wire(
            self._transport.execute(
                _request(
                    "seal_intern_async_handoff_for_review",
                    f"{self._PATH}/handoffs/review",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    def list_handoffs(self) -> tuple[InternMetaHandoff, ...]:
        """List the Async runtime's durable handoff review records."""
        return tuple(
            InternMetaHandoff.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    self._transport.execute(
                        _request("list_intern_async_handoffs", f"{self._PATH}/handoffs")
                    ),
                ),
                operation_id="list_intern_async_handoffs",
            )
        )

    def approve_handoff(self, handoff_id: str) -> InternMetaHandoff:
        """Approve one backend-owned Async handoff review."""
        return InternMetaHandoff.from_wire(
            self._transport.execute(
                _request(
                    "approve_intern_async_handoff",
                    f"{self._PATH}/handoffs/{handoff_id}/approve",
                    body=cast(JsonObject, {}),
                )
            )
        )

    def reject_handoff(self, handoff_id: str) -> InternMetaHandoff:
        """Reject one backend-owned Async handoff review."""
        return InternMetaHandoff.from_wire(
            self._transport.execute(
                _request(
                    "reject_intern_async_handoff",
                    f"{self._PATH}/handoffs/{handoff_id}/reject",
                    body=cast(JsonObject, {}),
                )
            )
        )

    def continue_handoff(
        self,
        handoff_id: str,
        request: InternMetaHandoffContinueRequest | None = None,
    ) -> InternMetaHandoff:
        """Continue a handoff with an optional typed continuation request."""
        body = cast(JsonObject, request.to_wire()) if request is not None else cast(JsonObject, {})
        return InternMetaHandoff.from_wire(
            self._transport.execute(
                _request(
                    "continue_intern_async_handoff",
                    f"{self._PATH}/handoffs/{handoff_id}/continue",
                    body=body,
                )
            )
        )

    def send(self, request: InternAsyncInstructionRequest) -> InternAsyncCommandReceipt:
        """Convert an Async instruction to a command and return its verified receipt."""
        return self.command(request.to_command())

    def pause(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        reason: str,
    ) -> InternAsyncCommandReceipt:
        """Pause Async work and free the sticky host **lease** (resume reacquires).

        The shared org exe.dev VM is retained until filestore backup exists;
        pause does not wipe Sync/Async guest workspaces on that box.
        """

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
        """Resume Async work through a generation-fenced command."""
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
        """Cancel Async work at the expected generation with an explicit reason.

        Cancel fences older pending effects and frees the sticky host **lease**.
        Like pause, it does not wipe machine memory: the shared org exe.dev VM
        is retained until filestore backup exists, so Sync/Async guest
        workspaces on that box survive.
        """

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
        """Answer one Async interaction with fenced input and optional context."""
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
        """Send a generation-fenced Async intervention instruction."""
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
        """Request an Async objective change through backend command authority."""
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
        """Request a backend-owned Async checkpoint at the expected generation."""
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
        """Read a bounded Async event page after a non-negative reconnect sequence."""
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

    def mcp_actions(self, *, limit: int = 100) -> tuple[JsonObject, ...]:
        """Read the shared runtime MCP-action ledger for this Async Intern."""

        runtime_id = self.get().async_runtime_id
        response = self._transport.execute(
            _request(
                "list_intern_runtime_mcp_actions",
                f"/smr/research-intern/runtimes/async/{runtime_id}/mcp-actions",
                query={"limit": _bounded_limit(limit)},
            )
        )
        items = array_value(
            cast(JsonValue, response),
            operation_id="list_intern_runtime_mcp_actions",
        )
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("Async Intern MCP-action ledger must contain objects")
        return tuple(cast(JsonObject, item) for item in items)

    def stream_events(
        self,
        *,
        after_sequence: int = 0,
        timeout_seconds: float = 30.0,
    ) -> Iterator[InternAsyncEvent]:
        """Stream Async SSE events and reject runtime identity or sequence drift."""
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
        """Collect bounded Async events and return the last durable reconnect sequence."""
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


class ResearchInternAcceptanceFixturesAPI:
    """Disposable Factory/Project/Effort/Run acceptance fixtures.

    One command provisions a bound, Factory-ready chain with the organization
    Intern attached (no tribal ids); the receipt is the durable evidence of
    what exists, and teardown preserves it through the retention window.
    """

    _PATH = "/smr/research-intern/fixtures"

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(self, request: InternAcceptanceFixtureRequest) -> InternAcceptanceFixtureReceipt:
        """Provision a typed acceptance fixture and return its backend-owned receipt."""
        return InternAcceptanceFixtureReceipt.from_wire(
            self._transport.execute(
                _request(
                    "create_intern_acceptance_fixture",
                    self._PATH,
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    def get(self, fixture_id: str) -> InternAcceptanceFixtureReceipt:
        """Retrieve a fixture receipt and reject a mismatched fixture identity."""
        receipt = InternAcceptanceFixtureReceipt.from_wire(
            self._transport.execute(
                _request(
                    "get_intern_acceptance_fixture",
                    f"{self._PATH}/{fixture_id}",
                )
            )
        )
        if receipt.fixture_id != fixture_id:
            raise ValueError("Intern acceptance fixture identity drifted")
        return receipt

    def teardown(self, fixture_id: str) -> InternAcceptanceFixtureReceipt:
        """Request fixture teardown while retaining its identity-checked evidence receipt."""
        receipt = InternAcceptanceFixtureReceipt.from_wire(
            self._transport.execute(
                _request(
                    "teardown_intern_acceptance_fixture",
                    f"{self._PATH}/{fixture_id}:teardown",
                )
            )
        )
        if receipt.fixture_id != fixture_id:
            raise ValueError("Intern acceptance fixture identity drifted")
        return receipt


class ResearchInternAPI:
    """One durable organization Intern and its many Factory memberships."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.meta_threads = ResearchInternMetaThreadsAPI(transport)
        self.sync_ = ResearchInternSyncRuntimeAPI(transport, self.meta_threads)
        self.async_ = ResearchInternAsyncRuntimeAPI(transport)
        self.program = InternProgramAPI(transport)
        self.factories = ResearchInternFactoriesAPI(transport)
        self.decisions = ResearchInternDecisionsAPI(transport)
        self.acceptance_receipts = ResearchInternAcceptanceReceiptsAPI(transport)
        self.fixtures = ResearchInternAcceptanceFixturesAPI(transport)

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
                    "get_public_research_intern_acceptance_receipt",
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


class AsyncResearchInternMetaThreadsAPI:
    """Native async graph and cross-lane protocol projections."""

    _PATH = "/smr/research-intern/meta-threads"

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self) -> tuple[InternMetaThread, ...]:
        """List backend-owned Sync and Async meta-thread projections."""
        return tuple(
            InternMetaThread.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    await self._transport.execute(_request("list_intern_meta_threads", self._PATH)),
                ),
                operation_id="list_intern_meta_threads",
            )
        )

    async def get(self, meta_thread_id: str) -> InternMetaThread:
        """Retrieve one meta-thread and reject a mismatched response identity."""
        thread = InternMetaThread.from_wire(
            await self._transport.execute(
                _request(
                    "get_intern_meta_thread",
                    f"{self._PATH}/{meta_thread_id}",
                )
            )
        )
        if thread.meta_thread_id != meta_thread_id:
            raise ValueError("Intern meta-thread identity drifted")
        return thread

    async def segments(self, meta_thread_id: str) -> tuple[InternMetaThreadSegment, ...]:
        """List the live and sealed segments of one meta-thread."""
        return tuple(
            InternMetaThreadSegment.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    await self._transport.execute(
                        _request(
                            "list_intern_meta_thread_segments",
                            f"{self._PATH}/{meta_thread_id}/segments",
                        )
                    ),
                ),
                operation_id="list_intern_meta_thread_segments",
            )
        )

    async def messages(
        self, meta_thread_id: str, *, limit: int = 200
    ) -> tuple[InternCrossMetaThreadMessage, ...]:
        """Read a bounded page of cross-meta-thread messages."""
        return tuple(
            InternCrossMetaThreadMessage.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    await self._transport.execute(
                        _request(
                            "list_intern_meta_thread_messages",
                            f"{self._PATH}/{meta_thread_id}/messages",
                            query={"limit": _bounded_limit(limit)},
                        )
                    ),
                ),
                operation_id="list_intern_meta_thread_messages",
            )
        )

    async def handoffs(self, meta_thread_id: str) -> tuple[InternMetaHandoff, ...]:
        """List durable cross-lane handoff records for one meta-thread."""
        return tuple(
            InternMetaHandoff.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    await self._transport.execute(
                        _request(
                            "list_intern_meta_thread_handoffs",
                            f"{self._PATH}/{meta_thread_id}/handoffs",
                        )
                    ),
                ),
                operation_id="list_intern_meta_thread_handoffs",
            )
        )

    async def send(
        self, request: InternCrossMetaThreadMessageCreateRequest
    ) -> InternCrossMetaThreadMessage:
        """Send a typed cross-lane message and verify its receipt identity."""
        message = InternCrossMetaThreadMessage.from_wire(
            await self._transport.execute(
                _request(
                    "create_intern_meta_thread_message",
                    f"{self._PATH}/messages",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if message.message_id != request.message_id:
            raise ValueError("Cross meta-thread message identity drifted")
        return message


class AsyncResearchInternSyncRuntimeAPI:
    """Native async transport for operator-present Sync sessions."""

    _PATH = "/smr/research-intern/sync-sessions"

    def __init__(
        self,
        transport: AsyncHttpTransport,
        meta_threads: AsyncResearchInternMetaThreadsAPI | None = None,
    ) -> None:
        self._transport = transport
        self._meta_threads = meta_threads or AsyncResearchInternMetaThreadsAPI(transport)

    async def branches(self) -> tuple[InternMetaThreadSegment, ...]:
        """List the Sync head and branch segments; require exactly one Sync meta-thread."""
        threads = await self._meta_threads.list()
        sync_threads = [thread for thread in threads if thread.kind is InternMetaThreadKind.SYNC]
        if len(sync_threads) != 1:
            raise ValueError("Research Intern must expose exactly one Sync meta-thread")
        return await self._meta_threads.segments(sync_threads[0].meta_thread_id)

    async def create(
        self,
        request: InternSyncSessionCreateRequest,
    ) -> InternSyncSession:
        """Create a Sync session from a typed backend request."""
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
        """List a bounded page of typed Sync session projections."""
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
        """Retrieve a Sync session and reject response identity drift."""
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

    async def presence(
        self,
        sync_session_id: str,
        *,
        connection_id: str,
        connection_generation: int = 1,
    ) -> InternSyncPresenceLease:
        """Acquire or renew this operator's presence lease on the session.

        Billed sync launches (rollouts, deploys) are gated on an active
        authenticated operator presence (``intern_sync_active_presence_required``).
        Hold and renew this lease (before ``expires_at``) for as long as the
        Intern is expected to launch work on your behalf.
        """

        lease = InternSyncPresenceLease.from_wire(
            await self._transport.execute(
                _request(
                    "acquire_intern_sync_presence",
                    f"{self._PATH}/{sync_session_id}/presence",
                    body={
                        "connection_id": connection_id,
                        "connection_generation": connection_generation,
                    },
                )
            )
        )
        if lease.sync_session_id != sync_session_id:
            raise ValueError("Sync Intern presence lease identity drifted")
        return lease

    async def release_presence(
        self,
        sync_session_id: str,
        *,
        connection_id: str,
        connection_generation: int = 1,
    ) -> InternSyncPresenceLease:
        """Release this operator's presence lease on the session."""

        lease = InternSyncPresenceLease.from_wire(
            await self._transport.execute(
                _request(
                    "release_intern_sync_presence",
                    f"{self._PATH}/{sync_session_id}/presence/release",
                    body={
                        "connection_id": connection_id,
                        "connection_generation": connection_generation,
                    },
                )
            )
        )
        if lease.sync_session_id != sync_session_id:
            raise ValueError("Sync Intern presence lease identity drifted")
        return lease

    async def approvals(self, sync_session_id: str) -> tuple[InternSyncApprovalCard, ...]:
        """List operator approval cards for this session.

        Billed MCP actions (rollout launches, deploys) park in
        ``awaiting_approval`` with ``capability_operator_approval_required``
        until the present operator decides them.
        """

        payload = await self._transport.execute(
            _request(
                "list_intern_sync_approvals",
                f"{self._PATH}/{sync_session_id}/approvals",
            )
        )
        cards = tuple(
            InternSyncApprovalCard.from_wire(item)
            for item in array_value(
                cast(JsonValue, payload), operation_id="list_intern_sync_approvals"
            )
        )
        if any(card.sync_session_id != sync_session_id for card in cards):
            raise ValueError("Sync Intern approval card identity drifted")
        return cards

    async def decide_approval(
        self,
        approval_id: str,
        *,
        decision: Literal["approve", "deny", "edit"],
        comment: str | None = None,
    ) -> InternSyncApprovalCard:
        """Decide one approval card: ``approve``, ``deny``, or ``edit``.

        Requires an active presence lease on the card's session
        (``intern_sync_active_presence_required`` otherwise).
        """

        body: JsonObject = {"decision": decision}
        if comment is not None:
            body["comment"] = comment
        card = InternSyncApprovalCard.from_wire(
            await self._transport.execute(
                _request(
                    "decide_intern_sync_approval",
                    f"/smr/research-intern/sync-approvals/{approval_id}/decision",
                    body=body,
                )
            )
        )
        if card.approval_id != approval_id:
            raise ValueError("Sync Intern approval card identity drifted")
        return card

    async def command(
        self,
        sync_session_id: str,
        request: InternSyncCommandRequest,
    ) -> InternSyncCommandReceipt:
        """Submit a fenced command and verify its command and runtime identities."""
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
        """Submit an operator message with an idempotency key and expected generation."""
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
        """Submit a fenced intervention with the requested state patch and context."""
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
        """Answer one interaction through a generation-fenced Sync command."""
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
        """Request a fenced pause and retain the operator rationale in the receipt.

        Pause and close act on the session, not the machine: the shared org
        exe.dev VM (one box for all of the org's Sync and Async work) is
        retained, so guest workspaces and Codex threads survive for resume.
        """

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
        """Request resumption at the expected Sync generation."""
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
        """Close a Sync session with an explicit outcome and rationale."""
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
        """Read a bounded event page after a non-negative reconnect sequence."""
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
        """Stream typed SSE events while validating runtime identity and contiguous sequence."""
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
        """Collect a bounded number of streamed events and return their reconnect cursor."""
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

    async def ensure(
        self,
        request: InternAsyncEnsureRequest,
        *,
        maximum_daily_cost_cents: int | None = None,
        maximum_monthly_cost_cents: int | None = None,
    ) -> InternAsyncRuntime:
        """Ensure the org Async Intern. Day/month kwargs override ``request.budget``."""

        ensure_request = _async_ensure_request_with_budget_overrides(
            request,
            maximum_daily_cost_cents=maximum_daily_cost_cents,
            maximum_monthly_cost_cents=maximum_monthly_cost_cents,
        )
        return InternAsyncRuntime.from_wire(
            await self._transport.execute(
                _request(
                    "ensure_intern_async_runtime",
                    self._PATH,
                    body=cast(JsonObject, ensure_request.to_wire()),
                )
            )
        )

    async def get(self) -> InternAsyncRuntime:
        """Retrieve the organization Intern's backend-owned Async runtime."""
        return InternAsyncRuntime.from_wire(
            await self._transport.execute(_request("get_intern_async_runtime", self._PATH))
        )

    async def command(self, request: InternAsyncCommandRequest) -> InternAsyncCommandReceipt:
        """Submit a fenced Async command and reject command receipt identity drift."""
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

    async def handoff_model(
        self, request: InternAsyncHandoffModelRequest
    ) -> InternAsyncCommandReceipt:
        """Change Async model/effort via spine handoff (no meta-thread id)."""

        receipt = InternAsyncCommandReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "handoff_intern_async_model",
                    f"{self._PATH}/handoff-model",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        if receipt.command_id != request.command_id:
            raise ValueError("Async Intern handoff-model receipt identity drifted")
        return receipt

    async def seal_handoff_for_review(
        self, request: InternAsyncHandoffReviewRequest
    ) -> InternMetaHandoff:
        """Attended seal: park model/effort switch at needs_review."""

        return InternMetaHandoff.from_wire(
            await self._transport.execute(
                _request(
                    "seal_intern_async_handoff_for_review",
                    f"{self._PATH}/handoffs/review",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    async def list_handoffs(self) -> tuple[InternMetaHandoff, ...]:
        """List the Async runtime's durable handoff review records."""
        return tuple(
            InternMetaHandoff.from_wire(item)
            for item in array_value(
                cast(
                    JsonValue,
                    await self._transport.execute(
                        _request("list_intern_async_handoffs", f"{self._PATH}/handoffs")
                    ),
                ),
                operation_id="list_intern_async_handoffs",
            )
        )

    async def approve_handoff(self, handoff_id: str) -> InternMetaHandoff:
        """Approve one backend-owned Async handoff review."""
        return InternMetaHandoff.from_wire(
            await self._transport.execute(
                _request(
                    "approve_intern_async_handoff",
                    f"{self._PATH}/handoffs/{handoff_id}/approve",
                    body=cast(JsonObject, {}),
                )
            )
        )

    async def reject_handoff(self, handoff_id: str) -> InternMetaHandoff:
        """Reject one backend-owned Async handoff review."""
        return InternMetaHandoff.from_wire(
            await self._transport.execute(
                _request(
                    "reject_intern_async_handoff",
                    f"{self._PATH}/handoffs/{handoff_id}/reject",
                    body=cast(JsonObject, {}),
                )
            )
        )

    async def continue_handoff(
        self,
        handoff_id: str,
        request: InternMetaHandoffContinueRequest | None = None,
    ) -> InternMetaHandoff:
        """Continue a handoff with an optional typed continuation request."""
        body = cast(JsonObject, request.to_wire()) if request is not None else cast(JsonObject, {})
        return InternMetaHandoff.from_wire(
            await self._transport.execute(
                _request(
                    "continue_intern_async_handoff",
                    f"{self._PATH}/handoffs/{handoff_id}/continue",
                    body=body,
                )
            )
        )

    async def send(self, request: InternAsyncInstructionRequest) -> InternAsyncCommandReceipt:
        """Convert an Async instruction to a command and return its verified receipt."""
        return await self.command(request.to_command())

    async def pause(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        expected_generation: int,
        reason: str,
    ) -> InternAsyncCommandReceipt:
        """Pause Async work and free the sticky host **lease** (resume reacquires).

        The shared org exe.dev VM is retained until filestore backup exists;
        pause does not wipe Sync/Async guest workspaces on that box.
        """

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
        """Resume Async work through a generation-fenced command."""
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
        """Cancel Async work at the expected generation with an explicit reason.

        Cancel fences older pending effects and frees the sticky host **lease**.
        Like pause, it does not wipe machine memory: the shared org exe.dev VM
        is retained until filestore backup exists, so Sync/Async guest
        workspaces on that box survive.
        """

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
        """Answer one Async interaction with fenced input and optional context."""
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
        """Send a generation-fenced Async intervention instruction."""
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
        """Request an Async objective change through backend command authority."""
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
        """Request a backend-owned Async checkpoint at the expected generation."""
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
        """Read a bounded Async event page after a non-negative reconnect sequence."""
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
        """Stream Async SSE events and reject runtime identity or sequence drift."""
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
        """Collect bounded Async events and return the last durable reconnect sequence."""
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


class AsyncResearchInternAcceptanceFixturesAPI:
    """Native asynchronous acceptance-fixture operations."""

    _PATH = "/smr/research-intern/fixtures"

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(
        self,
        request: InternAcceptanceFixtureRequest,
    ) -> InternAcceptanceFixtureReceipt:
        """Provision a typed acceptance fixture and return its backend-owned receipt."""
        return InternAcceptanceFixtureReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "create_intern_acceptance_fixture",
                    self._PATH,
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    async def get(self, fixture_id: str) -> InternAcceptanceFixtureReceipt:
        """Retrieve a fixture receipt and reject a mismatched fixture identity."""
        receipt = InternAcceptanceFixtureReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "get_intern_acceptance_fixture",
                    f"{self._PATH}/{fixture_id}",
                )
            )
        )
        if receipt.fixture_id != fixture_id:
            raise ValueError("Intern acceptance fixture identity drifted")
        return receipt

    async def teardown(self, fixture_id: str) -> InternAcceptanceFixtureReceipt:
        """Request fixture teardown while retaining its identity-checked evidence receipt."""
        receipt = InternAcceptanceFixtureReceipt.from_wire(
            await self._transport.execute(
                _request(
                    "teardown_intern_acceptance_fixture",
                    f"{self._PATH}/{fixture_id}:teardown",
                )
            )
        )
        if receipt.fixture_id != fixture_id:
            raise ValueError("Intern acceptance fixture identity drifted")
        return receipt


class AsyncResearchInternAPI:
    """Native asynchronous organization Research Intern operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.meta_threads = AsyncResearchInternMetaThreadsAPI(transport)
        self.sync_ = AsyncResearchInternSyncRuntimeAPI(transport, self.meta_threads)
        self.async_ = AsyncResearchInternAsyncRuntimeAPI(transport)
        self.program = AsyncInternProgramAPI(transport)
        self.factories = AsyncResearchInternFactoriesAPI(transport)
        self.decisions = AsyncResearchInternDecisionsAPI(transport)
        self.acceptance_receipts = AsyncResearchInternAcceptanceReceiptsAPI(transport)
        self.fixtures = AsyncResearchInternAcceptanceFixturesAPI(transport)

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
    "AsyncResearchInternAcceptanceFixturesAPI",
    "AsyncResearchInternAcceptanceReceiptsAPI",
    "AsyncResearchInternAPI",
    "AsyncResearchInternDecisionsAPI",
    "AsyncResearchInternFactoriesAPI",
    "ProjectComputerAPI",
    "ProjectDataBindingsAPI",
    "ResearchInternAcceptanceFixturesAPI",
    "ResearchInternAcceptanceReceiptsAPI",
    "ResearchInternAPI",
    "ResearchInternAsyncRuntimeAPI",
    "ResearchInternDecisionsAPI",
    "ResearchInternFactoriesAPI",
]
