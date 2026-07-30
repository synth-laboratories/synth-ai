"""Research Intern, Magi decision, and project resource operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
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
    ResearchInternFactoryMembershipResponse,
    ResearchInternPatchRequest,
    ResearchInternProvisionRequest,
    ResearchInternResponse,
    ResearchInternSessionCloseRequest,
    ResearchInternSessionCreateRequest,
    ResearchInternSessionResponse,
    ResearchInternSessionSyncResponse,
    ResearchInternTurnControl,
    ResearchInternTurnRequest,
    ResearchInternTurnResponse,
    ResearchInternTurnStatus,
)
from synth_ai.sdk.research.operations import (
    dataset_revision_publication_operation,
    research_operation,
)


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
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.SYNC,
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
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.SYNC,
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
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.SYNC,
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
        evidence_refs: list[str],
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
    """Reconnect position for one ordered Research Intern session event log."""

    session_id: str
    after_sequence: int = 0
    state_generation: int = 0
    last_event_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.after_sequence < 0 or self.state_generation < 0:
            raise ValueError("event cursor sequence and state generation must be non-negative")


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


@dataclass(frozen=True, slots=True)
class ResearchInternExchange:
    """One ordered operator-to-Intern exchange from backend authority."""

    operator_turn: ResearchInternOperatorTurn
    agent_event: ResearchInternEventResponse
    observed_events: tuple[ResearchInternEventResponse, ...]
    cursor: ResearchInternEventCursor


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

    def __init__(self, response: ResearchInternTurnResponse) -> None:
        message = (
            response.error.message
            if response.error is not None
            else f"Research Intern turn {response.turn_id} failed"
        )
        super().__init__(message)
        self.response = response


class ResearchInternSessionsAPI:
    """Durable reactive sessions and their ordered event logs."""

    def __init__(self, transport: HttpTransport, owner: ResearchInternAPI) -> None:
        self._transport = transport
        self._owner = owner

    def create(self, request: ResearchInternSessionCreateRequest) -> ResearchInternSessionResponse:
        """Create or replay one durable Intern session."""
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
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
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
        for event in events:
            if event.session_id != session_id or event.sequence != previous_sequence + 1:
                raise ValueError("Research Intern event page is not contiguous for its session")
            previous_sequence = event.sequence
        return events

    def sync(
        self,
        session_id: str,
        *,
        limit: int = 200,
    ) -> ResearchInternSessionSyncResponse:
        """Project a bounded page of canonical runtime transcript events."""
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

    def create_reactive(
        self,
        request: ResearchInternSessionCreateRequest,
    ) -> ResearchInternReactiveSession:
        """Create a reconnectable reactive view over one backend session."""
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
        ):
            raise ValueError("Research Intern event broke the reconnect cursor chain")
        self.cursor = ResearchInternEventCursor(
            session_id=event.session_id,
            after_sequence=event.sequence,
            state_generation=event.state_generation,
            last_event_id=event.event_id,
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

    def pause(
        self,
        *,
        idempotency_key: str,
        rationale: str,
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.SYNC,
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
        mode: MagiMode = MagiMode.SYNC,
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
    ) -> ResearchInternOperatorTurn:
        """Submit one backend-owned turn against the bound real runtime."""
        self._run_id()
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
        observed = self.synchronize()
        if self.cursor.after_sequence < response.reconnect_after_sequence:
            observed = (*observed, *self.synchronize())
        if self.cursor.after_sequence < response.reconnect_after_sequence:
            raise ValueError("Research Intern turn response is ahead of its reconnect event log")
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
        mode: MagiMode = MagiMode.SYNC,
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
        """Poll bounded event pages and return only requested event kinds."""
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
        """Wait for one backend-projected agent message."""
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
    ) -> ResearchInternExchange:
        """Submit one turn and require a canonical real-runtime agent reply."""
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
        )
        response = turn.response
        if response.status is ResearchInternTurnStatus.FAILED:
            raise ResearchInternTurnFailedError(response)
        if response.agent_event is None:
            raise ResearchInternPollingTimeoutError(
                f"no agent message observed for turn {response.turn_id}",
                response=response,
            )
        return ResearchInternExchange(
            operator_turn=turn,
            agent_event=response.agent_event,
            observed_events=turn.observed_events,
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


class ResearchInternAPI:
    """One durable organization Intern and its many Factory memberships."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.factories = ResearchInternFactoriesAPI(transport)
        self.decisions = ResearchInternDecisionsAPI(transport)
        self.sessions = ResearchInternSessionsAPI(transport, self)
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
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.SYNC,
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
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.SYNC,
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
        evidence_refs: list[str] | None = None,
        mode: MagiMode = MagiMode.SYNC,
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
        evidence_refs: list[str],
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
    """Native asynchronous Intern session and event-log operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(
        self,
        request: ResearchInternSessionCreateRequest,
    ) -> ResearchInternSessionResponse:
        """Create or replay one durable Intern session."""
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
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
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
        for event in events:
            if event.session_id != session_id or event.sequence != previous_sequence + 1:
                raise ValueError("Research Intern event page is not contiguous for its session")
            previous_sequence = event.sequence
        return events

    async def sync(
        self,
        session_id: str,
        *,
        limit: int = 200,
    ) -> ResearchInternSessionSyncResponse:
        """Project a bounded page of canonical runtime transcript events."""
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


class AsyncResearchInternAPI:
    """Native asynchronous organization Research Intern operations."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.factories = AsyncResearchInternFactoriesAPI(transport)
        self.decisions = AsyncResearchInternDecisionsAPI(transport)
        self.sessions = AsyncResearchInternSessionsAPI(transport)
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
    "AsyncResearchInternAcceptanceReceiptsAPI",
    "AsyncResearchInternAPI",
    "AsyncResearchInternDecisionsAPI",
    "AsyncResearchInternFactoriesAPI",
    "AsyncResearchInternSessionsAPI",
    "ProjectComputerAPI",
    "ProjectDataBindingsAPI",
    "ResearchInternAcceptanceReceiptsAPI",
    "ResearchInternAPI",
    "ResearchInternDecisionsAPI",
    "ResearchInternEventCursor",
    "ResearchInternEventObservation",
    "ResearchInternExchange",
    "ResearchInternFactoriesAPI",
    "ResearchInternOperatorTurn",
    "ResearchInternPollingTimeoutError",
    "ResearchInternReactiveSession",
    "ResearchInternSessionsAPI",
    "ResearchInternTurnFailedError",
]
