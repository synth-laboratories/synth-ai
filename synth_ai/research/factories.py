"""``client.research.factories`` — Factory domain (Tag under ``factories.tag``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.factories import (
    Factory,
    FactoryCandidate,
    FactoryCandidateGradingRequest,
    FactoryCandidateGradingStatus,
    FactoryChampionDecision,
    FactoryChampionEvent,
    FactoryChampionRollbackRequest,
    FactoryChampionSelectRequest,
    FactoryStatus,
    FactoryWakeDueResult,
)
from synth_ai.managed_research.models.tag import (
    TagFactoryContext,
    TagMessageRequest,
    TagScope,
    TagSession,
    TagSessionControlAction,
    TagSessionCreateRequest,
    TagSessionWatch,
    TagSteeringTarget,
)
from synth_ai.managed_research.sdk.client import ManagedResearchClient


class ResearchFactoriesTagSessionsMessagesAPI:
    """Send steering messages to an active Factory Tag session."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def send(
        self,
        session_id: str,
        message: TagMessageRequest | Mapping[str, Any] | dict[str, Any] | str,
        *,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        steering_target: TagSteeringTarget | str = TagSteeringTarget.ACTIVE_RUN,
    ) -> TagSession:
        """Post a message to a Tag session and return the updated session state.

        Args:
            session_id: Tag session id from ``sessions.create``.
            message: ``TagMessageRequest``, mapping, or plain string body.
            metadata: Optional message metadata.
            idempotency_key: Optional idempotency key for safe retries.

        Returns:
            Updated ``TagSession`` after the message is accepted.
        """
        return self._session.tag.send_message(
            session_id,
            message,
            metadata=metadata,
            idempotency_key=idempotency_key,
            steering_target=steering_target,
        )


class ResearchFactoriesTagSessionsAPI:
    """Create and inspect Factory Tag sessions."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._messages: ResearchFactoriesTagSessionsMessagesAPI | None = None

    @property
    def messages(self) -> ResearchFactoriesTagSessionsMessagesAPI:
        """Nested API for sending Tag session messages."""
        if self._messages is None:
            self._messages = ResearchFactoriesTagSessionsMessagesAPI(self._session)
        return self._messages

    def create(
        self,
        request: TagSessionCreateRequest,
    ) -> TagSession:
        """Start a Factory Tag session for a one-off research task.

        Args:
            request: Fully typed Tag session request.

        Returns:
            Created ``TagSession`` with ids needed for ``messages.send``.

        Example:
            session = research.factories.tag.sessions.create(
                TagSessionCreateRequest(
                    request="Summarize test failures",
                    factory_id=factory_id,
                    effort_id=effort_id,
                )
            )
            research.factories.tag.sessions.messages.send(
                session.session_id,
                "Return a bullet list of root causes.",
            )
        """
        return self._session.tag.create_session(request)

    def get(self, session_id: str) -> TagSession:
        """Fetch the current Tag session state and terminal receipt fields."""
        return self._session.tag.get_session(session_id)

    def list(
        self,
        *,
        factory_id: str | None = None,
        effort_id: str | None = None,
        limit: int = 50,
    ) -> tuple[TagSession, ...]:
        """List Tag sessions, optionally filtered by factory or effort.

        Args:
            factory_id: Only sessions bound to this factory.
            effort_id: Only sessions bound to this effort.
            limit: Maximum sessions returned (newest first).

        Returns:
            Tuple of ``TagSession`` records.
        """
        return self._session.tag.list_sessions(
            factory_id=factory_id,
            effort_id=effort_id,
            limit=limit,
        )

    def watch(self, session_id: str) -> TagSessionWatch:
        """Open a watch handle that polls the session until it is terminal.

        Args:
            session_id: Tag session to observe.

        Returns:
            ``TagSessionWatch`` iterator of session state snapshots.
        """
        return self._session.tag.watch_session(session_id)

    def control(
        self,
        session_id: str,
        action: TagSessionControlAction | str,
    ) -> TagSession:
        """Apply a control action (for example pause, resume, cancel) to a session.

        Args:
            session_id: Tag session to control.
            action: ``TagSessionControlAction`` or its string value.

        Returns:
            Updated ``TagSession`` after the action is applied.
        """
        return self._session.tag.control_session(session_id, action)

    def get_factory_context(self, session_id: str) -> TagFactoryContext:
        """Read the Factory champion and candidate context bound to a session."""
        return self._session.tag.get_factory_context(session_id=session_id)


class ResearchFactoriesTagScopesAPI:
    """Resolve default Tag scopes for an organization."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get_default(self) -> TagScope:
        """Return the org default Tag scope used when ``scope_id`` is omitted."""
        return self._session.tag.get_default_scope()

    def get_factory_context(self, scope_id: str = "default") -> TagFactoryContext:
        """Read Factory champion and candidate context for a Tag scope."""
        return self._session.tag.get_factory_context(scope_id=scope_id)


class ResearchFactoriesTagAPI:
    """Factory Tag namespace — delegate short research tasks from your IDE."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._sessions: ResearchFactoriesTagSessionsAPI | None = None
        self._scopes: ResearchFactoriesTagScopesAPI | None = None

    @property
    def sessions(self) -> ResearchFactoriesTagSessionsAPI:
        """Create Tag sessions and send steering messages."""
        if self._sessions is None:
            self._sessions = ResearchFactoriesTagSessionsAPI(self._session)
        return self._sessions

    @property
    def scopes(self) -> ResearchFactoriesTagScopesAPI:
        """Read Tag scope defaults for the org."""
        if self._scopes is None:
            self._scopes = ResearchFactoriesTagScopesAPI(self._session)
        return self._scopes


class ResearchFactoryCandidatesAPI:
    """Immutable Factory candidates and benchmark-owned grading intake."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        factory_id: str,
        *,
        grading_status: FactoryCandidateGradingStatus | str | None = None,
        effort_id: str | None = None,
        limit: int = 200,
    ) -> tuple[FactoryCandidate, ...]:
        """List candidates, optionally filtered by grading status or Effort."""
        return tuple(
            self._session.factories.list_candidates(
                factory_id,
                grading_status=grading_status,
                effort_id=effort_id,
                limit=limit,
            )
        )

    def record_grading(
        self,
        factory_id: str,
        candidate_id: str,
        request: FactoryCandidateGradingRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryCandidate:
        """Record a benchmark-owned grading result for one immutable candidate."""
        return self._session.factories.record_candidate_grading(
            factory_id,
            candidate_id,
            request,
        )


class ResearchFactoryChampionsAPI:
    """Deterministic champion selection and append-only decision history."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def select(
        self,
        factory_id: str,
        request: FactoryChampionSelectRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryChampionDecision:
        """Select the deterministic winner when it strictly beats the baseline."""
        return self._session.factories.select_champion(factory_id, request)

    def rollback(
        self,
        factory_id: str,
        request: FactoryChampionRollbackRequest | Mapping[str, Any] | dict[str, Any],
    ) -> FactoryChampionDecision:
        """Roll the champion pointer back and append the decision event."""
        return self._session.factories.rollback_champion(factory_id, request)

    def list_events(
        self,
        factory_id: str,
        *,
        limit: int = 100,
    ) -> tuple[FactoryChampionEvent, ...]:
        """List the append-only champion decision history, newest first."""
        return tuple(self._session.factories.list_champion_events(factory_id, limit=limit))


class ResearchFactoriesAPI:
    """Research Factory workflow API.

    The status projection is the shared workflow model for SDK, MCP, and the
    dashboard: experiments, outputs, decisions, limits, health, and next wake
    all come from the backend rather than being reconstructed by each client.
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._tag: ResearchFactoriesTagAPI | None = None
        self._candidates: ResearchFactoryCandidatesAPI | None = None
        self._champions: ResearchFactoryChampionsAPI | None = None

    @property
    def candidates(self) -> ResearchFactoryCandidatesAPI:
        """Immutable candidate discovery and benchmark-owned grading intake."""
        if self._candidates is None:
            self._candidates = ResearchFactoryCandidatesAPI(self._session)
        return self._candidates

    @property
    def champions(self) -> ResearchFactoryChampionsAPI:
        """Deterministic champion selection, rollback, and decision history."""
        if self._champions is None:
            self._champions = ResearchFactoryChampionsAPI(self._session)
        return self._champions

    @property
    def tag(self) -> ResearchFactoriesTagAPI:
        """Factory Tag — short-lived delegated research sessions."""
        if self._tag is None:
            self._tag = ResearchFactoriesTagAPI(self._session)
        return self._tag

    def list(self, *, include_archived: bool = False) -> tuple[Factory, ...]:
        """List Research Factories visible to the authenticated organization."""
        return tuple(self._session.factories.list(include_archived=include_archived))

    def get(self, factory_id: str) -> Factory:
        """Fetch one Research Factory."""
        return self._session.factories.get(factory_id)

    def status(self, factory_id: str) -> FactoryStatus:
        """Read the backend-owned Factory workflow projection."""
        return self._session.factories.status(factory_id)

    def preview_wake(
        self,
        factory_id: str,
        *,
        launch_request: Mapping[str, Any] | dict[str, Any] | None = None,
        limit: int = 10,
        allow_overlap: bool = False,
        continue_on_error: bool = True,
    ) -> FactoryWakeDueResult:
        """Preview due experiments and launch consequences without starting runs."""
        return self._session.factories.wake_due(
            factory_id,
            launch_request=launch_request,
            limit=limit,
            allow_overlap=allow_overlap,
            dry_run=True,
            continue_on_error=continue_on_error,
        )

    def wake_due(
        self,
        factory_id: str,
        *,
        preview: FactoryWakeDueResult,
    ) -> FactoryWakeDueResult:
        """Launch exactly the due experiments bound to a reviewed preview."""
        if preview.factory_id != factory_id:
            raise ValueError("preview factory_id does not match the requested Factory")
        if not preview.dry_run or not preview.confirmation_required:
            raise ValueError("wake_due requires a confirmation-ready dry-run preview")
        if preview.preview_id is None or preview.preview_token is None:
            raise ValueError("preview must include preview_id and preview_token")
        if preview.request_contract is None:
            raise ValueError("preview must include its resolved request contract")
        contract = preview.request_contract
        if contract.confirmed_preview_token is not None:
            raise ValueError("preview request_contract is not confirmation-ready")
        return self._session.factories.wake_due(
            factory_id,
            launch_request=contract.launch_request,
            limit=contract.limit,
            allow_overlap=contract.allow_overlap,
            dry_run=False,
            continue_on_error=contract.continue_on_error,
            confirmed_preview_token=preview.preview_token,
        )


__all__ = [
    "ResearchFactoriesAPI",
    "ResearchFactoriesTagAPI",
    "ResearchFactoriesTagScopesAPI",
    "ResearchFactoriesTagSessionsAPI",
    "ResearchFactoriesTagSessionsMessagesAPI",
    "ResearchFactoryCandidatesAPI",
    "ResearchFactoryChampionsAPI",
]
