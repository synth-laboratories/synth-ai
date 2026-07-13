"""``client.research.factories`` — Factory domain (Tag under ``factories.tag``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.factories import (
    FactoryCandidate,
    FactoryCandidateGradingRequest,
    FactoryCandidateGradingStatus,
    FactoryChampionDecision,
    FactoryChampionEvent,
    FactoryChampionRollbackRequest,
    FactoryChampionSelectRequest,
)
from synth_ai.managed_research.models.tag import (
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
        request: TagSessionCreateRequest | Mapping[str, Any] | dict[str, Any] | str,
        *,
        definition_of_done: str | None = None,
        scope_id: str | None = None,
        factory_id: str | None = None,
        effort_id: str | None = None,
        experiment_id: str | None = None,
        candidate_id: str | None = None,
        timebox_seconds: int | None = None,
        runbook_preset: str | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> TagSession:
        """Start a Factory Tag session for a one-off research task.

        Args:
            request: ``TagSessionCreateRequest``, mapping, or primary request
                string shown to the Tag worker.
            definition_of_done: Optional explicit DoD text.
            scope_id: Optional Tag scope override (defaults via ``scopes``).
            timebox_seconds: Optional wall-clock cap for the session.
            runbook_preset: Optional runbook preset slug.
            metadata: Optional session metadata.

        Returns:
            Created ``TagSession`` with ids needed for ``messages.send``.

        Example:
            session = research.factories.tag.sessions.create(
                "Summarize test failures",
                factory_id=factory_id,
                effort_id=effort_id,
            )
            research.factories.tag.sessions.messages.send(
                session.session_id,
                "Return a bullet list of root causes.",
            )
        """
        return self._session.tag.create_session(
            request,
            definition_of_done=definition_of_done,
            scope_id=scope_id,
            factory_id=factory_id,
            effort_id=effort_id,
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            timebox_seconds=timebox_seconds,
            runbook_preset=runbook_preset,
            metadata=metadata,
        )

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


class ResearchFactoriesTagScopesAPI:
    """Resolve default Tag scopes for an organization."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get_default(self) -> TagScope:
        """Return the org default Tag scope used when ``scope_id`` is omitted."""
        return self._session.tag.get_default_scope()


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
        request: FactoryCandidateGradingRequest
        | Mapping[str, Any]
        | dict[str, Any],
    ) -> FactoryCandidate:
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
        return self._session.factories.select_champion(factory_id, request)

    def rollback(
        self,
        factory_id: str,
        request: FactoryChampionRollbackRequest
        | Mapping[str, Any]
        | dict[str, Any],
    ) -> FactoryChampionDecision:
        return self._session.factories.rollback_champion(factory_id, request)

    def list_events(
        self,
        factory_id: str,
        *,
        limit: int = 100,
    ) -> tuple[FactoryChampionEvent, ...]:
        return tuple(
            self._session.factories.list_champion_events(factory_id, limit=limit)
        )


class ResearchFactoriesAPI:
    """Factory domain namespace (Tag ships under ``factories.tag``)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._tag: ResearchFactoriesTagAPI | None = None
        self._candidates: ResearchFactoryCandidatesAPI | None = None
        self._champions: ResearchFactoryChampionsAPI | None = None

    @property
    def candidates(self) -> ResearchFactoryCandidatesAPI:
        if self._candidates is None:
            self._candidates = ResearchFactoryCandidatesAPI(self._session)
        return self._candidates

    @property
    def champions(self) -> ResearchFactoryChampionsAPI:
        if self._champions is None:
            self._champions = ResearchFactoryChampionsAPI(self._session)
        return self._champions

    @property
    def tag(self) -> ResearchFactoriesTagAPI:
        """Factory Tag — short-lived delegated research sessions."""
        if self._tag is None:
            self._tag = ResearchFactoriesTagAPI(self._session)
        return self._tag


__all__ = [
    "ResearchFactoriesAPI",
    "ResearchFactoriesTagAPI",
    "ResearchFactoriesTagScopesAPI",
    "ResearchFactoriesTagSessionsAPI",
    "ResearchFactoriesTagSessionsMessagesAPI",
    "ResearchFactoryCandidatesAPI",
    "ResearchFactoryChampionsAPI",
]
