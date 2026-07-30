"""``research.advanced.factories`` — operator Factory capabilities.

Only what the stable front door does not carry lives here: Tag sessions, the
wake preview→confirm handshake, Effort creation, and the status projection.
Lifecycle CRUD, candidates, champions, lenses, and results are owned by
``research.factories`` (`synth_ai.sdk.research.factories` + the facade).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from synth_ai.sdk.research.contracts.factories import EffortRecurrence
from synth_ai.sdk.research.contracts.factory_operations import (
    Effort,
    EffortStatus,
    EffortType,
    FactoryStatus,
    FactoryWakeDueResult,
)
from synth_ai.sdk.research.contracts.tag import (
    TagFactoryContext,
    TagMessageRequest,
    TagScope,
    TagSession,
    TagSessionControlAction,
    TagSessionCreateRequest,
    TagSessionWatch,
    TagSteeringTarget,
)
from synth_ai.sdk.research.session.client import ResearchSession


class ResearchFactoriesTagSessionsMessagesAPI:
    """Send steering messages to an active Factory Tag session."""

    def __init__(self, session: ResearchSession) -> None:
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

    def __init__(self, session: ResearchSession) -> None:
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

    def __init__(self, session: ResearchSession) -> None:
        self._session = session

    def get_default(self) -> TagScope:
        """Return the org default Tag scope used when ``scope_id`` is omitted."""
        return self._session.tag.get_default_scope()

    def get_factory_context(self, scope_id: str = "default") -> TagFactoryContext:
        """Read Factory champion and candidate context for a Tag scope."""
        return self._session.tag.get_factory_context(scope_id=scope_id)


class ResearchFactoriesTagAPI:
    """Factory Tag namespace — delegate short research tasks from your IDE."""

    def __init__(self, session: ResearchSession) -> None:
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


class ResearchFactoriesAPI:
    """Operator Factory capabilities the stable contract does not carry.

    Lifecycle CRUD, candidates, champions, lenses, and results live on
    ``research.factories``. This namespace keeps Tag, Effort creation, the
    backend-owned status projection, and the wake preview→confirm handshake.
    """

    def __init__(self, session: ResearchSession) -> None:
        self._session = session
        self._tag: ResearchFactoriesTagAPI | None = None

    @property
    def tag(self) -> ResearchFactoriesTagAPI:
        """Factory Tag — short-lived delegated research sessions."""
        if self._tag is None:
            self._tag = ResearchFactoriesTagAPI(self._session)
        return self._tag

    def list_efforts(self, factory_id: str) -> tuple[Effort, ...]:
        """List Efforts owned by a Factory.

        Args:
            factory_id: Owning Factory id.

        Returns:
            Tuple of ``Effort`` records.
        """
        return tuple(self._session.factories.list_efforts(factory_id))

    def create_effort(
        self,
        factory_id: str,
        *,
        name: str,
        project_id: str | None = None,
        hypothesis_or_topic: str | None = None,
        effort_type: EffortType | str = EffortType.RESEARCH,
        status: EffortStatus | str = EffortStatus.ACTIVE,
        recurrence: EffortRecurrence | None = None,
        next_wake_at: datetime | str | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> Effort:
        """Create an Effort under a Factory.

        Args:
            factory_id: Owning Factory.
            name: Human-readable Effort name.
            project_id: Optional project; defaults to the Factory workspace project.
            hypothesis_or_topic: Optional research hypothesis text.
            effort_type: Effort type (default research).
            status: Initial Effort status.
            recurrence: Typed recurrence policy, including the exact launch
                SwarmSpec.
            next_wake_at: Optional first wake time.
            metadata: Optional metadata bag.

        Returns:
            The created ``Effort``.
        """
        return self._session.factories.create_effort(
            factory_id,
            name=name,
            project_id=project_id,
            hypothesis_or_topic=hypothesis_or_topic,
            effort_type=effort_type,
            status=status,
            recurrence=recurrence,
            next_wake_at=next_wake_at,
            metadata=metadata,
        )

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
        result = self._session.factories.wake_due(
            factory_id,
            launch_request=contract.launch_request,
            limit=contract.limit,
            allow_overlap=contract.allow_overlap,
            dry_run=False,
            continue_on_error=contract.continue_on_error,
            confirmed_preview_id=preview.preview_id,
            confirmed_preview_token=preview.preview_token,
        )
        if result.confirmed_preview_id != preview.preview_id or result.receipt_id is None:
            raise RuntimeError("wake receipt is not durably bound to the confirmed preview")
        return result


__all__ = [
    "ResearchFactoriesAPI",
    "ResearchFactoriesTagAPI",
    "ResearchFactoriesTagScopesAPI",
    "ResearchFactoriesTagSessionsAPI",
    "ResearchFactoriesTagSessionsMessagesAPI",
]
