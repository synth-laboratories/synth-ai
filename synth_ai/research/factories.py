"""``client.research.factories`` — Factory domain (Tag under ``factories.tag``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.tag import (
    TagMessageRequest,
    TagScope,
    TagSession,
    TagSessionCreateRequest,
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
            session = research.factories.tag.sessions.create("Summarize test failures")
            research.factories.tag.sessions.messages.send(
                session.session_id,
                "Return a bullet list of root causes.",
            )
        """
        return self._session.tag.create_session(
            request,
            definition_of_done=definition_of_done,
            scope_id=scope_id,
            timebox_seconds=timebox_seconds,
            runbook_preset=runbook_preset,
            metadata=metadata,
        )

    def get(self, session_id: str) -> TagSession:
        """Fetch the current Tag session state and terminal receipt fields."""
        return self._session.tag.get_session(session_id)


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


class ResearchFactoriesAPI:
    """Factory domain namespace (Tag ships under ``factories.tag``)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._tag: ResearchFactoriesTagAPI | None = None

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
]
