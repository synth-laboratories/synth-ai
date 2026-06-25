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
        return self._session.tag.send_message(
            session_id,
            message,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )


class ResearchFactoriesTagSessionsAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._messages: ResearchFactoriesTagSessionsMessagesAPI | None = None

    @property
    def messages(self) -> ResearchFactoriesTagSessionsMessagesAPI:
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
        return self._session.tag.create_session(
            request,
            definition_of_done=definition_of_done,
            scope_id=scope_id,
            timebox_seconds=timebox_seconds,
            runbook_preset=runbook_preset,
            metadata=metadata,
        )

    def get(self, session_id: str) -> TagSession:
        return self._session.tag.get_session(session_id)


class ResearchFactoriesTagScopesAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get_default(self) -> TagScope:
        return self._session.tag.get_default_scope()


class ResearchFactoriesTagAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._sessions: ResearchFactoriesTagSessionsAPI | None = None
        self._scopes: ResearchFactoriesTagScopesAPI | None = None

    @property
    def sessions(self) -> ResearchFactoriesTagSessionsAPI:
        if self._sessions is None:
            self._sessions = ResearchFactoriesTagSessionsAPI(self._session)
        return self._sessions

    @property
    def scopes(self) -> ResearchFactoriesTagScopesAPI:
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
