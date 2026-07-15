"""Synth Tag SDK namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
from synth_ai.managed_research.sdk._base import _ClientNamespace


class TagAPI(_ClientNamespace):
    def create_session(
        self,
        request: TagSessionCreateRequest,
    ) -> TagSession:
        if not isinstance(request, TagSessionCreateRequest):
            raise TypeError("request must be TagSessionCreateRequest")
        return TagSession.from_payload(
            self._client.create_tag_session(request.to_wire())
        )

    def get_session(self, session_id: str) -> TagSession:
        return TagSession.from_wire(self._client.get_tag_session(session_id))

    def list_sessions(
        self,
        *,
        factory_id: str | None = None,
        effort_id: str | None = None,
        limit: int = 50,
    ) -> tuple[TagSession, ...]:
        return tuple(
            TagSession.from_wire(item)
            for item in self._client.list_tag_sessions(
                factory_id=factory_id,
                effort_id=effort_id,
                limit=limit,
            )
        )

    def watch_session(self, session_id: str) -> TagSessionWatch:
        return TagSessionWatch.from_wire(self._client.watch_tag_session(session_id))

    def control_session(
        self,
        session_id: str,
        action: TagSessionControlAction | str,
    ) -> TagSession:
        return TagSession.from_wire(
            self._client.control_tag_session(
                session_id,
                action=str(getattr(action, "value", action)),
            )
        )

    def send_message(
        self,
        session_id: str,
        message: TagMessageRequest | Mapping[str, Any] | dict[str, Any] | str,
        *,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        steering_target: TagSteeringTarget | str = TagSteeringTarget.ACTIVE_RUN,
    ) -> TagSession:
        if isinstance(message, TagMessageRequest):
            payload = message.to_wire()
        elif isinstance(message, str):
            payload = TagMessageRequest(
                message=message,
                steering_target=steering_target,
                metadata=dict(metadata or {}),
                idempotency_key=idempotency_key,
            ).to_wire()
        elif isinstance(message, Mapping):
            payload = dict(message)
        else:
            raise TypeError("message must be TagMessageRequest, mapping, or string")
        return TagSession.from_wire(self._client.send_tag_message(session_id, payload))

    def get_default_scope(self) -> TagScope:
        return TagScope.from_wire(self._client.get_default_tag_scope())

    def get_factory_context(
        self,
        *,
        scope_id: str | None = None,
        session_id: str | None = None,
    ) -> TagFactoryContext:
        if (scope_id is None) == (session_id is None):
            raise ValueError("exactly one of scope_id or session_id is required")
        if scope_id is not None:
            payload = self._client.get_tag_scope_factory_context(scope_id)
        else:
            payload = self._client.get_tag_session_factory_context(str(session_id))
        return TagFactoryContext.from_payload(payload)


__all__ = ["TagAPI"]
