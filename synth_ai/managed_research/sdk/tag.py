"""Synth Tag SDK namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.tag import (
    TagMessageRequest,
    TagScope,
    TagSession,
    TagSessionCreateRequest,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace


class TagAPI(_ClientNamespace):
    def create_session(
        self,
        request: TagSessionCreateRequest | Mapping[str, Any] | dict[str, Any] | str,
        *,
        definition_of_done: str | None = None,
        scope_id: str | None = None,
        timebox_seconds: int | None = None,
        runbook_preset: str | None = None,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> TagSession:
        if isinstance(request, TagSessionCreateRequest):
            payload = request.to_wire()
        elif isinstance(request, str):
            payload = TagSessionCreateRequest(
                request=request,
                definition_of_done=definition_of_done,
                scope_id=scope_id,
                timebox_seconds=timebox_seconds,
                runbook_preset=runbook_preset,
                metadata=dict(metadata or {}),
            ).to_wire()
        elif isinstance(request, Mapping):
            payload = dict(request)
        else:
            raise TypeError("request must be TagSessionCreateRequest, mapping, or string")
        return TagSession.from_wire(self._client.create_tag_session(payload))

    def get_session(self, session_id: str) -> TagSession:
        return TagSession.from_wire(self._client.get_tag_session(session_id))

    def send_message(
        self,
        session_id: str,
        message: TagMessageRequest | Mapping[str, Any] | dict[str, Any] | str,
        *,
        metadata: Mapping[str, Any] | dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> TagSession:
        if isinstance(message, TagMessageRequest):
            payload = message.to_wire()
        elif isinstance(message, str):
            payload = TagMessageRequest(
                message=message,
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


__all__ = ["TagAPI"]
