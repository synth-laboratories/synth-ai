"""Scoped inference sessions; provider credentials stay in the core sealed store."""

# See: contracts/synth-v1.openapi.yaml#/paths/~1v1~1inference~1sessions
from __future__ import annotations

from ._generated.models import InferenceSession
from ._transport import Transport


class AsyncInference:
    def __init__(self, transport: Transport, project_id: str | None = None) -> None:
        self._transport = transport
        self._project_id = project_id

    async def session(
        self,
        *,
        project_id: str | None = None,
        preset: str = "hosted.openrouter_zdr_us",
        budget: str = "2.00",
        ttl: int = 3600,
        idempotency_key: str | None = None,
    ) -> InferenceSession:
        project = project_id or self._project_id
        if not project:
            raise ValueError("project_id is required")
        return await self._transport.request(
            "createInferenceSession",
            InferenceSession,
            body={
                "project_id": project,
                "preset": preset,
                "budget": {"amount": budget, "currency": "USD"},
                "ttl": ttl,
            },
            idempotency_key=idempotency_key,
        )
