"""``client.research.secrets`` — project secret refs (canonical over ``credentials``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient


class ResearchSecretsAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        project_id: str,
        *,
        kind: str | None = None,
    ) -> list[Any]:
        return self._session.secrets.list(project_id, kind=kind)

    def create(
        self,
        project_id: str,
        *,
        kind: str,
        label: str,
        provider: str | None = None,
        funding_source: str | None = None,
        credential_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        return self._session.secrets.create(
            project_id,
            kind=kind,
            label=label,
            provider=provider,
            funding_source=funding_source,
            credential_name=credential_name,
            metadata=metadata,
        )

    def delete(self, project_id: str, secret_id: str) -> dict[str, Any]:
        return self._session.secrets.delete(project_id, secret_id)


__all__ = ["ResearchSecretsAPI"]
