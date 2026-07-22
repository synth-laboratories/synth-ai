"""``client.research.secrets`` — project secret refs (canonical over ``credentials``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

from synth_ai.core.research._legacy.sdk.client import ManagedResearchClient


class ResearchSecretsAPI:
    """Manage project-scoped secret references for provider and rollout auth.

    Secrets are stored as org-managed refs — never pass raw credentials in run
    launch payloads. Attach refs here, then reference them from project setup.
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        project_id: str,
        *,
        kind: str | None = None,
    ) -> List[Any]:
        """List secret refs attached to a project.

        Args:
            project_id: Target project id.
            kind: Optional filter (for example ``provider``, ``repo``).

        Returns:
            List of secret ref records for the project.
        """
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
        """Create a secret ref on a project.

        Args:
            project_id: Target project id.
            kind: Secret category understood by the backend.
            label: Human-readable label shown in the UI and launch preflight.
            provider: Optional provider slug when ``kind`` is provider-scoped.
            funding_source: Optional billing/funding source id.
            credential_name: Optional named credential in org storage.
            metadata: Optional opaque metadata map.

        Returns:
            Created secret ref record.
        """
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
        """Delete a project secret ref.

        Args:
            project_id: Target project id.
            secret_id: Secret ref id returned from ``list`` or ``create``.

        Returns:
            Deletion acknowledgement payload.
        """
        return self._session.secrets.delete(project_id, secret_id)


__all__ = ["ResearchSecretsAPI"]
