"""Consumer-facing secret namespace.

This is a naming alias over the existing credential-ref resource surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

from synth_ai.core.research._legacy.models.types import Secret
from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class SecretsAPI(_ClientNamespace):
    def list(
        self,
        project_id: str,
        *,
        kind: str | None = None,
    ) -> List[Secret]:
        return [
            Secret.from_wire(item)
            for item in self._client.list_project_credential_refs(project_id, kind=kind)
        ]

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
    ) -> Secret:
        return Secret.from_wire(
            self._client.create_project_credential_ref(
                project_id,
                kind=kind,
                label=label,
                provider=provider,
                funding_source=funding_source,
                credential_name=credential_name,
                metadata=metadata,
            )
        )

    def delete(self, project_id: str, secret_id: str) -> dict[str, Any]:
        return self._client.delete_project_credential_ref(project_id, secret_id)


__all__ = ["SecretsAPI"]
