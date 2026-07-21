"""Credential-ref SDK namespace for Phase 3 resource surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.core.research._legacy.models.types import CredentialRef, RunCredentialBinding
from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class CredentialsAPI(_ClientNamespace):
    def list_project(
        self,
        project_id: str,
        *,
        kind: str | None = None,
    ) -> list[CredentialRef]:
        return [
            CredentialRef.from_wire(item)
            for item in self._client.list_project_credential_refs(
                project_id,
                kind=kind,
            )
        ]

    def create_project(
        self,
        project_id: str,
        *,
        kind: str,
        label: str,
        provider: str | None = None,
        funding_source: str | None = None,
        credential_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CredentialRef:
        return CredentialRef.from_wire(
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

    def patch_project(
        self,
        project_id: str,
        credential_ref_id: str,
        *,
        provider: str | None = None,
        funding_source: str | None = None,
        credential_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CredentialRef:
        return CredentialRef.from_wire(
            self._client.patch_project_credential_ref(
                project_id,
                credential_ref_id,
                provider=provider,
                funding_source=funding_source,
                credential_name=credential_name,
                metadata=metadata,
            )
        )

    def list_run_bindings(self, run_id: str) -> list[RunCredentialBinding]:
        return [
            RunCredentialBinding.from_wire(item)
            for item in self._client.list_run_credential_bindings(run_id)
        ]

    def create_run_binding(
        self,
        run_id: str,
        *,
        credential_ref_id: str,
    ) -> RunCredentialBinding:
        return RunCredentialBinding.from_wire(
            self._client.create_run_credential_binding(
                run_id,
                credential_ref_id=credential_ref_id,
            )
        )


__all__ = ["CredentialsAPI"]
