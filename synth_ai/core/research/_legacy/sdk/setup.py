"""Project-setup SDK namespace."""

from __future__ import annotations

from typing import Any

from synth_ai.core.research._legacy.models.types import SmrProjectSetup
from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class SetupAPI(_ClientNamespace):
    def get(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.get_project_setup(project_id))

    def prepare(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.prepare_project_setup(project_id))

    def start_onboarding(self, project_id: str) -> dict[str, Any]:
        return self._client.start_project_onboarding(project_id)

    def complete_onboarding_step(
        self,
        project_id: str,
        *,
        step: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.complete_project_onboarding_step(
            project_id,
            step=step,
            status=status,
            detail=detail,
        )

    def dry_run_onboarding(self, project_id: str) -> dict[str, Any]:
        return self._client.run_project_onboarding_dry_run(project_id)

    def get_onboarding_status(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_onboarding_status(project_id)


__all__ = ["SetupAPI"]
