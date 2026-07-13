"""Progress-oriented SDK namespace."""

from __future__ import annotations

from synth_ai.managed_research.models.types import SmrLaunchPreflight, SmrProjectSetup
from synth_ai.managed_research.sdk._base import _ClientNamespace


class ProgressAPI(_ClientNamespace):
    def get_project_setup(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.get_project_setup(project_id))

    def get_project_setup_authority(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.get_project_setup_authority(project_id))

    def prepare_project_setup(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.prepare_project_setup(project_id))

    def prepare_project_setup_authority(self, project_id: str) -> SmrProjectSetup:
        return SmrProjectSetup.from_wire(self._client.prepare_project_setup_authority(project_id))

    def get_launch_preflight(self, project_id: str, **kwargs: object) -> SmrLaunchPreflight:
        return SmrLaunchPreflight.from_wire(self._client.get_launch_preflight(project_id, **kwargs))


__all__ = ["ProgressAPI"]
