"""Read-only Environment catalog namespace."""

from __future__ import annotations

from synth_ai.managed_research.models.types import Environment, EnvironmentPreflight
from synth_ai.managed_research.sdk._base import _ClientNamespace


class EnvironmentsAPI(_ClientNamespace):
    def list(self, *, limit: int | None = None) -> list[Environment]:
        return [
            Environment.from_wire(item)
            for item in self._client.list_environments(limit=limit)
        ]

    def get(self, name: str, digest: str | None = None) -> Environment:
        return Environment.from_wire(
            self._client.get_environment(name=name, digest=digest)
        )

    def preflight(self, name: str, digest: str | None = None) -> EnvironmentPreflight:
        return EnvironmentPreflight.from_wire(
            self._client.preflight_environment(name=name, digest=digest)
        )


__all__ = ["EnvironmentsAPI"]
