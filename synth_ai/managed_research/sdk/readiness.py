"""Derived readiness namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk._base import _ClientNamespace


class ReadinessAPI(_ClientNamespace):
    def get(self, project_id: str) -> dict[str, Any]:
        return self._client.get_project_readiness(project_id)


__all__ = ["ReadinessAPI"]
