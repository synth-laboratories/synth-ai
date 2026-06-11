"""Project output namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk._base import _ClientNamespace


class OutputsAPI(_ClientNamespace):
    def list(self, project_id: str) -> list[dict[str, Any]]:
        return self._client.list_project_outputs(project_id)


__all__ = ["OutputsAPI"]
