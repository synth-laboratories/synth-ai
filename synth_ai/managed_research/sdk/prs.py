"""Project PR namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

from typing import Any, List

from synth_ai.managed_research.sdk._base import _ClientNamespace


class PrsAPI(_ClientNamespace):
    def list(self, project_id: str) -> List[dict[str, Any]]:
        return self._client.list_project_prs(project_id)

    def get(self, project_id: str, pr_id: str) -> dict[str, Any]:
        return self._client.get_project_pr(project_id, pr_id)


__all__ = ["PrsAPI"]
