from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SmrProjectsClient:
    parent: Any

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.parent.create_project(payload)

    def list(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.parent.list_projects(**kwargs)

    def get(self, project_id: str) -> dict[str, Any]:
        return self.parent.get_project(project_id)

    def status(self, project_id: str) -> dict[str, Any]:
        return self.parent.get_project_status(project_id)


__all__ = ["SmrProjectsClient"]
