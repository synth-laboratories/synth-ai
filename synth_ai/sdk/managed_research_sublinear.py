from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SmrSublinearClient:
    parent: Any

    def list_tasks(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.sublinear_list_tasks(project_id, **kwargs)

    def get_task(self, project_id: str, task_id: str) -> dict[str, Any]:
        return self.parent.sublinear_get_task(project_id, task_id)

    def create_task(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.sublinear_create_task(project_id, **kwargs)

    def update_task(self, project_id: str, task_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.sublinear_update_task(project_id, task_id, **kwargs)

    def list_comments(self, project_id: str, task_id: str) -> dict[str, Any]:
        return self.parent.sublinear_list_comments(project_id, task_id)

    def add_comment(self, project_id: str, task_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.sublinear_add_comment(project_id, task_id, **kwargs)


__all__ = ["SmrSublinearClient"]
