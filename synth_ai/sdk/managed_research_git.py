from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SmrGitClient:
    parent: Any

    def get_status(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_get_status(project_id, **kwargs)

    def list_tree(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_list_tree(project_id, **kwargs)

    def read_file(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_read_file(project_id, **kwargs)

    def get_diff(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_get_diff(project_id, **kwargs)

    def create_branch(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_create_branch(project_id, **kwargs)

    def write_files(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_write_files(project_id, **kwargs)

    def upload_files(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_upload_files(project_id, **kwargs)

    def create_commit(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_create_commit(project_id, **kwargs)

    def push(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_push(project_id, **kwargs)

    def list_pull_requests(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_list_pull_requests(project_id, **kwargs)

    def open_pull_request(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_open_pull_request(project_id, **kwargs)

    def comment_on_pull_request(self, project_id: str, pr_id: int, **kwargs: Any) -> dict[str, Any]:
        return self.parent.git_comment_on_pull_request(project_id, pr_id, **kwargs)


__all__ = ["SmrGitClient"]
