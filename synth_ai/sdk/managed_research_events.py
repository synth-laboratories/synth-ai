from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SmrEventsClient:
    parent: Any

    def list(self, project_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.parent.list_project_events(project_id, **kwargs)

    def get(self, project_id: str, event_id: str) -> dict[str, Any]:
        return self.parent.get_project_event(project_id, event_id)

    def post_message(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.post_project_message(project_id, **kwargs)

    def stream(self, project_id: str, **kwargs: Any) -> Any:
        return self.parent.stream_project_events(project_id, **kwargs)


__all__ = ["SmrEventsClient"]
