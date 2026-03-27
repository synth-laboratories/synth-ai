from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SmrRuntimeClient:
    parent: Any

    def list_messages(self, run_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.parent.list_runtime_messages(run_id, **kwargs)

    def enqueue_message(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.enqueue_runtime_message(run_id, **kwargs)

    def list_context(self, run_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.parent.list_runtime_context(run_id, **kwargs)

    def publish_context(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.parent.publish_runtime_context(run_id, **kwargs)


__all__ = ["SmrRuntimeClient"]
