"""Hosted Task Apps SDK — create, manage, and reference hosted task apps by ID."""

from __future__ import annotations

import os
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base

__all__ = [
    "TaskAppType",
    "TaskAppSpec",
    "TaskApp",
    "TaskAppsClient",
]


class TaskAppType(str, Enum):
    """Supported hosted task app types."""

    harbor_code = "harbor_code"
    harbor_browser = "harbor_browser"
    archipelago = "archipelago"
    openenv = "openenv"


class TaskAppSpec(BaseModel):
    """Specification for creating a hosted task app."""

    name: str = Field(..., description="Unique name for this task app within the org")
    task_type: TaskAppType = Field(..., description="Type of task app to provision")
    definition: dict[str, Any] = Field(
        default_factory=dict,
        description="Task app definition (environment setup, verifier config, etc.)",
    )
    environment_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional environment configuration overrides",
    )


class TaskApp(BaseModel):
    """Response model for a hosted task app."""

    id: str
    name: str
    task_type: str
    status: str
    internal_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


def _resolve_base_url(base_url: str | None) -> str:
    if base_url and base_url.strip():
        return normalize_backend_base(base_url)
    return normalize_backend_base(BACKEND_URL_BASE)


def _resolve_api_key(api_key: str | None) -> str:
    if api_key and api_key.strip():
        return api_key
    try:
        resolved = get_api_key("SYNTH_API_KEY", required=True)
    except Exception:
        resolved = os.environ.get("SYNTH_API_KEY", "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide or set SYNTH_API_KEY)")
    return resolved


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


class TaskAppsClient:
    """Client for managing hosted task apps.

    Usage::

        client = TaskAppsClient(api_key="sk_...")
        app = client.create(TaskAppSpec(
            name="my-code-task",
            task_type=TaskAppType.harbor_code,
            definition={"repo": "https://github.com/..."},
        ))
        print(app.id)  # task_app_abc123def456
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        self._base_url = _resolve_base_url(backend_base)
        self._prefix = f"{self._base_url}/api/v1/task-apps"

    def _headers(self) -> dict[str, str]:
        return _headers(self._api_key)

    def create(
        self,
        spec: TaskAppSpec,
        *,
        timeout: float = 30.0,
    ) -> TaskApp:
        """Create a new hosted task app."""
        import httpx

        payload = spec.model_dump(exclude_none=True)
        resp = httpx.post(
            self._prefix,
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return TaskApp.model_validate(resp.json())

    def get(
        self,
        task_app_id: str,
        *,
        timeout: float = 30.0,
    ) -> TaskApp:
        """Get a task app by ID."""
        import httpx

        resp = httpx.get(
            f"{self._prefix}/{task_app_id}",
            headers=self._headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        return TaskApp.model_validate(resp.json())

    def list(
        self,
        *,
        timeout: float = 30.0,
    ) -> list[TaskApp]:
        """List all task apps for the org."""
        import httpx

        resp = httpx.get(
            self._prefix,
            headers=self._headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data if isinstance(data, list) else data.get("items", [])
        return [TaskApp.model_validate(item) for item in items]

    def delete(
        self,
        task_app_id: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Delete a task app."""
        import httpx

        resp = httpx.delete(
            f"{self._prefix}/{task_app_id}",
            headers=self._headers(),
            timeout=timeout,
        )
        resp.raise_for_status()

    def wait_ready(
        self,
        task_app_id: str,
        *,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> TaskApp:
        """Poll until the task app reaches 'ready' status or a terminal state."""
        deadline = time.time() + timeout
        terminal = {"ready", "failed", "stopped"}
        while time.time() < deadline:
            app = self.get(task_app_id)
            if app.status in terminal:
                return app
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Task app {task_app_id} did not reach ready state within {timeout}s"
        )
