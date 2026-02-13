"""Hosted Containers SDK — create, manage, and reference hosted containers by ID."""

from __future__ import annotations

import os
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base

__all__ = [
    "ContainerType",
    "ContainerSpec",
    "Container",
    "ContainersClient",
]


class ContainerType(str, Enum):
    """Supported hosted container types."""

    harbor_code = "harbor_code"
    harbor_browser = "harbor_browser"
    archipelago = "archipelago"
    openenv = "openenv"


class ContainerSpec(BaseModel):
    """Specification for creating a hosted container."""

    name: str = Field(..., description="Unique name for this container within the org")
    task_type: ContainerType = Field(..., description="Type of container to provision")
    definition: dict[str, Any] = Field(
        default_factory=dict,
        description="Container definition (environment setup, verifier config, etc.)",
    )
    environment_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional environment configuration overrides",
    )
    internal_url: str | None = Field(
        default=None,
        description="URL where this container is running. If not provided, a placeholder is generated.",
    )


class Container(BaseModel):
    """Response model for a hosted container."""

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


class ContainersClient:
    """Client for managing hosted containers.

    Usage::

        client = ContainersClient(api_key="sk_...")
        app = client.create(ContainerSpec(
            name="my-code-task",
            task_type=ContainerType.harbor_code,
            definition={"repo": "https://github.com/..."},
        ))
        print(app.id)  # container_abc123def456
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        self._base_url = _resolve_base_url(backend_base).rstrip("/")
        self._prefix = f"{self._base_url}/api/v1/containers"

    def _headers(self) -> dict[str, str]:
        return _headers(self._api_key)

    def create(
        self,
        spec: ContainerSpec,
        *,
        timeout: float = 30.0,
    ) -> Container:
        """Create a new hosted container."""
        import httpx

        payload = spec.model_dump(exclude_none=True)
        resp = httpx.post(
            self._prefix,
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return Container.model_validate(resp.json())

    def get(
        self,
        container_id: str,
        *,
        timeout: float = 30.0,
    ) -> Container:
        """Get a container by ID."""
        import httpx

        resp = httpx.get(
            f"{self._prefix}/{container_id}",
            headers=self._headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        return Container.model_validate(resp.json())

    def list(
        self,
        *,
        timeout: float = 30.0,
    ) -> list[Container]:
        """List all containers for the org."""
        import httpx

        resp = httpx.get(
            self._prefix,
            headers=self._headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data if isinstance(data, list) else data.get("items", [])
        return [Container.model_validate(item) for item in items]

    def delete(
        self,
        container_id: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Delete a container."""
        import httpx

        resp = httpx.delete(
            f"{self._prefix}/{container_id}",
            headers=self._headers(),
            timeout=timeout,
        )
        resp.raise_for_status()

    def wait_ready(
        self,
        container_id: str,
        *,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> Container:
        """Poll until the container reaches 'ready' status or a terminal state."""
        deadline = time.time() + timeout
        terminal = {"ready", "failed", "stopped"}
        while time.time() < deadline:
            app = self.get(container_id)
            if app.status in terminal:
                return app
            time.sleep(poll_interval)
        raise TimeoutError(f"Container {container_id} did not reach ready state within {timeout}s")
