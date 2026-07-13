"""Hosted Containers SDK — create, manage, and reference hosted containers by ID.

Access via ``SynthClient().containers``.
"""

from __future__ import annotations

import asyncio
import builtins
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from synth_ai.sdk.base import SynthBaseClient

__all__ = [
    "AsyncContainersClient",
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


class ContainersClient(SynthBaseClient):
    """Create and manage hosted environment containers."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            backend_base=backend_base,
            timeout_seconds=timeout_seconds,
        )
        self._prefix = "/v1/containers"

    def create(
        self,
        spec: ContainerSpec,
        *,
        timeout_seconds: float | None = None,
    ) -> Container:
        """Provision a new hosted container from a :class:`ContainerSpec`."""
        payload = spec.model_dump(exclude_none=True)
        return self.cast_to(
            Container,
            self._request(
                "POST",
                self._prefix,
                json_body=payload,
                timeout_seconds=timeout_seconds,
            ),
        )

    def get(
        self,
        container_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> Container:
        """Retrieve a container by id."""
        return self.cast_to(
            Container,
            self._request(
                "GET",
                f"{self._prefix}/{container_id}",
                timeout_seconds=timeout_seconds,
            ),
        )

    def list(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> builtins.list[Container]:
        """List containers in the current organization."""
        data = self._request("GET", self._prefix, timeout_seconds=timeout_seconds)
        items = data if isinstance(data, list) else data.get("items", [])
        return [self.cast_to(Container, item) for item in items]

    def delete(
        self,
        container_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        """Delete a hosted container by id."""
        self._request(
            "DELETE",
            f"{self._prefix}/{container_id}",
            timeout_seconds=timeout_seconds,
        )

    def wait_ready(
        self,
        container_id: str,
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 2.0,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> Container:
        """Poll until the container reaches 'ready' status or a terminal state."""
        if timeout is not None:
            timeout_seconds = timeout
        if poll_interval is not None:
            poll_interval_seconds = poll_interval
        deadline = time.time() + timeout_seconds
        terminal = {"ready", "failed", "stopped"}
        while time.time() < deadline:
            app = self.get(container_id)
            if app.status in terminal:
                return app
            time.sleep(poll_interval_seconds)
        raise TimeoutError(
            f"Container {container_id} did not reach ready state within {timeout_seconds}s"
        )


class AsyncContainersClient:
    """Async adapter over :class:`ContainersClient` (thread-offloaded)."""

    def __init__(self, sync_client: ContainersClient) -> None:
        """Wrap a sync client for async call sites."""
        self._sync_client = sync_client

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._sync_client, name)
        if callable(attr):

            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(attr, *args, **kwargs)

            return _wrapped
        return attr
