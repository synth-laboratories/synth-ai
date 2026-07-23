"""Hosted Containers SDK — create, inspect, wait for, and delete containers.

Access this API through ``SynthClient().containers``. The client reads
``SYNTH_API_KEY`` when ``api_key`` is omitted and uses the environment-selected
backend when ``backend_base`` is omitted. An unconfigured development shell
defaults to ``http://localhost:8000``.

Availability:
    This is a compatibility client for deployments that expose
    ``/v1/containers``. The SDK's bundled OpenAPI file describes those routes,
    but the current production backend and its live OpenAPI contract route
    hosted workloads through ``/v1/pools``. Use ``SynthClient().pools`` for the
    portable production workflow, and use this client only when your target
    deployment exposes the compatibility routes.

Contract:
    Container names are organization-scoped. ``definition`` and
    ``environment_config`` are backend-owned JSON contracts for the selected
    ``task_type``; use the corresponding container guide rather than guessing
    fields.

Errors:
    HTTP failures raise ``httpx.HTTPStatusError``. A missing API key fails
    before the first request with a ``ValueError`` that names
    ``SYNTH_API_KEY``. Invalid response data raises
    ``pydantic.ValidationError`` instead of returning a partial model.
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
    """Validated request for creating one hosted container.

    Attributes:
        name: Organization-scoped container name. This field is required.
        task_type: Required provisioning substrate selected from ``ContainerType``.
        definition: Backend-owned definition for ``task_type``; defaults to an empty object.
        environment_config: Optional overrides; defaults to ``None`` and is omitted when unset.
        internal_url: Optional existing runtime URL. Defaults to ``None``; the
            service supplies a placeholder when omitted.
    """

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
    """Validated hosted-container response.

    Attributes:
        id: Stable container identifier used by ``get``, ``delete``, and
            ``wait_ready``.
        name: Organization-scoped container name.
        task_type: Provisioning substrate reported by the service.
        status: Current lifecycle state.
        internal_url: Runtime URL when one is available.
        created_at: Service creation timestamp when supplied.
        updated_at: Service update timestamp when supplied.
    """

    id: str
    name: str
    task_type: str
    status: str
    internal_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ContainersClient(SynthBaseClient):
    """Create and manage hosted environment containers.

    Args:
        api_key: Synth API key. Defaults to ``SYNTH_API_KEY``.
        backend_base: API base URL. Defaults to the environment-selected
            backend; an unconfigured development shell uses localhost.
        timeout_seconds: Default HTTP timeout in seconds. Defaults to ``30``.

    Raises:
        ValueError: The API key is missing.
        httpx.HTTPStatusError: The service rejects an operation.
        pydantic.ValidationError: A service response does not match ``Container``.
    """

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
        """Provision one hosted container.

        Args:
            spec: Validated container definition.
            timeout_seconds: Per-request timeout override. Defaults to the
                client timeout.

        Returns:
            The created ``Container``; provisioning may still be in progress.

        Raises:
            httpx.HTTPStatusError: Authentication, authorization, validation,
                conflict, quota, or service errors.
            pydantic.ValidationError: The response is not a valid ``Container``.
        """
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
        """Retrieve one container by ID.

        Args:
            container_id: Stable ID returned by ``create`` or ``list``.
            timeout_seconds: Per-request timeout override. Defaults to the
                client timeout.

        Returns:
            The current ``Container`` record.

        Raises:
            httpx.HTTPStatusError: The container is unavailable or the request
                is rejected.
            pydantic.ValidationError: The response is not a valid ``Container``.
        """
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
        """List containers in the current organization.

        Args:
            timeout_seconds: Per-request timeout override. Defaults to the
                client timeout.

        Returns:
            Validated ``Container`` records; an empty organization returns ``[]``.

        Raises:
            httpx.HTTPStatusError: The request is rejected.
            pydantic.ValidationError: Any returned item is not a valid ``Container``.
        """
        data = self._request("GET", self._prefix, timeout_seconds=timeout_seconds)
        items = data if isinstance(data, list) else data.get("items", [])
        return [self.cast_to(Container, item) for item in items]

    def delete(
        self,
        container_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        """Delete one hosted container by ID.

        Args:
            container_id: Stable ID returned by ``create`` or ``list``.
            timeout_seconds: Per-request timeout override. Defaults to the
                client timeout.

        Returns:
            ``None`` after the service accepts the deletion.

        Raises:
            httpx.HTTPStatusError: The container is unavailable or the request
                is rejected.
        """
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
        """Wait for ``ready``, ``failed``, or ``stopped``.

        Args:
            container_id: Stable ID returned by ``create`` or ``list``.
            timeout_seconds: Best-effort polling window. Defaults to ``300``;
                an in-flight read or sleep can finish after the nominal window.
            poll_interval_seconds: Delay between reads. Defaults to ``2``.
            timeout: Deprecated alias for ``timeout_seconds``.
            poll_interval: Deprecated alias for ``poll_interval_seconds``.

        Returns:
            The first terminal ``Container``; check ``status == "ready"`` before use.

        Raises:
            TimeoutError: No terminal state is observed during the polling window.
            httpx.HTTPStatusError: A polling request is rejected.
            pydantic.ValidationError: A polling response is invalid.
        """
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
    """Async adapter over `ContainersClient` (thread-offloaded)."""

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
