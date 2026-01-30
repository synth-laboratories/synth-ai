"""HarborDeploymentUploader - Upload deployments to Harbor API.

This module handles the API interactions for creating and managing
Harbor deployments. It provides both sync and async interfaces.

Example:
    >>> from synth_ai.sdk.harbor import HarborBuildSpec, upload_harbor_deployment
    >>>
    >>> spec = HarborBuildSpec(
    ...     name="my-deployment",
    ...     dockerfile_path="./Dockerfile",
    ...     context_dir=".",
    ...     entrypoint="python run.py",
    ... )
    >>>
    >>> result = upload_harbor_deployment(spec)
    >>> print(f"Created deployment: {result.deployment_id}")
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .build_spec import HarborBuildSpec, HarborDeploymentResult


class HarborAPIError(Exception):
    """Error from Harbor API."""

    def __init__(self, status_code: int, message: str, details: dict[str, Any] | None = None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(f"Harbor API error ({status_code}): {message}")


class HarborDeploymentUploader:
    """Upload and manage Harbor deployments via API.

    Handles authentication, request formatting, and error handling
    for Harbor deployment operations.

    Attributes:
        backend_url: Synth backend URL
        api_key: Synth API key for authentication
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        backend_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        """Initialize uploader with API credentials.

        Args:
            backend_url: Synth backend URL (default: from SYNTH_BACKEND_URL env)
            api_key: Synth API key (default: from SYNTH_API_KEY env)
            timeout: Request timeout in seconds (default: 300)

        Raises:
            ValueError: If api_key is not provided and not in environment
        """
        self.backend_url = backend_url or os.getenv("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
        self.api_key = api_key or os.getenv("SYNTH_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "api_key is required. Set SYNTH_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and raise on errors.

        Args:
            response: httpx Response object

        Returns:
            Parsed JSON response

        Raises:
            HarborAPIError: If response indicates an error
        """
        if response.status_code >= 400:
            data: Any | None = None
            try:
                data = response.json()
                message = data.get("detail", data.get("message", str(data)))
            except Exception:
                message = response.text or f"HTTP {response.status_code}"

            raise HarborAPIError(
                status_code=response.status_code,
                message=message,
                details=data if isinstance(data, dict) else {"raw": response.text},
            )

        return response.json()

    async def create_deployment_async(
        self,
        spec: HarborBuildSpec,
        auto_build: bool = True,
    ) -> HarborDeploymentResult:
        """Create a new Harbor deployment asynchronously.

        Args:
            spec: HarborBuildSpec defining the deployment
            auto_build: Whether to trigger build immediately (default: True)

        Returns:
            HarborDeploymentResult with deployment details

        Raises:
            HarborAPIError: If API request fails
            FileNotFoundError: If Dockerfile or context_dir doesn't exist
            ValueError: If package exceeds size limit
        """
        from .build_spec import HarborDeploymentResult
        from .packager import HarborPackager

        # Package the build context
        packager = HarborPackager(spec)
        context_base64 = packager.package()

        # Build request payload
        payload = spec.to_api_request(context_base64)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Create deployment
            response = await client.post(
                f"{self.backend_url}/api/harbor/deployments",
                json=payload,
                headers=self._get_headers(),
            )
            data = self._handle_response(response)

            deployment_id = data["id"]
            deployment_name = data.get("name") if isinstance(data, dict) else None
            build_id = None

            latest_build = data.get("latest_build") if isinstance(data, dict) else None
            if isinstance(latest_build, dict):
                build_id = latest_build.get("id") or latest_build.get("build_id")

            # Trigger build if requested and not already queued by create_deployment
            if auto_build and not build_id:
                build_payload = {
                    "dockerfile": payload.get("dockerfile"),
                    "context_tar_base64": payload.get("context_tar_base64"),
                    "context_url": payload.get("context_url"),
                    "force": False,
                }
                build_response = await client.post(
                    f"{self.backend_url}/api/harbor/deployments/{deployment_id}/build",
                    json=build_payload,
                    headers=self._get_headers(),
                )
                build_data = self._handle_response(build_response)
                build_id = build_data.get("build_id")

        return HarborDeploymentResult(
            deployment_id=deployment_id,
            deployment_name=deployment_name,
            build_id=build_id,
            name=data["name"],
            status=data["status"],
            snapshot_id=data.get("snapshot_id"),
        )

    def create_deployment(
        self,
        spec: HarborBuildSpec,
        auto_build: bool = True,
    ) -> HarborDeploymentResult:
        """Create a new Harbor deployment synchronously.

        Args:
            spec: HarborBuildSpec defining the deployment
            auto_build: Whether to trigger build immediately (default: True)

        Returns:
            HarborDeploymentResult with deployment details

        Raises:
            HarborAPIError: If API request fails
        """
        return asyncio.run(self.create_deployment_async(spec, auto_build))

    async def get_deployment_status_async(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment status asynchronously.

        Args:
            deployment_id: UUID of the deployment

        Returns:
            Status dictionary with deployment details and build history
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.backend_url}/api/harbor/deployments/{deployment_id}/status",
                headers=self._get_headers(),
            )
            return self._handle_response(response)

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment status synchronously."""
        return asyncio.run(self.get_deployment_status_async(deployment_id))

    async def wait_for_build_async(
        self,
        deployment_id: str,
        timeout_s: float = 600.0,
        poll_interval_s: float = 5.0,
    ) -> dict[str, Any]:
        """Wait for deployment build to complete.

        Args:
            deployment_id: UUID of the deployment
            timeout_s: Maximum time to wait in seconds
            poll_interval_s: Interval between status checks

        Returns:
            Final status dictionary

        Raises:
            TimeoutError: If build doesn't complete within timeout
            HarborAPIError: If build fails
        """
        start_time = time.time()

        async def _status_via_list() -> dict[str, Any]:
            deployments = await self.list_deployments_async(limit=100)
            for item in deployments:
                if item.get("name") == deployment_id or item.get("id") == deployment_id:
                    return item
            raise HarborAPIError(404, f"Deployment not found: {deployment_id}")

        while True:
            try:
                status = await self.get_deployment_status_async(deployment_id)
            except HarborAPIError as exc:
                if exc.status_code == 500:
                    status = await _status_via_list()
                else:
                    raise
            deployment_status = status.get("status", "unknown")

            if deployment_status == "ready":
                return status

            if deployment_status == "failed":
                error = status.get("error", "Build failed")
                raise HarborAPIError(500, f"Build failed: {error}", status)

            if time.time() - start_time > timeout_s:
                raise TimeoutError(
                    f"Build did not complete within {timeout_s}s. "
                    f"Current status: {deployment_status}"
                )

            await asyncio.sleep(poll_interval_s)

    def wait_for_build(
        self,
        deployment_id: str,
        timeout_s: float = 600.0,
        poll_interval_s: float = 5.0,
    ) -> dict[str, Any]:
        """Wait for deployment build to complete synchronously."""
        return asyncio.run(self.wait_for_build_async(deployment_id, timeout_s, poll_interval_s))

    async def trigger_build_async(
        self,
        deployment_id: str,
        *,
        dockerfile: str | None = None,
        context_tar_base64: str | None = None,
        context_url: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Trigger a new build for an existing deployment.

        Args:
            deployment_id: UUID of the deployment

        Returns:
            Build information with build_id
        """
        payload = {
            "dockerfile": dockerfile,
            "context_tar_base64": context_tar_base64,
            "context_url": context_url,
            "force": force,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.backend_url}/api/harbor/deployments/{deployment_id}/build",
                json=payload,
                headers=self._get_headers(),
            )
            return self._handle_response(response)

    def trigger_build(
        self,
        deployment_id: str,
        *,
        dockerfile: str | None = None,
        context_tar_base64: str | None = None,
        context_url: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Trigger a new build synchronously."""
        return asyncio.run(
            self.trigger_build_async(
                deployment_id,
                dockerfile=dockerfile,
                context_tar_base64=context_tar_base64,
                context_url=context_url,
                force=force,
            )
        )

    async def list_deployments_async(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List deployments with optional filtering.

        Args:
            status: Filter by status (pending, building, ready, failed)
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of deployment dictionaries
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.backend_url}/api/harbor/deployments",
                params=params,
                headers=self._get_headers(),
            )
            data = self._handle_response(response)
            if isinstance(data, dict) and "deployments" in data:
                return data["deployments"] or []
            if isinstance(data, list):
                return data
            return []

    def list_deployments(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List deployments synchronously."""
        return asyncio.run(self.list_deployments_async(status, limit, offset))


def upload_harbor_deployment(
    spec: HarborBuildSpec,
    api_key: str | None = None,
    backend_url: str | None = None,
    auto_build: bool = True,
    wait_for_ready: bool = False,
    build_timeout_s: float = 600.0,
) -> HarborDeploymentResult:
    """Upload a Harbor deployment from a build spec.

    This is the primary user-facing function for uploading deployments.
    It packages the build context, creates the deployment, and optionally
    waits for the build to complete.

    Args:
        spec: HarborBuildSpec defining the deployment
        api_key: Synth API key (default: from SYNTH_API_KEY env)
        backend_url: Synth backend URL (default: from SYNTH_BACKEND_URL env)
        auto_build: Whether to trigger build immediately (default: True)
        wait_for_ready: Whether to wait for build completion (default: False)
        build_timeout_s: Maximum time to wait for build (default: 600s)

    Returns:
        HarborDeploymentResult with deployment details

    Raises:
        HarborAPIError: If API request fails
        FileNotFoundError: If Dockerfile or context_dir doesn't exist
        ValueError: If package exceeds size limit
        TimeoutError: If wait_for_ready=True and build times out

    Example:
        >>> from synth_ai.sdk.harbor import HarborBuildSpec, upload_harbor_deployment
        >>>
        >>> spec = HarborBuildSpec(
        ...     name="my-agent",
        ...     dockerfile_path="./Dockerfile",
        ...     context_dir=".",
        ... )
        >>>
        >>> result = upload_harbor_deployment(spec, wait_for_ready=True)
        >>> print(f"Deployment ready: {result.deployment_id}")
    """
    uploader = HarborDeploymentUploader(backend_url=backend_url, api_key=api_key)
    result = uploader.create_deployment(spec, auto_build=auto_build)

    if wait_for_ready and auto_build:
        deployment_key = result.deployment_name or result.deployment_id
        status = uploader.wait_for_build(deployment_key, timeout_s=build_timeout_s)
        result.status = status.get("status", result.status)
        result.snapshot_id = status.get("snapshot_id", result.snapshot_id)

    return result


async def upload_harbor_deployment_async(
    spec: HarborBuildSpec,
    api_key: str | None = None,
    backend_url: str | None = None,
    auto_build: bool = True,
    wait_for_ready: bool = False,
    build_timeout_s: float = 600.0,
) -> HarborDeploymentResult:
    """Upload a Harbor deployment asynchronously.

    Async version of upload_harbor_deployment(). See that function for
    full documentation.
    """
    uploader = HarborDeploymentUploader(backend_url=backend_url, api_key=api_key)
    result = await uploader.create_deployment_async(spec, auto_build=auto_build)

    if wait_for_ready and auto_build:
        deployment_key = result.deployment_name or result.deployment_id
        status = await uploader.wait_for_build_async(deployment_key, timeout_s=build_timeout_s)
        result.status = status.get("status", result.status)
        result.snapshot_id = status.get("snapshot_id", result.snapshot_id)

    return result
