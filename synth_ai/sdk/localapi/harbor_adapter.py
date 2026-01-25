"""HarborExecutionBackend - Execute rollouts via Harbor API.

This module provides an execution backend that proxies rollout requests
to Harbor's API instead of executing locally. This allows the same
task app code to work both locally and with Harbor deployments.

Example:
    >>> from synth_ai.sdk.localapi import TaskAppConfig
    >>> from synth_ai.sdk.localapi.harbor_adapter import HarborExecutionBackend
    >>> from synth_ai.sdk.harbor import HarborDeploymentRef
    >>>
    >>> # Create Harbor backend
    >>> harbor_backend = HarborExecutionBackend(
    ...     deployment_ref=HarborDeploymentRef(deployment_id="abc-123"),
    ... )
    >>>
    >>> # Use in TaskAppConfig
    >>> config = TaskAppConfig(
    ...     app_id="my_task",
    ...     name="My Task",
    ...     description="...",
    ...     provide_taskset_description=lambda: {...},
    ...     provide_task_instances=lambda seeds: [...],
    ...     rollout=harbor_backend,  # Use Harbor instead of local execution
    ... )
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import Request

from ._impl.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
)

if TYPE_CHECKING:
    from synth_ai.sdk.harbor import HarborDeploymentRef


class HarborExecutionError(Exception):
    """Error during Harbor rollout execution."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class HarborExecutionBackend:
    """Execute rollouts via Harbor API instead of locally.

    This class implements the RolloutExecutor callable interface,
    allowing it to be used as the `rollout` parameter in TaskAppConfig.
    Instead of executing rollouts locally, it proxies them to a
    pre-uploaded Harbor deployment.

    Attributes:
        deployment_ref: Reference to the Harbor deployment
        timeout: Request timeout in seconds (default: 600)
        max_retries: Maximum retries on transient failures (default: 2)

    Example:
        >>> from synth_ai.sdk.harbor import HarborDeploymentRef
        >>> from synth_ai.sdk.localapi.harbor_adapter import HarborExecutionBackend
        >>>
        >>> backend = HarborExecutionBackend(
        ...     deployment_ref=HarborDeploymentRef(
        ...         deployment_id="abc-123",
        ...         backend_url="https://api-dev.usesynth.ai",
        ...     ),
        ... )
        >>>
        >>> # Use as rollout executor
        >>> response = await backend(rollout_request, fastapi_request)
    """

    def __init__(
        self,
        deployment_ref: HarborDeploymentRef,
        timeout: float = 600.0,
        max_retries: int = 2,
    ) -> None:
        """Initialize Harbor execution backend.

        Args:
            deployment_ref: Reference to an existing Harbor deployment
            timeout: Request timeout in seconds (default: 600)
            max_retries: Maximum retries on transient failures (default: 2)
        """
        self.deployment_ref = deployment_ref
        self.timeout = timeout
        self.max_retries = max_retries

        # Validate deployment_ref has required fields
        if not deployment_ref.deployment_id:
            raise ValueError("deployment_ref.deployment_id is required")
        if not deployment_ref.api_key:
            raise ValueError(
                "deployment_ref.api_key is required. "
                "Set SYNTH_API_KEY environment variable or pass api_key parameter."
            )

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Harbor API requests."""
        return {
            "Authorization": f"Bearer {self.deployment_ref.api_key}",
            "Content-Type": "application/json",
        }

    def _transform_request(self, request: RolloutRequest) -> dict[str, Any]:
        """Transform RolloutRequest to Harbor API format.

        Harbor's /rollout endpoint expects TaskApp format which includes
        a run_id field for GEPA/MIPRO compatibility.

        Args:
            request: RolloutRequest from the task app

        Returns:
            Dictionary payload for Harbor API
        """
        payload = request.model_dump(mode="json", exclude_none=True)
        # Harbor endpoint requires run_id for GEPA/MIPRO compatibility
        # Derive from trace_correlation_id if not present
        if "run_id" not in payload:
            # Use deployment ID as run_id prefix for grouping
            payload["run_id"] = f"harbor-{self.deployment_ref.deployment_id[:8]}"
        return payload

    def _transform_response(
        self, harbor_response: dict[str, Any], original_request: RolloutRequest
    ) -> RolloutResponse:
        """Transform Harbor API response to RolloutResponse.

        Args:
            harbor_response: Response from Harbor API
            original_request: Original RolloutRequest

        Returns:
            RolloutResponse compatible with task app contract
        """
        # Harbor response format:
        # {
        #   "trace_correlation_id": "...",
        #   "metrics": {"reward_mean": 0.85, "details": {...}},
        #   "success": true,
        #   "error": null,
        #   "execution_metadata": {...}
        # }

        # Also handle TaskApp format response:
        # {
        #   "trace_correlation_id": "...",
        #   "reward_info": {"outcome_reward": 0.85, ...},
        #   ...
        # }

        trace_correlation_id = harbor_response.get(
            "trace_correlation_id", original_request.trace_correlation_id
        )

        # Extract reward from various possible formats
        reward = 0.0
        details: dict[str, Any] = {}

        if "reward_info" in harbor_response:
            # TaskApp format response
            reward_info = harbor_response["reward_info"]
            reward = reward_info.get("outcome_reward", 0.0)
            details = reward_info.get("details", {})
        elif "metrics" in harbor_response:
            # Harbor format response
            metrics = harbor_response["metrics"]
            reward = metrics.get("reward_mean", metrics.get("outcome_reward", 0.0))
            details = metrics.get("details", {})

        # Add Harbor execution metadata to details
        if "execution_metadata" in harbor_response:
            details["harbor_execution"] = harbor_response["execution_metadata"]

        # Add success/error info
        if "success" in harbor_response:
            details["harbor_success"] = harbor_response["success"]
        if "error" in harbor_response and harbor_response["error"]:
            details["harbor_error"] = harbor_response["error"]

        return RolloutResponse(
            trace_correlation_id=trace_correlation_id,
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                details=details,
            ),
            trace=harbor_response.get("trace"),
            inference_url=harbor_response.get("inference_url"),
            artifact=harbor_response.get("artifact"),
            success_status=harbor_response.get("success_status"),
            status_detail=harbor_response.get("status_detail"),
        )

    async def execute_async(
        self, request: RolloutRequest, fastapi_request: Request | None = None
    ) -> RolloutResponse:
        """Execute a rollout via Harbor API asynchronously.

        Args:
            request: RolloutRequest to execute
            fastapi_request: Optional FastAPI request (for compatibility)

        Returns:
            RolloutResponse from Harbor

        Raises:
            HarborExecutionError: If Harbor API call fails
        """
        payload = self._transform_request(request)
        url = self.deployment_ref.rollout_url

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers=self._get_headers(),
                    )

                    if response.status_code == 429:
                        # Rate limited - wait and retry
                        if attempt < self.max_retries:
                            await asyncio.sleep(2**attempt)
                            continue
                        raise HarborExecutionError(
                            "Harbor queue at capacity (429)",
                            status_code=429,
                            details={"response": response.text},
                        )

                    if response.status_code >= 500 and attempt < self.max_retries:
                        # Server error - retry
                        await asyncio.sleep(2**attempt)
                        continue

                    if response.status_code >= 400:
                        try:
                            error_data = response.json()
                            message = error_data.get("detail", str(error_data))
                        except Exception:
                            message = response.text or f"HTTP {response.status_code}"

                        raise HarborExecutionError(
                            f"Harbor API error: {message}",
                            status_code=response.status_code,
                            details={"response": response.text},
                        )

                    harbor_response = response.json()
                    return self._transform_response(harbor_response, request)

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                raise HarborExecutionError(
                    f"Harbor request timed out after {self.timeout}s",
                    details={"timeout": self.timeout},
                ) from e

            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                raise HarborExecutionError(
                    f"Harbor request failed: {e}",
                    details={"error": str(e)},
                ) from e

        # Should not reach here, but just in case
        raise HarborExecutionError(
            f"Harbor request failed after {self.max_retries + 1} attempts",
            details={"last_error": str(last_error)},
        )

    def execute(
        self, request: RolloutRequest, fastapi_request: Request | None = None
    ) -> RolloutResponse:
        """Execute a rollout via Harbor API synchronously.

        Args:
            request: RolloutRequest to execute
            fastapi_request: Optional FastAPI request (for compatibility)

        Returns:
            RolloutResponse from Harbor
        """
        return asyncio.run(self.execute_async(request, fastapi_request))

    async def __call__(self, request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
        """Make the backend callable as a RolloutExecutor.

        This allows HarborExecutionBackend to be used directly as the
        `rollout` parameter in TaskAppConfig.

        Args:
            request: RolloutRequest to execute
            fastapi_request: FastAPI request object

        Returns:
            RolloutResponse from Harbor
        """
        return await self.execute_async(request, fastapi_request)

    async def check_deployment_ready(self) -> bool:
        """Check if the Harbor deployment is ready.

        Returns:
            True if deployment is in READY state

        Raises:
            HarborExecutionError: If status check fails
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self.deployment_ref.status_url,
                headers=self._get_headers(),
            )

            if response.status_code >= 400:
                raise HarborExecutionError(
                    f"Failed to check deployment status: {response.text}",
                    status_code=response.status_code,
                )

            data = response.json()
            return data.get("status") == "ready"


def create_harbor_rollout_executor(
    deployment_id: str,
    backend_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 600.0,
) -> HarborExecutionBackend:
    """Create a Harbor rollout executor for use in TaskAppConfig.

    Convenience function to create a HarborExecutionBackend with
    common defaults.

    Args:
        deployment_id: UUID of the Harbor deployment
        backend_url: Synth backend URL (default: from SYNTH_BACKEND_URL env)
        api_key: Synth API key (default: from SYNTH_API_KEY env)
        timeout: Request timeout in seconds (default: 600)

    Returns:
        HarborExecutionBackend ready for use as rollout executor

    Example:
        >>> from synth_ai.sdk.localapi import TaskAppConfig
        >>> from synth_ai.sdk.localapi.harbor_adapter import create_harbor_rollout_executor
        >>>
        >>> config = TaskAppConfig(
        ...     app_id="my_task",
        ...     name="My Task",
        ...     description="...",
        ...     provide_taskset_description=lambda: {...},
        ...     provide_task_instances=lambda seeds: [...],
        ...     rollout=create_harbor_rollout_executor("abc-123-def"),
        ... )
    """
    from synth_ai.sdk.harbor import HarborDeploymentRef

    ref = HarborDeploymentRef(
        deployment_id=deployment_id,
        backend_url=backend_url,
        api_key=api_key,
    )

    return HarborExecutionBackend(deployment_ref=ref, timeout=timeout)
