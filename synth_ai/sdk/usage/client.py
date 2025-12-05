"""Usage client for fetching org usage and limits.

This module provides the UsageClient class that communicates with the
Synth backend to fetch usage data and check limits.
"""

from __future__ import annotations

import os
from typing import Any

from synth_ai.core.errors import UsageLimitError
from synth_ai.core.http import AsyncHttpClient, http_request

from .models import OrgUsage, UsageMetric


def _get_base_url() -> str:
    """Get the backend base URL from environment."""
    return os.getenv("BACKEND_BASE_URL", "https://api.usesynth.ai")


def _get_api_key() -> str:
    """Get the API key from environment."""
    return os.getenv("SYNTH_API_KEY", "")


class UsageClient:
    """Client for fetching org usage and limits.

    Usage:
        # Sync usage
        client = UsageClient()
        usage = client.get()
        print(usage.tier)
        print(usage.apis.inference.tokens_per_day.remaining)

        # Async usage
        async with UsageClient() as client:
            usage = await client.get_async()

        # Check a specific limit
        client.check("prompt_opt", "jobs_per_day")  # raises UsageLimitError if exhausted
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the usage client.

        Args:
            base_url: Backend URL (defaults to BACKEND_BASE_URL env var)
            api_key: API key (defaults to SYNTH_API_KEY env var)
        """
        self._base_url = base_url or _get_base_url()
        self._api_key = api_key or _get_api_key()
        self._async_client: AsyncHttpClient | None = None

    async def __aenter__(self) -> UsageClient:
        """Enter async context."""
        self._async_client = AsyncHttpClient(self._base_url, self._api_key)
        await self._async_client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Exit async context."""
        if self._async_client is not None:
            await self._async_client.__aexit__(exc_type, exc, tb)
            self._async_client = None

    def get(self) -> OrgUsage:
        """Fetch current usage (synchronous).

        Returns:
            OrgUsage object with all usage metrics and limits

        Raises:
            HTTPError: If the API request fails
        """
        url = f"{self._base_url}/api/v1/usage"
        headers = {"authorization": f"Bearer {self._api_key}"}
        status, data = http_request("GET", url, headers=headers)

        if status != 200:
            from synth_ai.core.errors import HTTPError
            raise HTTPError(
                status=status,
                url=url,
                message="Failed to fetch usage",
                detail=data if isinstance(data, dict) else None,
            )

        if isinstance(data, dict):
            return OrgUsage.from_dict(data)
        raise ValueError(f"Unexpected response type: {type(data)}")

    async def get_async(self) -> OrgUsage:
        """Fetch current usage (asynchronous).

        Returns:
            OrgUsage object with all usage metrics and limits

        Raises:
            HTTPError: If the API request fails
        """
        if self._async_client is None:
            raise RuntimeError("Must use as async context manager")

        data = await self._async_client.get("/api/v1/usage")
        return OrgUsage.from_dict(data)

    def check(self, api: str, metric: str) -> UsageMetric:
        """Check if a specific limit has capacity remaining.

        Args:
            api: API name (inference, judges, prompt_opt, rl, sft, research)
            metric: Metric name (e.g., requests_per_min, jobs_per_day)

        Returns:
            The UsageMetric if capacity is available

        Raises:
            UsageLimitError: If the limit is exhausted
            ValueError: If the api/metric combination is invalid
        """
        usage = self.get()
        return self._check_metric(usage, api, metric)

    async def check_async(self, api: str, metric: str) -> UsageMetric:
        """Check if a specific limit has capacity remaining (async).

        Args:
            api: API name (inference, judges, prompt_opt, rl, sft, research)
            metric: Metric name (e.g., requests_per_min, jobs_per_day)

        Returns:
            The UsageMetric if capacity is available

        Raises:
            UsageLimitError: If the limit is exhausted
            ValueError: If the api/metric combination is invalid
        """
        usage = await self.get_async()
        return self._check_metric(usage, api, metric)

    def _check_metric(self, usage: OrgUsage, api: str, metric: str) -> UsageMetric:
        """Internal method to check a metric and raise if exhausted."""
        m = usage.get_metric(api, metric)
        if m is None:
            raise ValueError(f"Unknown api/metric: {api}/{metric}")

        if m.is_exhausted:
            raise UsageLimitError(
                limit_type=metric,
                api=api,
                current=m.used,
                limit=m.limit,
                tier=usage.tier,
            )

        return m


__all__ = ["UsageClient"]
