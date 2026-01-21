"""Async client for Graph Optimization jobs."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from .graph_optimization_config import GraphOptimizationConfig


class GraphOptimizationClient:
    """Client for Graph Optimization Job API.

    This client interacts with the backend to run graph optimization jobs.
    The client is agnostic to graph internals - it just manages jobs.

    Example:
        async with GraphOptimizationClient("http://localhost:8000", api_key) as client:
            config = GraphOptimizationConfig.from_toml("config.toml")
            job_id = await client.start_job(config)

            async for event in client.stream_events(job_id):
                print(event["type"], event.get("data", {}))

            result = await client.get_result(job_id)
            print(f"Best score: {result['best_score']}")
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Backend API URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> GraphOptimizationClient:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with GraphOptimizationClient(...) as client:'"
            )
        return self._client

    def _get_api_prefix(self, algorithm: str) -> str:
        """Get the API prefix for an algorithm."""
        prefixes = {
            "graph_evolve": "/graph-evolve",
            "graph_gepa": "/graph-evolve",  # Backwards compat: map old name to new endpoint
        }
        return prefixes.get(algorithm, f"/{algorithm.replace('_', '-')}")

    async def start_job(self, config: GraphOptimizationConfig) -> str:
        """Start a graph optimization job."""
        client = self._ensure_client()
        prefix = self._get_api_prefix(config.algorithm)
        response = await client.post(
            f"{prefix}/jobs",
            json=config.to_request_dict(),
        )
        response.raise_for_status()
        data = response.json()
        return data["job_id"]

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        client = self._ensure_client()
        response = await client.get(f"/graph-evolve/jobs/{job_id}/status")
        response.raise_for_status()
        return response.json()

    async def get_result(self, job_id: str) -> Dict[str, Any]:
        """Get job result."""
        client = self._ensure_client()
        response = await client.get(f"/graph-evolve/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()

    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job."""
        client = self._ensure_client()
        response = await client.delete(f"/graph-evolve/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    async def stream_events(
        self,
        job_id: str,
        timeout: float = 600.0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream events from a running job via SSE."""
        client = self._ensure_client()

        async with client.stream(
            "GET",
            f"/graph-evolve/jobs/{job_id}/events",
            timeout=timeout,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
