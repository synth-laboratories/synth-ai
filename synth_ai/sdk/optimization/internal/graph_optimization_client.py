"""Async client for Graph Optimization jobs."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from synth_ai.core.errors import HTTPError
from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.rust_core.sse import stream_sse_events
from synth_ai.core.rust_core.urls import ensure_api_base

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
        self._client: Optional[RustCoreHttpClient] = None

    async def __aenter__(self) -> GraphOptimizationClient:
        self._client = RustCoreHttpClient(
            base_url=self.base_url,
            api_key=self.api_key or "",
            timeout=self.timeout,
            shared=True,
            use_api_base=True,
        )
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    def _ensure_client(self) -> RustCoreHttpClient:
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

    def _parse_json(
        self, payload: Any, *, context: str, expect_dict: bool = True
    ) -> Dict[str, Any]:
        if expect_dict and not isinstance(payload, dict):
            raise RuntimeError(f"{context} returned unexpected JSON type: {type(payload).__name__}")
        return payload if isinstance(payload, dict) else {}

    async def start_job(self, config: GraphOptimizationConfig) -> str:
        """Start a graph optimization job."""
        client = self._ensure_client()
        prefix = self._get_api_prefix(config.algorithm)
        try:
            data = await client.post_json(f"{prefix}/jobs", json=config.to_request_dict())
        except HTTPError as exc:
            raise RuntimeError(f"Graph optimization submission failed: {exc}") from exc
        data = self._parse_json(data, context="Graph optimization submission")
        job_id = data.get("job_id")
        if not job_id:
            raise RuntimeError(f"Job submission missing job_id: {data}")
        return job_id

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        client = self._ensure_client()
        try:
            data = await client.get(f"/graph-evolve/jobs/{job_id}/status")
        except HTTPError as exc:
            raise RuntimeError(f"Graph optimization status failed: {exc}") from exc
        return self._parse_json(data, context="Graph optimization status")

    async def get_result(self, job_id: str) -> Dict[str, Any]:
        """Get job result."""
        client = self._ensure_client()
        try:
            data = await client.get(f"/graph-evolve/jobs/{job_id}/result")
        except HTTPError as exc:
            raise RuntimeError(f"Graph optimization result failed: {exc}") from exc
        return self._parse_json(data, context="Graph optimization result")

    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job."""
        client = self._ensure_client()
        try:
            data = await client.delete(f"/graph-evolve/jobs/{job_id}")
        except HTTPError as exc:
            raise RuntimeError(f"Graph optimization cancel failed: {exc}") from exc
        return self._parse_json(data, context="Graph optimization cancel", expect_dict=False)

    async def stream_events(
        self,
        job_id: str,
        timeout: float = 600.0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream events from a running job via SSE."""
        base = ensure_api_base(self.base_url).rstrip("/")
        url = f"{base}/graph-evolve/jobs/{job_id}/events/stream"
        headers: dict[str, str] = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        async for event in stream_sse_events(url, headers=headers, timeout=timeout):
            if isinstance(event, dict):
                yield event
