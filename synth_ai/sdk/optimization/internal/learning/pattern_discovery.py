"""Client for backend pattern discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.utils.env import get_api_key, get_backend_url


@dataclass
class PatternDiscoveryRequest:
    job_id: str
    max_calls: int | None = None
    filter_noise: bool | None = None
    cluster_threshold: float | None = None
    min_support: int | None = None
    max_patterns: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"job_id": self.job_id}
        if self.max_calls is not None:
            payload["max_calls"] = self.max_calls
        if self.filter_noise is not None:
            payload["filter_noise"] = self.filter_noise
        if self.cluster_threshold is not None:
            payload["cluster_threshold"] = self.cluster_threshold
        if self.min_support is not None:
            payload["min_support"] = self.min_support
        if self.max_patterns is not None:
            payload["max_patterns"] = self.max_patterns
        return payload


class PatternDiscoveryClient:
    """Client wrapper for backend prompt pattern discovery."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        if base_url is None:
            base_url = get_backend_url()
        base_url = base_url.strip().rstrip("/")
        if base_url.endswith("/api"):
            base_url = base_url[:-4]

        if api_key is None:
            api_key = get_api_key("SYNTH_API_KEY", required=True)

        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout

    async def discover(self, request: PatternDiscoveryRequest) -> dict[str, Any]:
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json(
                "/api/prompt-learning/patterns/discover",
                json=request.to_payload(),
            )


async def get_eval_patterns(
    job_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    if base_url is None:
        base_url = get_backend_url()
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/api"):
        base_url = base_url[:-4]

    if api_key is None:
        api_key = get_api_key("SYNTH_API_KEY", required=True)

    async with RustCoreHttpClient(base_url, api_key, timeout=timeout) as http:
        data = await http.get_json(f"/api/eval/jobs/{job_id}/results")
    summary = data.get("summary") if isinstance(data, dict) else None
    if isinstance(summary, dict):
        return summary.get("pattern_discovery")
    return None
