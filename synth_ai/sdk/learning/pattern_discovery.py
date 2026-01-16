"""Client for backend pattern discovery."""

from dataclasses import dataclass
from typing import Any

from synth_ai.core.env import get_api_key
from synth_ai.core.http import AsyncHttpClient
from synth_ai.core.urls import (
    synth_api_v1_base,
    synth_eval_job_results_url,
    synth_prompt_learning_patterns_discover_url,
)


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
        synth_user_key: str | None = None,
        *,
        timeout: float = 30.0,
        synth_base_url: str | None = None,
    ) -> None:
        if synth_user_key is None:
            synth_user_key = get_api_key("SYNTH_API_KEY", required=True)

        self._synth_base_url = synth_base_url
        self._synth_user_key = synth_user_key
        self._timeout = timeout

    async def discover(self, request: PatternDiscoveryRequest) -> dict[str, Any]:
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.post_json(
                synth_prompt_learning_patterns_discover_url(self._synth_base_url),
                json=request.to_payload(),
            )


async def get_eval_patterns(
    job_id: str,
    *,
    synth_user_key: str | None = None,
    timeout: float = 30.0,
    synth_base_url: str | None = None,
) -> dict[str, Any] | None:
    if synth_user_key is None:
        synth_user_key = get_api_key("SYNTH_API_KEY", required=True)

    async with AsyncHttpClient(
        synth_api_v1_base(synth_base_url), synth_user_key, timeout=timeout
    ) as http:
        data = await http.get_json(synth_eval_job_results_url(job_id, synth_base_url))
    summary = data.get("summary") if isinstance(data, dict) else None
    if isinstance(summary, dict):
        return summary.get("pattern_discovery")
    return None
