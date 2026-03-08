"""Internal graph optimization implementation.

Public API: Use `synth_ai.sdk.optimization.GraphOptimizationJob` instead.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from synth_ai.core.rust_core.urls import ensure_api_base
from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.sdk.optimization.models import GraphJobStatus as JobStatus
from synth_ai.sdk.optimization.models import GraphOptimizationResult

from .graph_optimization_client import GraphOptimizationClient
from .graph_optimization_config import GraphOptimizationConfig
from .graph_optimization_converters import (
    ConversionError,
    ConversionResult,
    ConversionWarning,
    convert_openai_sft,
    preview_conversion,
)
from .utils import http_delete, http_get, http_post, parse_json_response, run_sync


@dataclass
class GraphOptimizationJobConfig:
    """Configuration for a graph optimization job."""

    backend_url: str
    api_key: str
    config_path: Optional[Path] = None
    config_dict: Optional[Dict[str, Any]] = None
    config: GraphOptimizationConfig = field(init=False)

    def __post_init__(self) -> None:
        has_path = self.config_path is not None
        has_dict = self.config_dict is not None

        if has_path and has_dict:
            raise ValueError("Provide either config_path OR config_dict, not both")
        if not has_path and not has_dict:
            raise ValueError("Either config_path or config_dict is required")

        if has_path:
            self.config = GraphOptimizationConfig.from_toml(self.config_path)  # type: ignore[arg-type]
        else:
            self.config = GraphOptimizationConfig.model_validate(self.config_dict)

        if not self.backend_url:
            raise ValueError("backend_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        self.backend_url = ensure_api_base(self.backend_url)


class GraphOptimizationJob:
    """High-level SDK class for graph optimization jobs."""

    def __init__(
        self,
        config: GraphOptimizationJobConfig,
        job_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self._job_id = job_id

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> GraphOptimizationJob:
        if not backend_url:
            backend_url = BACKEND_URL_BASE
        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError("api_key is required (provide or set SYNTH_API_KEY)")

        job_config = GraphOptimizationJobConfig(
            backend_url=backend_url,
            api_key=api_key,
            config_path=Path(config_path),
        )
        return cls(job_config)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> GraphOptimizationJob:
        if not backend_url:
            backend_url = BACKEND_URL_BASE
        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError("api_key is required (provide or set SYNTH_API_KEY)")

        job_config = GraphOptimizationJobConfig(
            backend_url=backend_url,
            api_key=api_key,
            config_dict=config_dict,
        )
        return cls(job_config)

    @property
    def job_id(self) -> Optional[str]:
        return self._job_id

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "X-API-Key": self.config.api_key,
        }

    def submit(self) -> str:
        """Submit the job and return job_id."""
        url = f"{self.config.backend_url.rstrip('/')}/graph-evolve/jobs"
        resp = http_post(
            url, headers=self._headers(), json_body=self.config.config.to_request_dict()
        )
        data = parse_json_response(resp, context="Graph optimization submission")
        job_id = data.get("job_id")
        if not job_id:
            raise RuntimeError(f"Job submission missing job_id: {data}")
        self._job_id = job_id
        return job_id

    def get_status(self) -> Dict[str, Any]:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        url = f"{self.config.backend_url.rstrip('/')}/graph-evolve/jobs/{self._job_id}/status"
        resp = http_get(url, headers=self._headers())
        return parse_json_response(resp, context="Graph optimization status")

    def get_result(self) -> GraphOptimizationResult:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        url = f"{self.config.backend_url.rstrip('/')}/graph-evolve/jobs/{self._job_id}/result"
        resp = http_get(url, headers=self._headers())
        data = parse_json_response(resp, context="Graph optimization result")
        return GraphOptimizationResult.from_response(self._job_id, data)

    def cancel(self) -> Dict[str, Any]:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        url = f"{self.config.backend_url.rstrip('/')}/graph-evolve/jobs/{self._job_id}"
        resp = http_delete(url, headers=self._headers(), timeout=30)
        return parse_json_response(resp, context="Graph optimization cancel")

    def poll_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 5.0,
        progress: bool = False,
    ) -> GraphOptimizationResult:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        start_time = time.time()
        last_data: Dict[str, Any] = {}
        error_count = 0
        max_errors = 5
        import logging

        logger = logging.getLogger(__name__)

        while time.time() - start_time < timeout:
            try:
                status = self.get_status()
                last_data = status
                error_count = 0
                job_status = JobStatus.from_string(status.get("status", "pending"))
                if progress:
                    msg = status.get("status", "pending")
                    gen = status.get("current_generation") or status.get("generation")
                    if gen is not None:
                        msg = f"{msg} | generation {gen}"
                    logger.info("[poll] %s", msg)
                if job_status.is_terminal:
                    return self.get_result()
            except Exception as exc:
                error_count += 1
                logger.warning(
                    "Polling error %s/%s for job %s: %s",
                    error_count,
                    max_errors,
                    self._job_id,
                    exc,
                )
                if error_count >= max_errors:
                    raise RuntimeError(
                        f"Polling failed after {error_count} consecutive errors."
                    ) from exc

            time.sleep(interval)

        if progress:
            logger.warning("Polling timeout after %.0fs for job %s", timeout, self._job_id)

        return GraphOptimizationResult.from_response(self._job_id, last_data)

    async def stream_until_complete_async(
        self,
        *,
        timeout: float = 3600.0,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GraphOptimizationResult:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        async with GraphOptimizationClient(
            self.config.backend_url, api_key=self.config.api_key
        ) as client:
            async for event in client.stream_events(self._job_id, timeout=timeout):
                if on_event:
                    on_event(event)
            result = await client.get_result(self._job_id)
            return GraphOptimizationResult.from_response(self._job_id, result)

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GraphOptimizationResult:
        return run_sync(
            self.stream_until_complete_async(timeout=timeout, on_event=on_event),
            label="stream_until_complete() (use stream_until_complete_async in async contexts)",
        )


__all__ = [
    "GraphOptimizationJob",
    "GraphOptimizationJobConfig",
    "GraphOptimizationResult",
    "GraphOptimizationClient",
    "GraphOptimizationConfig",
    "ConversionError",
    "ConversionResult",
    "ConversionWarning",
    "convert_openai_sft",
    "preview_conversion",
]
