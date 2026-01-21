"""Internal graph optimization implementation.

Public API: Use `synth_ai.sdk.optimization.GraphOptimizationJob` instead.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from synth_ai.core.utils.urls import BACKEND_URL_BASE

from .graph_optimization_client import GraphOptimizationClient
from .graph_optimization_config import GraphOptimizationConfig
from .graph_optimization_converters import (
    ConversionError,
    ConversionResult,
    ConversionWarning,
    convert_openai_sft,
    preview_conversion,
)
from .utils import http_get, http_post


class JobStatus(str, Enum):
    """Status of a graph optimization job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> JobStatus:
        try:
            return cls(status.lower())
        except ValueError:
            return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def is_success(self) -> bool:
        return self == JobStatus.COMPLETED


@dataclass
class GraphOptimizationResult:
    """Typed result from a graph optimization job."""

    job_id: str
    status: JobStatus
    best_score: Optional[float] = None
    best_yaml: Optional[str] = None
    best_snapshot_id: Optional[str] = None
    generations_completed: Optional[int] = None
    total_candidates_evaluated: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, job_id: str, data: Dict[str, Any]) -> GraphOptimizationResult:
        status_str = data.get("status", "pending")
        status = JobStatus.from_string(status_str)
        best_score = data.get("best_score") or data.get("best_reward")
        return cls(
            job_id=job_id,
            status=status,
            best_score=best_score,
            best_yaml=data.get("best_yaml"),
            best_snapshot_id=data.get("best_snapshot_id"),
            generations_completed=data.get("generations_completed"),
            total_candidates_evaluated=data.get("total_candidates_evaluated"),
            duration_seconds=data.get("duration_seconds"),
            error=data.get("error"),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        return self.status.is_success

    @property
    def failed(self) -> bool:
        return self.status == JobStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


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
        return {"Authorization": f"Bearer {self.config.api_key}"}

    def submit(self) -> str:
        """Submit the job and return job_id."""
        url = f"{self.config.backend_url.rstrip('/')}/graph-evolve/jobs"
        resp = http_post(
            url, headers=self._headers(), json_body=self.config.config.to_request_dict()
        )
        if not resp.ok:
            raise RuntimeError(f"Job submission failed: HTTP {resp.status_code} {resp.text}")
        data = resp.json()
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
        if not resp.ok:
            raise RuntimeError(f"Status request failed: HTTP {resp.status_code} {resp.text}")
        return resp.json()

    def get_result(self) -> GraphOptimizationResult:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        url = f"{self.config.backend_url.rstrip('/')}/graph-evolve/jobs/{self._job_id}/result"
        resp = http_get(url, headers=self._headers())
        if not resp.ok:
            raise RuntimeError(f"Result request failed: HTTP {resp.status_code} {resp.text}")
        data = resp.json()
        return GraphOptimizationResult.from_response(self._job_id, data)

    def cancel(self) -> Dict[str, Any]:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        url = f"{self.config.backend_url.rstrip('/')}/graph-evolve/jobs/{self._job_id}"
        resp = requests.delete(url, headers=self._headers(), timeout=30)
        if not resp.ok:
            raise RuntimeError(f"Cancel failed: HTTP {resp.status_code} {resp.text}")
        return resp.json()

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

        while time.time() - start_time < timeout:
            try:
                status = self.get_status()
                last_data = status
                job_status = JobStatus.from_string(status.get("status", "pending"))
                if progress:
                    msg = status.get("status", "pending")
                    gen = status.get("current_generation") or status.get("generation")
                    if gen is not None:
                        msg = f"{msg} | generation {gen}"
                    print(f"[poll] {msg}")
                if job_status.is_terminal:
                    return self.get_result()
            except Exception as exc:
                if progress:
                    print(f"[poll] error: {exc}")

            time.sleep(interval)

        if progress:
            print(f"[poll] timeout after {timeout:.0f}s")

        return GraphOptimizationResult.from_response(self._job_id, last_data)

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GraphOptimizationResult:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        async def _run() -> GraphOptimizationResult:
            async with GraphOptimizationClient(
                self.config.backend_url, api_key=self.config.api_key
            ) as client:
                async for event in client.stream_events(self._job_id, timeout=timeout):
                    if on_event:
                        on_event(event)
                result = await client.get_result(self._job_id)
                return GraphOptimizationResult.from_response(self._job_id, result)

        return asyncio.run(_run())


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
