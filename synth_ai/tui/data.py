"""Data access for TUI monitoring views."""

from __future__ import annotations

from typing import Any, Dict, List

from synth_ai.core._utils.http import AsyncHttpClient
from synth_ai.core.telemetry import log_info

from .models import ActionResult, JobDetail, JobEvent, JobSummary, coerce_events


class PromptLearningDataClient:
    """Thin API client for prompt learning monitoring."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def list_jobs(self, *, limit: int = 50, offset: int = 0) -> List[JobSummary]:
        params = {"limit": limit, "offset": offset}
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            try:
                payload = await http.get("/api/prompt-learning/online/jobs", params=params)
            except Exception as exc:
                log_info("tui.list_jobs.failed", ctx={"error": type(exc).__name__})
                return []
        jobs = _extract_job_list(payload)
        return [JobSummary.from_api(job) for job in jobs]

    async def get_job(self, job_id: str) -> JobDetail:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(f"/api/prompt-learning/online/jobs/{job_id}")
        return JobDetail.from_api(payload if isinstance(payload, dict) else {})

    async def get_events(
        self, job_id: str, *, since_seq: int = 0, limit: int = 500
    ) -> list[JobEvent]:
        params = {"since_seq": since_seq, "limit": limit}
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(f"/api/prompt-learning/online/jobs/{job_id}/events", params=params)
        raw_events = _extract_events(payload)
        return coerce_events(raw_events)

    async def get_metrics(self, job_id: str) -> Dict[str, Any]:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/prompt-learning/online/jobs/{job_id}/metrics")

    async def list_artifacts(self, job_id: str) -> list[Dict[str, Any]]:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(f"/api/prompt-learning/online/jobs/{job_id}/artifacts")
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, list):
                return artifacts
        return []

    async def get_snapshot(self, job_id: str, snapshot_id: str) -> Dict[str, Any]:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(
                f"/api/prompt-learning/online/jobs/{job_id}/snapshots/{snapshot_id}"
            )

    async def cancel_job(self, job_id: str) -> ActionResult:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            try:
                payload = await http.post_json(
                    f"/api/prompt-learning/online/jobs/{job_id}/cancel", json={}
                )
                return ActionResult(ok=True, message="Cancel requested", payload=_to_dict(payload))
            except Exception as exc:
                log_info(
                    "tui.cancel_job.failed",
                    ctx={"job_id": job_id, "error": type(exc).__name__},
                )
        return ActionResult(ok=False, message="Cancel failed or unsupported", payload={})


def _extract_job_list(payload: Any) -> list[Dict[str, Any]]:
    if isinstance(payload, list):
        return [job for job in payload if isinstance(job, dict)]
    if isinstance(payload, dict):
        for key in ("jobs", "data", "results"):
            items = payload.get(key)
            if isinstance(items, list):
                return [job for job in items if isinstance(job, dict)]
    return []


def _extract_events(payload: Any) -> list[Dict[str, Any]]:
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if isinstance(payload, dict):
        events = payload.get("events")
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]
    return []


def _to_dict(payload: Any) -> Dict[str, Any]:
    return payload if isinstance(payload, dict) else {}

