from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import click

from synth_ai.core.telemetry import flush_logger, log_error, log_info

from .utils import ensure_api_base, fmt_duration, http_get, sleep


@dataclass(slots=True)
class PollOutcome:
    status: str
    payload: Mapping[str, Any]


class JobPoller:
    def __init__(
        self, base_url: str, api_key: str, *, interval: float = 5.0, timeout: float = 3600.0
    ) -> None:
        self.base_url = ensure_api_base(base_url)
        self.api_key = api_key
        self.interval = interval
        self.timeout = timeout
        ctx: dict[str, Any] = {
            "base_url": self.base_url,
            "interval": interval,
            "timeout": timeout,
        }
        log_info("JobPoller initialized", ctx=ctx)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def poll(self, path: str) -> PollOutcome:
        ctx: dict[str, Any] = {"path": path, "base_url": self.base_url, "timeout": self.timeout}
        log_info("poll invoked", ctx=ctx)
        elapsed = 0.0
        status = "unknown"
        info: Mapping[str, Any] = {}
        click.echo("Polling job status...")
        while elapsed <= self.timeout:
            try:
                # Normalize URL to avoid double /api/api
                if self.base_url.endswith("/api") and path.startswith("/api"):
                    # Remove /api from path since base_url already has it
                    path_normalized = path[4:].lstrip("/")
                    url = f"{self.base_url}/{path_normalized}"
                else:
                    # Ensure proper path joining
                    path_clean = path.lstrip("/")
                    url = f"{self.base_url}/{path_clean}"
                resp = http_get(url, headers=self._headers())
                info = (
                    resp.json()
                    if resp.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                status = (info.get("status") or info.get("state") or "").lower()
                timestamp = datetime.now().strftime("%H:%M:%S")
                click.echo(f"[poll] {timestamp} {elapsed:.0f}s status={status}")
                if status in {"succeeded", "failed", "cancelled", "canceled", "completed"}:
                    break
            except Exception as exc:  # pragma: no cover - network failures
                ctx["error"] = type(exc).__name__
                log_error("poll request failed", ctx=ctx)
                click.echo(f"[poll] error: {exc}")
            sleep(self.interval)
            elapsed += self.interval
        else:
            ctx["elapsed"] = elapsed
            log_error("poll timeout", ctx=ctx)
            click.echo(f"[poll] timeout after {fmt_duration(self.timeout)}")
        ctx["status"] = status
        ctx["elapsed"] = elapsed
        log_info("poll completed", ctx=ctx)
        flush_logger()
        return PollOutcome(status=status, payload=info)


class RLJobPoller(JobPoller):
    def poll_job(self, job_id: str) -> PollOutcome:
        ctx: dict[str, Any] = {"job_id": job_id, "job_type": "rl"}
        log_info("RLJobPoller.poll_job invoked", ctx=ctx)
        return super().poll(f"/rl/jobs/{job_id}")


class SFTJobPoller(JobPoller):
    def poll_job(self, job_id: str) -> PollOutcome:
        ctx: dict[str, Any] = {"job_id": job_id, "job_type": "sft"}
        log_info("SFTJobPoller.poll_job invoked", ctx=ctx)
        return super().poll(f"/learning/jobs/{job_id}")


class PromptLearningJobPoller(JobPoller):
    """Poller for prompt learning jobs (MIPRO and GEPA)."""

    def poll_job(self, job_id: str) -> PollOutcome:
        """Poll a prompt learning job by ID.

        Args:
            job_id: Job ID (e.g., "pl_9c58b711c2644083")

        Returns:
            PollOutcome with status and payload
        """
        ctx: dict[str, Any] = {"job_id": job_id, "job_type": "prompt_learning"}
        log_info("PromptLearningJobPoller.poll_job invoked", ctx=ctx)
        return super().poll(f"/api/prompt-learning/online/jobs/{job_id}")


__all__ = [
    "PollOutcome",
    "RLJobPoller",
    "SFTJobPoller",
    "PromptLearningJobPoller",
]
