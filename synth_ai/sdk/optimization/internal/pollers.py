from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import click

from .utils import ensure_api_base, fmt_duration, http_get, parse_json_response, sleep


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

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def poll(self, path: str) -> PollOutcome:
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
                info = parse_json_response(resp, context=f"Job status ({url})")
                status = (info.get("status") or info.get("state") or "").lower()
                timestamp = datetime.now().strftime("%H:%M:%S")
                click.echo(f"[poll] {timestamp} {elapsed:.0f}s status={status}")
                if status in {"succeeded", "failed", "cancelled", "canceled", "completed"}:
                    break
            except Exception as exc:  # pragma: no cover - network failures
                click.echo(f"[poll] error: {exc}")
            sleep(self.interval)
            elapsed += self.interval
        else:
            click.echo(f"[poll] timeout after {fmt_duration(self.timeout)}")
        return PollOutcome(status=status, payload=info)


class PromptLearningJobPoller(JobPoller):
    """Poller for prompt learning jobs (GEPA)."""

    def poll_job(self, job_id: str) -> PollOutcome:
        """Poll a prompt learning job by ID.

        Args:
            job_id: Job ID (e.g., "pl_9c58b711c2644083")

        Returns:
            PollOutcome with status and payload
        """
        return super().poll(f"/api/prompt-learning/online/jobs/{job_id}")


class EvalJobPoller(JobPoller):
    """Poller for evaluation jobs.

    Polls the backend eval job API to check job status until completion.

    Example:
        >>> poller = EvalJobPoller(
        ...     base_url="https://api.usesynth.ai",
        ...     api_key="sk_live_...",
        ...     interval=2.0,
        ...     timeout=1200.0,
        ... )
        >>> outcome = poller.poll_job("eval-abc123")
        >>> if outcome.status == "completed":
        ...     print(outcome.payload)

    See Also:
        - `synth_ai.sdk.api.eval.EvalJob`: High-level eval job API
        - Backend API: GET /api/eval/jobs/{job_id}
    """

    def poll_job(self, job_id: str) -> PollOutcome:
        """Poll an eval job by ID.

        Args:
            job_id: Job ID (e.g., "eval-abc123")

        Returns:
            PollOutcome with status and payload
        """
        return super().poll(f"/api/eval/jobs/{job_id}")


__all__ = [
    "PollOutcome",
    "JobPoller",
    "PromptLearningJobPoller",
    "EvalJobPoller",
]
