from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import click

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
                resp = http_get(f"{self.base_url}{path}", headers=self._headers())
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
                click.echo(f"[poll] error: {exc}")
            sleep(self.interval)
            elapsed += self.interval
        else:
            click.echo(f"[poll] timeout after {fmt_duration(self.timeout)}")
        return PollOutcome(status=status, payload=info)


class RLJobPoller(JobPoller):
    def poll_job(self, job_id: str) -> PollOutcome:
        return super().poll(f"/rl/jobs/{job_id}")


class SFTJobPoller(JobPoller):
    def poll_job(self, job_id: str) -> PollOutcome:
        return super().poll(f"/learning/jobs/{job_id}")


__all__ = [
    "PollOutcome",
    "RLJobPoller",
    "SFTJobPoller",
]
