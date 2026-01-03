"""State and polling controller for TUI apps."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
import threading
import concurrent.futures
from typing import Awaitable

from synth_ai.core.telemetry import log_info

from .data import PromptLearningDataClient
from .models import JobDetail, JobEvent, JobListFilter, JobSummary, ActionResult


@dataclass(slots=True)
class PromptLearningTuiState:
    jobs: List[JobSummary] = field(default_factory=list)
    selected_job: JobDetail | None = None
    events: List[JobEvent] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    filters: JobListFilter = field(default_factory=JobListFilter)
    last_error: str | None = None
    status_message: str | None = None
    last_refresh_ts: float | None = None

    def update_status(self, message: str) -> None:
        self.status_message = message


@dataclass(slots=True)
class PromptLearningTuiSnapshot:
    jobs: List[JobSummary]
    selected_job: JobDetail | None
    events: List[JobEvent]
    metrics: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    status_message: str | None
    last_error: str | None
    last_refresh_ts: float | None


class PromptLearningTuiController:
    """Polling controller with shared state for the prompt learning TUI."""

    def __init__(
        self,
        *,
        client: PromptLearningDataClient,
        refresh_interval: float = 5.0,
        event_interval: float = 2.0,
        job_limit: int = 50,
        initial_job_id: str | None = None,
    ) -> None:
        self.client = client
        self.refresh_interval = max(refresh_interval, 1.0)
        self.event_interval = max(event_interval, 0.5)
        self.job_limit = job_limit
        self.state = PromptLearningTuiState()
        self.state.filters.limit = job_limit
        self._stop = asyncio.Event()
        self._last_seq: int = 0
        self._initial_job_id = initial_job_id
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> PromptLearningTuiSnapshot:
        with self._lock:
            return PromptLearningTuiSnapshot(
                jobs=list(self.state.jobs),
                selected_job=self.state.selected_job,
                events=list(self.state.events),
                metrics=dict(self.state.metrics),
                artifacts=list(self.state.artifacts),
                status_message=self.state.status_message,
                last_error=self.state.last_error,
                last_refresh_ts=self.state.last_refresh_ts,
            )

    def apply_action_result(self, result: ActionResult) -> None:
        with self._lock:
            self.state.status_message = result.message
            if not result.ok:
                self.state.last_error = result.message
            payload = result.payload
            artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
            if isinstance(artifacts, list):
                self.state.artifacts = artifacts

    def submit_select(self, job_id: str) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self.select_job(job_id), self._loop)

    def submit_refresh(self) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self.refresh_jobs(), self._loop)

    def submit_action(self, coro: Awaitable[ActionResult]) -> concurrent.futures.Future:
        if not self._loop:
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_result(ActionResult(ok=False, message="Controller not running", payload={}))
            return future
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def run_forever(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self.refresh_jobs()
        if self._initial_job_id:
            await self.select_job(self._initial_job_id)
        elif self.state.jobs:
            await self.select_job(self.state.jobs[0].job_id)
        await asyncio.gather(
            self._job_refresh_loop(),
            self._event_loop(),
        )

    async def refresh_jobs(self) -> None:
        try:
            jobs = await self.client.list_jobs(limit=self.state.filters.limit)
        except Exception as exc:
            with self._lock:
                self.state.last_error = f"Failed to load jobs: {type(exc).__name__}"
            return
        filtered = [job for job in jobs if self.state.filters.matches(job)]
        with self._lock:
            self.state.jobs = filtered
            self.state.last_refresh_ts = time.time()
            if self.state.selected_job:
                self.state.selected_job = next(
                    (j for j in filtered if j.job_id == self.state.selected_job.job_id),
                    self.state.selected_job,
                )

    async def select_job(self, job_id: str) -> None:
        self._last_seq = 0
        with self._lock:
            self.state.events = []
        try:
            detail = await self.client.get_job(job_id)
            metrics = await self.client.get_metrics(job_id)
        except Exception as exc:
            with self._lock:
                self.state.last_error = f"Failed to load job {job_id}: {type(exc).__name__}"
            return
        with self._lock:
            self.state.selected_job = detail
            self.state.metrics = metrics
            self.state.status_message = f"Selected job {job_id}"

    async def refresh_selected_job(self) -> None:
        with self._lock:
            selected = self.state.selected_job
        if not selected:
            return
        job_id = selected.job_id
        try:
            detail = await self.client.get_job(job_id)
        except Exception as exc:
            with self._lock:
                self.state.last_error = f"Failed to refresh job {job_id}: {type(exc).__name__}"
            return
        with self._lock:
            self.state.selected_job = detail

    async def _job_refresh_loop(self) -> None:
        while not self._stop.is_set():
            await self.refresh_jobs()
            await self.refresh_selected_job()
            await asyncio.sleep(self.refresh_interval)

    async def _event_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                selected = self.state.selected_job
            if not selected:
                await asyncio.sleep(self.event_interval)
                continue
            job_id = selected.job_id
            try:
                events = await self.client.get_events(
                    job_id,
                    since_seq=self._last_seq,
                    limit=200,
                )
                if events:
                    with self._lock:
                        self.state.events.extend(events)
                        self._last_seq = max(self._last_seq, max(e.seq for e in events))
            except Exception as exc:
                log_info("tui.events.failed", ctx={"job_id": job_id, "error": type(exc).__name__})
            await asyncio.sleep(self.event_interval)
