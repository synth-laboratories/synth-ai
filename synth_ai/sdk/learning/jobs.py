from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from synth_ai.core._utils.http import AsyncHttpClient, sleep

from .constants import TERMINAL_EVENT_FAILURE, TERMINAL_EVENT_SUCCESS, TERMINAL_STATUSES


def _api_base(b: str) -> str:
    b = (b or "").rstrip("/")
    return b if b.endswith("/api") else f"{b}/api"


class JobsApiResolver:
    def __init__(self, base_url: str, *, strict: bool) -> None:
        self._base = _api_base(base_url)
        self._strict = strict

    def status_urls(self, job_id: str) -> list[str]:
        if self._strict:
            return [f"{self._base}/learning/jobs/{job_id}"]
        return [
            f"{self._base}/learning/jobs/{job_id}",
            f"{self._base}/rl/jobs/{job_id}",
            f"{self._base}/orchestration/jobs/{job_id}",
        ]

    def events_urls(self, job_id: str, since: int) -> list[str]:
        if self._strict:
            return [f"{self._base}/learning/jobs/{job_id}/events?since_seq={since}&limit=200"]
        return [
            f"{self._base}/learning/jobs/{job_id}/events?since_seq={since}&limit=200",
            f"{self._base}/orchestration/jobs/{job_id}/events?since_seq={since}&limit=200",
            # RL /jobs/{id}/events is SSE in backend; avoid in JSON poller
        ]

    def metrics_url(self, job_id: str, after_step: int) -> str:
        return f"{self._base}/learning/jobs/{job_id}/metrics?after_step={after_step}&limit=200"


class JobHandle:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        job_id: str,
        *,
        strict: bool = True,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.job_id = job_id
        self.strict = strict
        self.timeout = timeout

    async def poll_until_terminal(
        self,
        *,
        interval_seconds: float = 2.0,
        max_seconds: float | None = None,
        empty_polls_threshold: int = 5,
        startup_deadline_s: int = 45,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_metric: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        last_seq_by_stream: dict[str, int] = {}
        events_job_id: str | None = None
        last_status: str | None = None
        last_step_by_name: dict[str, int] = {}
        empty_polls = 0
        saw_any_event = False
        start_t = time.time()
        resolver = JobsApiResolver(self.base_url, strict=self.strict)
        detected_fine_tuned_model: str | None = None

        async with AsyncHttpClient(self.base_url, self.api_key, timeout=self.timeout) as http:
            while True:
                # Status
                status_data: dict[str, Any] | None = None
                for su in resolver.status_urls(self.job_id):
                    try:
                        status_data = await http.get(su)
                        if isinstance(status_data, dict):
                            break
                    except Exception:
                        continue
                status = str((status_data or {}).get("status") or "").lower()
                if status_data:
                    linked = status_data.get("linked_job_id")
                    if isinstance(linked, str) and linked and linked != events_job_id:
                        events_job_id = linked
                    # Capture fine_tuned_model if already present on status
                    if not detected_fine_tuned_model:
                        ftm = status_data.get("fine_tuned_model")
                        if isinstance(ftm, str) and ftm:
                            detected_fine_tuned_model = ftm
                if status and status != last_status:
                    last_status = status
                    if on_event:
                        with suppress(Exception):
                            on_event({"type": "job.status", "message": status})

                # Events
                stream_ids = [self.job_id]
                if events_job_id and events_job_id not in stream_ids:
                    stream_ids.append(events_job_id)
                total_events_this_cycle = 0
                terminal_event_seen = False
                terminal_event_status: str | None = None
                for ev_id in stream_ids:
                    since = last_seq_by_stream.get(ev_id, 0)
                    for eu in resolver.events_urls(ev_id, since):
                        try:
                            ev_js = await http.get(eu)
                        except Exception:
                            continue
                        events = (ev_js or {}).get("events") or (ev_js or {}).get("data") or []
                        if not isinstance(events, list):
                            events = []
                        total_events_this_cycle += len(events)
                        if events:
                            saw_any_event = True
                        for e in events:
                            seq_val = int(e.get("seq") or 0)
                            if seq_val <= last_seq_by_stream.get(ev_id, 0):
                                continue
                            last_seq_by_stream[ev_id] = seq_val
                            if on_event:
                                with suppress(Exception):
                                    on_event(e)
                            et = str(e.get("type") or e.get("event_type") or "").lower()
                            # Capture fine_tuned_model from event data when available
                            if not detected_fine_tuned_model:
                                data_obj = e.get("data") or {}
                                if isinstance(data_obj, dict):
                                    ftm = data_obj.get("fine_tuned_model")
                                    if isinstance(ftm, str) and ftm:
                                        detected_fine_tuned_model = ftm
                            if et in TERMINAL_EVENT_SUCCESS:
                                terminal_event_seen = True
                                terminal_event_status = "succeeded"
                            elif et in TERMINAL_EVENT_FAILURE:
                                terminal_event_seen = True
                                terminal_event_status = "failed"

                # Metrics
                try:
                    after = max(last_step_by_name.values()) if last_step_by_name else -1
                    mu = resolver.metrics_url(self.job_id, after)
                    md = await http.get(mu)
                    for p in (md or {}).get("points", []):
                        name = str(p.get("name") or "")
                        step = int(p.get("step") or -1)
                        if step <= last_step_by_name.get(name, -1):
                            continue
                        last_step_by_name[name] = step
                        if on_metric:
                            with suppress(Exception):
                                on_metric(p)
                except Exception:
                    pass

                # Terminal decisions
                if terminal_event_seen or (status and status in TERMINAL_STATUSES):
                    # Best-effort enrichment of final result with fine_tuned_model
                    result_status = terminal_event_status or status or "completed"
                    final_res: dict[str, Any] = {"status": result_status, "job_id": self.job_id}
                    if not detected_fine_tuned_model:
                        # Briefly try to re-fetch status to see if fine_tuned_model is persisted
                        try:
                            for su in resolver.status_urls(self.job_id):
                                final_status = await http.get(su)
                                if isinstance(final_status, dict):
                                    ftm2 = final_status.get("fine_tuned_model")
                                    if isinstance(ftm2, str) and ftm2:
                                        detected_fine_tuned_model = ftm2
                                        break
                        except Exception:
                            pass
                    if detected_fine_tuned_model:
                        final_res["fine_tuned_model"] = detected_fine_tuned_model
                    return final_res

                # Guards (relaxed): do not abort on consecutive empty polls
                if total_events_this_cycle == 0:
                    empty_polls += 1
                else:
                    empty_polls = 0
                if not saw_any_event and (time.time() - start_t) > int(startup_deadline_s):
                    raise AssertionError(
                        f"No events observed within startup window ({startup_deadline_s}s). Investigate event streaming."
                    )
                await sleep(interval_seconds)
                if max_seconds is not None and (time.time() - start_t) >= max_seconds:
                    raise TimeoutError(
                        f"Polling timed out after {max_seconds}s for job {self.job_id}"
                    )
