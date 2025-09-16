from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
import os
import time

from ..http import AsyncHttpClient, HTTPError, sleep


def _api_base(b: str) -> str:
    b = (b or "").rstrip("/")
    return b if b.endswith("/api") else f"{b}/api"


class RlClient:
    """Lightweight RL client for provider-agnostic job control.

    Notes:
    - Uses learning/* for status/events/metrics and rl/* for creation/start.
    - Trainer endpoints are resolved server-side via trainer_id.
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 600.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def resolve_trainer_start_url(self, trainer_id: str) -> str:
        """GET /api/rl/services/{id} → { training_start_url }"""
        path = f"/api/rl/services/{trainer_id}"
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=30.0) as http:
            js = await http.get(path)
        if not isinstance(js, dict):
            raise HTTPError(status=500, url=path, message="invalid_service_response", body_snippet=str(js)[:200])
        start_url = js.get("training_start_url")
        if not isinstance(start_url, str) or not start_url:
            raise HTTPError(status=500, url=path, message="missing_training_start_url", body_snippet=str(js)[:200])
        return start_url

    async def create_job(
        self,
        *,
        model: str,
        task_app_url: str,
        trainer: Dict[str, Any],
        trainer_id: Optional[str] = None,
        job_config_id: Optional[str] = None,
        inline_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body = {
            "job_type": "rl",
            "data": {
                "model": model,
                "endpoint_base_url": task_app_url,
                **({"job_config_id": job_config_id} if job_config_id else {}),
                **({"config": inline_config} if inline_config else {}),
                "trainer": {
                    "batch_size": int(trainer.get("batch_size", 1)),
                    "group_size": max(2, int(trainer.get("group_size", 2))),
                },
            },
        }
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.post_json(f"{_api_base(self._base_url)}/rl/jobs", json=body)
        if not isinstance(js, dict):
            raise HTTPError(status=500, url="/api/rl/jobs", message="invalid_create_response", body_snippet=str(js)[:200])
        return js

    async def start_job_if_supported(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = f"{_api_base(self._base_url)}/rl/jobs/{job_id}/start"
        try:
            async with AsyncHttpClient(self._base_url, self._api_key, timeout=30.0) as http:
                return await http.post_json(path, json={})
        except HTTPError as he:  # noqa: PERF203
            if he.status == 404:
                return None
            raise

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=30.0) as http:
            return await http.get(f"{_api_base(self._base_url)}/learning/jobs/{job_id}")

    async def get_events(self, job_id: str, *, since_seq: int = 0, limit: int = 200) -> List[Dict[str, Any]]:
        params = {"since_seq": since_seq, "limit": limit}
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=30.0) as http:
            try:
                js = await http.get(f"{_api_base(self._base_url)}/learning/jobs/{job_id}/events", params=params)
            except HTTPError as he:
                try:
                    print(
                        f"[poll] events HTTPError status={he.status} url={he.url} since_seq={since_seq} body={(he.body_snippet or '')[:200]}"
                    )
                except Exception:
                    pass
                raise
        if isinstance(js, dict):
            evs = js.get("events") or js.get("data")
            if isinstance(evs, list):
                return evs
        return []

    async def get_metrics(self, job_id: str, *, after_step: int = -1, limit: int = 200) -> List[Dict[str, Any]]:
        params = {"after_step": after_step, "limit": limit}
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=30.0) as http:
            js = await http.get(f"{_api_base(self._base_url)}/learning/jobs/{job_id}/metrics", params=params)
        if isinstance(js, dict) and isinstance(js.get("points"), list):
            return js["points"]
        return []

    async def poll_until_terminal(
        self,
        job_id: str,
        *,
        interval_seconds: float = 2.0,
        max_seconds: float | None = None,
        empty_polls_threshold: int = 5,
        startup_deadline_s: int = 45,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_metric: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        last_seq_by_stream: Dict[str, int] = {}
        events_job_id: Optional[str] = None
        last_status: Optional[str] = None
        last_step_by_name: Dict[str, int] = {}
        empty_polls = 0
        saw_any_event = False
        start_t = time.time()
        terminal = {"succeeded", "failed", "cancelled", "canceled", "error", "completed"}

        while True:
            status_data: Optional[Dict[str, Any]] = None
            try:
                status_data = await self.get_job(job_id)
            except Exception:
                status_data = None
            if status_data is None:
                try:
                    print(f"[poll] get_job returned None base={self._base_url} job_id={job_id}")
                except Exception:
                    pass
            status = str((status_data or {}).get("status") or "").lower()
            if status_data:
                linked = status_data.get("linked_job_id")
                if isinstance(linked, str) and linked and linked != events_job_id:
                    events_job_id = linked
                    try:
                        print(f"[poll] discovered linked_job_id stream={events_job_id}")
                    except Exception:
                        pass
            if status and status != last_status:
                last_status = status
                # Status transitions only to avoid log spam
                if on_event:
                    try:
                        on_event({"type": "rl.status", "message": status})
                    except Exception:
                        pass

            # Events
            stream_ids = [job_id]
            if events_job_id and events_job_id not in stream_ids:
                stream_ids.append(events_job_id)
            try:
                print(f"[poll] streams={stream_ids} intervals={interval_seconds}s since_map={last_seq_by_stream} empty_polls={empty_polls}")
            except Exception:
                pass
            total_events_this_cycle = 0
            terminal_event_seen = False
            terminal_event_status: Optional[str] = None
            for ev_id in stream_ids:
                since = last_seq_by_stream.get(ev_id, 0)
                try:
                    events = await self.get_events(ev_id, since_seq=since, limit=200)
                except HTTPError as he:
                    try:
                        print(f"[poll] get_events error status={he.status} url={he.url} since={since} body={(he.body_snippet or '')[:200]}")
                    except Exception:
                        pass
                    events = []
                except Exception as e:
                    try:
                        print(f"[poll] get_events unexpected error ev_id={ev_id} since={since} err={type(e).__name__}: {e}")
                    except Exception:
                        pass
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
                        try:
                            on_event(e)
                        except Exception:
                            pass
                    et = str(e.get("type") or e.get("event_type") or "").lower()
                    if et in ("rl.job.completed", "workflow.completed", "rl.train.completed"):
                        terminal_event_seen = True
                        terminal_event_status = "succeeded"
                    elif et in ("rl.job.failed", "workflow.failed"):
                        terminal_event_seen = True
                        terminal_event_status = "failed"

            # Metrics
            try:
                after = max(last_step_by_name.values()) if last_step_by_name else -1
                points = await self.get_metrics(job_id, after_step=after, limit=200)
                for p in points:
                    name = str(p.get("name") or "")
                    step = int(p.get("step") or -1)
                    if step <= last_step_by_name.get(name, -1):
                        continue
                    last_step_by_name[name] = step
                    if on_metric:
                        try:
                            on_metric(p)
                        except Exception:
                            pass
            except Exception:
                pass

            if terminal_event_seen:
                return {"status": terminal_event_status or status or "completed", "job_id": job_id}
            if status and status in terminal:
                return {"status": status, "job_id": job_id}

            if total_events_this_cycle == 0:
                empty_polls += 1
            else:
                empty_polls = 0
            if empty_polls >= max(1, int(empty_polls_threshold)):
                try:
                    print(
                        f"[poll] threshold hit: empty_polls={empty_polls} >= {empty_polls_threshold} streams={stream_ids} last_seq_map={last_seq_by_stream}"
                    )
                except Exception:
                    pass
                raise AssertionError(f"No new events detected for {empty_polls_threshold} consecutive polls. Check event ingestion.")

            if not saw_any_event and (time.time() - start_t) > int(startup_deadline_s):
                try:
                    print(
                        f"[poll] startup window exceeded: {startup_deadline_s}s base={self._base_url} job={job_id} streams={stream_ids} last_seq_map={last_seq_by_stream}"
                    )
                except Exception:
                    pass
                raise AssertionError(f"No events observed within startup window ({startup_deadline_s}s). Investigate event streaming.")

            await sleep(interval_seconds)
            if max_seconds is not None and (time.time() - start_t) >= max_seconds:
                raise TimeoutError(f"Polling timed out after {max_seconds}s for job {job_id}")


