import warnings
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, TypedDict

from synth_ai.core.errors import HTTPError
from synth_ai.core.rust_core.http import RustCoreHttpClient, sleep
from synth_ai.sdk.shared.models import UnsupportedModelError, normalize_model_identifier


class LearningClient:
    """Client for learning/training jobs.

    Note: SFT-specific job creation has been moved to the research repo.
    Use this client for general learning job operations.
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def upload_training_file(self, path: str | Path, *, purpose: str = "fine-tune") -> str:
        p = Path(path)
        content = p.read_bytes()
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            data = {"purpose": purpose}
            files = {"file": (p.name, content, _infer_content_type(p.name))}
            js = await http.post_multipart("/api/learning/files", data=data, files=files)
        if not isinstance(js, dict) or "id" not in js:
            raise HTTPError(
                status=500,
                url="/api/learning/files",
                message="invalid_upload_response",
                body_snippet=str(js)[:200],
            )
        return str(js["id"])

    async def create_job(
        self,
        *,
        job_type: str,
        model: str,
        training_file_id: str,
        hyperparameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        validation_file: str | None = None,
    ) -> dict[str, Any]:
        """Create a learning job.

        Note: For SFT-specific jobs with full validation, use the research repo's SFT client.
        """
        try:
            normalized_model = normalize_model_identifier(model, allow_finetuned_prefixes=True)
        except UnsupportedModelError as exc:
            raise ValueError(str(exc)) from exc

        body: dict[str, Any] = {
            "job_type": job_type,
            "model": normalized_model,
            "training_file_id": training_file_id,
            "hyperparameters": hyperparameters or {},
            "metadata": metadata or {},
        }
        if validation_file:
            body["validation_file"] = validation_file

        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json("/api/learning/jobs", json=body)

    async def start_job(self, job_id: str) -> dict[str, Any]:
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json(f"/api/learning/jobs/{job_id}/start", json={})

    async def get_job(self, job_id: str) -> dict[str, Any]:
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/learning/jobs/{job_id}")

    async def get_events(
        self, job_id: str, *, since_seq: int = 0, limit: int = 200
    ) -> list[dict[str, Any]]:
        params = {"since_seq": since_seq, "limit": limit}
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.get(f"/api/learning/jobs/{job_id}/events", params=params)
        if isinstance(js, dict) and isinstance(js.get("events"), list):
            return js["events"]
        return []

    async def get_metrics(
        self,
        job_id: str,
        *,
        name: str | None = None,
        after_step: int | None = None,
        limit: int = 500,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if name is not None:
            params["name"] = name
        if after_step is not None:
            params["after_step"] = after_step
        if run_id is not None:
            warnings.warn(
                "run_id is deprecated, use trace_correlation_id instead. Will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            params["run_id"] = run_id
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.get(f"/api/learning/jobs/{job_id}/metrics", params=params)
        if isinstance(js, dict) and isinstance(js.get("points"), list):
            return js["points"]
        return []

    async def get_timeline(self, job_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        params = {"limit": limit}
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.get(f"/api/learning/jobs/{job_id}/timeline", params=params)
        if isinstance(js, dict) and isinstance(js.get("events"), list):
            return js["events"]
        return []

    async def poll_until_terminal(
        self,
        job_id: str,
        *,
        interval_seconds: float = 2.0,
        max_seconds: float | None = 3600,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        last_seq = 0
        elapsed = 0.0
        while True:
            # Events
            events = await self.get_events(job_id, since_seq=last_seq, limit=200)
            for e in events:
                if isinstance(e, dict) and isinstance(e.get("seq"), int):
                    last_seq = max(last_seq, int(e["seq"]))
                if on_event:
                    with suppress(Exception):
                        on_event(e)

            # Status
            job = await self.get_job(job_id)
            status = str(job.get("status") or "").lower()
            if status in {"succeeded", "failed", "canceled", "cancelled"}:
                return job

            # Sleep and time budget
            await sleep(interval_seconds)
            elapsed += interval_seconds
            if max_seconds is not None and elapsed >= max_seconds:
                raise TimeoutError(f"Polling timed out after {elapsed} seconds for job {job_id}")

    # --- Optional diagnostics ---
    async def pricing_preflight(
        self, *, job_type: str, gpu_type: str, estimated_seconds: float, container_count: int
    ) -> dict[str, Any]:
        body = {
            "job_type": job_type,
            "gpu_type": gpu_type,
            "estimated_seconds": float(estimated_seconds or 0.0),
            "container_count": int(container_count or 1),
        }
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.post_json("/api/v1/pricing/preflight", json=body)
        if not isinstance(js, dict):
            raise HTTPError(
                status=500,
                url="/api/v1/pricing/preflight",
                message="invalid_preflight_response",
                body_snippet=str(js)[:200],
            )
        return js

    async def balance_autumn_normalized(self) -> dict[str, Any]:
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.get("/api/v1/balance/autumn-normalized")
        if not isinstance(js, dict):
            raise HTTPError(
                status=500,
                url="/api/v1/balance/autumn-normalized",
                message="invalid_balance_response",
                body_snippet=str(js)[:200],
            )
        return js


class FineTunedModelInfo(TypedDict, total=False):
    id: str
    base_model: str | None
    created_at: int | None
    job_id: str | None
    status: str | None


class LearningClient(LearningClient):  # type: ignore[misc]
    async def list_fine_tuned_models(self) -> list[FineTunedModelInfo]:
        """Return completed fine‑tuned models for the caller's organization.

        Calls backend route `/api/learning/models` and returns a compact list.
        """
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.get("/api/learning/models")
        if isinstance(js, dict) and isinstance(js.get("data"), list):
            out: list[FineTunedModelInfo] = []
            for item in js["data"]:
                if not isinstance(item, dict):
                    continue
                rec: FineTunedModelInfo = {
                    "id": str(item.get("id")),
                    "base_model": item.get("base_model"),
                    "created_at": item.get("created_at"),
                    "job_id": item.get("job_id"),
                    "status": item.get("status"),
                }
                if rec.get("id"):
                    out.append(rec)
            return out
        # Fallback: empty list on unexpected shape
        return []


def _infer_content_type(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".jsonl"):
        return "application/jsonl"
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".txt"):
        return "text/plain"
    return "application/octet-stream"
