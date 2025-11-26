from __future__ import annotations

from pathlib import Path
from typing import Any

from synth_ai.core._utils.http import AsyncHttpClient, HTTPError

from .config import prepare_sft_job_payload
from .data import validate_jsonl_or_raise


class FtClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def upload_training_file(self, path: str | Path, *, purpose: str = "fine-tune") -> str:
        p = Path(path)
        if p.suffix.lower() == ".jsonl" and purpose == "fine-tune":
            validate_jsonl_or_raise(p, min_messages=2)
        content = p.read_bytes()
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
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

    async def create_sft_job(
        self,
        *,
        model: str,
        training_file_id: str,
        hyperparameters: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = prepare_sft_job_payload(
            model=model,
            training_file=training_file_id,
            hyperparameters=hyperparameters,
            metadata=metadata,
            training_type="sft_offline",
            training_file_field="training_file_id",
            require_training_file=True,
        )
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json("/api/learning/jobs", json=body)

    async def start_job(self, job_id: str) -> dict[str, Any]:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json(f"/api/learning/jobs/{job_id}/start", json={})

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get the status and details of an SFT job.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Job details including status, progress, etc.
        """
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/learning/jobs/{job_id}")

    async def list_jobs(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List SFT jobs.
        
        Args:
            limit: Max number of jobs to return
            offset: Pagination offset
            
        Returns:
            List of job objects
        """
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            result = await http.get(f"/api/learning/jobs?limit={limit}&offset={offset}")
            return result if isinstance(result, list) else result.get("jobs", [])


def _infer_content_type(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".jsonl"):
        return "application/jsonl"
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".txt"):
        return "text/plain"
    return "application/octet-stream"
