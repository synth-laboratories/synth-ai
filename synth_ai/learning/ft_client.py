from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..http import AsyncHttpClient, HTTPError


class FtClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def upload_training_file(self, path: str | Path, *, purpose: str = "fine-tune") -> str:
        p = Path(path)
        content = p.read_bytes()
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            data = {"purpose": purpose}
            files = {"file": (p.name, content, _infer_content_type(p.name))}
            js = await http.post_multipart("/api/learning/files", data=data, files=files)
        if not isinstance(js, dict) or "id" not in js:
            raise HTTPError(status=500, url="/api/learning/files", message="invalid_upload_response", body_snippet=str(js)[:200])
        return str(js["id"])

    async def create_sft_job(
        self,
        *,
        model: str,
        training_file_id: str,
        hyperparameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body = {
            "training_type": "sft_offline",
            "model": model,
            "training_file_id": training_file_id,
            "hyperparameters": dict(hyperparameters or {}),
            "metadata": dict(metadata or {}),
        }
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json("/api/learning/jobs", json=body)

    async def start_job(self, job_id: str) -> Dict[str, Any]:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json(f"/api/learning/jobs/{job_id}/start", json={})


def _infer_content_type(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".jsonl"):
        return "application/jsonl"
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".txt"):
        return "text/plain"
    return "application/octet-stream"


