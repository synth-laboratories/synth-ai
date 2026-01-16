from pathlib import Path
from typing import Any

from synth_ai.core.http import AsyncHttpClient, HTTPError
from synth_ai.core.urls import (
    synth_api_v1_base,
    synth_learning_files_url,
    synth_learning_job_start_url,
    synth_learning_job_url,
    synth_learning_jobs_url,
)

from .config import prepare_sft_job_payload
from .data import validate_jsonl_or_raise


class FtClient:
    def __init__(
        self,
        synth_user_key: str | None = None,
        *,
        timeout: float = 30.0,
        synth_base_url: str | None = None,
    ) -> None:
        self._synth_base_url = synth_base_url
        if synth_user_key is None:
            raise ValueError("synth_user_key is required")
        self._synth_user_key = synth_user_key
        self._timeout = timeout

    async def upload_training_file(self, path: str | Path, *, purpose: str = "fine-tune") -> str:
        p = Path(path)
        if p.suffix.lower() == ".jsonl" and purpose == "fine-tune":
            validate_jsonl_or_raise(p, min_messages=2)
        content = p.read_bytes()
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            data = {"purpose": purpose}
            files = {"file": (p.name, content, _infer_content_type(p.name))}
            js = await http.post_multipart(
                synth_learning_files_url(self._synth_base_url), data=data, files=files
            )
        if not isinstance(js, dict) or "id" not in js:
            raise HTTPError(
                status=500,
                url=synth_learning_files_url(self._synth_base_url),
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
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.post_json(synth_learning_jobs_url(self._synth_base_url), json=body)

    async def start_job(self, job_id: str) -> dict[str, Any]:
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.post_json(
                synth_learning_job_start_url(job_id, self._synth_base_url), json={}
            )

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get the status and details of an SFT job.

        Args:
            job_id: The job ID to check

        Returns:
            Job details including status, progress, etc.
        """
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            return await http.get(synth_learning_job_url(job_id, self._synth_base_url))

    async def list_jobs(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List SFT jobs.

        Args:
            limit: Max number of jobs to return
            offset: Pagination offset

        Returns:
            List of job objects
        """
        async with AsyncHttpClient(
            synth_api_v1_base(self._synth_base_url), self._synth_user_key, timeout=self._timeout
        ) as http:
            result = await http.get(
                f"{synth_learning_jobs_url(self._synth_base_url)}?limit={limit}&offset={offset}"
            )
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
