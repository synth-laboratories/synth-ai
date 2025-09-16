from __future__ import annotations

from typing import Any, Dict, Optional

from synth_ai.http import AsyncHttpClient


class FilesApi:
    def __init__(self, http: AsyncHttpClient) -> None:
        self._http = http

    async def upload(self, *, filename: str, content: bytes, purpose: str, content_type: Optional[str] = None, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        data = {"purpose": purpose}
        files = {"file": (filename, content, content_type)}
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http.post_multipart("/api/files", data=data, files=files, headers=headers)

    async def list(self, *, purpose: Optional[str] = None, after: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if purpose is not None:
            params["purpose"] = purpose
        if after is not None:
            params["after"] = after
        params["limit"] = limit
        return await self._http.get("/api/files", params=params)

    async def retrieve(self, file_id: str) -> Dict[str, Any]:
        return await self._http.get(f"/api/files/{file_id}")

    async def delete(self, file_id: str) -> Any:
        return await self._http.delete(f"/api/files/{file_id}")

    async def list_jobs(self, file_id: str, *, after: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if after is not None:
            params["after"] = after
        return await self._http.get(f"/api/files/{file_id}/jobs", params=params)


class SftJobsApi:
    def __init__(self, http: AsyncHttpClient) -> None:
        self._http = http

    async def create(
        self,
        *,
        training_file: str,
        model: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        integrations: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "training_file": training_file,
            "model": model,
        }
        if validation_file is not None:
            payload["validation_file"] = validation_file
        if hyperparameters is not None:
            payload["hyperparameters"] = hyperparameters
        if suffix is not None:
            payload["suffix"] = suffix
        if integrations is not None:
            payload["integrations"] = integrations
        if metadata is not None:
            payload["metadata"] = metadata
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http.post_json("/api/sft/jobs", json=payload, headers=headers)

    async def list(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        file_id: Optional[str] = None,
        created_after: Optional[int] = None,
        created_before: Optional[int] = None,
        after: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if model is not None:
            params["model"] = model
        if file_id is not None:
            params["file_id"] = file_id
        if created_after is not None:
            params["created_after"] = created_after
        if created_before is not None:
            params["created_before"] = created_before
        if after is not None:
            params["after"] = after
        return await self._http.get("/api/sft/jobs", params=params)

    async def retrieve(self, job_id: str) -> Dict[str, Any]:
        return await self._http.get(f"/api/sft/jobs/{job_id}")

    async def cancel(self, job_id: str) -> Dict[str, Any]:
        return await self._http.post_json(f"/api/sft/jobs/{job_id}/cancel", json={})

    async def list_events(self, job_id: str, *, since_seq: int = 0, limit: int = 200) -> Dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        return await self._http.get(f"/api/sft/jobs/{job_id}/events", params=params)

    async def checkpoints(self, job_id: str, *, after: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if after is not None:
            params["after"] = after
        return await self._http.get(f"/api/sft/jobs/{job_id}/checkpoints", params=params)


class RlJobsApi:
    def __init__(self, http: AsyncHttpClient) -> None:
        self._http = http

    async def create(
        self,
        *,
        model: str,
        endpoint_base_url: str,
        trainer_id: str,
        trainer: Optional[Dict[str, Any]] = None,
        job_config_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "endpoint_base_url": endpoint_base_url,
            "trainer_id": trainer_id,
        }
        if trainer is not None:
            payload["trainer"] = trainer
        if job_config_id is not None:
            payload["job_config_id"] = job_config_id
        if config is not None:
            payload["config"] = config
        if metadata is not None:
            payload["metadata"] = metadata
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http.post_json("/api/rl/jobs", json=payload, headers=headers)

    async def list(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        created_after: Optional[int] = None,
        created_before: Optional[int] = None,
        after: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if model is not None:
            params["model"] = model
        if created_after is not None:
            params["created_after"] = created_after
        if created_before is not None:
            params["created_before"] = created_before
        if after is not None:
            params["after"] = after
        return await self._http.get("/api/rl/jobs", params=params)

    async def retrieve(self, job_id: str) -> Dict[str, Any]:
        return await self._http.get(f"/api/rl/jobs/{job_id}")

    async def cancel(self, job_id: str) -> Dict[str, Any]:
        return await self._http.post_json(f"/api/rl/jobs/{job_id}/cancel", json={})

    async def list_events(self, job_id: str, *, since_seq: int = 0, limit: int = 200) -> Dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        return await self._http.get(f"/api/rl/jobs/{job_id}/events", params=params)

    async def metrics(self, job_id: str, *, after_step: int = -1, limit: int = 200) -> Dict[str, Any]:
        params = {"after_step": after_step, "limit": limit}
        return await self._http.get(f"/api/rl/jobs/{job_id}/metrics", params=params)


class ModelsApi:
    def __init__(self, http: AsyncHttpClient) -> None:
        self._http = http

    async def list(
        self,
        *,
        source: Optional[str] = None,
        base_model: Optional[str] = None,
        status: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if source is not None:
            params["source"] = source
        if base_model is not None:
            params["base_model"] = base_model
        if status is not None:
            params["status"] = status
        if after is not None:
            params["after"] = after
        return await self._http.get("/api/models", params=params)

    async def retrieve(self, model_id: str) -> Dict[str, Any]:
        return await self._http.get(f"/api/models/{model_id}")

    async def delete(self, model_id: str) -> Any:
        return await self._http.delete(f"/api/models/{model_id}")

    async def list_jobs(self, model_id: str, *, after: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if after is not None:
            params["after"] = after
        return await self._http.get(f"/api/models/{model_id}/jobs", params=params)


class JobsClient:
    """High-level client aggregating job APIs.

    Usage:
        async with JobsClient(base_url, api_key) as c:
            await c.files.list()
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0, http: Optional[AsyncHttpClient] = None) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._http = http or AsyncHttpClient(base_url, api_key, timeout=timeout)
        self.files = FilesApi(self._http)
        self.sft = SftJobsApi(self._http)
        self.rl = RlJobsApi(self._http)
        self.models = ModelsApi(self._http)

    async def __aenter__(self) -> "JobsClient":
        await self._http.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self._http.__aexit__(exc_type, exc, tb)
