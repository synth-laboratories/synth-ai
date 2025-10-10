from __future__ import annotations

from typing import Any

from synth_ai.api.models.supported import normalize_model_identifier
from synth_ai.http import AsyncHttpClient
from synth_ai.learning.sft.config import prepare_sft_job_payload


class FilesApi:
    def __init__(self, http: AsyncHttpClient) -> None:
        self._http = http

    async def upload(
        self,
        *,
        filename: str,
        content: bytes,
        purpose: str,
        content_type: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        data = {"purpose": purpose}
        files = {"file": (filename, content, content_type)}
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http.post_multipart(
            "/api/files", data=data, files=files, headers=headers
        )

    async def list(
        self, *, purpose: str | None = None, after: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if purpose is not None:
            params["purpose"] = purpose
        if after is not None:
            params["after"] = after
        params["limit"] = limit
        return await self._http.get("/api/files", params=params)

    async def retrieve(self, file_id: str) -> dict[str, Any]:
        return await self._http.get(f"/api/files/{file_id}")

    async def delete(self, file_id: str) -> Any:
        return await self._http.delete(f"/api/files/{file_id}")

    async def list_jobs(
        self, file_id: str, *, after: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
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
        validation_file: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        integrations: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload = prepare_sft_job_payload(
            model=model,
            training_file=training_file,
            hyperparameters=hyperparameters,
            metadata=metadata,
            training_type=None,
            validation_file=validation_file,
            suffix=suffix,
            integrations=integrations,
            training_file_field="training_file",
            require_training_file=True,
        )
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http.post_json("/api/sft/jobs", json=payload, headers=headers)

    async def list(
        self,
        *,
        status: str | None = None,
        model: str | None = None,
        file_id: str | None = None,
        created_after: int | None = None,
        created_before: int | None = None,
        after: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
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

    async def retrieve(self, job_id: str) -> dict[str, Any]:
        return await self._http.get(f"/api/sft/jobs/{job_id}")

    async def cancel(self, job_id: str) -> dict[str, Any]:
        return await self._http.post_json(f"/api/sft/jobs/{job_id}/cancel", json={})

    async def list_events(
        self, job_id: str, *, since_seq: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        return await self._http.get(f"/api/sft/jobs/{job_id}/events", params=params)

    async def checkpoints(
        self, job_id: str, *, after: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
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
        trainer: dict[str, Any] | None = None,
        job_config_id: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": normalize_model_identifier(model),
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
        status: str | None = None,
        model: str | None = None,
        created_after: int | None = None,
        created_before: int | None = None,
        after: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
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

    async def retrieve(self, job_id: str) -> dict[str, Any]:
        return await self._http.get(f"/api/rl/jobs/{job_id}")

    async def cancel(self, job_id: str) -> dict[str, Any]:
        return await self._http.post_json(f"/api/rl/jobs/{job_id}/cancel", json={})

    async def list_events(
        self, job_id: str, *, since_seq: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        params = {"since_seq": since_seq, "limit": limit}
        return await self._http.get(f"/api/rl/jobs/{job_id}/events", params=params)

    async def metrics(
        self, job_id: str, *, after_step: int = -1, limit: int = 200
    ) -> dict[str, Any]:
        params = {"after_step": after_step, "limit": limit}
        return await self._http.get(f"/api/rl/jobs/{job_id}/metrics", params=params)


class ModelsApi:
    def __init__(self, http: AsyncHttpClient) -> None:
        self._http = http

    async def list(
        self,
        *,
        source: str | None = None,
        base_model: str | None = None,
        status: str | None = None,
        after: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if source is not None:
            params["source"] = source
        if base_model is not None:
            params["base_model"] = base_model
        if status is not None:
            params["status"] = status
        if after is not None:
            params["after"] = after
        return await self._http.get("/api/models", params=params)

    async def retrieve(self, model_id: str) -> dict[str, Any]:
        return await self._http.get(f"/api/models/{model_id}")

    async def delete(self, model_id: str) -> Any:
        return await self._http.delete(f"/api/models/{model_id}")

    async def list_jobs(
        self, model_id: str, *, after: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if after is not None:
            params["after"] = after
        return await self._http.get(f"/api/models/{model_id}/jobs", params=params)


class JobsClient:
    """High-level client aggregating job APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        http: AsyncHttpClient | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._http = http or AsyncHttpClient(base_url, api_key, timeout=timeout)
        self.files = FilesApi(self._http)
        self.sft = SftJobsApi(self._http)
        self.rl = RlJobsApi(self._http)
        self.models = ModelsApi(self._http)

    async def __aenter__(self) -> JobsClient:
        await self._http.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self._http.__aexit__(exc_type, exc, tb)
