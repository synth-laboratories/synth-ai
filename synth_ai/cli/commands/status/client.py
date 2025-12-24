"""HTTP client for status commands."""

from __future__ import annotations

from typing import Any

import httpx

from .config import BackendConfig
from .errors import StatusAPIError
from .utils import build_headers


class StatusAPIClient:
    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "StatusAPIClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=build_headers(self._config.api_key),
                timeout=self._config.timeout,
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._client is not None
        resp = await self._client.get(path, params=params)
        if resp.status_code >= 400:
            detail = resp.json().get("detail", "")
            raise StatusAPIError(detail or "Request failed", status_code=resp.status_code)
        return resp.json()

    async def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._client is not None
        resp = await self._client.post(path, json=payload or {})
        if resp.status_code >= 400:
            detail = resp.json().get("detail", "")
            raise StatusAPIError(detail or "Request failed", status_code=resp.status_code)
        return resp.json()

    async def list_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        created_after: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if job_type is not None:
            params["type"] = job_type
        if created_after is not None:
            params["created_after"] = created_after
        if limit is not None:
            params["limit"] = limit
        payload = await self._get("/learning/jobs", params=params or None)
        return payload.get("jobs", [])

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._get(f"/learning/jobs/{job_id}")

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self._post(f"/learning/jobs/{job_id}/cancel")

    async def list_models(
        self,
        *,
        limit: int | None = None,
        model_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if model_type:
            payload = await self._get(f"/learning/models/{model_type}")
            return payload.get("models", [])
        params = {"limit": limit} if limit is not None else None
        payload = await self._get("/learning/models", params=params)
        return payload.get("models", [])

    async def list_job_runs(self, job_id: str) -> list[dict[str, Any]]:
        payload = await self._get(f"/jobs/{job_id}/runs")
        return payload.get("runs", [])

