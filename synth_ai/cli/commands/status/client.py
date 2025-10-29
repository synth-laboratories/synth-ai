"""Async HTTP client for Synth status and listing endpoints."""

from __future__ import annotations

from typing import Any

import httpx

from .config import BackendConfig
from .errors import StatusAPIError


class StatusAPIClient:
    """Thin wrapper around httpx.AsyncClient with convenience methods."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        timeout = httpx.Timeout(config.timeout)
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=config.headers,
            timeout=timeout,
        )

    async def __aenter__(self) -> StatusAPIClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.__aexit__(*args)

    async def close(self) -> None:
        await self._client.aclose()

    # Jobs -----------------------------------------------------------------

    async def list_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        created_after: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if job_type:
            params["type"] = job_type
        if created_after:
            params["created_after"] = created_after
        if limit:
            params["limit"] = limit
        resp = await self._client.get("/learning/jobs", params=params)
        return self._json_list(resp, key="jobs")

    async def get_job(self, job_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/learning/jobs/{job_id}")
        return self._json(resp)

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/learning/jobs/{job_id}/status")
        return self._json(resp)

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        resp = await self._client.post(f"/learning/jobs/{job_id}/cancel")
        return self._json(resp)

    async def get_job_config(self, job_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/learning/jobs/{job_id}/config")
        return self._json(resp)

    async def get_job_metrics(self, job_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/learning/jobs/{job_id}/metrics")
        return self._json(resp)

    async def get_job_timeline(self, job_id: str) -> list[dict[str, Any]]:
        resp = await self._client.get(f"/learning/jobs/{job_id}/timeline")
        return self._json_list(resp, key="timeline")

    async def list_job_runs(self, job_id: str) -> list[dict[str, Any]]:
        resp = await self._client.get(f"/jobs/{job_id}/runs")
        return self._json_list(resp, key="runs")

    async def get_job_events(
        self,
        job_id: str,
        *,
        since: str | None = None,
        limit: int | None = None,
        after: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if since:
            params["since"] = since
        if limit:
            params["limit"] = limit
        if after:
            params["after"] = after
        if run_id:
            params["run"] = run_id
        resp = await self._client.get(f"/learning/jobs/{job_id}/events", params=params)
        return self._json_list(resp, key="events")

    # Files ----------------------------------------------------------------

    async def list_files(
        self,
        *,
        purpose: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if purpose:
            params["purpose"] = purpose
        if limit:
            params["limit"] = limit
        resp = await self._client.get("/files", params=params)
        data = self._json(resp)
        if isinstance(data, dict):
            for key in ("files", "data", "items"):
                if isinstance(data.get(key), list):
                    return list(data[key])
        if isinstance(data, list):
            return list(data)
        return []

    async def get_file(self, file_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/files/{file_id}")
        return self._json(resp)

    # Models ---------------------------------------------------------------

    async def list_models(
        self,
        *,
        limit: int | None = None,
        model_type: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        endpoint = "/learning/models/rl" if model_type == "rl" else "/learning/models"
        resp = await self._client.get(endpoint, params=params)
        return self._json_list(resp, key="models")

    async def get_model(self, model_id: str) -> dict[str, Any]:
        resp = await self._client.get(f"/learning/models/{model_id}")
        return self._json(resp)

    # Helpers --------------------------------------------------------------

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._extract_detail(exc.response)
            raise StatusAPIError(detail, exc.response.status_code if exc.response else None) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise StatusAPIError("Backend response was not valid JSON") from exc
        if isinstance(data, dict):
            return data
        return {"data": data}

    def _json_list(self, response: httpx.Response, *, key: str | None = None) -> list[dict[str, Any]]:
        payload = self._json(response)
        if key and isinstance(payload.get(key), list):
            return list(payload[key])
        if isinstance(payload.get("data"), list):
            return list(payload["data"])
        if isinstance(payload.get("results"), list):
            return list(payload["results"])
        if isinstance(payload, list):
            return list(payload)
        return []

    @staticmethod
    def _extract_detail(response: httpx.Response | None) -> str:
        if response is None:
            return "Backend request failed"
        try:
            data = response.json()
            if isinstance(data, dict):
                for key in ("detail", "message", "error"):
                    if data.get(key):
                        return str(data[key])
            return response.text
        except ValueError:
            return response.text
