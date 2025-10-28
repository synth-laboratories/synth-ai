"""Async HTTP client for interacting with Task Apps."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from pydantic import BaseModel

from .contracts import RolloutRequest, RolloutResponse, TaskInfo
from .json import to_jsonable


def _prepare_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json", by_alias=True)
    return to_jsonable(payload)


class TaskAppClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 600.0,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(1, retries)
        self._client: httpx.AsyncClient | None = None
        self.env = _TaskAppEnvironmentClient(self)

    async def __aenter__(self) -> TaskAppClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        # Primary key
        primary = (self.api_key or "").strip()
        if primary:
            headers["X-API-Key"] = primary
            # Also set Authorization for clients that read bearer tokens
            headers.setdefault("Authorization", f"Bearer {primary}")
        # Include ALL available environment keys via CSV in X-API-Keys
        keys: list[str] = []
        if primary:
            keys.append(primary)
        aliases = (os.getenv("ENVIRONMENT_API_KEY_ALIASES") or "").strip()
        if aliases:
            for part in aliases.split(","):
                trimmed = part.strip()
                if trimmed and trimmed not in keys:
                    keys.append(trimmed)
        if keys:
            headers["X-API-Keys"] = ",".join(keys)
        return headers

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | list[tuple[str, Any]] | None = None,
        json_payload: Any = None,
    ) -> httpx.Response:
        client = await self._ensure_client()
        payload = _prepare_payload(json_payload)
        headers = self._headers()
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = await client.request(
                    method,
                    path,
                    headers=headers,
                    params=params,
                    json=payload,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if 500 <= exc.response.status_code < 600 and attempt + 1 < self.retries:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    last_exc = exc
                    continue
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt + 1 >= self.retries:
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
        if last_exc:  # pragma: no cover - defensive
            raise last_exc
        raise RuntimeError("Unreachable code in TaskAppClient._request")

    async def health(self) -> dict[str, Any]:
        response = await self._request("GET", "/health")
        return response.json()

    async def info(self) -> dict[str, Any]:
        response = await self._request("GET", "/info")
        return response.json()

    async def task_info(self, seeds: list[int] | None = None) -> TaskInfo | list[TaskInfo]:
        params: list[tuple[str, Any]] | None = None
        if seeds:
            params = [("seed", seed) for seed in seeds]
        response = await self._request("GET", "/task_info", params=params)
        data = response.json()
        if isinstance(data, list):
            return [TaskInfo.model_validate(item) for item in data]
        return TaskInfo.model_validate(data)

    async def rollout(self, request: RolloutRequest) -> RolloutResponse:
        response = await self._request("POST", "/rollout", json_payload=request)
        data = response.json()
        return RolloutResponse.model_validate(data)


class _TaskAppEnvironmentClient:
    def __init__(self, client: TaskAppClient) -> None:
        self._client = client

    async def initialize(self, env_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client._request(
            "POST", f"/env/{env_name}/initialize", json_payload=payload
        )
        return response.json()

    async def step(self, env_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client._request(
            "POST", f"/env/{env_name}/step", json_payload=payload
        )
        return response.json()

    async def terminate(
        self, env_name: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response = await self._client._request(
            "POST", f"/env/{env_name}/terminate", json_payload=payload or {}
        )
        return response.json()
