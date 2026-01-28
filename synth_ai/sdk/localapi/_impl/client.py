"""Internal task app client.

Public API: Use `synth_ai.sdk.localapi.client` instead.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel

try:
    import synth_ai_py as _synth_ai_py
except Exception:  # pragma: no cover - optional rust bindings
    _synth_ai_py = None

from synth_ai.core.rust_core.http import RustCoreHttpClient

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
        self._client: RustCoreHttpClient | None = None
        self._rust_client = None
        self.env = _TaskAppEnvironmentClient(self)

    async def __aenter__(self) -> TaskAppClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _ensure_client(self) -> RustCoreHttpClient:
        if _synth_ai_py is not None and self._rust_client is None:
            self._rust_client = _synth_ai_py.TaskAppClientPy(
                self.base_url,
                self.api_key,
                int(self.timeout),
            )
        if self._client is None and self._rust_client is None:
            self._client = RustCoreHttpClient(
                base_url=self.base_url,
                api_key=self.api_key or "",
                timeout=self.timeout,
                shared=True,
                use_api_base=False,
            )
            await self._client.__aenter__()
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
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def _call_rust(self, method: str, path: str, payload: Any = None) -> Any:
        if self._rust_client is None:
            raise RuntimeError("Rust TaskAppClient is not available")
        if method == "GET":
            return await asyncio.to_thread(self._rust_client.get, path)
        if method == "POST":
            return await asyncio.to_thread(self._rust_client.post, path, payload or {})
        raise ValueError(f"Unsupported method: {method}")

    async def health(self) -> dict[str, Any]:
        if self._rust_client is not None:
            return await asyncio.to_thread(self._rust_client.health)
        client = await self._ensure_client()
        return await client.get("/health")

    async def is_healthy(self) -> bool:
        if self._rust_client is not None:
            return await asyncio.to_thread(self._rust_client.is_healthy)
        try:
            data = await self.health()
        except Exception:
            return False
        return bool(data.get("healthy"))

    async def info(self) -> dict[str, Any]:
        if self._rust_client is not None:
            return await asyncio.to_thread(self._rust_client.info)
        client = await self._ensure_client()
        return await client.get("/info")

    async def task_info(self, seeds: list[int] | None = None) -> TaskInfo | list[TaskInfo]:
        if self._rust_client is not None:
            data = await asyncio.to_thread(self._rust_client.task_info, seeds)
        else:
            params: list[tuple[str, Any]] | None = None
            if seeds:
                params = [("seed", seed) for seed in seeds]
            client = await self._ensure_client()
            data = await client.get("/task_info", params=params)
        if isinstance(data, list):
            return [TaskInfo.model_validate(item) for item in data]
        return TaskInfo.model_validate(data)

    async def rollout(self, request: RolloutRequest) -> RolloutResponse:
        payload = _prepare_payload(request)
        if self._rust_client is not None:
            data = await asyncio.to_thread(self._rust_client.rollout, payload)
        else:
            client = await self._ensure_client()
            data = await client.post_json("/rollout", json=payload)
        return RolloutResponse.model_validate(data)

    async def taskset_info(self) -> dict[str, Any]:
        if self._rust_client is not None:
            return await asyncio.to_thread(self._rust_client.taskset_info)
        client = await self._ensure_client()
        return await client.get("/task_info")

    async def done(self) -> dict[str, Any]:
        if self._rust_client is not None:
            return await asyncio.to_thread(self._rust_client.done)
        client = await self._ensure_client()
        return await client.post_json("/done", json={})

    async def wait_for_healthy(
        self,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 2.0,
    ) -> None:
        if self._rust_client is not None:
            await asyncio.to_thread(
                self._rust_client.wait_for_healthy,
                timeout_seconds,
                poll_interval_seconds,
            )
            return
        loop = asyncio.get_running_loop()
        start = loop.time()
        while True:
            if (loop.time() - start) >= timeout_seconds:
                raise TimeoutError(
                    f"Task app at {self.base_url} did not become healthy within {timeout_seconds} seconds"
                )
            if await self.is_healthy():
                return
            await asyncio.sleep(poll_interval_seconds)


class LocalAPIClient(TaskAppClient):
    """Alias for TaskAppClient with LocalAPI naming."""


class _TaskAppEnvironmentClient:
    def __init__(self, client: TaskAppClient) -> None:
        self._client = client

    async def initialize(self, env_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client._rust_client is not None:
            return await asyncio.to_thread(
                self._client._rust_client.env_initialize, env_name, payload
            )
        client = await self._client._ensure_client()
        return await client.post_json(f"/env/{env_name}/initialize", json=payload)

    async def step(self, env_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client._rust_client is not None:
            return await asyncio.to_thread(self._client._rust_client.env_step, env_name, payload)
        client = await self._client._ensure_client()
        return await client.post_json(f"/env/{env_name}/step", json=payload)

    async def terminate(
        self, env_name: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = payload or {}
        if self._client._rust_client is not None:
            return await asyncio.to_thread(
                self._client._rust_client.env_terminate, env_name, payload
            )
        client = await self._client._ensure_client()
        return await client.post_json(f"/env/{env_name}/terminate", json=payload)

    async def reset(self, env_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        if self._client._rust_client is not None:
            return await asyncio.to_thread(self._client._rust_client.env_reset, env_name, payload)
        client = await self._client._ensure_client()
        return await client.post_json(f"/env/{env_name}/reset", json=payload)
