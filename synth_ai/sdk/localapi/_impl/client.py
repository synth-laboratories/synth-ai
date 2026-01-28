"""Internal task app client.

Public API: Use `synth_ai.sdk.localapi.client` instead.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from .contracts import RolloutRequest, RolloutResponse, TaskInfo
from .json import to_jsonable

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for sdk.localapi.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None:
        raise RuntimeError("synth_ai_py is required for LocalAPI. Install rust bindings.")
    return synth_ai_py


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
        self._rust_client: Any = None
        self.env = _TaskAppEnvironmentClient(self)
        rust = _require_rust()
        self._rust_client = rust.TaskAppClient(self.base_url, self.api_key, int(timeout))

    async def __aenter__(self) -> TaskAppClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        return

    async def health(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._rust_client.health)

    async def info(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._rust_client.info)

    async def task_info(self, seeds: list[int] | None = None) -> TaskInfo | list[TaskInfo]:
        result = await asyncio.to_thread(self._rust_client.task_info, seeds)
        if isinstance(result, list):
            return [TaskInfo.model_validate(item) for item in result]
        return TaskInfo.model_validate(result)

    async def rollout(self, request: RolloutRequest) -> RolloutResponse:
        payload = _prepare_payload(request)
        result = await asyncio.to_thread(self._rust_client.rollout, payload)
        return RolloutResponse.model_validate(result)

    async def done(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._rust_client.done)

    async def get(self, path: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._rust_client.get, path)

    async def post(self, path: str, body: Any) -> dict[str, Any]:
        payload = _prepare_payload(body)
        return await asyncio.to_thread(self._rust_client.post, path, payload)

    async def wait_for_healthy(self, timeout_seconds: float, poll_interval_seconds: float) -> None:
        await asyncio.to_thread(
            self._rust_client.wait_for_healthy, timeout_seconds, poll_interval_seconds
        )


class LocalAPIClient(TaskAppClient):
    """Alias for TaskAppClient with LocalAPI naming."""


class _TaskAppEnvironmentClient:
    def __init__(self, client: TaskAppClient) -> None:
        self._rust_env = client._rust_client.env()

    async def initialize(self, env_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload_value = _prepare_payload(payload)
        return await asyncio.to_thread(self._rust_env.initialize, env_name, payload_value)

    async def step(self, env_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload_value = _prepare_payload(payload)
        return await asyncio.to_thread(self._rust_env.step, env_name, payload_value)

    async def terminate(
        self, env_name: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload_value = _prepare_payload(payload or {})
        return await asyncio.to_thread(self._rust_env.terminate, env_name, payload_value)

    async def reset(self, env_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload_value = _prepare_payload(payload)
        return await asyncio.to_thread(self._rust_env.reset, env_name, payload_value)
