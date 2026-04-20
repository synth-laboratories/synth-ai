"""horizons-private facade over container pool APIs."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from synth_ai.sdk.pools import ContainerPoolsClient, PoolTarget


class HorizonsPrivateClient:
    """Thin ergonomic facade for horizons-private workloads via container pools."""

    def __init__(self, pools: ContainerPoolsClient) -> None:
        self._pools = pools

    @staticmethod
    def _ensure_target(payload: dict[str, Any]) -> dict[str, Any]:
        out = dict(payload)
        target = str(out.get("target") or "").strip().lower()
        expected = PoolTarget.HORIZONS_PRIVATE.value
        if not target:
            out["target"] = expected
            return out
        if target != expected:
            raise ValueError(
                f"horizons_private runtime requires target={expected!r}, got {target!r}"
            )
        return out

    def create_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._pools.create_pool(self._ensure_target(request))

    def create_session(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._pools.create_rollout(pool_id, request)

    def run(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.create_session(pool_id, request)

    def get(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._pools.get_rollout(pool_id, rollout_id)

    def list(
        self,
        pool_id: str,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._pools.list_rollouts(pool_id, state=state, limit=limit, cursor=cursor)

    def cancel(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._pools.cancel_rollout(pool_id, rollout_id)

    def events(
        self, pool_id: str, rollout_id: str, *, cursor: str | None = None
    ) -> Iterator[dict[str, Any]]:
        return self._pools.stream_rollout_events(pool_id, rollout_id, cursor=cursor)

    def traces(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._pools.get_rollout_artifacts(pool_id, rollout_id)

    def artifacts(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._pools.get_rollout_artifacts(pool_id, rollout_id)

    def usage(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._pools.get_rollout_usage(pool_id, rollout_id)

    def summary(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return self._pools.get_rollout_summary(pool_id, rollout_id)


class AsyncHorizonsPrivateClient:
    """Async adapter around ``HorizonsPrivateClient``."""

    def __init__(self, sync_client: HorizonsPrivateClient) -> None:
        self._sync = sync_client

    async def create_runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_runtime, request)

    async def create_session(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_session, pool_id, request)

    async def run(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.run, pool_id, request)

    async def get(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get, pool_id, rollout_id)

    async def list(
        self,
        pool_id: str,
        *,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.list, pool_id, state=state, limit=limit, cursor=cursor
        )

    async def cancel(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.cancel, pool_id, rollout_id)

    async def events(
        self, pool_id: str, rollout_id: str, *, cursor: str | None = None
    ) -> list[dict[str, Any]]:
        def _collect() -> list[dict[str, Any]]:
            return list(self._sync.events(pool_id, rollout_id, cursor=cursor))

        return await asyncio.to_thread(_collect)

    async def traces(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.traces, pool_id, rollout_id)

    async def artifacts(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.artifacts, pool_id, rollout_id)

    async def usage(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.usage, pool_id, rollout_id)

    async def summary(self, pool_id: str, rollout_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.summary, pool_id, rollout_id)


__all__ = [
    "AsyncHorizonsPrivateClient",
    "HorizonsPrivateClient",
]
