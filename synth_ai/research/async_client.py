"""Async Research API (parity stub — sync hero is canonical today)."""

from __future__ import annotations

import asyncio
from typing import Any

from synth_ai.research.client import ResearchClient


class _AsyncNamespaceProxy:
    def __init__(self, sync_obj: Any) -> None:
        self._sync_obj = sync_obj
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        attr = getattr(self._sync_obj, name)
        if callable(attr):

            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(attr, *args, **kwargs)

            self._cache[name] = _wrapped
            return _wrapped
        proxy = _AsyncNamespaceProxy(attr)
        self._cache[name] = proxy
        return proxy


class AsyncResearchClient:
    """Async adapter over ``ResearchClient`` (thread-offloaded sync hero)."""

    def __init__(self, client: ResearchClient) -> None:
        """Wrap a sync :class:`ResearchClient` for async call sites."""
        self._client = client

    @property
    def session(self) -> _AsyncNamespaceProxy:
        """Async proxy for the backing monolith session (advanced)."""
        return _AsyncNamespaceProxy(self._client.session)

    @property
    def factories(self) -> _AsyncNamespaceProxy:
        """Async proxy for ``client.research.factories``."""
        return _AsyncNamespaceProxy(self._client.factories)

    @property
    def projects(self) -> _AsyncNamespaceProxy:
        """Async proxy for ``client.research.projects``."""
        return _AsyncNamespaceProxy(self._client.projects)

    @property
    def runs(self) -> _AsyncNamespaceProxy:
        """Async proxy for ``client.research.runs``."""
        return _AsyncNamespaceProxy(self._client.runs)

    @property
    def limits(self) -> _AsyncNamespaceProxy:
        """Async proxy for ``client.research.limits``."""
        return _AsyncNamespaceProxy(self._client.limits)

    @property
    def secrets(self) -> _AsyncNamespaceProxy:
        """Async proxy for ``client.research.secrets``."""
        return _AsyncNamespaceProxy(self._client.secrets)

    async def close(self) -> None:
        """Close the underlying research session."""
        await asyncio.to_thread(self._client.close)


__all__ = ["AsyncResearchClient"]
