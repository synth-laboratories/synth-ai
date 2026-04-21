"""Anthropic-view managed-agents client (via backend BFF proxy)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base


class ManagedAgentsAnthropicClient:
    """Sync client for backend Anthropic-view managed-agents proxy endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        timeout: float = 30.0,
        anthropic_version: str | None = None,
    ) -> None:
        self._api_key = (api_key or get_api_key(required=False) or "").strip()
        if not self._api_key:
            raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
        self._backend_base = normalize_backend_base(backend_base or BACKEND_URL_BASE)
        self._timeout = timeout
        self._anthropic_version = (
            anthropic_version or os.getenv("ANTHROPIC_VERSION") or "2023-06-01"
        ).strip()
        self._prefix = "/api/managed-agents/anthropic/v1"

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "anthropic-version": self._anthropic_version,
            "anthropic-beta": "managed-agents-2026-04-01",
        }
        if extra:
            headers.update(extra)
        return headers

    def _path(self, path: str) -> str:
        cleaned = str(path or "").strip()
        if not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"
        return f"{self._prefix}{cleaned}"

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = httpx.request(
            method=method.upper(),
            url=join_url(self._backend_base, self._path(path)),
            headers=self._headers(headers),
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            return response.json()
        return {"content": response.text}

    def stream(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        with httpx.stream(
            "GET",
            join_url(self._backend_base, self._path(path)),
            headers=self._headers(headers),
            params=params,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if not payload:
                    continue
                if payload == "[DONE]":
                    break
                yield json.loads(payload)

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def list_environments(self, **params: Any) -> dict[str, Any]:
        return self.request("GET", "/environments", params=params or None)

    def create_environment(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/environments", json_body=request)

    def get_environment(self, environment_id: str) -> dict[str, Any]:
        return self.request("GET", f"/environments/{environment_id}")

    def update_environment(self, environment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/environments/{environment_id}", json_body=request)

    def archive_environment(self, environment_id: str) -> dict[str, Any]:
        return self.request("POST", f"/environments/{environment_id}/archive", json_body={})

    def list_agents(self, **params: Any) -> dict[str, Any]:
        return self.request("GET", "/agents", params=params or None)

    def create_agent(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/agents", json_body=request)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self.request("GET", f"/agents/{agent_id}")

    def update_agent(self, agent_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/agents/{agent_id}", json_body=request)

    def archive_agent(self, agent_id: str) -> dict[str, Any]:
        return self.request("POST", f"/agents/{agent_id}/archive", json_body={})

    def list_sessions(self, **params: Any) -> dict[str, Any]:
        return self.request("GET", "/sessions", params=params or None)

    def create_session(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/sessions", json_body=request)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}")

    def update_session(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}", json_body=request)

    def archive_session(self, session_id: str) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}/archive", json_body={})

    def post_session_events(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}/events", json_body=request)

    def list_session_events(self, session_id: str, **params: Any) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}/events", params=params or None)

    def stream_session_events(
        self,
        session_id: str,
        **params: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.stream(f"/sessions/{session_id}/events/stream", params=params or None)

    def stream_session(self, session_id: str, **params: Any) -> Iterator[dict[str, Any]]:
        return self.stream(f"/sessions/{session_id}/stream", params=params or None)


class AsyncManagedAgentsAnthropicClient:
    """Async adapter around ``ManagedAgentsAnthropicClient``."""

    def __init__(self, sync_client: ManagedAgentsAnthropicClient) -> None:
        self._sync = sync_client

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self._sync.request,
            method,
            path,
            json_body=json_body,
            params=params,
            headers=headers,
        )

    async def health(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.health)

    async def list_environments(self, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_environments, **params)

    async def create_environment(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_environment, request)

    async def get_environment(self, environment_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_environment, environment_id)

    async def update_environment(
        self, environment_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update_environment, environment_id, request)

    async def archive_environment(self, environment_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.archive_environment, environment_id)

    async def list_agents(self, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_agents, **params)

    async def create_agent(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_agent, request)

    async def get_agent(self, agent_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_agent, agent_id)

    async def update_agent(self, agent_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update_agent, agent_id, request)

    async def archive_agent(self, agent_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.archive_agent, agent_id)

    async def list_sessions(self, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_sessions, **params)

    async def create_session(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_session, request)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_session, session_id)

    async def update_session(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update_session, session_id, request)

    async def archive_session(self, session_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.archive_session, session_id)

    async def post_session_events(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.post_session_events, session_id, request)

    async def list_session_events(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_session_events, session_id, **params)

    async def stream_session_events(
        self,
        session_id: str,
        **params: Any,
    ) -> list[dict[str, Any]]:
        def _collect() -> list[dict[str, Any]]:
            return list(self._sync.stream_session_events(session_id, **params))

        return await asyncio.to_thread(_collect)

    async def stream_session(self, session_id: str, **params: Any) -> list[dict[str, Any]]:
        def _collect() -> list[dict[str, Any]]:
            return list(self._sync.stream_session(session_id, **params))

        return await asyncio.to_thread(_collect)


__all__ = [
    "AsyncManagedAgentsAnthropicClient",
    "ManagedAgentsAnthropicClient",
]
