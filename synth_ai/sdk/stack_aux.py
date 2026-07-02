"""Stack auxiliary inference client (Responses wire, passthrough to Baseten)."""

from __future__ import annotations

from typing import Any

import httpx

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base

STACK_AUX_PREFIX = "/api/v1/stack-aux/openai/v1"


class _StackAuxResponsesClient:
    def __init__(self, raw: StackAuxClient) -> None:
        self._raw = raw

    def create(
        self,
        *,
        model: str,
        input: Any,
        metadata: dict[str, Any] | None = None,
        actor_role: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "input": input, **extra}
        if metadata:
            body["metadata"] = metadata
        role = (actor_role or (metadata or {}).get("actor_role") or "").strip()
        headers = {"X-Stack-Actor-Role": role} if role else None
        return self._raw.request("POST", "/responses", json_body=body, headers=headers)


class StackAuxClient:
    """Sync client for Stack auxiliary inference on usesynth.ai API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = (api_key or get_api_key(required=False) or "").strip()
        if not self._api_key:
            raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
        self._backend_base = normalize_backend_base(backend_base or BACKEND_URL_BASE)
        self._timeout = timeout
        self.responses = _StackAuxResponsesClient(self)

    @staticmethod
    def _normalize_path(path: str) -> str:
        cleaned = str(path or "").strip()
        if not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"
        return cleaned

    @staticmethod
    def _parse_response(response: httpx.Response) -> Any:
        if not response.content:
            return {}
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            return response.json()
        return {"content": response.text}

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if extra:
            headers.update(extra)
        return headers

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
            url=join_url(
                self._backend_base,
                f"{STACK_AUX_PREFIX}{self._normalize_path(path)}",
            ),
            headers=self._headers(headers),
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return self._parse_response(response)


class AsyncStackAuxClient:
    """Async wrapper around StackAuxClient."""

    def __init__(self, sync_client: StackAuxClient) -> None:
        self._sync = sync_client
        self.responses = _AsyncStackAuxResponsesClient(sync_client)


class _AsyncStackAuxResponsesClient:
    def __init__(self, sync_client: StackAuxClient) -> None:
        self._sync = sync_client.responses

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self._sync.create, **kwargs)
