from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from synth_ai.core.errors import HTTPError
from synth_ai.core.rust_core.urls import ensure_api_base, normalize_base_url

_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None


def _build_shared_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=200,
            max_keepalive_connections=200,
            keepalive_expiry=30.0,
        ),
        timeout=httpx.Timeout(
            connect=30.0,
            read=300.0,
            write=30.0,
            pool=30.0,
        ),
        follow_redirects=False,
    )


def get_shared_http_client() -> httpx.AsyncClient:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        _SHARED_HTTP_CLIENT = _build_shared_client()
    return _SHARED_HTTP_CLIENT


class RustCoreHttpClient:
    """Rust-core-aligned HTTP client (Python compatibility layer)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        *,
        shared: bool = True,
        use_api_base: bool = True,
    ) -> None:
        self._base_url = normalize_base_url(base_url)
        self._api_key = api_key
        self._timeout = timeout
        self._shared = shared
        self._use_api_base = use_api_base
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> RustCoreHttpClient:
        if self._client is None:
            self._client = get_shared_http_client() if self._shared else _build_shared_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None and not self._shared:
            await self._client.aclose()
        if not self._shared:
            self._client = None

    def _abs(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        base = ensure_api_base(self._base_url) if self._use_api_base else self._base_url
        if self._use_api_base and base.endswith("/api") and path.startswith("/api"):
            path = path[4:]
        return f"{base.rstrip('/')}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["X-API-Key"] = self._api_key
        user_id = os.getenv("SYNTH_USER_ID") or os.getenv("X_USER_ID")
        if user_id:
            headers["X-User-ID"] = user_id
        org_id = os.getenv("SYNTH_ORG_ID") or os.getenv("X_ORG_ID")
        if org_id:
            headers["X-Org-ID"] = org_id
        return headers

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = get_shared_http_client() if self._shared else _build_shared_client()
        return self._client

    async def request_raw(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | list[tuple[str, Any]] | None = None,
        json_payload: Any = None,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, tuple[str, bytes, str | None]] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        client = self._ensure_client()
        url = self._abs(path)
        merged_headers = {**self._headers(), **(headers or {})}
        return await client.request(
            method,
            url,
            params=params,
            json=json_payload,
            headers=merged_headers,
            data=data,
            files=files,
            timeout=timeout or self._timeout,
        )

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        resp = await self.request_raw("GET", path, params=params)
        return await self._handle_response(resp)

    async def post_json(self, path: str, *, json: dict[str, Any]) -> Any:
        resp = await self.request_raw("POST", path, json_payload=json)
        return await self._handle_response(resp)

    async def post_multipart(
        self,
        path: str,
        *,
        data: dict[str, Any],
        files: dict[str, tuple[str, bytes, str | None]],
    ) -> Any:
        resp = await self.request_raw("POST", path, data=data, files=files)
        return await self._handle_response(resp)

    async def delete(self, path: str) -> Any:
        resp = await self.request_raw("DELETE", path)
        return await self._handle_response(resp)

    async def _handle_response(self, resp: httpx.Response) -> Any:
        text = resp.text
        body_snippet = text[:200] if text else None

        if 200 <= resp.status_code < 300:
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                try:
                    return resp.json()
                except Exception:
                    return text
            return text

        detail: Any | None = None
        try:
            detail = resp.json()
        except Exception:
            detail = None

        raise HTTPError(
            status=resp.status_code,
            url=str(resp.url),
            message="request_failed",
            body_snippet=body_snippet,
            detail=detail,
        )


def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    with httpx.Client(verify=False, timeout=30.0) as client:
        resp = client.request(method, url, headers=headers, json=body)
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.status_code, resp.json()
        return resp.status_code, resp.text


async def sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


__all__ = ["RustCoreHttpClient", "get_shared_http_client", "http_request", "sleep"]
