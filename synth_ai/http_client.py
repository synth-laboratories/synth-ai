from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp


@dataclass
class HTTPError(Exception):
    status: int
    url: str
    message: str
    body_snippet: str | None = None
    detail: Any | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = f"HTTP {self.status} for {self.url}: {self.message}"
        if self.body_snippet:
            base += f" | body[0:200]={self.body_snippet[:200]}"
        return base


class AsyncHttpClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncHttpClient":
        if self._session is None:
            headers = {"authorization": f"Bearer {self._api_key}"}
            self._session = aiohttp.ClientSession(headers=headers, timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _abs(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        # If base_url already ends with /api and path starts with /api, remove duplicate
        if self._base_url.endswith("/api") and path.startswith("/api"):
            path = path[4:]  # Remove leading /api
        return f"{self._base_url}/{path.lstrip('/')}"

    async def get(self, path: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        url = self._abs(path)
        assert self._session is not None, "AsyncHttpClient must be used as an async context manager"
        async with self._session.get(url, params=params, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def post_json(self, path: str, *, json: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
        url = self._abs(path)
        assert self._session is not None, "AsyncHttpClient must be used as an async context manager"
        async with self._session.post(url, json=json, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def post_multipart(self, path: str, *, data: Dict[str, Any], files: Dict[str, tuple[str, bytes, str | None]], headers: Optional[Dict[str, str]] = None) -> Any:
        url = self._abs(path)
        assert self._session is not None, "AsyncHttpClient must be used as an async context manager"
        form = aiohttp.FormData()
        for k, v in data.items():
            form.add_field(k, str(v))
        for field, (filename, content, content_type) in files.items():
            form.add_field(field, content, filename=filename, content_type=content_type or "application/octet-stream")
        async with self._session.post(url, data=form, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def delete(self, path: str, *, headers: Optional[Dict[str, str]] = None) -> Any:
        url = self._abs(path)
        assert self._session is not None, "AsyncHttpClient must be used as an async context manager"
        async with self._session.delete(url, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def _handle_response(self, resp: aiohttp.ClientResponse, url: str) -> Any:
        text = await resp.text()
        body_snippet = text[:200] if text else None
        if 200 <= resp.status < 300:
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                try:
                    return await resp.json()
                except Exception:
                    # Fallback to text
                    return text
            return text
        # error
        detail: Any | None = None
        try:
            detail = await resp.json()
        except Exception:
            detail = None
        raise HTTPError(status=resp.status, url=url, message="request_failed", body_snippet=body_snippet, detail=detail)


async def sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


