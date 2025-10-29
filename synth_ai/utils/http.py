from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import aiohttp

__all__ = ["HTTPError", "AsyncHttpClient", "http_request", "sleep"]


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
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> AsyncHttpClient:
        if self._session is None:
            headers = {
                "authorization": f"Bearer {self._api_key}",
                "accept": "application/json",
            }
            user_id = os.getenv("SYNTH_USER_ID") or os.getenv("X_USER_ID") or os.getenv("USER_ID")
            if user_id:
                headers["X-User-ID"] = user_id
            org_id = os.getenv("SYNTH_ORG_ID") or os.getenv("X_ORG_ID") or os.getenv("ORG_ID")
            if org_id:
                headers["X-Org-ID"] = org_id
            self._session = aiohttp.ClientSession(headers=headers, timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _abs(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if self._base_url.endswith("/api") and path.startswith("/api"):
            path = path[4:]
        return f"{self._base_url}/{path.lstrip('/')}"

    async def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = self._abs(path)
        assert self._session is not None, "AsyncHttpClient must be used as an async context manager"
        async with self._session.get(url, params=params, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def post_json(
        self, path: str, *, json: dict[str, Any], headers: dict[str, str] | None = None
    ) -> Any:
        url = self._abs(path)
        assert self._session is not None, "AsyncHttpClient must be used as an async context manager"
        async with self._session.post(url, json=json, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def post_multipart(
        self,
        path: str,
        *,
        data: dict[str, Any],
        files: dict[str, tuple[str, bytes, str | None]],
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = self._abs(path)
        assert self._session is not None, "AsyncHttpClient must be used as an async context manager"
        form = aiohttp.FormData()
        for k, v in data.items():
            form.add_field(k, str(v))
        for field, (filename, content, content_type) in files.items():
            form.add_field(
                field,
                content,
                filename=filename,
                content_type=content_type or "application/octet-stream",
            )
        async with self._session.post(url, data=form, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def delete(self, path: str, *, headers: dict[str, str] | None = None) -> Any:
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
                    return text
            return text
        detail: Any | None = None
        try:
            detail = await resp.json()
        except Exception:
            detail = None
        raise HTTPError(
            status=resp.status,
            url=url,
            message="request_failed",
            body_snippet=body_snippet,
            detail=detail,
        )


def http_request(
    method: str, url: str, headers: dict[str, str] | None = None, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any] | str]:
    import json as _json
    import ssl
    import urllib.error
    import urllib.request

    data = None
    if body is not None:
        data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=data)
    try:
        ctx = ssl._create_unverified_context()
        if os.getenv("SYNTH_SSL_VERIFY", "0") == "1":
            ctx = None
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            code = getattr(resp, "status", 200)
            txt = resp.read().decode("utf-8", errors="ignore")
            try:
                return int(code), _json.loads(txt)
            except Exception:
                return int(code), txt
    except urllib.error.HTTPError as exc:  # Capture 4xx/5xx bodies
        txt = exc.read().decode("utf-8", errors="ignore")
        try:
            return int(exc.code or 0), _json.loads(txt)
        except Exception:
            return int(exc.code or 0), txt
    except Exception as exc:
        return 0, str(exc)


async def sleep(seconds: float) -> None:
    """Small async sleep helper preserved for backwards compatibility."""

    await asyncio.sleep(max(float(seconds or 0.0), 0.0))
