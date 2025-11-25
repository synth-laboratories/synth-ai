"""HTTP client utilities for Synth AI SDK.

This module provides async HTTP client functionality used by SDK modules
for communicating with the Synth backend.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp

from .errors import HTTPError


class AsyncHttpClient:
    """Async HTTP client for Synth API calls.

    Usage:
        async with AsyncHttpClient(base_url, api_key) as client:
            result = await client.get("/api/jobs/123")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            base_url: Base URL for the API (without trailing /api)
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> AsyncHttpClient:
        if self._session is None:
            headers = {"authorization": f"Bearer {self._api_key}"}
            # Optional dev overrides for user/org context
            user_id = os.getenv("SYNTH_USER_ID") or os.getenv("X_USER_ID")
            if user_id:
                headers["X-User-ID"] = user_id
            org_id = os.getenv("SYNTH_ORG_ID") or os.getenv("X_ORG_ID")
            if org_id:
                headers["X-Org-ID"] = org_id
            self._session = aiohttp.ClientSession(
                headers=headers, timeout=self._timeout
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _abs(self, path: str) -> str:
        """Convert relative path to absolute URL."""
        if path.startswith(("http://", "https://")):
            return path
        # Handle /api prefix
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
        """Make a GET request."""
        url = self._abs(path)
        assert self._session is not None, "Must use as async context manager"
        async with self._session.get(url, params=params, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def post_json(
        self,
        path: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make a POST request with JSON body."""
        url = self._abs(path)
        assert self._session is not None, "Must use as async context manager"
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
        """Make a POST request with multipart form data."""
        url = self._abs(path)
        assert self._session is not None, "Must use as async context manager"
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

    async def delete(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Make a DELETE request."""
        url = self._abs(path)
        assert self._session is not None, "Must use as async context manager"
        async with self._session.delete(url, headers=headers) as resp:
            return await self._handle_response(resp, url)

    async def _handle_response(
        self, resp: aiohttp.ClientResponse, url: str
    ) -> Any:
        """Handle HTTP response, raising HTTPError on failure."""
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

        # Error response
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


async def sleep(seconds: float) -> None:
    """Async sleep helper."""
    await asyncio.sleep(seconds)


def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    """Synchronous HTTP request using stdlib.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL to request
        headers: Optional headers dict
        body: Optional JSON body dict

    Returns:
        Tuple of (status_code, response_data)
    """
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
    except urllib.error.HTTPError as exc:
        txt = exc.read().decode("utf-8", errors="ignore")
        try:
            return int(exc.code or 0), _json.loads(txt)
        except Exception:
            return int(exc.code or 0), txt
    except Exception as exc:
        return 0, str(exc)


__all__ = [
    "AsyncHttpClient",
    "HTTPError",
    "http_request",
    "sleep",
]

