from __future__ import annotations

import asyncio
import os
from typing import Any

from synth_ai.core.errors import HTTPError
from synth_ai.core.rust_core.urls import ensure_api_base, normalize_base_url

try:
    import synth_ai_py as _synth_ai_py
except Exception:  # pragma: no cover - optional rust bindings
    _synth_ai_py = None


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
        self._rust_client = None

    async def __aenter__(self) -> RustCoreHttpClient:
        if _synth_ai_py is None:
            raise RuntimeError("synth_ai_py is required for RustCoreHttpClient")
        if self._rust_client is None:
            self._rust_client = _synth_ai_py.HttpClient(
                ensure_api_base(self._base_url) if self._use_api_base else self._base_url,
                self._api_key or "",
                int(self._timeout or 30.0),
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self._shared:
            self._rust_client = None

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
    ):
        raise RuntimeError("request_raw is not supported in rust-backed client")

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        if self._rust_client is None:
            raise RuntimeError("synth_ai_py is required for RustCoreHttpClient")
        try:
            return await asyncio.to_thread(self._rust_client.get_json, self._abs(path), params)
        except Exception as exc:
            raise _wrap_rust_error(exc, self._abs(path)) from exc

    async def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return await self.get(path, params=params)

    async def post_json(self, path: str, *, json: dict[str, Any]) -> Any:
        if self._rust_client is None:
            raise RuntimeError("synth_ai_py is required for RustCoreHttpClient")
        try:
            return await asyncio.to_thread(self._rust_client.post_json, self._abs(path), json)
        except Exception as exc:
            raise _wrap_rust_error(exc, self._abs(path)) from exc

    async def post_multipart(
        self,
        path: str,
        *,
        data: dict[str, Any],
        files: dict[str, tuple[str, bytes, str | None]],
    ) -> Any:
        if self._rust_client is None:
            raise RuntimeError("synth_ai_py is required for RustCoreHttpClient")
        try:
            return await asyncio.to_thread(
                self._rust_client.post_multipart, self._abs(path), data, files
            )
        except Exception as exc:
            raise _wrap_rust_error(exc, self._abs(path)) from exc

    async def delete(self, path: str) -> Any:
        if self._rust_client is None:
            raise RuntimeError("synth_ai_py is required for RustCoreHttpClient")
        try:
            await asyncio.to_thread(self._rust_client.delete, self._abs(path))
            return None
        except Exception as exc:
            raise _wrap_rust_error(exc, self._abs(path)) from exc


def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    """Make a simple HTTP request to any URL using Rust HTTP client."""
    if _synth_ai_py is None:
        raise RuntimeError("synth_ai_py is required for HTTP requests")
    # Extract base URL (scheme + host + port) for client initialization
    from urllib.parse import urlparse

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    # Extract API key from headers if provided
    api_key = ""
    if headers:
        api_key = (
            headers.get("X-API-Key")
            or headers.get("x-api-key")
            or headers.get("Authorization", "").removeprefix("Bearer ").strip()
            or ""
        )
    client = _synth_ai_py.HttpClient(base, api_key, 30)
    try:
        if method.upper() == "GET":
            return 200, client.get_json(url, None)
        if method.upper() == "DELETE":
            client.delete(url)
            return 200, ""
        return 200, client.post_json(url, body or {})
    except Exception as exc:  # pragma: no cover - defensive
        return 0, str(exc)


async def sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _wrap_rust_error(exc: Exception, url: str) -> HTTPError:
    message = str(exc)
    status = 0
    if message.startswith("HTTP "):
        try:
            status = int(message.split(" ", 2)[1])
        except Exception:
            status = 0
    return HTTPError(
        status=status,
        url=url,
        message="request_failed",
        body_snippet=message[:200],
        detail=message,
    )


__all__ = ["RustCoreHttpClient", "http_request", "sleep"]
