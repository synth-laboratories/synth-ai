from __future__ import annotations

import asyncio
from typing import Any

from synth_ai.core.rust_core.urls import ensure_api_base, normalize_base_url

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for rust_core.http.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None:
        raise RuntimeError("synth_ai_py is required for RustCoreHttpClient.")
    return synth_ai_py


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
        self._use_api_base = use_api_base
        self._shared = shared
        self._rust_client: Any | None = None

    async def __aenter__(self) -> RustCoreHttpClient:
        rust = _require_rust()
        if self._rust_client is None:
            base = ensure_api_base(self._base_url) if self._use_api_base else self._base_url
            self._rust_client = rust.HttpClient(base, self._api_key or "", int(self._timeout))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return

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
    ) -> Any:
        raise RuntimeError("request_raw is not supported in rust-only mode.")

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        if self._rust_client is None:
            await self.__aenter__()
        return await asyncio.to_thread(self._rust_client.get_json, path, params)

    async def post_json(self, path: str, *, json: dict[str, Any]) -> Any:
        if self._rust_client is None:
            await self.__aenter__()
        return await asyncio.to_thread(self._rust_client.post_json, path, json)

    async def post_multipart(
        self,
        path: str,
        *,
        data: dict[str, Any],
        files: dict[str, tuple[str, bytes, str | None]],
    ) -> Any:
        if self._rust_client is None:
            await self.__aenter__()
        return await asyncio.to_thread(self._rust_client.post_multipart, path, data, files)

    async def delete(self, path: str) -> Any:
        if self._rust_client is None:
            await self.__aenter__()
        await asyncio.to_thread(self._rust_client.delete, path)
        return None


def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    raise RuntimeError("http_request is not supported in rust-only mode.")


def get_shared_http_client() -> None:
    raise RuntimeError("get_shared_http_client is not supported in rust-only mode.")


async def sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


__all__ = ["RustCoreHttpClient", "http_request", "get_shared_http_client", "sleep"]
