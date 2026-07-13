"""Shared HTTP transport for infra SDK clients."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base

ModelT = TypeVar("ModelT", bound=BaseModel)


def resolve_api_key(api_key: str | None) -> str:
    if api_key and api_key.strip():
        return api_key.strip()
    resolved = (get_api_key(required=False) or "").strip()
    if not resolved:
        raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
    return resolved


def resolve_backend_base(base_url: str | None) -> str:
    if base_url and base_url.strip():
        return normalize_backend_base(base_url)
    return normalize_backend_base(BACKEND_URL_BASE)


class SynthBaseClient:
    """One httpx pool, shared request helpers, and typed ``cast_to`` for infra SDKs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._backend_base = resolve_backend_base(backend_base or base_url)
        self._timeout_seconds = timeout_seconds
        self._http_client: httpx.Client | None = None

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def backend_base(self) -> str:
        return self._backend_base

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    def _client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(
                base_url=self._backend_base,
                timeout=self._timeout_seconds,
            )
        return self._http_client

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        response = self._client().request(
            method,
            path if path.startswith("/") else f"/{path}",
            headers=self._headers(),
            json=json_body,
            params=params,
            timeout=timeout_seconds if timeout_seconds is not None else self._timeout_seconds,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    def _stream(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        with self._client().stream(
            "GET",
            path if path.startswith("/") else f"/{path}",
            headers=self._headers(),
            params=params,
            timeout=timeout_seconds if timeout_seconds is not None else self._timeout_seconds,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if payload:
                    yield json.loads(payload)

    def cast_to(self, model: type[ModelT], payload: Any) -> ModelT:
        return model.model_validate(payload)

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> SynthBaseClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


__all__ = [
    "SynthBaseClient",
    "join_url",
    "resolve_api_key",
    "resolve_backend_base",
]
