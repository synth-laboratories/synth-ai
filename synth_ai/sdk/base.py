"""Shared HTTP transport for infra SDK clients."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NoReturn, TypeVar, cast

import httpx
from pydantic import BaseModel

from synth_ai.core.auth.credentials import resolve_api_credential
from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.errors import AuthenticationError
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base

ModelT = TypeVar("ModelT", bound=BaseModel)


def resolve_api_key(api_key: str | None) -> str:
    try:
        return resolve_api_credential(api_key).value
    except AuthenticationError as error:
        raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)") from error


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
        self._transport: HttpTransport | None = None

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
        return self._open_transport().client

    def _open_transport(self) -> HttpTransport:
        if self._transport is None:
            self._transport = HttpTransport(
                base_url=self._backend_base,
                headers=self._headers(),
                timeout_seconds=self._timeout_seconds,
                error_handler=_raise_infra_error,
                exception_handler=_raise_infra_transport_exception,
            )
        return self._transport

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
        return self._open_transport().request_json(
            method.upper(),
            path if path.startswith("/") else f"/{path}",
            json_body=cast(JsonObject | None, json_body),
            params=cast(dict[str, JsonValue] | None, params),
            timeout_seconds=timeout_seconds,
        )

    def _stream(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        for event in self._open_transport().stream_sse(
            path if path.startswith("/") else f"/{path}",
            params=cast(dict[str, JsonValue] | None, params),
            timeout_seconds=timeout_seconds,
        ):
            payload = event.json_data()
            if not isinstance(payload, dict):
                raise ValueError("SSE data payload must be a JSON object")
            yield payload

    def cast_to(self, model: type[ModelT], payload: Any) -> ModelT:
        return model.model_validate(payload)

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def __enter__(self) -> SynthBaseClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _raise_infra_error(response: httpx.Response) -> NoReturn:
    response.raise_for_status()
    raise RuntimeError("unreachable: successful response passed to error handler")


def _raise_infra_transport_exception(
    method: str,
    path: str,
    error: httpx.HTTPError,
) -> NoReturn:
    raise error


__all__ = [
    "SynthBaseClient",
    "join_url",
    "resolve_api_key",
    "resolve_backend_base",
]
