"""OpenAI Agents SDK compatibility client with dual transport support."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base

TRANSPORT_MODE_BACKEND_BFF = "backend_bff"
TRANSPORT_MODE_DIRECT_HP = "direct_hp"
TRANSPORT_MODE_AUTO = "auto"
VALID_TRANSPORT_MODES = {
    TRANSPORT_MODE_BACKEND_BFF,
    TRANSPORT_MODE_DIRECT_HP,
    TRANSPORT_MODE_AUTO,
}
FALLBACK_STATUS_CODES = {404, 405, 501}

BACKEND_BFF_PREFIX = "/api/managed-agents/openai/v1"
DIRECT_HP_PREFIX = "/openai/v1"


class OpenAIAgentsSdkClient:
    """Sync client for OpenAI Agents SDK compatibility routes."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        timeout: float = 30.0,
        transport_mode: str = TRANSPORT_MODE_AUTO,
        openai_organization: str | None = None,
        openai_project: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self._api_key = (api_key or os.getenv("SYNTH_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
        self._backend_base = normalize_backend_base(backend_base or BACKEND_URL_BASE)
        self._timeout = timeout
        self._transport_mode = self._validate_transport_mode(transport_mode)
        self._openai_organization = str(openai_organization or "").strip() or None
        self._openai_project = str(openai_project or "").strip() or None
        self._request_id = str(request_id or "").strip() or None

    @staticmethod
    def _validate_transport_mode(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in VALID_TRANSPORT_MODES:
            allowed = ", ".join(sorted(VALID_TRANSPORT_MODES))
            raise ValueError(f"transport_mode must be one of: {allowed}")
        return normalized

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

    @staticmethod
    def _should_fallback(exc: Exception) -> bool:
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        return exc.response.status_code in FALLBACK_STATUS_CODES

    @staticmethod
    def _prefix_for_mode(mode: str) -> str:
        if mode == TRANSPORT_MODE_BACKEND_BFF:
            return BACKEND_BFF_PREFIX
        return DIRECT_HP_PREFIX

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if self._openai_organization:
            headers["OpenAI-Organization"] = self._openai_organization
        if self._openai_project:
            headers["OpenAI-Project"] = self._openai_project
        if self._request_id:
            headers["X-Client-Request-Id"] = self._request_id
        if extra:
            headers.update(extra)
        return headers

    def _request_once(
        self,
        *,
        prefix: str,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = httpx.request(
            method=method.upper(),
            url=join_url(self._backend_base, f"{prefix}{self._normalize_path(path)}"),
            headers=self._headers(headers),
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return self._parse_response(response)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if self._transport_mode != TRANSPORT_MODE_AUTO:
            return self._request_once(
                prefix=self._prefix_for_mode(self._transport_mode),
                method=method,
                path=path,
                json_body=json_body,
                params=params,
                headers=headers,
            )
        try:
            return self._request_once(
                prefix=BACKEND_BFF_PREFIX,
                method=method,
                path=path,
                json_body=json_body,
                params=params,
                headers=headers,
            )
        except Exception as exc:
            if not self._should_fallback(exc):
                raise
            return self._request_once(
                prefix=DIRECT_HP_PREFIX,
                method=method,
                path=path,
                json_body=json_body,
                params=params,
                headers=headers,
            )

    @staticmethod
    def _parse_sse_data(payload: str) -> Any:
        text = str(payload or "").strip()
        if not text:
            return ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _stream_once(
        self,
        *,
        prefix: str,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        with httpx.stream(
            method.upper(),
            join_url(self._backend_base, f"{prefix}{self._normalize_path(path)}"),
            headers=self._headers(headers),
            params=params,
            json=json_body,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        ) as response:
            response.raise_for_status()
            event_name: str | None = None
            event_id: str | None = None
            retry_ms: int | None = None
            data_lines: list[str] = []

            def _emit() -> dict[str, Any] | None:
                nonlocal event_name, event_id, retry_ms, data_lines
                if not data_lines and event_name is None and event_id is None and retry_ms is None:
                    return None
                joined = "\n".join(data_lines).strip()
                event_name_value = event_name or "message"
                event_name = None
                event_id_value = event_id
                event_id = None
                retry_value = retry_ms
                retry_ms = None
                data_lines = []
                if joined == "[DONE]":
                    return {"event": event_name_value, "data": "[DONE]"}
                frame: dict[str, Any] = {
                    "event": event_name_value,
                    "data": self._parse_sse_data(joined),
                }
                if event_id_value is not None:
                    frame["id"] = event_id_value
                if retry_value is not None:
                    frame["retry"] = retry_value
                return frame

            for raw_line in response.iter_lines():
                if raw_line is None:
                    continue
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, (bytes, bytearray))
                    else str(raw_line)
                )
                if line == "":
                    frame = _emit()
                    if frame is None:
                        continue
                    if frame.get("data") == "[DONE]":
                        break
                    yield frame
                    continue
                if line.startswith(":"):
                    continue
                key, _, raw_value = line.partition(":")
                value = raw_value.lstrip(" ")
                if key == "event":
                    event_name = value or None
                    continue
                if key == "data":
                    data_lines.append(value)
                    continue
                if key == "id":
                    event_id = value or None
                    continue
                if key == "retry":
                    try:
                        retry_ms = int(value)
                    except ValueError:
                        retry_ms = None
                    continue
            trailing = _emit()
            if trailing is not None and trailing.get("data") != "[DONE]":
                yield trailing

    def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if self._transport_mode != TRANSPORT_MODE_AUTO:
            return self._stream_once(
                prefix=self._prefix_for_mode(self._transport_mode),
                method=method,
                path=path,
                json_body=json_body,
                params=params,
                headers=headers,
            )

        def _iter_auto() -> Iterator[dict[str, Any]]:
            try:
                yield from self._stream_once(
                    prefix=BACKEND_BFF_PREFIX,
                    method=method,
                    path=path,
                    json_body=json_body,
                    params=params,
                    headers=headers,
                )
            except Exception as exc:
                if not self._should_fallback(exc):
                    raise
                yield from self._stream_once(
                    prefix=DIRECT_HP_PREFIX,
                    method=method,
                    path=path,
                    json_body=json_body,
                    params=params,
                    headers=headers,
                )

        return _iter_auto()

    # /responses family
    def create_response(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/responses", json_body=request)

    def stream_response(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        payload = dict(request)
        payload["stream"] = True
        return self.stream("POST", "/responses", json_body=payload)

    def get_response(self, response_id: str) -> dict[str, Any]:
        return self.request("GET", f"/responses/{response_id}")

    def cancel_response(self, response_id: str) -> dict[str, Any]:
        return self.request("POST", f"/responses/{response_id}/cancel", json_body={})

    def delete_response(self, response_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/responses/{response_id}")

    def list_response_input_items(
        self,
        response_id: str,
        **params: Any,
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            f"/responses/{response_id}/input_items",
            params=params or None,
        )

    def create_response_input_tokens(
        self,
        response_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/responses/{response_id}/input_tokens",
            json_body=request,
        )

    def compact_response(self, response_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/responses/{response_id}/compact", json_body=request)

    # /conversations family
    def create_conversation(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", "/conversations", json_body=request or {})

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self.request("GET", f"/conversations/{conversation_id}")

    def update_conversation(
        self,
        conversation_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self.request("POST", f"/conversations/{conversation_id}", json_body=request)

    def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/conversations/{conversation_id}")

    def create_conversation_item(
        self,
        conversation_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self.request("POST", f"/conversations/{conversation_id}/items", json_body=request)

    def list_conversation_items(
        self,
        conversation_id: str,
        **params: Any,
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            f"/conversations/{conversation_id}/items",
            params=params or None,
        )

    def get_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        return self.request("GET", f"/conversations/{conversation_id}/items/{item_id}")

    def delete_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        return self.request("DELETE", f"/conversations/{conversation_id}/items/{item_id}")


class AsyncOpenAIAgentsSdkClient:
    """Async adapter around ``OpenAIAgentsSdkClient``.

    All non-streaming methods delegate to the sync client via ``asyncio.to_thread``.
    ``stream_response`` is a true async generator backed by ``httpx.AsyncClient``.
    """

    def __init__(self, sync_client: OpenAIAgentsSdkClient) -> None:
        self._sync = sync_client

    # ------------------------------------------------------------------
    # Internal async SSE streaming (httpx.AsyncClient based)
    # ------------------------------------------------------------------

    async def _async_stream_once(
        self,
        *,
        prefix: str,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        url = join_url(
            self._sync._backend_base,
            f"{prefix}{self._sync._normalize_path(path)}",
        )
        async with httpx.AsyncClient() as client, client.stream(
            method.upper(),
            url,
            headers=self._sync._headers(headers),
            params=params,
            json=json_body,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        ) as response:
            response.raise_for_status()
            event_name: str | None = None
            event_id: str | None = None
            retry_ms: int | None = None
            data_lines: list[str] = []

            def _emit() -> dict[str, Any] | None:
                nonlocal event_name, event_id, retry_ms, data_lines
                if (
                    not data_lines
                    and event_name is None
                    and event_id is None
                    and retry_ms is None
                ):
                    return None
                joined = "\n".join(data_lines).strip()
                ev = event_name or "message"
                ei = event_id
                rv = retry_ms
                event_name = None  # type: ignore[assignment]
                event_id = None  # type: ignore[assignment]
                retry_ms = None  # type: ignore[assignment]
                data_lines.clear()
                if joined == "[DONE]":
                    return {"event": ev, "data": "[DONE]"}
                frame: dict[str, Any] = {
                    "event": ev,
                    "data": self._sync._parse_sse_data(joined),
                }
                if ei is not None:
                    frame["id"] = ei
                if rv is not None:
                    frame["retry"] = rv
                return frame

            async for raw_line in response.aiter_lines():
                if raw_line is None:
                    continue
                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
                if line == "":
                    frame = _emit()
                    if frame is None:
                        continue
                    if frame.get("data") == "[DONE]":
                        return
                    yield frame
                    continue
                if line.startswith(":"):
                    continue
                key, _, raw_value = line.partition(":")
                value = raw_value.lstrip(" ")
                if key == "event":
                    event_name = value or None
                elif key == "data":
                    data_lines.append(value)
                elif key == "id":
                    event_id = value or None
                elif key == "retry":
                    try:
                        retry_ms = int(value)
                    except ValueError:
                        retry_ms = None
            trailing = _emit()
            if trailing is not None and trailing.get("data") != "[DONE]":
                yield trailing

    async def _async_stream(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if self._sync._transport_mode != TRANSPORT_MODE_AUTO:
            async for frame in self._async_stream_once(
                prefix=self._sync._prefix_for_mode(self._sync._transport_mode),
                method=method,
                path=path,
                json_body=json_body,
                params=params,
                headers=headers,
            ):
                yield frame
            return

        # auto mode: BFF first; fallback to direct HP only on FALLBACK_STATUS_CODES.
        # raise_for_status() fires before any frames are yielded, so the fallback
        # is safe — no frames have been emitted when the exception propagates.
        try:
            async for frame in self._async_stream_once(
                prefix=BACKEND_BFF_PREFIX,
                method=method,
                path=path,
                json_body=json_body,
                params=params,
                headers=headers,
            ):
                yield frame
        except Exception as exc:
            if not self._sync._should_fallback(exc):
                raise
            async for frame in self._async_stream_once(
                prefix=DIRECT_HP_PREFIX,
                method=method,
                path=path,
                json_body=json_body,
                params=params,
                headers=headers,
            ):
                yield frame

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

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

    async def create_response(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_response, request)

    async def stream_response(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Async generator that streams SSE frames from the /responses endpoint."""
        payload = dict(request)
        payload["stream"] = True
        async for frame in self._async_stream("POST", "/responses", json_body=payload):
            yield frame

    async def get_response(self, response_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_response, response_id)

    async def cancel_response(self, response_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.cancel_response, response_id)

    async def delete_response(self, response_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.delete_response, response_id)

    async def list_response_input_items(
        self,
        response_id: str,
        **params: Any,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_response_input_items, response_id, **params)

    async def create_response_input_tokens(
        self,
        response_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.create_response_input_tokens, response_id, request
        )

    async def compact_response(self, response_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.compact_response, response_id, request)

    async def create_conversation(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_conversation, request)

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_conversation, conversation_id)

    async def update_conversation(
        self,
        conversation_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update_conversation, conversation_id, request)

    async def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.delete_conversation, conversation_id)

    async def create_conversation_item(
        self,
        conversation_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.create_conversation_item, conversation_id, request
        )

    async def list_conversation_items(
        self,
        conversation_id: str,
        **params: Any,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.list_conversation_items, conversation_id, **params
        )

    async def get_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_conversation_item, conversation_id, item_id)

    async def delete_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.delete_conversation_item, conversation_id, item_id
        )


__all__ = [
    "AsyncOpenAIAgentsSdkClient",
    "BACKEND_BFF_PREFIX",
    "DIRECT_HP_PREFIX",
    "FALLBACK_STATUS_CODES",
    "OpenAIAgentsSdkClient",
    "TRANSPORT_MODE_AUTO",
    "TRANSPORT_MODE_BACKEND_BFF",
    "TRANSPORT_MODE_DIRECT_HP",
]
