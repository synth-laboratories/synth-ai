"""Anthropic-view managed-agents client (via backend BFF proxy)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from synth_ai.core.utils.env import get_api_key
from synth_ai.core.utils.urls import BACKEND_URL_BASE, join_url, normalize_backend_base


DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MANAGED_AGENTS_BETA = (
    "managed-agents-2026-04-01,"
    "managed-agents-2026-04-01-research-preview,"
    "files-api-2025-04-14"
)


def _raise_for_status_with_body(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        if len(body) > 2000:
            body = f"{body[:2000]}..."
        message = f"{exc} Response body: {body}" if body else str(exc)
        raise httpx.HTTPStatusError(message, request=response.request, response=response) from exc


def _page_data(page: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = page.get("data")
    if raw_items is None:
        raw_items = page.get("events")
    return [dict(item) for item in list(raw_items or []) if isinstance(item, dict)]


def _is_terminal_failure_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("type") or "")
    if event_type == "session.error":
        return True
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    failure_text = " ".join(
        str(value or "").lower()
        for value in (
            payload.get("failure_type"),
            payload.get("failure_reason"),
            payload.get("last_failure"),
            error.get("kind"),
            error.get("message"),
        )
    )
    return any(
        needle in failure_text
        for needle in (
            "retry_exhausted",
            "missing_api_key",
            "provider_empty_response",
        )
    )


def _is_terminal_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("type") or "")
    if event_type == "session.turn_released" or _is_terminal_failure_event(event):
        return True
    if event_type == "session.status_idle":
        stop_reason = event.get("stop_reason")
        if not isinstance(stop_reason, dict):
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            stop_reason = payload.get("stop_reason")
        return isinstance(stop_reason, dict) and stop_reason.get("type") == "completed"
    return False


def _event_error(event: dict[str, Any]) -> dict[str, Any] | None:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    raw_error = payload.get("error") if isinstance(payload.get("error"), dict) else None
    if raw_error is not None:
        return dict(raw_error)
    if not _is_terminal_failure_event(event):
        return None
    error = {
        key: value
        for key, value in payload.items()
        if key in {"failure_type", "failure_reason", "last_failure"}
        and value not in {None, ""}
    }
    if "message" not in error:
        error["message"] = str(event.get("type") or "session failed")
    return error


@dataclass(frozen=True)
class ManagedAgentRun:
    session: dict[str, Any]
    post_response: dict[str, Any]
    events: list[dict[str, Any]]
    terminal_event: dict[str, Any] | None
    status: str
    error: dict[str, Any] | None = None

    @property
    def session_id(self) -> str:
        return str(self.session.get("id") or "")


class ManagedAgentsAnthropicClient:
    """Sync client for backend Anthropic-view managed-agents proxy endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        timeout: float = 30.0,
        anthropic_version: str | None = None,
        path_prefix: str = "/api/managed-agents/anthropic/v1",
        require_api_key: bool = True,
        anthropic_beta: str | None = None,
    ) -> None:
        self._api_key = (api_key or get_api_key(required=False) or "").strip()
        if require_api_key and not self._api_key:
            raise ValueError("api_key is required (provide explicitly or set SYNTH_API_KEY)")
        self._backend_base = normalize_backend_base(backend_base or BACKEND_URL_BASE)
        self._timeout = timeout
        self._anthropic_version = (
            anthropic_version or os.getenv("ANTHROPIC_VERSION") or DEFAULT_ANTHROPIC_VERSION
        ).strip()
        self._anthropic_beta = (
            anthropic_beta
            or os.getenv("ANTHROPIC_BETA")
            or DEFAULT_MANAGED_AGENTS_BETA
        ).strip()
        self._prefix = path_prefix.rstrip("/")

    @classmethod
    def from_horizons_private(
        cls,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        anthropic_version: str | None = None,
        anthropic_beta: str | None = None,
    ) -> ManagedAgentsAnthropicClient:
        return cls(
            api_key=api_key or "",
            backend_base=base_url,
            timeout=timeout,
            anthropic_version=anthropic_version or DEFAULT_ANTHROPIC_VERSION,
            anthropic_beta=anthropic_beta or DEFAULT_MANAGED_AGENTS_BETA,
            path_prefix="/anthropic/v1",
            require_api_key=False,
        )

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "anthropic-version": self._anthropic_version,
            "anthropic-beta": self._anthropic_beta,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if extra:
            headers.update(extra)
        return headers

    def _path(self, path: str) -> str:
        cleaned = str(path or "").strip()
        if not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"
        return f"{self._prefix}{cleaned}"

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
            url=join_url(self._backend_base, self._path(path)),
            headers=self._headers(headers),
            json=json_body,
            params=params,
            timeout=self._timeout,
        )
        _raise_for_status_with_body(response)
        if not response.content:
            return {}
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            return response.json()
        return {"content": response.text}

    def request_bytes(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        response = httpx.request(
            method=method.upper(),
            url=join_url(self._backend_base, self._path(path)),
            headers=self._headers(headers),
            params=params,
            timeout=self._timeout,
        )
        _raise_for_status_with_body(response)
        return response.content

    def stream(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        with httpx.stream(
            "GET",
            join_url(self._backend_base, self._path(path)),
            headers=self._headers(headers),
            params=params,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
        ) as response:
            _raise_for_status_with_body(response)
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if not payload:
                    continue
                if payload == "[DONE]":
                    break
                yield json.loads(payload)

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def list_environments(self, **params: Any) -> dict[str, Any]:
        return self.request("GET", "/environments", params=params or None)

    def create_environment(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/environments", json_body=request)

    def get_environment(self, environment_id: str) -> dict[str, Any]:
        return self.request("GET", f"/environments/{environment_id}")

    def update_environment(self, environment_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/environments/{environment_id}", json_body=request)

    def archive_environment(self, environment_id: str) -> dict[str, Any]:
        return self.request("POST", f"/environments/{environment_id}/archive", json_body={})

    def list_agents(self, **params: Any) -> dict[str, Any]:
        return self.request("GET", "/agents", params=params or None)

    def create_agent(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/agents", json_body=request)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self.request("GET", f"/agents/{agent_id}")

    def update_agent(self, agent_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/agents/{agent_id}", json_body=request)

    def archive_agent(self, agent_id: str) -> dict[str, Any]:
        return self.request("POST", f"/agents/{agent_id}/archive", json_body={})

    def list_sessions(self, **params: Any) -> dict[str, Any]:
        return self.request("GET", "/sessions", params=params or None)

    def create_session(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/sessions", json_body=request)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}")

    def update_session(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}", json_body=request)

    def archive_session(self, session_id: str) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}/archive", json_body={})

    def post_session_events(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}/events", json_body=request)

    def list_session_events(self, session_id: str, **params: Any) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}/events", params=params or None)

    def list_threads(self, **params: Any) -> dict[str, Any]:
        return self.request("GET", "/threads", params=params or None)

    def list_session_threads(self, session_id: str, **params: Any) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}/threads", params=params or None)

    def list_session_thread_events(
        self,
        session_id: str,
        agent_thread_id: str,
        **params: Any,
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            f"/sessions/{session_id}/threads/{agent_thread_id}/events",
            params=params or None,
        )

    def stream_session_events(
        self,
        session_id: str,
        **params: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.stream(f"/sessions/{session_id}/events/stream", params=params or None)

    def stream_session(self, session_id: str, **params: Any) -> Iterator[dict[str, Any]]:
        return self.stream(f"/sessions/{session_id}/stream", params=params or None)

    def create_file(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/files", json_body=request)

    def add_session_resource(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}/resources", json_body=request)

    def list_session_resources(self, session_id: str, **params: Any) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}/resources", params=params or None)

    def get_session_resource(self, session_id: str, resource_id: str) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}/resources/{resource_id}")

    def update_session_resource(
        self,
        session_id: str,
        resource_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/sessions/{session_id}/resources/{resource_id}",
            json_body=request,
        )

    def delete_session_resource(self, session_id: str, resource_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/sessions/{session_id}/resources/{resource_id}")

    def list_files(self, **params: Any) -> dict[str, Any]:
        return self.request("GET", "/files", params=params or None)

    def list_session_files(self, session_id: str, **params: Any) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}/files", params=params or None)

    def get_file(self, file_id: str) -> dict[str, Any]:
        return self.request("GET", f"/files/{file_id}")

    def delete_file(self, file_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/files/{file_id}")

    def download_file_content(self, file_id: str) -> bytes:
        return self.request_bytes("GET", f"/files/{file_id}/content")

    def iter_session_events(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        cursor = int(after_sequence)
        while True:
            page = self.list_session_events(session_id, after_sequence=cursor, limit=limit)
            events = _page_data(page)
            yield from events
            next_after = int(page.get("next_after_sequence") or 0)
            if len(events) < limit or next_after <= cursor:
                break
            cursor = next_after

    def collect_session_events(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return list(
            self.iter_session_events(
                session_id,
                after_sequence=after_sequence,
                limit=limit,
            )
        )

    def run_until_done(
        self,
        *,
        agent: str,
        environment_id: str | None,
        message: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        resources: list[dict[str, Any]] | None = None,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 2.0,
    ) -> ManagedAgentRun:
        import time

        session_request: dict[str, Any] = {
            "agent": agent,
            "metadata": metadata or {},
        }
        if environment_id:
            session_request["environment_id"] = environment_id
        if title:
            session_request["title"] = title
        if resources:
            session_request["resources"] = resources
        session = self.create_session(session_request)
        session_id = str(session["id"])
        post_response = self.post_session_events(
            session_id,
            {
                "events": [
                    {
                        "type": "user.message",
                        "content": [{"type": "text", "text": message}],
                    }
                ]
            },
        )
        posted_events = _page_data(post_response)
        first_sequence = min(
            [int(event.get("sequence") or 0) for event in posted_events] or [1]
        )
        deadline = time.time() + timeout_seconds
        terminal_event: dict[str, Any] | None = None
        events: list[dict[str, Any]] = []
        while time.time() < deadline:
            events = self.collect_session_events(
                session_id,
                after_sequence=max(0, first_sequence - 1),
            )
            relevant = [
                event for event in events if int(event.get("sequence") or 0) >= first_sequence
            ]
            terminal_events = [event for event in relevant if _is_terminal_event(event)]
            if terminal_events:
                failure_events = [
                    event for event in terminal_events if _is_terminal_failure_event(event)
                ]
                terminal_event = (failure_events or terminal_events)[-1]
                break
            session_state = self.get_session(session_id)
            last_failure = session_state.get("last_failure")
            if isinstance(last_failure, dict) and last_failure:
                terminal_event = {
                    "type": "session.error",
                    "sequence": max([int(event.get("sequence") or 0) for event in events] or [0]),
                    "payload": {"error": last_failure},
                    "stop_reason": session_state.get("stop_reason"),
                }
                break
            stop_reason = session_state.get("stop_reason")
            if (
                str(session_state.get("status") or "") in {"idle", "terminated"}
                and isinstance(stop_reason, dict)
                and stop_reason.get("type") == "completed"
            ):
                terminal_event = {
                    "type": "session.status_idle",
                    "sequence": max([int(event.get("sequence") or 0) for event in events] or [0]),
                    "payload": {"stop_reason": stop_reason},
                    "stop_reason": stop_reason,
                }
                break
            time.sleep(poll_interval_seconds)
        else:
            raise TimeoutError(
                f"managed agents session {session_id} timed out after {timeout_seconds}s"
            )
        error = _event_error(terminal_event) if terminal_event else None
        status = (
            "failed"
            if terminal_event and _is_terminal_failure_event(terminal_event)
            else "succeeded"
        )
        return ManagedAgentRun(
            session=session,
            post_response=post_response,
            events=events,
            terminal_event=terminal_event,
            status=status,
            error=error if isinstance(error, dict) else None,
        )

    def download_session_files(
        self,
        *,
        session_id: str,
        names: list[str],
        output_dir: str | Path,
    ) -> list[dict[str, str]]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            files = _page_data(self.list_session_files(session_id))
        except Exception:
            files = _page_data(self.list_files(scope="session", scope_id=session_id))
        session_files = [
            item
            for item in files
            if str(item.get("scope_id") or item.get("session_id") or session_id) == session_id
        ]
        downloaded: list[dict[str, str]] = []
        for expected_name in names:
            match = None
            for item in session_files:
                item_name = str(item.get("name") or item.get("filename") or "")
                if (
                    item_name == expected_name
                    or item_name.endswith("/" + expected_name)
                    or Path(item_name).name == expected_name
                ):
                    match = item
                    break
            if match is None:
                continue
            file_id = str(match["id"])
            filename = Path(str(match.get("name") or match.get("filename") or file_id)).name
            path = output_path / filename
            path.write_bytes(self.download_file_content(file_id))
            downloaded.append({"file_id": file_id, "filename": filename, "path": str(path)})
        return downloaded


class AsyncManagedAgentsAnthropicClient:
    """Async adapter around ``ManagedAgentsAnthropicClient``."""

    def __init__(self, sync_client: ManagedAgentsAnthropicClient) -> None:
        self._sync = sync_client

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

    async def health(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.health)

    async def list_environments(self, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_environments, **params)

    async def create_environment(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_environment, request)

    async def get_environment(self, environment_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_environment, environment_id)

    async def update_environment(
        self, environment_id: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update_environment, environment_id, request)

    async def archive_environment(self, environment_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.archive_environment, environment_id)

    async def list_agents(self, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_agents, **params)

    async def create_agent(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_agent, request)

    async def get_agent(self, agent_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_agent, agent_id)

    async def update_agent(self, agent_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update_agent, agent_id, request)

    async def archive_agent(self, agent_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.archive_agent, agent_id)

    async def list_sessions(self, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_sessions, **params)

    async def create_session(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_session, request)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_session, session_id)

    async def update_session(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.update_session, session_id, request)

    async def archive_session(self, session_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.archive_session, session_id)

    async def post_session_events(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.post_session_events, session_id, request)

    async def list_session_events(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_session_events, session_id, **params)

    async def list_threads(self, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_threads, **params)

    async def list_session_threads(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_session_threads, session_id, **params)

    async def list_session_thread_events(
        self,
        session_id: str,
        agent_thread_id: str,
        **params: Any,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.list_session_thread_events,
            session_id,
            agent_thread_id,
            **params,
        )

    async def create_file(self, request: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.create_file, request)

    async def add_session_resource(
        self,
        session_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.add_session_resource, session_id, request)

    async def list_session_resources(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_session_resources, session_id, **params)

    async def get_session_resource(self, session_id: str, resource_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.get_session_resource,
            session_id,
            resource_id,
        )

    async def update_session_resource(
        self,
        session_id: str,
        resource_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.update_session_resource,
            session_id,
            resource_id,
            request,
        )

    async def delete_session_resource(self, session_id: str, resource_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._sync.delete_session_resource,
            session_id,
            resource_id,
        )

    async def list_files(self, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_files, **params)

    async def list_session_files(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.list_session_files, session_id, **params)

    async def get_file(self, file_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_file, file_id)

    async def delete_file(self, file_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.delete_file, file_id)

    async def download_file_content(self, file_id: str) -> bytes:
        return await asyncio.to_thread(self._sync.download_file_content, file_id)

    async def stream_session_events(
        self,
        session_id: str,
        **params: Any,
    ) -> list[dict[str, Any]]:
        def _collect() -> list[dict[str, Any]]:
            return list(self._sync.stream_session_events(session_id, **params))

        return await asyncio.to_thread(_collect)

    async def stream_session(self, session_id: str, **params: Any) -> list[dict[str, Any]]:
        def _collect() -> list[dict[str, Any]]:
            return list(self._sync.stream_session(session_id, **params))

        return await asyncio.to_thread(_collect)

    async def run_until_done(self, **kwargs: Any) -> ManagedAgentRun:
        return await asyncio.to_thread(self._sync.run_until_done, **kwargs)

    async def download_session_files(self, **kwargs: Any) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._sync.download_session_files, **kwargs)


__all__ = [
    "AsyncManagedAgentsAnthropicClient",
    "ManagedAgentRun",
    "ManagedAgentsAnthropicClient",
]
