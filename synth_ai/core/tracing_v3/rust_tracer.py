"""Rust-backed SessionTracer wrapper for tracing v3."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from typing import Any

from .decorators import set_session_id, set_session_tracer, set_turn_number
from .hooks import HookManager

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for tracing_v3.rust_tracer.") from exc


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    if is_dataclass(value):
        return asdict(value)
    return value


def _storage_from_url(url: str, token: str | None) -> Any:
    if url == ":memory:":
        return synth_ai_py.LibsqlTraceStorage.memory()
    if url.startswith(("libsql://", "http://", "https://")):
        auth = token or os.getenv("TURSO_AUTH_TOKEN") or ""
        return synth_ai_py.LibsqlTraceStorage.turso(url, auth)
    return synth_ai_py.LibsqlTraceStorage.file(url)


def _resolve_storage(
    db_url: str | None,
    storage: Any,
    storage_config: Any,
) -> Any | None:
    if storage is not None:
        return storage

    if storage_config is not None:
        conn = getattr(storage_config, "connection_string", None)
        token = getattr(storage_config, "turso_auth_token", None)
        if conn:
            return _storage_from_url(conn, token)

    if db_url:
        token = None
        if storage_config is not None:
            token = getattr(storage_config, "turso_auth_token", None)
        return _storage_from_url(db_url, token)

    return None


class SessionTracer:
    """Async-compatible wrapper over Rust core SessionTracer."""

    def __init__(
        self,
        hooks: Any | None = None,
        db_url: str | None = None,
        auto_save: bool = True,
        storage: Any | None = None,
        storage_config: Any | None = None,
    ) -> None:
        if synth_ai_py is None:
            raise RuntimeError("synth_ai_py is required for tracing_v3.rust_tracer.")

        self.hooks = hooks or HookManager()
        resolved_storage = _resolve_storage(db_url, storage, storage_config)
        if resolved_storage is None:
            self._tracer = synth_ai_py.SessionTracer.memory()
        else:
            self._tracer = synth_ai_py.SessionTracer(resolved_storage, auto_save)

        self._current_session_id: str | None = None
        self._current_step_id: str | None = None
        self._last_trace: Any | None = None

    @property
    def current_session(self) -> Any | None:
        return self._last_trace

    @property
    def current_step(self) -> Any | None:
        return self._current_step_id

    async def initialize(self) -> None:
        return None

    async def start_session(
        self, session_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> str:
        session_id = await asyncio.to_thread(self._tracer.start_session, session_id, metadata or {})
        self._current_session_id = session_id
        set_session_id(session_id)
        set_session_tracer(self)
        await self.hooks.trigger("session_start", session_id=session_id, metadata=metadata or {})
        return session_id

    async def start_timestep(
        self,
        step_id: str,
        turn_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        await asyncio.to_thread(self._tracer.start_timestep, step_id, turn_number, metadata or {})
        self._current_step_id = step_id
        if turn_number is not None:
            set_turn_number(turn_number)
        await self.hooks.trigger(
            "timestep_start", step=step_id, session_id=self._current_session_id
        )
        return step_id

    async def end_timestep(self, step_id: str | None = None) -> None:
        await asyncio.to_thread(self._tracer.end_timestep)
        await self.hooks.trigger("timestep_end", step=step_id, session_id=self._current_session_id)
        self._current_step_id = None

    async def record_event(self, event: Any) -> int | None:
        payload = _to_jsonable(event)
        event_id = await asyncio.to_thread(self._tracer.record_event, payload)
        await self.hooks.trigger("event_recorded", event_obj=event)
        return event_id

    async def record_message(self, message: Any) -> int | None:
        payload = _to_jsonable(message)
        message_id = await asyncio.to_thread(self._tracer.record_message, payload)
        await self.hooks.trigger("message_recorded", message=message)
        return message_id

    async def record_outcome_reward(self, reward: Any) -> int | None:
        payload = _to_jsonable(reward)
        return await asyncio.to_thread(self._tracer.record_outcome_reward, payload)

    async def record_event_reward(
        self,
        event_id: int,
        reward: Any,
        message_id: int | None = None,
        turn_number: int | None = None,
    ) -> int | None:
        payload = _to_jsonable(reward)
        return await asyncio.to_thread(
            self._tracer.record_event_reward, event_id, payload, message_id, turn_number
        )

    async def end_session(self, save: bool | None = None) -> Any:
        if save is None or save:
            await self.hooks.trigger("before_save", session=self._last_trace)
        trace = await asyncio.to_thread(
            self._tracer.end_session, save if save is not None else True
        )
        if save is None or save:
            await self.hooks.trigger("after_save", session=trace)
        await self.hooks.trigger("session_end", session=trace)
        self._last_trace = trace
        self._current_session_id = None
        self._current_step_id = None
        set_session_id(None)
        set_turn_number(None)
        set_session_tracer(None)
        return trace

    async def get_session(self, session_id: str) -> Any:
        return await asyncio.to_thread(self._tracer.get_session, session_id)

    async def query_traces(self, query: str, params: dict[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(self._tracer.query, query, params or None)

    async def get_session_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM session_traces ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        records = await self.query_traces(query)
        if hasattr(records, "to_dict"):
            try:
                return records.to_dict("records")
            except Exception:
                pass
        return records

    async def delete_session(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._tracer.delete_session, session_id)

    @asynccontextmanager
    async def session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        save: bool | None = None,
    ):
        session_id = await self.start_session(session_id, metadata)
        try:
            yield session_id
        finally:
            await self.end_session(save=save)

    @asynccontextmanager
    async def timestep(
        self,
        step_id: str,
        turn_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        step = await self.start_timestep(step_id, turn_number, metadata)
        try:
            yield step
        finally:
            await self.end_timestep(step_id)

    async def close(self) -> None:
        return None
