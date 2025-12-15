"""LibSQL-native trace manager prototype.

This module provides the Turso/libsql-backed trace storage implementation. It
mirrors the public surface area of the historical SQLAlchemy manager while
executing all operations directly via libsql.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
import libsql
from sqlalchemy.engine import make_url

from ..abstractions import (
    EnvironmentEvent,
    LMCAISEvent,
    RuntimeEvent,
    SessionMessageContent,
    SessionTrace,
)
from ..config import CONFIG
from ..storage.base import TraceStorage
from .models import analytics_views

if TYPE_CHECKING:
    from sqlite3 import Connection as LibsqlConnection
else:  # pragma: no cover - runtime fallback for typing only
    LibsqlConnection = Any  # type: ignore[assignment]

_LIBSQL_CONNECT_ATTR = getattr(libsql, "connect", None)
if _LIBSQL_CONNECT_ATTR is None:  # pragma: no cover - defensive guard
    raise RuntimeError("libsql.connect is required for NativeLibsqlTraceManager")
_libsql_connect: Callable[..., LibsqlConnection] = cast(
    Callable[..., LibsqlConnection],
    _LIBSQL_CONNECT_ATTR,
)

try:  # pragma: no cover - exercised only when pandas present
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ConnectionTarget:
    """Resolved connection target for libsql."""

    database: str
    sync_url: str | None = None
    auth_token: str | None = None


def _strip_auth_component(url: str) -> tuple[str, str | None]:
    """Remove auth_token query parameter from URL, returning the token separately."""
    parsed = urlparse(url)
    if not parsed.query:
        return url, None

    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    token = params.pop("auth_token", None)
    query = urlencode(params, doseq=True)
    sanitised = urlunparse(parsed._replace(query=query))
    return sanitised, token


def _resolve_connection_target(db_url: str | None, auth_token: str | None) -> _ConnectionTarget:
    """Normalise the configured database URL."""
    url = db_url or CONFIG.db_url
    sanitised, token_from_url = _strip_auth_component(url)
    effective_token = auth_token or token_from_url or CONFIG.auth_token

    # SQLAlchemy-compatible libsql scheme (`sqlite+libsql://<endpoint or path>`)
    if sanitised.startswith("sqlite+libsql://"):
        raise RuntimeError("sqlite+libsql scheme is no longer supported; use libsql://")

    # Plain SQLite files: file://, /absolute/path, or relative path
    # libsql.connect() handles these without sync_url or auth_token
    if sanitised.startswith("file://") or sanitised.startswith("/") or "://" not in sanitised:
        # Strip file:// prefix if present, libsql.connect handles both formats
        db_path = sanitised.replace("file://", "") if sanitised.startswith("file://") else sanitised
        return _ConnectionTarget(database=db_path, sync_url=None, auth_token=None)

    # Native libsql URLs (`libsql://...`).
    if sanitised.startswith("libsql://"):
        return _ConnectionTarget(database=sanitised, sync_url=sanitised, auth_token=effective_token)

    # Fallback to SQLAlchemy URL parsing for anything else we missed.
    try:
        parsed = make_url(sanitised)
        driver = parsed.drivername.lower()
        if driver.startswith("sqlite"):
            database = parsed.database or ""
            if database and database not in {":memory:", ":memory"}:
                # Absolute paths are passed through; relative paths are resolved to cwd
                if database.startswith("/"):
                    db_path = database
                else:
                    db_path = str(Path(database).expanduser().resolve())
            elif database in {":memory:", ":memory"}:
                db_path = ":memory:"
            else:
                raise RuntimeError("SQLite URL missing database path.")
            return _ConnectionTarget(database=db_path, sync_url=None, auth_token=None)
        if driver.startswith("libsql"):
            database = parsed.render_as_string(hide_password=False)
            return _ConnectionTarget(database=database, sync_url=database, auth_token=effective_token)
    except Exception:  # pragma: no cover - defensive guardrail
        logger.debug("Unable to parse db_url via SQLAlchemy", exc_info=True)

    # Python libsql client uses HTTP API for http:// URLs, not Hrana WebSocket
    # For local sqld with http:// URL, we need to ensure it points to the HTTP API port
    # sqld uses two ports: Hrana WebSocket (e.g. 8080) and HTTP API (e.g. 8081)
    # libsql.connect() with http:// uses HTTP API, so URL should point to HTTP API port
    if sanitised.startswith(("http://", "https://", "libsql://")):
        return _ConnectionTarget(database=sanitised, sync_url=sanitised, auth_token=effective_token)
    raise RuntimeError(f"Unsupported tracing database URL: {sanitised}")


def _json_dumps(value: Any) -> str | None:
    """Serialise Python objects as JSON compatible with the existing schema."""

    def _default(obj: Any):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), default=_default)


def _maybe_datetime(value: Any) -> Any:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return value


def _load_json(value: Any) -> Any:
    if value is None or isinstance(value, dict | list):
        return value or {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value


_TABLE_DEFINITIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        description TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        configuration TEXT,
        metadata TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS systems (
        system_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        system_type VARCHAR,
        description TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS system_versions (
        version_id VARCHAR PRIMARY KEY,
        system_id VARCHAR NOT NULL,
        version_number VARCHAR NOT NULL,
        commit_hash VARCHAR,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        configuration TEXT,
        metadata TEXT,
        FOREIGN KEY(system_id) REFERENCES systems(system_id),
        UNIQUE(system_id, version_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experimental_systems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id VARCHAR NOT NULL,
        system_id VARCHAR NOT NULL,
        version_id VARCHAR NOT NULL,
        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
        FOREIGN KEY(system_id) REFERENCES systems(system_id),
        FOREIGN KEY(version_id) REFERENCES system_versions(version_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_traces (
        session_id VARCHAR PRIMARY KEY,
        created_at DATETIME NOT NULL,
        num_timesteps INTEGER NOT NULL,
        num_events INTEGER NOT NULL,
        num_messages INTEGER NOT NULL,
        metadata TEXT,
        experiment_id VARCHAR,
        embedding VECTOR,
        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_timesteps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id VARCHAR NOT NULL,
        step_id VARCHAR NOT NULL,
        step_index INTEGER NOT NULL,
        turn_number INTEGER,
        started_at DATETIME,
        completed_at DATETIME,
        num_events INTEGER,
        num_messages INTEGER,
        step_metadata TEXT,
        UNIQUE(session_id, step_id),
        FOREIGN KEY(session_id) REFERENCES session_traces(session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id VARCHAR NOT NULL,
        timestep_id INTEGER,
        event_type VARCHAR NOT NULL,
        system_instance_id VARCHAR,
        event_time FLOAT,
        message_time INTEGER,
        created_at DATETIME,
        model_name VARCHAR,
        provider VARCHAR,
        input_tokens INTEGER,
        output_tokens INTEGER,
        total_tokens INTEGER,
        cost_usd INTEGER,
        latency_ms INTEGER,
        span_id VARCHAR,
        trace_id VARCHAR,
        call_records TEXT,
        reward FLOAT,
        terminated BOOLEAN,
        truncated BOOLEAN,
        system_state_before TEXT,
        system_state_after TEXT,
        metadata TEXT,
        event_metadata TEXT,
        embedding VECTOR,
        CHECK (event_type IN ('cais', 'environment', 'runtime')),
        FOREIGN KEY(session_id) REFERENCES session_traces(session_id),
        FOREIGN KEY(timestep_id) REFERENCES session_timesteps(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id VARCHAR NOT NULL,
        timestep_id INTEGER,
        message_type VARCHAR NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME,
        event_time FLOAT,
        message_time INTEGER,
        metadata TEXT,
        embedding VECTOR,
        CHECK (message_type IN ('user', 'assistant', 'system', 'tool_use', 'tool_result')),
        FOREIGN KEY(session_id) REFERENCES session_traces(session_id),
        FOREIGN KEY(timestep_id) REFERENCES session_timesteps(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS outcome_rewards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id VARCHAR NOT NULL,
        total_reward FLOAT NOT NULL,
        achievements_count INTEGER NOT NULL,
        total_steps INTEGER NOT NULL,
        created_at DATETIME NOT NULL,
        reward_metadata TEXT,
        annotation TEXT,
        FOREIGN KEY(session_id) REFERENCES session_traces(session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_rewards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id INTEGER NOT NULL,
        session_id VARCHAR NOT NULL,
        message_id INTEGER,
        turn_number INTEGER,
        reward_value FLOAT NOT NULL,
        reward_type VARCHAR,
        "key" VARCHAR,
        annotation TEXT,
        source VARCHAR,
        created_at DATETIME NOT NULL,
        FOREIGN KEY(event_id) REFERENCES events(id),
        FOREIGN KEY(session_id) REFERENCES session_traces(session_id),
        FOREIGN KEY(message_id) REFERENCES messages(id)
    )
    """
)


_INDEX_DEFINITIONS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_session_created ON session_traces (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_session_experiment ON session_traces (experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_timestep_session_step ON session_timesteps (session_id, step_id)",
    "CREATE INDEX IF NOT EXISTS idx_timestep_turn ON session_timesteps (turn_number)",
    "CREATE INDEX IF NOT EXISTS idx_event_session_step ON events (session_id, timestep_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_type ON events (event_type)",
    "CREATE INDEX IF NOT EXISTS idx_event_created ON events (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_event_model ON events (model_name)",
    "CREATE INDEX IF NOT EXISTS idx_event_trace ON events (trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_message_session_step ON messages (session_id, timestep_id)",
    "CREATE INDEX IF NOT EXISTS idx_message_type ON messages (message_type)",
    "CREATE INDEX IF NOT EXISTS idx_message_timestamp ON messages (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_created ON experiments (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_experiment_name ON experiments (name)",
    "CREATE INDEX IF NOT EXISTS idx_system_name ON systems (name)",
    "CREATE INDEX IF NOT EXISTS idx_system_type ON systems (system_type)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_system_version ON system_versions (system_id, version_number)",
    "CREATE INDEX IF NOT EXISTS idx_version_system ON system_versions (system_id)",
    "CREATE INDEX IF NOT EXISTS idx_version_created ON system_versions (created_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_system ON experimental_systems (experiment_id, system_id)",
    "CREATE INDEX IF NOT EXISTS idx_experimental_system ON experimental_systems (experiment_id, system_id)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_rewards_session ON outcome_rewards (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_rewards_total ON outcome_rewards (total_reward)",
    "CREATE INDEX IF NOT EXISTS idx_event_rewards_session ON event_rewards (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_rewards_event ON event_rewards (event_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_rewards_type ON event_rewards (reward_type)",
    'CREATE INDEX IF NOT EXISTS idx_event_rewards_key ON event_rewards ("key")',
)


class NativeLibsqlTraceManager(TraceStorage):
    """Libsql-backed trace manager."""

    def __init__(
        self,
        db_url: str | None = None,
        *,
        auth_token: str | None = None,
    ):
        self._config_auth_token = auth_token
        self._target = _resolve_connection_target(db_url, auth_token)
        self._conn: LibsqlConnection | None = None
        self._conn_lock = asyncio.Lock()
        self._op_lock = asyncio.Lock()
        self._initialized = False

    def _open_connection(self) -> LibsqlConnection:
        """Open a libsql connection for the resolved target."""
        kwargs: dict[str, Any] = {}
        if self._target.sync_url and self._target.sync_url.startswith("libsql://"):
            kwargs["sync_url"] = self._target.sync_url
        if self._target.auth_token:
            kwargs["auth_token"] = self._target.auth_token
        # Disable automatic background sync; ReplicaSync drives this explicitly.
        kwargs.setdefault("sync_interval", 0)
        logger.debug("Opening libsql connection to %s", self._target.database)
        return _libsql_connect(self._target.database, **kwargs)

    async def initialize(self):
        """Initialise the backend."""
        async with self._conn_lock:
            if self._initialized:
                return

            # Fast-fail preflight: if using remote endpoint or local sqld, check health
            # Skip health check for plain SQLite files (sync_url is None)
            if self._target.sync_url:
                try:
                    parsed = urlparse(self._target.database or "")
                    # Check for local sqld: http://, https://, or libsql://
                    if parsed.scheme in ("http", "https", "libsql"):
                        host_port = parsed.netloc or ""
                        host = (host_port.split(":", 1)[0] or "").strip().lower()
                        if host in {"127.0.0.1", "localhost"} and host_port:
                            # For http:// URLs, the port should already be the HTTP API port
                            # For libsql:// URLs, we need to calculate health check port
                            if ":" in host_port:
                                port = int(host_port.split(":", 1)[1])
                                if parsed.scheme == "libsql":
                                    # libsql:// uses Hrana port, health check is on HTTP API port (Hrana + 1)
                                    health_url = f"http://{host}:{port + 1}/health"
                                else:
                                    # http:// already points to HTTP API port
                                    health_url = f"http://{host}:{port}/health"
                            else:
                                health_url = f"http://{host_port}/health"
                            try:
                                async with httpx.AsyncClient(timeout=httpx.Timeout(1.0)) as client:
                                    resp = await client.get(health_url)
                                if resp.status_code != 200:
                                    raise RuntimeError(
                                        f"Tracing backend unhealthy at {health_url} (status={resp.status_code})"
                                    )
                            except Exception as exc:  # pragma: no cover - network env dependent
                                raise RuntimeError(
                                    f"Tracing backend not reachable at {health_url}. "
                                    f"Start sqld with both ports: sqld --db-path <path> --hrana-listen-addr {host}:HRANA_PORT --http-listen-addr {host}:HTTP_PORT "
                                    f"or disable tracing (TASKAPP_TRACING_ENABLED=0)."
                                ) from exc
                except Exception:
                    # Propagate any preflight failure to abort early
                    raise

            # Establish a libsql connection for future native operations.
            self._conn = self._open_connection()
            self._ensure_schema()
            self._initialized = True

    async def close(self):
        """Close the libsql connection."""
        async with self._conn_lock:
            if self._conn:
                logger.debug("Closing libsql connection to %s", self._target.database)
                self._conn.close()
                self._conn = None
            self._initialized = False

    # ------------------------------------------------------------------
    # Delegated operations (to be swapped with native libsql versions).
    # ------------------------------------------------------------------

    async def insert_session_trace(self, trace: SessionTrace) -> str:
        await self.initialize()
        
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        _logger.info(f"[TRACE_DEBUG] insert_session_trace START: session_id={trace.session_id}, {len(trace.markov_blanket_message_history)} messages")

        session_exists = await self._session_exists(trace.session_id)
        _logger.info(f"[TRACE_DEBUG] Session exists: {session_exists}")
        
        step_id_map: dict[str, int] = {}
        
        if session_exists:
            _logger.warning(f"[TRACE_DEBUG] Session {trace.session_id} already exists, skipping events/timesteps, only updating messages!")
            # Don't return early - we need to save messages!
            # Just update metadata
            async with self._op_lock:
                conn = self._conn
                assert conn is not None
                conn.execute(
                    "UPDATE session_traces SET metadata = ? WHERE session_id = ?",
                    (_json_dumps(trace.metadata or {}), trace.session_id),
                )
                conn.commit()
            # Skip events and timesteps to ensure idempotency
        else:
            created_at = trace.created_at or datetime.now(UTC)

            async with self._op_lock:
                conn = self._conn
                assert conn is not None
                conn.execute(
                    """
                    INSERT INTO session_traces (
                        session_id,
                        created_at,
                        num_timesteps,
                        num_events,
                        num_messages,
                        metadata
                    )
                    VALUES (?, ?, 0, 0, 0, ?)
                    """,
                    (
                        trace.session_id,
                        created_at.isoformat(),
                        _json_dumps(trace.metadata or {}),
                    ),
                )
                conn.commit()
                _logger.info("[TRACE_DEBUG] Session row inserted")

            # Only insert timesteps and events if this is a new session
            for step in trace.session_time_steps:
                step_db_id = await self.ensure_timestep(
                    trace.session_id,
                    step_id=step.step_id,
                    step_index=step.step_index,
                    turn_number=step.turn_number,
                    started_at=step.timestamp,
                    completed_at=step.completed_at,
                    metadata=step.step_metadata or {},
                )
                step_id_map[step.step_id] = step_db_id

            for event in trace.event_history:
                step_ref = None
                metadata = event.metadata or {}
                if isinstance(metadata, dict):
                    step_ref = metadata.get("step_id")
                timestep_db_id = step_id_map.get(step_ref) if step_ref else None
                await self.insert_event_row(
                    trace.session_id,
                    timestep_db_id=timestep_db_id,
                    event=event,
                    metadata_override=event.metadata or {},
                )

        import logging as _logging
        _logger = _logging.getLogger(__name__)
        _logger.info(f"[TRACE_DEBUG] insert_session_trace: saving {len(trace.markov_blanket_message_history)} messages (session_exists={session_exists})")
        
        # Only insert messages if this is a new session (for idempotency)
        if not session_exists:
            for idx, msg in enumerate(trace.markov_blanket_message_history):
                metadata = dict(getattr(msg, "metadata", {}) or {})
                step_ref = metadata.get("step_id")
                content_value = msg.content
                if isinstance(msg.content, SessionMessageContent):
                    if msg.content.json_payload:
                        metadata.setdefault("json_payload", msg.content.json_payload)
                        content_value = msg.content.json_payload
                    else:
                        content_value = msg.content.as_text()
                        if msg.content.text:
                            metadata.setdefault("text", msg.content.text)
                elif not isinstance(content_value, str):
                    try:
                        content_value = json.dumps(content_value, ensure_ascii=False)
                    except (TypeError, ValueError):
                        content_value = str(content_value)

                _logger.info(f"[TRACE_DEBUG]   Message {idx+1}: type={msg.message_type}, content_len={len(str(content_value))}")
                
                try:
                    await self.insert_message_row(
                        trace.session_id,
                        timestep_db_id=step_id_map.get(step_ref) if step_ref else None,
                        message_type=msg.message_type,
                        content=content_value,
                        event_time=msg.time_record.event_time,
                        message_time=msg.time_record.message_time,
                        metadata=metadata,
                    )
                    _logger.info(f"[TRACE_DEBUG]   Message {idx+1}: saved successfully")
                except Exception as exc:
                    _logger.error(f"[TRACE_DEBUG]   Message {idx+1}: FAILED TO SAVE: {exc}", exc_info=True)
                    raise
        else:
            _logger.info("[TRACE_DEBUG] Skipping message insertion for existing session (idempotency)")

        async with self._op_lock:
            conn = self._conn
            assert conn is not None
            conn.execute(
                "UPDATE session_traces SET num_timesteps = ?, num_events = ?, num_messages = ?, metadata = ? WHERE session_id = ?",
                (
                    len(trace.session_time_steps),
                    len(trace.event_history),
                    len(trace.markov_blanket_message_history),
                    _json_dumps(trace.metadata or {}),
                    trace.session_id,
                ),
            )
            conn.commit()

        return trace.session_id

    async def get_session_trace(self, session_id: str) -> dict[str, Any] | None:
        await self.initialize()

        async with self._op_lock:
            conn = self._conn
            assert conn is not None

            session_cursor = conn.execute(
                """
                SELECT session_id,
                       created_at,
                       num_timesteps,
                       num_events,
                       num_messages,
                       metadata
                FROM session_traces
                WHERE session_id = ?
                """,
                (session_id,),
            )
            session_row = session_cursor.fetchone()
            session_cursor.close()

            if not session_row:
                return None

            session_columns = ["session_id", "created_at", "num_timesteps", "num_events", "num_messages", "metadata"]
            session_data = dict(zip(session_columns, session_row, strict=True))

            timestep_cursor = conn.execute(
                """
                SELECT step_id,
                       step_index,
                       turn_number,
                       started_at,
                       completed_at,
                       step_metadata
                FROM session_timesteps
                WHERE session_id = ?
                ORDER BY step_index ASC
                """,
                (session_id,),
            )
            timestep_rows = timestep_cursor.fetchall()
            timestep_cursor.close()

        return {
            "session_id": session_data["session_id"],
            "created_at": _maybe_datetime(session_data["created_at"]),
            "num_timesteps": session_data["num_timesteps"],
            "num_events": session_data["num_events"],
            "num_messages": session_data["num_messages"],
            "metadata": _load_json(session_data["metadata"]),
            "timesteps": [
                {
                    "step_id": row[0],
                    "step_index": row[1],
                    "turn_number": row[2],
                    "started_at": _maybe_datetime(row[3]),
                    "completed_at": _maybe_datetime(row[4]),
                    "metadata": _load_json(row[5]),
                }
                for row in timestep_rows
            ],
        }

    async def _session_exists(self, session_id: str) -> bool:
        await self.initialize()
        async with self._op_lock:
            conn = self._conn
            assert conn is not None
            cursor = conn.execute(
                "SELECT 1 FROM session_traces WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            return row is not None

    @staticmethod
    def _normalise_params(params: dict[str, Any] | None) -> dict[str, Any]:
        if not params:
            return {}
        normalised: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, datetime):
                normalised[key] = value.isoformat()
            else:
                normalised[key] = value
        return normalised

    @staticmethod
    def _prepare_query_params(query: str, params: dict[str, Any] | list[Any] | tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
        if isinstance(params, dict):
            keys: list[str] = []

            def _replace(match: re.Match[str]) -> str:
                key = match.group(1)
                keys.append(key)
                return "?"

            new_query = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", _replace, query)
            if not keys:
                raise ValueError("No named parameters found in query for provided mapping")
            values = tuple(params[key] for key in keys)
            return new_query, values
        if isinstance(params, list | tuple):
            return query, tuple(params)
        raise TypeError("Unsupported parameter type for query execution")

    def _ensure_schema(self) -> None:
        if not self._conn:
            raise RuntimeError("Connection not initialised")

        for ddl in _TABLE_DEFINITIONS:
            self._conn.execute(ddl)
        self._apply_schema_migrations()
        for ddl in _INDEX_DEFINITIONS:
            self._conn.execute(ddl)
        for view_sql in analytics_views.values():
            self._conn.execute(view_sql)
        self._conn.commit()

    def _apply_schema_migrations(self) -> None:
        """Apply forward-compatible schema changes for existing databases."""
        self._migrate_outcome_rewards()

    @staticmethod
    def _col_value(row: Any, *, index: int, key: str) -> Any:
        try:
            return row[key]  # type: ignore[index]
        except Exception:
            return row[index]

    def _migrate_outcome_rewards(self) -> None:
        conn = self._conn
        if conn is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Connection not initialised")

        cursor = conn.execute("PRAGMA table_info(outcome_rewards)")
        rows = cursor.fetchall()
        cursor.close()
        if not rows:
            return

        columns: dict[str, str] = {}
        for row in rows:
            name = str(self._col_value(row, index=1, key="name"))
            col_type = self._col_value(row, index=2, key="type")
            columns[name] = (str(col_type) if col_type is not None else "").strip().upper()

        total_reward_type = columns.get("total_reward", "")
        has_annotation = "annotation" in columns

        # If the DB was created with total_reward INTEGER, rebuild the table to
        # update the declared type and add the annotation column in one step.
        if total_reward_type == "INTEGER":
            conn.execute("BEGIN")
            try:
                conn.execute("ALTER TABLE outcome_rewards RENAME TO outcome_rewards_old")
                conn.execute(
                    """
                    CREATE TABLE outcome_rewards (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id VARCHAR NOT NULL,
                        total_reward FLOAT NOT NULL,
                        achievements_count INTEGER NOT NULL,
                        total_steps INTEGER NOT NULL,
                        created_at DATETIME NOT NULL,
                        reward_metadata TEXT,
                        annotation TEXT,
                        FOREIGN KEY(session_id) REFERENCES session_traces(session_id)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO outcome_rewards (
                        id,
                        session_id,
                        total_reward,
                        achievements_count,
                        total_steps,
                        created_at,
                        reward_metadata,
                        annotation
                    )
                    SELECT
                        id,
                        session_id,
                        CAST(total_reward AS FLOAT),
                        achievements_count,
                        total_steps,
                        created_at,
                        reward_metadata,
                        NULL
                    FROM outcome_rewards_old
                    """
                )
                conn.execute("DROP TABLE outcome_rewards_old")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return

        # Otherwise, add annotation column if missing.
        if not has_annotation:
            conn.execute("ALTER TABLE outcome_rewards ADD COLUMN annotation TEXT")

    async def query_traces(self, query: str, params: dict[str, Any] | None = None) -> Any:
        await self.initialize()

        async with self._op_lock:
            conn = self._conn
            assert conn is not None
            normalised = self._normalise_params(params)
            if normalised:
                prepared_query, prepared_params = self._prepare_query_params(query, normalised)
                cursor = conn.execute(prepared_query, prepared_params)
            else:
                cursor = conn.execute(query)
            try:
                description = cursor.description or []
                columns = [col[0] for col in description]
                rows = cursor.fetchall()
            finally:
                cursor.close()

        if not rows:
            if pd is not None:
                return pd.DataFrame(columns=list(columns))  # type: ignore[arg-type]
            return []

        records = [dict(zip(columns, row, strict=True)) for row in rows]
        if pd is not None:
            return pd.DataFrame(records)
        return records

    async def get_model_usage(
        self,
        start_date=None,
        end_date=None,
        model_name=None,
    ) -> Any:
        query = """
            SELECT * FROM model_usage_stats
            WHERE 1=1
        """
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
            query += " AND last_used >= :start_date"
        if end_date:
            params["end_date"] = end_date
            query += " AND first_used <= :end_date"
        if model_name:
            params["model_name"] = model_name
            query += " AND model_name = :model_name"
        query += " ORDER BY usage_count DESC"
        return await self.query_traces(query, params)

    async def delete_session(self, session_id: str) -> bool:
        await self.initialize()

        async with self._op_lock:
            conn = self._conn
            assert conn is not None

            cursor = conn.execute(
                "SELECT 1 FROM session_traces WHERE session_id = ?", (session_id,)
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            if not exists:
                return False

            conn.execute("DELETE FROM event_rewards WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM outcome_rewards WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_timesteps WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_traces WHERE session_id = ?", (session_id,))
            conn.commit()
            return True

    # Experiment helpers -------------------------------------------------
    async def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> str:
        await self.initialize()

        async with self._op_lock:
            conn = self._conn
            assert conn is not None
            conn.execute(
                """
                INSERT INTO experiments (experiment_id, name, description, configuration)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    configuration = excluded.configuration
                """,
                (
                    experiment_id,
                    name,
                    description,
                    _json_dumps(configuration or {}),
                ),
            )
            conn.commit()
        return experiment_id

    async def link_session_to_experiment(self, session_id: str, experiment_id: str):
        await self.initialize()

        async with self._op_lock:
            conn = self._conn
            assert conn is not None
            conn.execute(
                "UPDATE session_traces SET experiment_id = ? WHERE session_id = ?",
                (experiment_id, session_id),
            )
            conn.commit()

    async def get_sessions_by_experiment(
        self, experiment_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        await self.initialize()

        sql = """
            SELECT session_id,
                   created_at,
                   num_timesteps,
                   num_events,
                   num_messages,
                   metadata
            FROM session_traces
            WHERE experiment_id = ?
            ORDER BY created_at DESC
        """
        params: list[Any] = [experiment_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        async with self._op_lock:
            conn = self._conn
            assert conn is not None
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            cursor.close()

        return [
            {
                "session_id": row[0],
                "created_at": _maybe_datetime(row[1]),
                "num_timesteps": row[2],
                "num_events": row[3],
                "num_messages": row[4],
                "metadata": _load_json(row[5]),
            }
            for row in rows
        ]

    async def batch_insert_sessions(
        self, traces: list[SessionTrace], batch_size: int | None = None
    ) -> list[str]:
        batch_size = batch_size or CONFIG.batch_size
        inserted: list[str] = []

        for i in range(0, len(traces), batch_size):
            chunk = traces[i : i + batch_size]
            for trace in chunk:
                session_id = await self.insert_session_trace(trace)
                inserted.append(session_id)
        return inserted

    # Incremental helpers -----------------------------------------------
    async def ensure_session(
        self,
        session_id: str,
        *,
        created_at=None,
        metadata=None,
    ) -> None:
        await self.initialize()

        created_at_val = (created_at or datetime.now(UTC)).isoformat()
        metadata_json = _json_dumps(metadata or {})

        async with self._op_lock:
            conn = self._conn

            assert conn is not None
            conn.execute(
                """
                INSERT INTO session_traces (
                    session_id, created_at, num_timesteps, num_events, num_messages, metadata
                )
                VALUES (?, ?, 0, 0, 0, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, created_at_val, metadata_json),
            )
            conn.commit()

    async def ensure_timestep(
        self,
        session_id: str,
        *,
        step_id: str,
        step_index: int,
        turn_number: int | None = None,
        started_at=None,
        completed_at=None,
        metadata=None,
    ) -> int:
        await self.initialize()

        started_at_val = (started_at or datetime.now(UTC)).isoformat()
        completed_at_val = completed_at.isoformat() if completed_at else None
        metadata_json = _json_dumps(metadata or {})

        async with self._op_lock:
            conn = self._conn

            assert conn is not None
            cur = conn.execute(
                """
                SELECT id FROM session_timesteps
                WHERE session_id = ? AND step_id = ?
                """,
                (session_id, step_id),
            )
            row = cur.fetchone()
            if row:
                return int(row[0])

            cur = conn.execute(
                """
                INSERT INTO session_timesteps (
                    session_id,
                    step_id,
                    step_index,
                    turn_number,
                    started_at,
                    completed_at,
                    num_events,
                    num_messages,
                    step_metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?)
                """,
                (
                    session_id,
                    step_id,
                    step_index,
                    turn_number,
                    started_at_val,
                    completed_at_val,
                    metadata_json,
                ),
            )
            timestep_id = int(cur.lastrowid or 0)
            conn.execute(
                """
                UPDATE session_traces
                SET num_timesteps = num_timesteps + 1
                WHERE session_id = ?
                """,
                (session_id,),
            )
            conn.commit()
            return timestep_id

    async def insert_event_row(
        self,
        session_id: str,
        *,
        timestep_db_id: int | None,
        event: Any,
        metadata_override: dict[str, Any] | None = None,
    ) -> int:
        await self.initialize()

        if not isinstance(event, EnvironmentEvent | LMCAISEvent | RuntimeEvent):
            raise TypeError(f"Unsupported event type for native manager: {type(event)!r}")

        metadata_json = metadata_override or event.metadata or {}
        event_extra_metadata = getattr(event, "event_metadata", None)
        system_state_before = getattr(event, "system_state_before", None)
        system_state_after = getattr(event, "system_state_after", None)

        payload: dict[str, Any] = {
            "session_id": session_id,
            "timestep_id": timestep_db_id,
            "system_instance_id": event.system_instance_id,
            "event_time": event.time_record.event_time,
            "message_time": event.time_record.message_time,
            "metadata": metadata_json,
            "event_metadata": event_extra_metadata,
            "system_state_before": system_state_before,
            "system_state_after": system_state_after,
        }

        if isinstance(event, LMCAISEvent):
            call_records = None
            if getattr(event, "call_records", None):
                # Handle both dataclass instances and dicts (from deserialization)
                call_records = [
                    asdict(record) if not isinstance(record, dict) else record
                    for record in event.call_records
                ]
            payload.update(
                {
                    "event_type": "cais",
                    "model_name": event.model_name,
                    "provider": event.provider,
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                    "total_tokens": event.total_tokens,
                    "cost_usd": int(event.cost_usd * 100) if event.cost_usd is not None else None,
                    "latency_ms": event.latency_ms,
                    "span_id": event.span_id,
                    "trace_id": event.trace_id,
                    "call_records": call_records,
                }
            )
        elif isinstance(event, EnvironmentEvent):
            payload.update(
                {
                    "event_type": "environment",
                    "reward": event.reward,
                    "terminated": event.terminated,
                    "truncated": event.truncated,
                }
            )
        elif isinstance(event, RuntimeEvent):
            payload.update(
                {
                    "event_type": "runtime",
                    "metadata": {**(event.metadata or {}), "actions": event.actions},
                }
            )

        async with self._op_lock:
            conn = self._conn

            assert conn is not None
            cur = conn.execute(
                """
                INSERT INTO events (
                    session_id,
                    timestep_id,
                    event_type,
                    system_instance_id,
                    event_time,
                    message_time,
                    model_name,
                    provider,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cost_usd,
                    latency_ms,
                    span_id,
                    trace_id,
                    call_records,
                    reward,
                    terminated,
                    truncated,
                    system_state_before,
                    system_state_after,
                    metadata,
                    event_metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["session_id"],
                    payload["timestep_id"],
                    payload.get("event_type"),
                    payload["system_instance_id"],
                    payload["event_time"],
                    payload["message_time"],
                    payload.get("model_name"),
                    payload.get("provider"),
                    payload.get("input_tokens"),
                    payload.get("output_tokens"),
                    payload.get("total_tokens"),
                    payload.get("cost_usd"),
                    payload.get("latency_ms"),
                    payload.get("span_id"),
                    payload.get("trace_id"),
                    _json_dumps(payload.get("call_records")),
                    payload.get("reward"),
                    payload.get("terminated"),
                    payload.get("truncated"),
                    _json_dumps(payload.get("system_state_before")),
                    _json_dumps(payload.get("system_state_after")),
                    _json_dumps(payload.get("metadata")),
                    _json_dumps(payload.get("event_metadata")),
                ),
            )
            event_id = int(cur.lastrowid or 0)
            conn.execute(
                """
                UPDATE session_traces
                SET num_events = num_events + 1
                WHERE session_id = ?
                """,
                (session_id,),
            )
            if timestep_db_id is not None:
                conn.execute(
                    """
                    UPDATE session_timesteps
                    SET num_events = num_events + 1
                    WHERE id = ?
                    """,
                    (timestep_db_id,),
                )
            conn.commit()
            return event_id

    async def insert_message_row(
        self,
        session_id: str,
        *,
        timestep_db_id: int | None,
        message_type: str,
        content: Any,
        event_time: float | None = None,
        message_time: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        await self.initialize()

        metadata_payload = dict(metadata or {})
        if isinstance(content, SessionMessageContent):
            if content.json_payload:
                metadata_payload.setdefault("json_payload", content.json_payload)
                content_value = content.json_payload
            else:
                content_value = content.as_text()
                if content.text:
                    metadata_payload.setdefault("text", content.text)
        else:
            content_value = content
            if not isinstance(content_value, str):
                try:
                    content_value = json.dumps(content_value, ensure_ascii=False)
                except (TypeError, ValueError):
                    content_value = str(content_value)

        async with self._op_lock:
            conn = self._conn

            assert conn is not None
            cur = conn.execute(
                """
                INSERT INTO messages (
                    session_id,
                    timestep_id,
                    message_type,
                    content,
                    event_time,
                    message_time,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    timestep_db_id,
                    message_type,
                    content_value,
                    event_time,
                    message_time,
                    _json_dumps(metadata_payload),
                ),
            )
            message_id = int(cur.lastrowid or 0)
            conn.execute(
                """
                UPDATE session_traces
                SET num_messages = num_messages + 1
                WHERE session_id = ?
                """,
                (session_id,),
            )
            if timestep_db_id is not None:
                conn.execute(
                    """
                    UPDATE session_timesteps
                    SET num_messages = num_messages + 1
                    WHERE id = ?
                    """,
                    (timestep_db_id,),
                )
            conn.commit()
            return message_id

    async def insert_outcome_reward(
        self,
        session_id: str,
        *,
        total_reward: float,
        achievements_count: int,
        total_steps: int,
        reward_metadata: dict | None = None,
        annotation: dict[str, Any] | None = None,
    ) -> int:
        await self.initialize()

        async with self._op_lock:
            conn = self._conn

            assert conn is not None
            cur = conn.execute(
                """
                INSERT INTO outcome_rewards (
                    session_id,
                    total_reward,
                    achievements_count,
                    total_steps,
                    created_at,
                    reward_metadata,
                    annotation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    total_reward,
                    achievements_count,
                    total_steps,
                    datetime.now(UTC).isoformat(),
                    _json_dumps(reward_metadata),
                    _json_dumps(annotation),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    async def insert_event_reward(
        self,
        session_id: str,
        *,
        event_id: int,
        message_id: int | None = None,
        turn_number: int | None = None,
        reward_value: float = 0.0,
        reward_type: str | None = None,
        key: str | None = None,
        annotation: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> int:
        await self.initialize()

        async with self._op_lock:
            conn = self._conn

            assert conn is not None
            cur = conn.execute(
                """
                INSERT INTO event_rewards (
                    event_id,
                    session_id,
                    message_id,
                    turn_number,
                    reward_value,
                    reward_type,
                    key,
                    annotation,
                    source,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    message_id,
                    turn_number,
                    reward_value,
                    reward_type,
                    key,
                    _json_dumps(annotation),
                    source,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
