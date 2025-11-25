"""Configuration helpers for tracing v3.

This module centralises the logic for discovering which datastore the tracer
should use. Historically the project defaulted to a local SQLite file which
breaks under parallel load. The new resolver inspects environment variables
and defaults to Turso/libSQL whenever credentials are supplied, while keeping a
SQLite fallback for contributors without remote access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from synth_ai.core.tracing_v3.constants import canonical_trace_db_path

# STARTUP DIAGNOSTIC - Commented out to reduce noise
# print(f"[TRACING_V3_CONFIG_LOADED] Python={sys.version_info.major}.{sys.version_info.minor} MODAL_IS_REMOTE={os.getenv('MODAL_IS_REMOTE')}", flush=True)

# ---------------------------------------------------------------------------
# DSN resolution helpers
# ---------------------------------------------------------------------------

_CANONICAL_DB_PATH = canonical_trace_db_path()
_DEFAULT_TRACE_DIR = Path(os.getenv("SYNTH_TRACES_DIR", _CANONICAL_DB_PATH.parent))


def _normalise_path(path: Path) -> Path:
    """Resolve relative paths and expand user/home markers."""
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _is_modal_environment() -> bool:
    """Detect if running in Modal container.
    
    Modal automatically sets MODAL_IS_REMOTE=1 in all deployed containers.
    We check this first, then fall back to other Modal env vars.
    """
    # Modal sets this in all deployed containers
    if os.getenv("MODAL_IS_REMOTE") == "1":
        return True
    
    # Additional Modal env vars as fallback
    return bool(
        os.getenv("MODAL_TASK_ID")
        or os.getenv("MODAL_ENVIRONMENT")
        or os.getenv("SERVICE", "").upper() == "MODAL"
    )


def _split_auth_from_url(url: str) -> tuple[str, str | None]:
    """Strip any auth_token query parameter from a DSN."""
    parsed = urlparse(url)
    if not parsed.query:
        return url, None

    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    token = params.pop("auth_token", None)
    query = urlencode(params, doseq=True)
    # urlunparse will omit the '?' automatically when query is empty
    sanitised = urlunparse(parsed._replace(query=query))
    return sanitised, token


def _default_sqlite_url(*, ensure_dir: bool = True) -> tuple[str, str | None]:
    """Generate a SQLite URL from SYNTH_TRACES_DIR if set, otherwise raise."""
    traces_dir = os.getenv("SYNTH_TRACES_DIR")
    if traces_dir:
        dir_path = _normalise_path(Path(traces_dir))
        if ensure_dir:
            dir_path.mkdir(parents=True, exist_ok=True)
        db_path = dir_path / "synth_traces.db"
        sqlite_url = f"sqlite+aiosqlite:///{db_path}"
        return sqlite_url, None
    raise RuntimeError("SQLite fallback is disabled; configure LIBSQL_URL or run sqld locally.")


def resolve_trace_db_settings(*, ensure_dir: bool = True) -> tuple[str, str | None]:
    """Resolve the tracing database URL and optional auth token.

    Resolution order:
      1. `SYNTH_TRACES_DB` (explicit DSN override)
      2. `LIBSQL_URL` / `TURSO_DATABASE_URL` (remote libSQL endpoints)
      3. `TURSO_LOCAL_DB_URL` (legacy env for local sqld)
      4. Modal environment: plain SQLite file (no sqld, no auth)
      5. Local dev: sqld default
    """
    import logging
    logger = logging.getLogger(__name__)

    explicit = os.getenv("SYNTH_TRACES_DB")
    if explicit:
        logger.info(f"[TRACE_CONFIG] Using explicit SYNTH_TRACES_DB: {explicit}")
        return _split_auth_from_url(explicit)

    remote = os.getenv("LIBSQL_URL") or os.getenv("TURSO_DATABASE_URL")
    if remote:
        logger.info(f"[TRACE_CONFIG] Using remote Turso: {remote}")
        url, token = _split_auth_from_url(remote)
        if token:
            return url, token
        env_token = os.getenv("LIBSQL_AUTH_TOKEN") or os.getenv("TURSO_AUTH_TOKEN")
        return url, env_token

    local_override = os.getenv("TURSO_LOCAL_DB_URL")
    if local_override:
        logger.info(f"[TRACE_CONFIG] Using TURSO_LOCAL_DB_URL: {local_override}")
        url, token = _split_auth_from_url(local_override)
        if token:
            return url, token
        env_token = os.getenv("LIBSQL_AUTH_TOKEN") or os.getenv("TURSO_AUTH_TOKEN")
        return url, env_token

    # Check for SYNTH_TRACES_DIR to generate SQLite URL
    traces_dir = os.getenv("SYNTH_TRACES_DIR")
    if traces_dir:
        try:
            sqlite_url, _ = _default_sqlite_url(ensure_dir=ensure_dir)
            logger.info(f"[TRACE_CONFIG] Using SQLite from SYNTH_TRACES_DIR: {sqlite_url}")
            return sqlite_url, None
        except RuntimeError:
            pass  # Fall through to other options

    # Modal environment: use plain SQLite file (no sqld daemon, no auth required)
    is_modal = _is_modal_environment()
    logger.info(f"[TRACE_CONFIG] Modal detection: {is_modal} (MODAL_IS_REMOTE={os.getenv('MODAL_IS_REMOTE')})")
    if is_modal:
        logger.info("[TRACE_CONFIG] Using Modal SQLite: file:/tmp/synth_traces.db")
        return "file:/tmp/synth_traces.db", None

    # Local dev: default to sqld HTTP API
    default_url = os.getenv("LIBSQL_DEFAULT_URL", "http://127.0.0.1:8081")
    logger.info(f"[TRACE_CONFIG] Using local sqld: {default_url}")
    return default_url, None


def resolve_trace_db_url(*, ensure_dir: bool = True) -> str:
    """Return just the DSN, discarding any auth token."""
    url, _ = resolve_trace_db_settings(ensure_dir=ensure_dir)
    return url


def resolve_trace_db_auth_token() -> str | None:
    """Return the resolved auth token for the tracing datastore."""
    _, token = resolve_trace_db_settings()
    return token


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

DEFAULT_DB_FILE = str(_normalise_path(_DEFAULT_TRACE_DIR) / _CANONICAL_DB_PATH.name)


@dataclass
class TursoConfig:
    """Configuration for Turso/sqld connection."""

    # Default values matching serve.sh
    DEFAULT_DB_FILE = DEFAULT_DB_FILE
    DEFAULT_HTTP_PORT = 8080

    # Resolve DB URL and auth token from environment (libSQL preferred)
    db_url: str = field(default_factory=resolve_trace_db_url)

    # Remote database sync configuration
    sync_url: str = os.getenv("LIBSQL_SYNC_URL") or os.getenv("TURSO_SYNC_URL", "")
    auth_token: str = resolve_trace_db_auth_token() or ""
    sync_interval: int = int(
        os.getenv("TURSO_SYNC_SECONDS", "2")
    )  # 2 seconds for responsive local development

    # Connection pool settings
    pool_size: int = int(os.getenv("TURSO_POOL_SIZE", "8"))
    max_overflow: int = int(os.getenv("TURSO_MAX_OVERFLOW", "16"))
    pool_timeout: float = float(os.getenv("TURSO_POOL_TIMEOUT", "30.0"))
    pool_recycle: int = int(os.getenv("TURSO_POOL_RECYCLE", "3600"))

    # SQLite settings
    foreign_keys: bool = os.getenv("TURSO_FOREIGN_KEYS", "true").lower() == "true"
    journal_mode: str = os.getenv("TURSO_JOURNAL_MODE", "WAL")

    # Performance settings
    echo_sql: bool = os.getenv("TURSO_ECHO_SQL", "false").lower() == "true"
    batch_size: int = int(os.getenv("TURSO_BATCH_SIZE", "1000"))

    # Daemon settings (for local sqld) - match serve.sh defaults
    sqld_binary: str = os.getenv("SQLD_BINARY", "sqld")
    sqld_db_path: str = os.getenv("SQLD_DB_PATH", DEFAULT_DB_FILE)
    sqld_http_port: int = int(os.getenv("SQLD_HTTP_PORT", "8080"))
    sqld_idle_shutdown: int = int(os.getenv("SQLD_IDLE_SHUTDOWN", "0"))  # 0 = no idle shutdown

    def get_connect_args(self) -> dict[str, str]:
        """Get SQLAlchemy connection arguments."""
        args: dict[str, str] = {}
        if self.auth_token:
            args["auth_token"] = self.auth_token
        return args

    def get_engine_kwargs(self) -> dict[str, Any]:
        """Get SQLAlchemy engine creation kwargs."""
        kwargs: dict[str, Any] = {
            "echo": self.echo_sql,
            "future": True,
        }

        # Only add pool settings for non-SQLite URLs
        if not (self.db_url.startswith("sqlite") or ":memory:" in self.db_url):
            kwargs.update(
                {
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout,
                    "pool_recycle": self.pool_recycle,
                }
            )

        return kwargs


# Global config instance
CONFIG = TursoConfig()
