"""Configuration helpers for tracing v3 (SQLite-only)."""

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


def _strip_auth_from_url(url: str) -> tuple[str, str | None]:
    """Strip auth_token query parameter from a DSN."""
    parsed = urlparse(url)
    if not parsed.query:
        return url, None
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    token = params.pop("auth_token", None)
    query = urlencode(params, doseq=True)
    sanitised = urlunparse(parsed._replace(query=query))
    return sanitised, token


def _default_sqlite_url(*, ensure_dir: bool = True) -> tuple[str, str | None]:
    """Generate a SQLite URL from SYNTH_TRACES_DIR if set."""
    traces_dir = os.getenv("SYNTH_TRACES_DIR")
    if traces_dir:
        dir_path = _normalise_path(Path(traces_dir))
        if ensure_dir:
            dir_path.mkdir(parents=True, exist_ok=True)
        db_path = dir_path / "synth_traces.db"
        sqlite_url = f"sqlite+aiosqlite:///{db_path}"
        return sqlite_url, None
    raise RuntimeError("SQLite fallback is disabled; set SYNTH_TRACES_DIR or SYNTH_TRACES_DB.")


def resolve_trace_db_settings(*, ensure_dir: bool = True) -> tuple[str, str | None]:
    """Resolve the tracing database URL (SQLite-only)."""
    import logging

    logger = logging.getLogger(__name__)

    explicit = os.getenv("SYNTH_TRACES_DB")
    if explicit:
        logger.info(f"[TRACE_CONFIG] Using explicit SYNTH_TRACES_DB: {explicit}")
        url, token = _strip_auth_from_url(explicit)
        return url, token

    traces_dir = os.getenv("SYNTH_TRACES_DIR")
    if traces_dir:
        sqlite_url, _ = _default_sqlite_url(ensure_dir=ensure_dir)
        logger.info(f"[TRACE_CONFIG] Using SQLite from SYNTH_TRACES_DIR: {sqlite_url}")
        return sqlite_url, None

    is_modal = _is_modal_environment()
    logger.info(
        f"[TRACE_CONFIG] Modal detection: {is_modal} (MODAL_IS_REMOTE={os.getenv('MODAL_IS_REMOTE')})"
    )
    if is_modal:
        logger.info("[TRACE_CONFIG] Using Modal SQLite: file:/tmp/synth_traces.db")
        return "file:/tmp/synth_traces.db", None

    # Default to a local sqlite file under the canonical traces dir.
    default_path = _normalise_path(_DEFAULT_TRACE_DIR / _CANONICAL_DB_PATH.name)
    if ensure_dir:
        default_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_url = f"sqlite+aiosqlite:///{default_path}"
    logger.info(f"[TRACE_CONFIG] Using default SQLite: {sqlite_url}")
    return sqlite_url, None


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
    """Configuration for tracing (SQLite-only)."""

    DEFAULT_DB_FILE = DEFAULT_DB_FILE

    # SQLite connection URL
    db_url: str = field(default_factory=resolve_trace_db_url)

    # Performance settings
    echo_sql: bool = os.getenv("TRACE_ECHO_SQL", "false").lower() == "true"
    batch_size: int = int(os.getenv("TRACE_BATCH_SIZE", "1000"))

    def get_connect_args(self) -> dict[str, str]:
        """Get SQLAlchemy connection arguments (none for SQLite)."""
        return {}

    def get_engine_kwargs(self) -> dict[str, Any]:
        """Get SQLAlchemy engine creation kwargs."""
        return {"echo": self.echo_sql, "future": True}


# Global config instance
CONFIG = TursoConfig()
