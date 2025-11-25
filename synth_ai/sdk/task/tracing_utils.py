"""Utilities for wiring tracing_v3 into task apps."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from synth_ai.core.tracing_v3.constants import TRACE_DB_DIR, canonical_trace_db_name


def tracing_env_enabled(default: bool = False) -> bool:
    """Return True when tracing is enabled for task apps via environment variable."""

    raw = os.getenv("TASKAPP_TRACING_ENABLED")
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def resolve_tracing_db_url() -> str | None:
    """Resolve tracing database URL using centralized tracing_v3 config logic.
    
    This delegates to synth_ai.core.tracing_v3.config.resolve_trace_db_settings() which
    handles Modal detection, remote Turso, local sqld, and SQLite fallbacks.
    """
    try:
        from synth_ai.core.tracing_v3.config import resolve_trace_db_settings
        db_url, _ = resolve_trace_db_settings(ensure_dir=True)
        return db_url
    except ImportError:
        # Fallback if tracing_v3 is not available (shouldn't happen in normal usage)
        db_url = (
            os.getenv("TURSO_LOCAL_DB_URL")
            or os.getenv("LIBSQL_URL")
            or os.getenv("SYNTH_TRACES_DB")
        )
        if db_url:
            return db_url
        
        # Auto-provision local sqld location for callers that rely on trace directories.
        base_dir = TRACE_DB_DIR.expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        candidate = base_dir / canonical_trace_db_name(timestamp=datetime.now())
        os.environ["TASKAPP_TRACE_DB_PATH"] = str(candidate)
        os.environ.setdefault("SQLD_DB_PATH", str(candidate))
        
        default_url = os.getenv("LIBSQL_DEFAULT_URL", "http://127.0.0.1:8081")
        return default_url


def build_tracer_factory(
    make_tracer: Callable[..., Any], *, enabled: bool, db_url: str | None
) -> Callable[[], Any] | None:
    """Return a factory that instantiates a tracer when enabled, else None."""

    if not enabled:
        return None

    def _factory() -> Any:
        return make_tracer(db_url=db_url) if db_url else make_tracer()

    return _factory


def resolve_sft_output_dir() -> str | None:
    """Resolve location for writing SFT records, creating directory if requested."""

    raw = os.getenv("TASKAPP_SFT_OUTPUT_DIR") or os.getenv("SFT_OUTPUT_DIR")
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return str(path)


def unique_sft_path(base_dir: str, *, run_id: str) -> Path:
    """Return a unique JSONL path for an SFT record batch."""

    from datetime import datetime

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{run_id}_{timestamp}.jsonl"
    return Path(base_dir) / name
