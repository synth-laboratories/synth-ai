"""Utilities for wiring tracing_v3 into task apps."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


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
    """Resolve tracing database URL and prefer async drivers for SQLite."""

    db_url = os.getenv("TURSO_LOCAL_DB_URL")
    if db_url:
        return db_url

    sqld_path = os.getenv("SQLD_DB_PATH")
    if sqld_path:
        path = Path(sqld_path).expanduser()
        if path.is_dir():
            candidate = path / "dbs" / "default" / "data"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{candidate}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{path}"

    fallback_path = Path("traces/v3/synth_ai.db").expanduser()
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{fallback_path}"


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
