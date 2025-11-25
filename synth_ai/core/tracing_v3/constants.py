from __future__ import annotations

from datetime import datetime
from pathlib import Path

TRACE_DB_DIR = Path("traces")
TRACE_DB_BASENAME = "turso_task_app_traces"


def canonical_trace_db_name(*, timestamp: datetime | None = None) -> str:
    """Return the canonical trace database filename (with optional timestamp suffix)."""

    if timestamp is None:
        return f"{TRACE_DB_BASENAME}.db"
    return f"{TRACE_DB_BASENAME}_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.db"


def canonical_trace_db_path(*, timestamp: datetime | None = None) -> Path:
    """Return the canonical trace database path within the default trace directory."""

    return TRACE_DB_DIR / canonical_trace_db_name(timestamp=timestamp)
