"""Configuration helpers for the experiment queue service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_DB_PATH = Path("traces") / "v3" / "synth_ai.db"


def _candidate_db_paths() -> list[Path]:
    """Build candidate paths using env overrides."""
    env_vars = (
        "EXPERIMENT_QUEUE_DB_PATH",
        "EXPERIMENT_QUEUE_SQLITE_PATH",
        "SYNTH_AI_EXPERIMENT_DB_PATH",
        "SQLD_DB_PATH",
    )
    candidates: list[Path] = []
    for name in env_vars:
        value = os.getenv(name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.append(DEFAULT_DB_PATH)
    return candidates


def _resolve_sqlite_file(base_path: Path) -> Path:
    """
    Resolve the actual SQLite file path managed by sqld.

    sqld typically creates `{db_path}/dbs/default/data`. If the directory
    does not exist yet (e.g., sqld not started), fall back to the provided path.
    """
    base_path = base_path.expanduser()
    if base_path.is_dir():
        sqld_candidate = (base_path / "dbs" / "default" / "data").resolve()
        if sqld_candidate.exists():
            return sqld_candidate

    resolved = base_path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@dataclass(frozen=True, slots=True)
class ExperimentQueueConfig:
    """Resolved configuration for the experiment queue runtime."""

    sqlite_path: Path
    broker_url: str
    result_backend_url: str

    @property
    def sqlalchemy_url(self) -> str:
        """Return SQLAlchemy connection string for ORM usage."""
        return f"sqlite:///{self.sqlite_path}"


@lru_cache(maxsize=1)
def load_config() -> ExperimentQueueConfig:
    """Resolve configuration once per process."""
    sqlite_path: Path | None = None
    for candidate in _candidate_db_paths():
        sqlite_path = _resolve_sqlite_file(candidate)
        if sqlite_path:
            break

    if sqlite_path is None:
        sqlite_path = _resolve_sqlite_file(DEFAULT_DB_PATH)

    # Redis broker and backend (no longer using SQLite for Celery)
    broker_url = os.getenv("EXPERIMENT_QUEUE_BROKER_URL", "redis://localhost:6379/0")
    backend_url = os.getenv("EXPERIMENT_QUEUE_RESULT_BACKEND_URL", "redis://localhost:6379/1")

    return ExperimentQueueConfig(
        sqlite_path=sqlite_path,
        broker_url=broker_url,
        result_backend_url=backend_url,
    )


def reset_config_cache() -> None:
    """Clear the cached config to force reload from environment variables.
    
    Call this before importing/using the Celery app if EXPERIMENT_QUEUE_DB_PATH
    or EXPERIMENT_QUEUE_TRAIN_CMD environment variables have changed.
    """
    load_config.cache_clear()
