from __future__ import annotations

from contextlib import asynccontextmanager

from synth_ai.tracing_v3.storage.factory import StorageConfig, create_storage


@asynccontextmanager
async def open_storage(db_url: str):
    """Async context manager that yields an initialized tracing storage."""

    storage = create_storage(StorageConfig(connection_string=db_url))
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.close()


def storage_from_sqlite_path(path: str):
    """Return an uninitialized storage manager for a sqlite path."""

    return create_storage(StorageConfig(connection_string=f"sqlite+aiosqlite:///{path}"))


__all__ = ["open_storage", "storage_from_sqlite_path"]
