"""Factory for creating storage instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TraceStorage
from .config import StorageBackend, StorageConfig

if TYPE_CHECKING:
    pass


def create_storage(config: StorageConfig | None = None) -> TraceStorage:
    """Create a storage instance based on configuration.

    Args:
        config: Storage configuration (uses default if not provided)

    Returns:
        A TraceStorage implementation

    Raises:
        ValueError: If the backend is not supported
    """
    if config is None:
        from .config import STORAGE_CONFIG

        config = STORAGE_CONFIG

    connection_string = config.get_connection_string()

    if config.backend == StorageBackend.SQLITE:
        # Lazy import to avoid circular dependency
        from ..turso.native_manager import SQLiteTraceManager

        return SQLiteTraceManager(db_url=connection_string)
    elif config.backend == StorageBackend.POSTGRES:
        # Future: PostgreSQL implementation
        raise NotImplementedError("PostgreSQL backend not yet implemented")
    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")
