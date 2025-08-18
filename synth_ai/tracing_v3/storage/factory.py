"""Factory for creating storage instances."""


from ..turso.manager import AsyncSQLTraceManager
from .base import TraceStorage
from .config import StorageBackend, StorageConfig


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

    if config.backend == StorageBackend.TURSO:
        # Turso uses the AsyncSQLTraceManager
        return AsyncSQLTraceManager(db_url=config.get_connection_string())
    elif config.backend == StorageBackend.SQLITE:
        # For pure SQLite, we can still use AsyncSQLTraceManager
        # but with a file-based URL
        return AsyncSQLTraceManager(db_url=config.get_connection_string())
    elif config.backend == StorageBackend.POSTGRES:
        # Future: PostgreSQL implementation
        raise NotImplementedError("PostgreSQL backend not yet implemented")
    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")
