"""Factory for creating storage instances."""

from ..turso.native_manager import NativeLibsqlTraceManager
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

    connection_string = config.get_connection_string()

    if config.backend == StorageBackend.TURSO_NATIVE:
        backend_config = config.get_backend_config()
        return NativeLibsqlTraceManager(
            db_url=connection_string,
            auth_token=backend_config.get("auth_token"),
        )
    elif config.backend == StorageBackend.SQLITE:
        return NativeLibsqlTraceManager(db_url=connection_string)
    elif config.backend == StorageBackend.POSTGRES:
        # Future: PostgreSQL implementation
        raise NotImplementedError("PostgreSQL backend not yet implemented")
    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")
