"""Factory for creating trace storage instances."""
from typing import Optional
from .base import TraceStorage
from .config import TraceStorageConfig, get_config


def create_storage(backend: Optional[str] = None, config: Optional[TraceStorageConfig] = None) -> TraceStorage:
    """Create a trace storage instance based on configuration.
    
    Args:
        backend: Override backend type from config
        config: Configuration to use (defaults to global config)
        
    Returns:
        TraceStorage instance
        
    Raises:
        ValueError: If backend is not supported
    """
    if config is None:
        config = get_config()
    
    backend_type = backend or config.backend
    
    if backend_type == "duckdb":
        from ..duckdb.manager import DuckDBTraceManager
        return DuckDBTraceManager(config=config.duckdb)
    else:
        raise ValueError(f"Unsupported storage backend: {backend_type}")


def get_default_storage() -> TraceStorage:
    """Get the default storage instance based on current configuration."""
    return create_storage()