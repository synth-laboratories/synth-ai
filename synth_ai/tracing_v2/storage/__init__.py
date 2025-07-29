"""Storage abstraction layer for tracing_v2."""
from .base import TraceStorage
from .config import TraceStorageConfig, DuckDBConfig, get_config, set_config, configure
from .factory import create_storage, get_default_storage
from .types import EventType, MessageType, Provider
from .exceptions import (
    TraceStorageError, SessionNotFoundError, SessionAlreadyExistsError,
    DatabaseConnectionError, SchemaInitializationError, QueryExecutionError,
    DataValidationError
)

__all__ = [
    "TraceStorage",
    "TraceStorageConfig", 
    "DuckDBConfig",
    "get_config",
    "set_config", 
    "configure",
    "create_storage",
    "get_default_storage",
    "EventType",
    "MessageType",
    "Provider",
    "TraceStorageError",
    "SessionNotFoundError",
    "SessionAlreadyExistsError",
    "DatabaseConnectionError",
    "SchemaInitializationError",
    "QueryExecutionError",
    "DataValidationError"
]