"""Storage configuration for tracing v3."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import resolve_trace_db_auth_token, resolve_trace_db_settings


class StorageBackend(str, Enum):
    """Supported storage backends."""

    TURSO_NATIVE = "turso_native"
    SQLITE = "sqlite"
    POSTGRES = "postgres"  # Future support


def _is_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class StorageConfig:
    """Configuration for storage backend."""

    connection_string: str | None = None
    backend: StorageBackend | None = None
    turso_auth_token: str | None = field(default=None)

    # Common settings
    pool_size: int = int(os.getenv("STORAGE_POOL_SIZE", "8"))
    echo_sql: bool = os.getenv("STORAGE_ECHO_SQL", "false").lower() == "true"
    batch_size: int = int(os.getenv("STORAGE_BATCH_SIZE", "1000"))

    # Performance settings
    enable_compression: bool = os.getenv("STORAGE_COMPRESSION", "false").lower() == "true"
    max_content_length: int = int(os.getenv("STORAGE_MAX_CONTENT_LENGTH", "1000000"))  # 1MB

    def __post_init__(self):
        # Allow legacy override while keeping compatibility with existing TURSO_NATIVE env flag
        native_env = os.getenv("TURSO_NATIVE")
        native_flag = _is_enabled(native_env) if native_env is not None else None
        resolved_url: str | None = self.connection_string
        resolved_token: str | None = self.turso_auth_token

        if resolved_url is None:
            resolved_url, inferred_token = resolve_trace_db_settings()
            self.connection_string = resolved_url
            resolved_token = inferred_token

        if resolved_token is None:
            resolved_token = resolve_trace_db_auth_token()

        self.turso_auth_token = resolved_token or ""

        if self.backend is None:
            self.backend = self._infer_backend(self.connection_string or "")

        if native_flag is False:
            raise RuntimeError("TURSO_NATIVE=false is no longer supported; only Turso/libSQL backend is available.")

        # Allow both TURSO_NATIVE and SQLITE backends (both use libsql.connect)
        if self.backend not in (StorageBackend.TURSO_NATIVE, StorageBackend.SQLITE):
            raise RuntimeError(f"Unsupported backend: {self.backend}. Only Turso/libSQL and SQLite are supported.")

    @staticmethod
    def _infer_backend(connection_string: str) -> StorageBackend:
        """Infer backend type from the connection string."""
        scheme = connection_string.split(":", 1)[0].lower()
        
        # Plain SQLite files: file://, /absolute/path, or no scheme
        if (
            scheme == "file"
            or scheme.startswith("sqlite")
            or connection_string.startswith("/")
            or "://" not in connection_string
        ):
            return StorageBackend.SQLITE
        
        # Turso/sqld: libsql://, http://, https://
        if scheme.startswith("libsql") or "libsql" in scheme or scheme in ("http", "https"):
            return StorageBackend.TURSO_NATIVE
        
        raise RuntimeError(f"Unsupported tracing backend scheme: {scheme}")

    def get_connection_string(self) -> str:
        """Get the appropriate connection string for the backend."""
        if self.connection_string:
            return self.connection_string

        if self.backend == StorageBackend.TURSO_NATIVE:
            return self.connection_string or ""
        raise ValueError(f"Unsupported backend: {self.backend}")

    def get_backend_config(self) -> dict[str, Any]:
        """Get backend-specific configuration."""
        if self.backend == StorageBackend.TURSO_NATIVE:
            config = {}
            if self.turso_auth_token:
                config["auth_token"] = self.turso_auth_token
            return config
        return {}


# Global storage configuration
STORAGE_CONFIG = StorageConfig()
