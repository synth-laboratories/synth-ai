"""Storage configuration for tracing v3."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


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

    backend: StorageBackend = StorageBackend.TURSO_NATIVE
    connection_string: str | None = None

    # Turso-specific settings
    turso_url: str = os.getenv("TURSO_DATABASE_URL", "sqlite+libsql://http://127.0.0.1:8080")
    turso_auth_token: str = os.getenv("TURSO_AUTH_TOKEN", "")

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

        if native_flag is False:
            self.backend = StorageBackend.SQLITE

    def get_connection_string(self) -> str:
        """Get the appropriate connection string for the backend."""
        if self.connection_string:
            return self.connection_string

        if self.backend == StorageBackend.TURSO_NATIVE:
            return self.turso_url
        if self.backend == StorageBackend.SQLITE:
            return "sqlite+aiosqlite:///traces.db"
        if self.backend == StorageBackend.POSTGRES:
            return os.getenv("POSTGRES_URL", "postgresql+asyncpg://localhost/traces")
        raise ValueError(f"Unknown backend: {self.backend}")

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
