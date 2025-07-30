"""Storage configuration for tracing v3."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
from enum import Enum


class StorageBackend(str, Enum):
    """Supported storage backends."""

    TURSO = "turso"
    SQLITE = "sqlite"
    POSTGRES = "postgres"  # Future support


@dataclass
class StorageConfig:
    """Configuration for storage backend."""

    backend: StorageBackend = StorageBackend.TURSO
    connection_string: Optional[str] = None

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

    def get_connection_string(self) -> str:
        """Get the appropriate connection string for the backend."""
        if self.connection_string:
            return self.connection_string

        if self.backend == StorageBackend.TURSO:
            return self.turso_url
        elif self.backend == StorageBackend.SQLITE:
            return "sqlite+aiosqlite:///traces.db"
        elif self.backend == StorageBackend.POSTGRES:
            return os.getenv("POSTGRES_URL", "postgresql+asyncpg://localhost/traces")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def get_backend_config(self) -> Dict[str, Any]:
        """Get backend-specific configuration."""
        if self.backend == StorageBackend.TURSO:
            config = {}
            if self.turso_auth_token:
                config["auth_token"] = self.turso_auth_token
            return config
        return {}


# Global storage configuration
STORAGE_CONFIG = StorageConfig()
