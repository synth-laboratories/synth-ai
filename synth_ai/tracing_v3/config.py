"""Configuration for tracing v3 with Turso/sqld."""

from dataclasses import dataclass
import os


@dataclass
class TursoConfig:
    """Configuration for Turso/sqld connection."""

    # Default values matching serve.sh
    DEFAULT_DB_FILE = "traces/v3/synth_ai.db"
    DEFAULT_HTTP_PORT = 8080

    # Local embedded database for async SQLAlchemy
    # Resolve to the actual SQLite file used by sqld if the base path is a directory
    def _resolve_sqlite_db_url() -> str:  # type: ignore[no-redef]
        base_path = os.path.abspath(os.getenv("SQLD_DB_PATH", "traces/v3/synth_ai.db"))
        # If sqld is managing this DB, the real SQLite file lives under dbs/default/data
        candidate = os.path.join(base_path, "dbs", "default", "data")
        if os.path.isdir(base_path) and os.path.exists(candidate):
            return f"sqlite+aiosqlite:///{candidate}"
        return f"sqlite+aiosqlite:///{base_path}"

    # Use env override if provided; otherwise resolve based on SQLD layout
    db_url: str = os.getenv("TURSO_LOCAL_DB_URL", _resolve_sqlite_db_url())

    # Remote database sync configuration
    sync_url: str = os.getenv("TURSO_DATABASE_URL", "")
    auth_token: str = os.getenv("TURSO_AUTH_TOKEN", "")
    sync_interval: int = int(os.getenv("TURSO_SYNC_SECONDS", "2"))  # 2 seconds for responsive local development

    # Connection pool settings
    pool_size: int = int(os.getenv("TURSO_POOL_SIZE", "8"))
    max_overflow: int = int(os.getenv("TURSO_MAX_OVERFLOW", "16"))
    pool_timeout: float = float(os.getenv("TURSO_POOL_TIMEOUT", "30.0"))
    pool_recycle: int = int(os.getenv("TURSO_POOL_RECYCLE", "3600"))

    # SQLite settings
    foreign_keys: bool = os.getenv("TURSO_FOREIGN_KEYS", "true").lower() == "true"
    journal_mode: str = os.getenv("TURSO_JOURNAL_MODE", "WAL")

    # Performance settings
    echo_sql: bool = os.getenv("TURSO_ECHO_SQL", "false").lower() == "true"
    batch_size: int = int(os.getenv("TURSO_BATCH_SIZE", "1000"))

    # Daemon settings (for local sqld) - match serve.sh defaults
    sqld_binary: str = os.getenv("SQLD_BINARY", "sqld")
    sqld_db_path: str = os.getenv("SQLD_DB_PATH", "traces/v3/synth_ai.db")
    sqld_http_port: int = int(os.getenv("SQLD_HTTP_PORT", "8080"))
    sqld_idle_shutdown: int = int(os.getenv("SQLD_IDLE_SHUTDOWN", "0"))  # 0 = no idle shutdown

    def get_connect_args(self) -> dict:
        """Get SQLAlchemy connection arguments."""
        args = {}
        if self.auth_token:
            args["auth_token"] = self.auth_token
        return args

    def get_engine_kwargs(self) -> dict:
        """Get SQLAlchemy engine creation kwargs."""
        kwargs = {
            "echo": self.echo_sql,
            "future": True,
        }

        # Only add pool settings for non-SQLite URLs
        if not (self.db_url.startswith("sqlite") or ":memory:" in self.db_url):
            kwargs.update(
                {
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout,
                    "pool_recycle": self.pool_recycle,
                }
            )

        return kwargs


# Global config instance
CONFIG = TursoConfig()
