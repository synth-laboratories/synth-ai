"""
Centralized database configuration for v3 tracing.
"""

import logging
import os
import shutil
from typing import TYPE_CHECKING, Optional

from synth_ai.core.tracing_v3.constants import canonical_trace_db_path

if TYPE_CHECKING:
    from .turso.daemon import SqldDaemon

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Centralized database configuration management."""

    # Default values from serve.sh
    DEFAULT_DB_FILE = str(canonical_trace_db_path())
    DEFAULT_HTTP_PORT = 8080

    def __init__(
        self, db_path: str | None = None, http_port: int | None = None, use_sqld: bool = True
    ):
        """
        Initialize database configuration.

        Args:
            db_path: Path to database file. If None, uses DEFAULT_DB_FILE from serve.sh.
            http_port: Hrana WebSocket port for sqld daemon (env: SQLD_HTTP_PORT). If None, uses DEFAULT_HTTP_PORT.
            use_sqld: Whether to use sqld daemon or direct SQLite.
        """
        self.use_sqld = use_sqld and self._sqld_binary_available()
        # Note: SQLD_HTTP_PORT is actually the hrana port (8080), not the HTTP API port
        self.hrana_port = http_port or int(os.getenv("SQLD_HTTP_PORT", self.DEFAULT_HTTP_PORT))
        self._daemon: SqldDaemon | None = None

        # Set up database path to match serve.sh configuration
        if db_path is None:
            # Use the same database file as serve.sh
            self.db_file = os.getenv("SQLD_DB_PATH", self.DEFAULT_DB_FILE)
            # For sqld, db_base_path is just the filename without directory
            self.db_base_path = self.db_file
        else:
            self.db_file = db_path
            self.db_base_path = db_path

    @property
    def database_url(self) -> str:
        """Get the SQLAlchemy database URL."""
        # Always use direct file access with aiosqlite
        # (sqld HTTP interface is not compatible with SQLAlchemy)

        # Check if sqld is running - it creates a directory structure
        abs_path = os.path.abspath(self.db_file)
        sqld_data_path = os.path.join(abs_path, "dbs", "default", "data")

        if not os.path.exists(sqld_data_path) and not os.path.exists(abs_path):
            raise RuntimeError(
                "sqld data directory not found. Run `sqld --db-path <path>` before using the tracing database."
            )

        # Use http:// for local sqld HTTP API port
        # sqld has two ports: hrana_port (Hrana WebSocket) and hrana_port+1 (HTTP API)
        # Python libsql client uses HTTP API with http:// URLs
        http_api_port = self.hrana_port + 1
        return f"http://127.0.0.1:{http_api_port}"

    def _sqld_binary_available(self) -> bool:
        """Check if the sqld (Turso) binary is available on PATH."""
        # Respect explicit SQLD_BINARY override when present
        binary_override = os.getenv("SQLD_BINARY")
        candidates = [binary_override, "sqld", "libsql-server"]

        for candidate in candidates:
            if candidate and shutil.which(candidate):
                return True

        if binary_override:
            raise RuntimeError(
                f"Configured SQLD_BINARY='{binary_override}' but the executable was not found on PATH."
            )
        raise RuntimeError(
            "sqld binary not detected; install Turso's sqld or set SQLD_BINARY so that libSQL can be used."
        )

    def start_daemon(self, wait_time: float = 2.0):
        """
        Start the sqld daemon if configured.

        Args:
            wait_time: Time to wait for daemon startup.

        Returns:
            The daemon instance.
        """
        if not self.use_sqld:
            raise RuntimeError("Database not configured to use sqld daemon")

        if self._daemon is None:
            # Import here to avoid circular dependency
            from .turso.daemon import SqldDaemon

            self._daemon = SqldDaemon(db_path=self.db_base_path, hrana_port=self.hrana_port)

        self._daemon.start()

        # Wait for daemon to be ready
        import time

        time.sleep(wait_time)

        return self._daemon

    def stop_daemon(self):
        """Stop the sqld daemon if running."""
        if self._daemon:
            self._daemon.stop()
            self._daemon = None

    def get_daemon_and_url(self, wait_time: float = 2.0) -> tuple[Optional["SqldDaemon"], str]:
        """
        Get daemon (starting if needed) and database URL.

        Returns:
            Tuple of (daemon or None, database_url)
        """
        daemon = None
        if self.use_sqld:
            daemon = self.start_daemon(wait_time)

        return daemon, self.database_url


# Global default configuration
_default_config: DatabaseConfig | None = None


def get_default_db_config() -> DatabaseConfig:
    """Get the default database configuration."""
    global _default_config
    if _default_config is None:
        # Check environment variable for database path
        db_path = os.environ.get("SYNTH_AI_V3_DB_PATH")
        use_sqld = os.environ.get("SYNTH_AI_V3_USE_SQLD", "true").lower() == "true"

        # Check if sqld is already running (started by serve.sh)
        import subprocess

        sqld_hrana_port = int(os.getenv("SQLD_HTTP_PORT", DatabaseConfig.DEFAULT_HTTP_PORT))
        sqld_http_port = sqld_hrana_port + 1
        sqld_running = False
        try:
            # Check for either hrana or http port in the process command line
            result = subprocess.run(
                ["pgrep", "-f", f"sqld.*(--hrana-listen-addr.*:{sqld_hrana_port}|--http-listen-addr.*:{sqld_http_port})"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # sqld is already running, don't start a new one
                sqld_running = True
                use_sqld = False
                logger.debug(f"✅ Detected sqld already running on ports {sqld_hrana_port} (hrana) and {sqld_http_port} (http)")
        except Exception as e:
            logger.debug(f"Could not check for sqld process: {e}")

        if not sqld_running and use_sqld:
            logger.warning("sqld service not detected. Start the Turso daemon (./serve.sh) before running tracing workloads.")

        _default_config = DatabaseConfig(db_path=db_path, use_sqld=use_sqld)

    return _default_config


def set_default_db_config(config: DatabaseConfig):
    """Set the default database configuration."""
    global _default_config
    _default_config = config
