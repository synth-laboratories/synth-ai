"""
Centralized database configuration for v3 tracing.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .turso.daemon import SqldDaemon

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Centralized database configuration management."""

    # Default values from serve.sh
    DEFAULT_DB_FILE = "traces/v3/synth_ai.db"
    DEFAULT_HTTP_PORT = 8080

    def __init__(
        self, db_path: str | None = None, http_port: int | None = None, use_sqld: bool = True
    ):
        """
        Initialize database configuration.

        Args:
            db_path: Path to database file. If None, uses DEFAULT_DB_FILE from serve.sh.
            http_port: HTTP port for sqld daemon. If None, uses DEFAULT_HTTP_PORT from serve.sh.
            use_sqld: Whether to use sqld daemon or direct SQLite.
        """
        self.use_sqld = use_sqld
        self.http_port = http_port or int(os.getenv("SQLD_HTTP_PORT", self.DEFAULT_HTTP_PORT))
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

        if os.path.exists(sqld_data_path):
            # sqld is managing the database
            logger.debug(f"✅ Using sqld-managed database at: {sqld_data_path}")
            actual_db_path = sqld_data_path
        else:
            # Direct SQLite file
            if not os.path.exists(abs_path):
                logger.debug(f"⚠️  Database file not found at: {abs_path}")
                logger.debug("🔧 Make sure to run './serve.sh' to start the turso/sqld service")
            else:
                logger.debug(f"📁 Using direct SQLite file at: {abs_path}")
            actual_db_path = abs_path

        # SQLite URLs need 3 slashes for absolute paths
        return f"sqlite+aiosqlite:///{actual_db_path}"

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

            self._daemon = SqldDaemon(db_path=self.db_base_path, http_port=self.http_port)

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

        sqld_port = int(os.getenv("SQLD_HTTP_PORT", DatabaseConfig.DEFAULT_HTTP_PORT))
        sqld_running = False
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"sqld.*--http-listen-addr.*:{sqld_port}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # sqld is already running, don't start a new one
                sqld_running = True
                use_sqld = False
                logger.debug(f"✅ Detected sqld already running on port {sqld_port}")
        except Exception as e:
            logger.debug(f"Could not check for sqld process: {e}")

        if not sqld_running and use_sqld:
            logger.warning("⚠️  sqld service not detected!")
            logger.warning("🔧 Please start the turso/sqld service by running:")
            logger.warning("   ./serve.sh")
            logger.warning("")
            logger.warning("This will start:")
            logger.warning("  - sqld daemon (SQLite server) on port 8080")
            logger.warning("  - Environment service on port 8901")

        _default_config = DatabaseConfig(db_path=db_path, use_sqld=use_sqld)

    return _default_config


def set_default_db_config(config: DatabaseConfig):
    """Set the default database configuration."""
    global _default_config
    _default_config = config
