"""sqld daemon management utilities."""

import logging
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time

import requests
from requests import RequestException

from ..config import CONFIG

logger = logging.getLogger(__name__)


class SqldDaemon:
    """Manages local sqld daemon lifecycle."""

    def __init__(
        self,
        db_path: str | None = None,
        http_port: int | None = None,
        hrana_port: int | None = None,
        binary_path: str | None = None,
    ):
        """Initialize sqld daemon manager.

        Args:
            db_path: Path to database file (uses config default if not provided)
            http_port: HTTP port for health/API (uses config default + 1 if not provided)
            hrana_port: Hrana WebSocket port for libsql connections (uses config default if not provided)
            binary_path: Path to sqld binary (auto-detected if not provided)
        """
        self.db_path = db_path or CONFIG.sqld_db_path
        self.hrana_port = hrana_port or CONFIG.sqld_http_port  # Main port for libsql://
        self.http_port = http_port or (self.hrana_port + 1)  # HTTP API on next port
        self.binary_path = binary_path or self._find_binary()
        self.process: subprocess.Popen[str] | None = None

    def _find_binary(self) -> str:
        """Find sqld binary in PATH, auto-installing if needed.
        
        Search order:
        1. CONFIG.sqld_binary in PATH
        2. libsql-server in PATH
        3. Common install locations (~/.turso/bin, /usr/local/bin, etc.)
        4. Auto-install via synth_ai.utils.sqld (if interactive terminal)
        
        Returns:
            Path to sqld binary
            
        Raises:
            RuntimeError: If binary not found and auto-install fails/disabled
        """
        # Check PATH first
        binary = shutil.which(CONFIG.sqld_binary) or shutil.which("libsql-server")
        if binary:
            logger.debug(f"Found sqld binary in PATH: {binary}")
            return binary
        
        # Check common install locations
        try:
            from synth_ai.cli.lib.sqld import find_sqld_binary
            binary = find_sqld_binary()
            if binary:
                logger.debug(f"Found sqld binary in common location: {binary}")
                return binary
        except ImportError:
            logger.debug("synth_ai.utils.sqld not available, skipping common location check")
        
        # Try auto-install if enabled and interactive
        auto_install_enabled = os.getenv("SYNTH_AI_AUTO_INSTALL_SQLD", "true").lower() == "true"
        
        if auto_install_enabled and sys.stdin.isatty():
            try:
                from synth_ai.cli.lib.sqld import install_sqld
                logger.info("sqld binary not found. Attempting automatic installation...")
                
                # Use click if available for better UX, otherwise proceed automatically
                try:
                    import click
                    if not click.confirm(
                        "sqld not found. Install automatically via Homebrew?",
                        default=True
                    ):
                        raise RuntimeError("User declined automatic installation")
                except ImportError:
                    # click not available, auto-install without prompt
                    logger.info("Installing sqld automatically (non-interactive mode)")
                
                binary = install_sqld()
                logger.info(f"Successfully installed sqld to: {binary}")
                return binary
                
            except Exception as exc:
                logger.warning(f"Auto-install failed: {exc}")
                # Fall through to error message below
        elif not auto_install_enabled:
            logger.debug("Auto-install disabled via SYNTH_AI_AUTO_INSTALL_SQLD=false")
        elif not sys.stdin.isatty():
            logger.debug("Non-interactive terminal, skipping auto-install prompt")
        
        # If we get here, all methods failed
        raise RuntimeError(
            "sqld binary not found. Install using one of these methods:\n"
            "\n"
            "Quick install (recommended):\n"
            "  synth-ai turso\n"
            "\n"
            "Manual install:\n"
            "  brew install turso-tech/tools/sqld\n"
            "  # or\n"
            "  curl -sSfL https://get.tur.so/install.sh | bash && turso dev\n"
            "\n"
            "For CI/CD environments:\n"
            "  Set SYNTH_AI_AUTO_INSTALL_SQLD=false and pre-install sqld"
        )

    def start(self, wait_for_ready: bool = True) -> subprocess.Popen:
        """Start the sqld daemon."""
        if self.process and self.process.poll() is None:
            return self.process

        # Avoid port conflicts by selecting free ports when needed
        if not self._port_available(self.hrana_port):
            self.hrana_port = self._find_free_port()
        if not self._port_available(self.http_port) or self.http_port == self.hrana_port:
            self.http_port = self._find_free_port()
            # Ensure distinct ports
            if self.http_port == self.hrana_port:
                self.http_port = self._find_free_port()

        db_file = pathlib.Path(self.db_path).resolve()
        db_file.parent.mkdir(parents=True, exist_ok=True)

        args = [
            self.binary_path,
            "--db-path",
            str(db_file),
            "--hrana-listen-addr",
            f"127.0.0.1:{self.hrana_port}",
            "--http-listen-addr",
            f"127.0.0.1:{self.http_port}",
        ]

        # No replication for local-only mode
        if CONFIG.sqld_idle_shutdown > 0:
            args.extend(["--idle-shutdown-timeout-s", str(CONFIG.sqld_idle_shutdown)])

        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if wait_for_ready:
            self._wait_for_ready()

        return self.process

    def _wait_for_ready(self, timeout: float = 10.0):
        """Wait for daemon to be ready to accept connections."""
        start_time = time.time()
        health_url = f"http://127.0.0.1:{self.http_port}/health"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    return
            except RequestException:
                pass

            # Check if process crashed
            if self.process and self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"sqld daemon failed to start:\nstdout: {stdout}\nstderr: {stderr}"
                )

            time.sleep(0.1)

        raise TimeoutError(f"sqld daemon did not become ready within {timeout} seconds")

    @staticmethod
    def _port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def stop(self, timeout: float = 5.0):
        """Stop the sqld daemon gracefully."""
        if not self.process:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        self.process = None

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self.process is not None and self.process.poll() is None

    def get_hrana_port(self) -> int:
        """Get the Hrana WebSocket port for libsql:// connections."""
        return self.hrana_port

    def get_http_port(self) -> int:
        """Get the HTTP API port for health checks."""
        return self.http_port

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Convenience functions
_daemon: SqldDaemon | None = None


def start_sqld(
    db_path: str | None = None,
    port: int | None = None,
    hrana_port: int | None = None,
    http_port: int | None = None,
) -> SqldDaemon:
    """Start a global sqld daemon instance.
    
    Args:
        db_path: Path to database file
        port: Legacy parameter - used as hrana_port if hrana_port not specified
        hrana_port: Hrana WebSocket port for libsql:// connections
        http_port: HTTP API port for health checks
    """
    global _daemon
    if _daemon and _daemon.is_running():
        return _daemon

    # Support legacy 'port' parameter by using it as hrana_port
    final_hrana_port = hrana_port or port
    _daemon = SqldDaemon(db_path=db_path, hrana_port=final_hrana_port, http_port=http_port)
    _daemon.start()
    return _daemon


def stop_sqld():
    """Stop the global sqld daemon instance."""
    global _daemon
    if _daemon:
        _daemon.stop()
        _daemon = None


def get_daemon() -> SqldDaemon | None:
    """Get the global daemon instance."""
    return _daemon
