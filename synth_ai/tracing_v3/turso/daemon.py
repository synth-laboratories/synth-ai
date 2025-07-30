"""sqld daemon management utilities."""

import subprocess
import pathlib
import shutil
import sys
import time
import requests
from typing import Optional
from ..config import CONFIG


class SqldDaemon:
    """Manages local sqld daemon lifecycle."""

    def __init__(self, db_path: str = None, http_port: int = None, binary_path: str = None):
        """Initialize sqld daemon manager.
        
        Args:
            db_path: Path to database file (uses config default if not provided)
            http_port: HTTP port for daemon (uses config default if not provided)  
            binary_path: Path to sqld binary (auto-detected if not provided)
        """
        self.db_path = db_path or CONFIG.sqld_db_path
        self.http_port = http_port or CONFIG.sqld_http_port
        self.binary_path = binary_path or self._find_binary()
        self.process: Optional[subprocess.Popen] = None

    def _find_binary(self) -> str:
        """Find sqld binary in PATH."""
        binary = shutil.which(CONFIG.sqld_binary) or shutil.which("libsql-server")
        if not binary:
            raise RuntimeError(
                f"sqld binary not found in PATH. Install with: brew install turso-tech/tools/sqld"
            )
        return binary

    def start(self, wait_for_ready: bool = True) -> subprocess.Popen:
        """Start the sqld daemon."""
        if self.process and self.process.poll() is None:
            return self.process

        db_file = pathlib.Path(self.db_path).resolve()
        db_file.parent.mkdir(parents=True, exist_ok=True)

        args = [
            self.binary_path,
            "--db-path",
            str(db_file),
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
            except requests.exceptions.RequestException:
                pass

            # Check if process crashed
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"sqld daemon failed to start:\nstdout: {stdout}\nstderr: {stderr}"
                )

            time.sleep(0.1)

        raise TimeoutError(f"sqld daemon did not become ready within {timeout} seconds")

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

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Convenience functions
_daemon: Optional[SqldDaemon] = None


def start_sqld(db_path: str = None, port: int = None) -> SqldDaemon:
    """Start a global sqld daemon instance."""
    global _daemon
    if _daemon and _daemon.is_running():
        return _daemon

    _daemon = SqldDaemon(db_path=db_path, http_port=port)
    _daemon.start()
    return _daemon


def stop_sqld():
    """Stop the global sqld daemon instance."""
    global _daemon
    if _daemon:
        _daemon.stop()
        _daemon = None


def get_daemon() -> Optional[SqldDaemon]:
    """Get the global daemon instance."""
    return _daemon
