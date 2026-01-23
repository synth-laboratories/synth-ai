"""cloudflared connector management with warm reuse.

This module manages the cloudflared process lifecycle:
1. Starting the connector with a tunnel token
2. Monitoring connection state
3. Keeping the connector warm between leases
4. Automatic cleanup on idle timeout

The connector is designed to be long-lived to minimize latency
for subsequent lease activations.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Optional

from .cleanup import track_process
from .errors import (
    ConnectorConnectionError,
    ConnectorNotInstalledError,
    ConnectorTokenError,
)
from .types import ConnectorState, ConnectorStatus

logger = logging.getLogger(__name__)

# Connection timeout for cloudflared
DEFAULT_CONNECTION_TIMEOUT = 60.0

# Idle timeout before stopping the connector (when no active leases)
DEFAULT_IDLE_TIMEOUT = 120.0

# Patterns to detect connection state
CONNECTED_PATTERNS = [
    re.compile(r"Registered tunnel connection", re.IGNORECASE),
    re.compile(r"Connection .* registered", re.IGNORECASE),
    re.compile(r"INF\s+.*registered", re.IGNORECASE),
]

ERROR_PATTERNS = [
    (re.compile(r"failed to connect", re.IGNORECASE), "Connection failed"),
    (re.compile(r"invalid.*token", re.IGNORECASE), "Invalid token"),
    (re.compile(r"tunnel.*not.*found", re.IGNORECASE), "Tunnel not found"),
    (re.compile(r"unauthorized", re.IGNORECASE), "Unauthorized"),
    (re.compile(r"rate.*limit", re.IGNORECASE), "Rate limited"),
]


class TunnelConnector:
    """Manages a cloudflared connector process.

    The connector is designed to be warm-reusable:
    - It stays running between lease activations
    - It automatically stops after an idle timeout
    - It can be restarted with a new token if needed

    Example:
        connector = TunnelConnector()
        await connector.start(tunnel_token)
        # ... use the tunnel ...
        await connector.stop()  # Or let it idle-timeout
    """

    def __init__(
        self,
        cloudflared_path: Optional[str] = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    ):
        """Initialize the connector.

        Args:
            cloudflared_path: Path to cloudflared binary (auto-detected if None)
            idle_timeout: Seconds to wait before stopping on idle
        """
        self._cloudflared_path = cloudflared_path
        self._idle_timeout = idle_timeout

        self._process: Optional[subprocess.Popen[bytes]] = None
        self._state = ConnectorState.STOPPED
        self._error: Optional[str] = None
        self._connected_at: Optional[datetime] = None
        self._current_token: Optional[str] = None
        self._active_leases: set[str] = set()
        self._idle_timer: Optional[asyncio.Task[Any]] = None

        self._lock = threading.Lock()
        self._output_lines: list[str] = []
        self._output_reader_task: Optional[asyncio.Task[Any]] = None

    @property
    def status(self) -> ConnectorStatus:
        """Get the current connector status."""
        return ConnectorStatus(
            state=self._state,
            pid=self._process.pid if self._process else None,
            connected_at=self._connected_at,
            error=self._error,
        )

    @property
    def is_connected(self) -> bool:
        """Check if the connector is connected."""
        return self._state == ConnectorState.CONNECTED

    @property
    def is_running(self) -> bool:
        """Check if the process is running (may not be connected yet)."""
        return self._state in (ConnectorState.STARTING, ConnectorState.CONNECTED)

    def _get_cloudflared_path(self) -> str:
        """Get the path to cloudflared binary."""
        if self._cloudflared_path:
            return self._cloudflared_path

        # Try to find cloudflared
        from .cloudflare import get_cloudflared_path

        path = get_cloudflared_path()
        if not path:
            raise ConnectorNotInstalledError()
        return path

    async def start(
        self,
        tunnel_token: str,
        *,
        timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        force_restart: bool = False,
    ) -> None:
        """Start the connector with the given tunnel token.

        If the connector is already running with the same token,
        this is a no-op (unless force_restart=True).

        Args:
            tunnel_token: The tunnel token from lease creation
            timeout: Connection timeout in seconds
            force_restart: If True, restart even if already running

        Raises:
            ConnectorNotInstalledError: If cloudflared is not installed
            ConnectorTokenError: If the token is invalid
            ConnectorConnectionError: If connection fails
        """
        # Check if already running with same token
        if self.is_running and self._current_token == tunnel_token and not force_restart:
            logger.debug("[CONNECTOR] Already running with same token")
            self._cancel_idle_timer()
            return

        # Stop existing process if running
        if self.is_running:
            await self.stop()

        cloudflared_path = self._get_cloudflared_path()
        self._state = ConnectorState.STARTING
        self._error = None
        self._connected_at = None
        self._current_token = tunnel_token
        self._output_lines.clear()

        logger.info("[CONNECTOR] Starting cloudflared connector")

        try:
            # Start cloudflared process
            cmd = [
                cloudflared_path,
                "tunnel",
                "run",
                "--token",
                tunnel_token,
            ]

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
            )
            track_process(self._process)

            # Start output reader
            self._output_reader_task = asyncio.create_task(self._read_output())

            # Wait for connection
            await self._wait_for_connection(timeout)

            self._state = ConnectorState.CONNECTED
            self._connected_at = datetime.now()
            logger.info(
                "[CONNECTOR] Connected to Cloudflare edge (pid=%d)",
                self._process.pid,
            )

        except Exception as e:
            self._state = ConnectorState.ERROR
            self._error = str(e)
            await self._cleanup_process()
            raise

    async def _read_output(self) -> None:
        """Read output from cloudflared process."""
        if not self._process:
            return

        async def read_stream(stream: Any, name: str) -> None:
            while True:
                try:
                    line = await asyncio.get_event_loop().run_in_executor(None, stream.readline)
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded:
                        self._output_lines.append(decoded)
                        # Keep only last 100 lines
                        if len(self._output_lines) > 100:
                            self._output_lines = self._output_lines[-100:]
                        logger.debug("[CONNECTOR] %s: %s", name, decoded)
                except Exception:
                    break

        if self._process.stdout:
            asyncio.create_task(read_stream(self._process.stdout, "stdout"))
        if self._process.stderr:
            asyncio.create_task(read_stream(self._process.stderr, "stderr"))

    async def _wait_for_connection(self, timeout: float) -> None:
        """Wait for cloudflared to connect to the edge."""
        start_time = time.time()
        poll_interval = 0.1

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise ConnectorConnectionError(
                    "Timeout waiting for cloudflared to connect",
                    timeout=timeout,
                )

            # Check if process died
            if self._process and self._process.poll() is not None:
                error = self._detect_error()
                if "token" in error.lower():
                    raise ConnectorTokenError(error)
                raise ConnectorConnectionError(f"cloudflared exited: {error}")

            # Check for connection in output
            for line in self._output_lines:
                for pattern in CONNECTED_PATTERNS:
                    if pattern.search(line):
                        return

                # Check for errors
                for error_pattern, error_msg in ERROR_PATTERNS:
                    if error_pattern.search(line):
                        if "token" in error_msg.lower():
                            raise ConnectorTokenError(error_msg)
                        raise ConnectorConnectionError(error_msg)

            await asyncio.sleep(poll_interval)

    def _detect_error(self) -> str:
        """Detect error from output lines."""
        for line in reversed(self._output_lines):
            for error_pattern, error_msg in ERROR_PATTERNS:
                if error_pattern.search(line):
                    return f"{error_msg}: {line}"
        return "Unknown error"

    async def stop(self) -> None:
        """Stop the connector."""
        self._cancel_idle_timer()
        await self._cleanup_process()
        self._state = ConnectorState.STOPPED
        self._current_token = None
        self._active_leases.clear()
        logger.info("[CONNECTOR] Stopped")

    async def _cleanup_process(self) -> None:
        """Clean up the cloudflared process."""
        if self._output_reader_task:
            self._output_reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._output_reader_task
            self._output_reader_task = None

        if self._process:
            try:
                self._process.terminate()
                # Wait briefly for graceful shutdown
                await asyncio.get_event_loop().run_in_executor(None, self._process.wait, 2.0)
            except Exception:
                with contextlib.suppress(Exception):
                    self._process.kill()
            self._process = None

    def register_lease(self, lease_id: str) -> None:
        """Register an active lease with the connector.

        This cancels any pending idle timeout.

        Args:
            lease_id: The lease ID to register
        """
        self._active_leases.add(lease_id)
        self._cancel_idle_timer()

    def unregister_lease(self, lease_id: str) -> None:
        """Unregister a lease from the connector.

        If no leases remain, starts the idle timeout.

        Args:
            lease_id: The lease ID to unregister
        """
        self._active_leases.discard(lease_id)
        if not self._active_leases:
            self._start_idle_timer()

    def _cancel_idle_timer(self) -> None:
        """Cancel the idle timeout timer."""
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _start_idle_timer(self) -> None:
        """Start the idle timeout timer."""
        self._cancel_idle_timer()

        async def idle_shutdown() -> None:
            await asyncio.sleep(self._idle_timeout)
            if not self._active_leases:
                logger.info("[CONNECTOR] Idle timeout reached, stopping connector")
                await self.stop()

        self._idle_timer = asyncio.create_task(idle_shutdown())

    def get_logs(self, lines: int = 50) -> list[str]:
        """Get recent log lines from cloudflared.

        Args:
            lines: Number of lines to return

        Returns:
            List of recent log lines
        """
        return self._output_lines[-lines:]


# Global connector instance
_connector: Optional[TunnelConnector] = None
_connector_lock = threading.Lock()


def get_connector() -> TunnelConnector:
    """Get or create the global connector instance.

    Returns:
        TunnelConnector instance
    """
    global _connector
    with _connector_lock:
        if _connector is None:
            _connector = TunnelConnector()
        return _connector


async def ensure_connector_running(
    tunnel_token: str,
    *,
    timeout: float = DEFAULT_CONNECTION_TIMEOUT,
) -> TunnelConnector:
    """Ensure the connector is running and connected.

    Args:
        tunnel_token: The tunnel token
        timeout: Connection timeout

    Returns:
        Connected TunnelConnector instance
    """
    connector = get_connector()
    await connector.start(tunnel_token, timeout=timeout)
    return connector
