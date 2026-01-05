"""Port management utilities.

This module provides utilities for checking port availability and
killing processes that are using specific ports.

Example:
    from synth_ai.sdk.tunnels import kill_port, is_port_available, find_available_port
    from synth_ai.sdk.tunnels.ports import PortConflictBehavior, acquire_port

    # Check if port is available
    if not is_port_available(8001):
        kill_port(8001)  # Free the port

    # Or find an available port automatically
    port = find_available_port(8001)

    # Or use the unified acquire_port function with configurable behavior
    port = acquire_port(8001, on_conflict=PortConflictBehavior.FIND_NEW)
"""

from __future__ import annotations

import socket
import subprocess
import sys
from enum import Enum
from typing import Optional


class PortConflictBehavior(str, Enum):
    """Behavior when a requested port is already in use.

    Attributes:
        FAIL: Raise an error if the port is in use
        EVICT: Kill the process using the port and take it
        FIND_NEW: Find the next available port starting from the requested one
    """

    FAIL = "fail"
    EVICT = "evict"
    FIND_NEW = "find_new"


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available for binding.

    Args:
        port: Port number to check
        host: Host to check (default: 0.0.0.0 to match uvicorn's default)

    Returns:
        True if the port is available, False if in use
    """
    # Check both 0.0.0.0 (all interfaces) and 127.0.0.1 (localhost)
    # A server on either will block the port
    hosts_to_check = ["0.0.0.0", "127.0.0.1"] if host == "0.0.0.0" else [host]

    for check_host in hosts_to_check:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Don't use SO_REUSEADDR - we want to detect if port is truly in use
                sock.settimeout(1)
                sock.bind((check_host, port))
        except OSError:
            return False
    return True


def find_available_port(
    start_port: int, host: str = "0.0.0.0", max_attempts: int = 100
) -> int:
    """Find an available port starting from start_port.

    Args:
        start_port: Port number to start searching from
        host: Host to check (default: 127.0.0.1)
        max_attempts: Maximum number of ports to try

    Returns:
        First available port number

    Raises:
        RuntimeError: If no available port found within max_attempts
    """
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port, host):
            return port
    raise RuntimeError(f"No available port found starting from {start_port}")


def kill_port(port: int, host: str = "0.0.0.0") -> bool:
    """Kill any process using the specified port.

    This is a best-effort operation that attempts to free a port by
    terminating whatever process is using it. Use with caution.

    Args:
        port: Port number to free
        host: Host (unused, for API consistency)

    Returns:
        True if a process was killed, False if port was already free

    Note:
        This may not work on all systems or require elevated privileges.
        Prefer using find_available_port() instead when possible.
    """
    if is_port_available(port, host):
        return False  # Already free

    try:
        if sys.platform == "win32":
            # Windows: use netstat + taskkill
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, check=False
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True,
                            check=False,
                        )
                        return True
        else:
            # Unix-like (macOS, Linux): use lsof + kill
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split():
                    subprocess.run(
                        ["kill", "-9", pid], capture_output=True, check=False
                    )
                return True
    except Exception:
        pass

    return False


def acquire_port(
    port: int,
    on_conflict: PortConflictBehavior = PortConflictBehavior.FAIL,
    host: str = "0.0.0.0",
    max_search: int = 100,
) -> int:
    """Acquire a port with configurable conflict handling.

    This is the recommended way to get a port for your server. It handles
    port conflicts gracefully based on the specified behavior.

    Args:
        port: Desired port number
        on_conflict: What to do if the port is in use:
            - FAIL: Raise PortInUseError
            - EVICT: Kill the process using the port
            - FIND_NEW: Find the next available port
        host: Host to check (default: 127.0.0.1)
        max_search: Maximum ports to search when using FIND_NEW

    Returns:
        The port number that was acquired (may differ from requested if FIND_NEW)

    Raises:
        PortInUseError: If port is in use and on_conflict is FAIL
        RuntimeError: If EVICT fails or no port found with FIND_NEW

    Example:
        # Strict mode - fail if port taken
        port = acquire_port(8001, on_conflict=PortConflictBehavior.FAIL)

        # Aggressive mode - evict previous process
        port = acquire_port(8001, on_conflict=PortConflictBehavior.EVICT)

        # Flexible mode - find any available port
        port = acquire_port(8001, on_conflict=PortConflictBehavior.FIND_NEW)
    """
    # Check if port is available
    if is_port_available(port, host):
        return port

    # Port is in use - handle based on behavior
    if on_conflict == PortConflictBehavior.FAIL:
        raise PortInUseError(port, host)

    elif on_conflict == PortConflictBehavior.EVICT:
        killed = kill_port(port, host)
        if killed:
            # Give the OS a moment to release the port
            import time
            time.sleep(0.5)
            if is_port_available(port, host):
                return port
        raise RuntimeError(
            f"Failed to evict process on port {port}. "
            f"Try using PortConflictBehavior.FIND_NEW instead."
        )

    elif on_conflict == PortConflictBehavior.FIND_NEW:
        return find_available_port(port, host, max_search)

    else:
        raise ValueError(f"Unknown conflict behavior: {on_conflict}")


class PortInUseError(Exception):
    """Raised when a port is already in use and FAIL behavior is specified."""

    def __init__(self, port: int, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        super().__init__(
            f"Port {port} is already in use. "
            f"Use PortConflictBehavior.EVICT to kill the existing process, "
            f"or PortConflictBehavior.FIND_NEW to find an available port."
        )
