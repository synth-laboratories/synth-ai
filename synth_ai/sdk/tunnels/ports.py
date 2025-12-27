"""Port management utilities.

This module provides utilities for checking port availability and
killing processes that are using specific ports.

Example:
    from synth_ai.sdk.tunnels import kill_port, is_port_available, find_available_port

    # Check if port is available
    if not is_port_available(8001):
        kill_port(8001)  # Free the port

    # Or find an available port automatically
    port = find_available_port(8001)
"""

from __future__ import annotations

import socket
import subprocess
import sys


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is available for binding.

    Args:
        port: Port number to check
        host: Host to check (default: 127.0.0.1)

    Returns:
        True if the port is available, False if in use
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(
    start_port: int, host: str = "127.0.0.1", max_attempts: int = 100
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


def kill_port(port: int, host: str = "127.0.0.1") -> bool:
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
