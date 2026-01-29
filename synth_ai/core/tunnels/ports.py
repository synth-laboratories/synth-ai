"""Port management types.

This module provides types for port conflict handling.
The actual port functions are implemented in Rust (synth_ai_py) and
exposed via the rust.py wrapper module.

Example:
    from synth_ai.core.tunnels import kill_port, is_port_available, find_available_port
    from synth_ai.core.tunnels.ports import PortConflictBehavior, acquire_port

    # Check if port is available
    if not is_port_available(8001):
        kill_port(8001)  # Free the port

    # Or find an available port automatically
    port = find_available_port(8001)

    # Or use the unified acquire_port function with configurable behavior
    port = acquire_port(8001, on_conflict=PortConflictBehavior.FIND_NEW)
"""

from __future__ import annotations

from enum import Enum


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
