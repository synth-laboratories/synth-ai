"""Process tracking with automatic atexit cleanup.

This module provides a simple way to track cloudflared processes and ensure
they are terminated when Python exits (via atexit) or when cleanup_all() is
called explicitly.

Example:
    from synth_ai.sdk.tunnels import open_managed_tunnel, track_process

    # Start a cloudflared process and track it
    proc = track_process(open_managed_tunnel(tunnel_token))

    # Process will be automatically terminated when Python exits
    # Or you can clean up early:
    # cleanup_all()
"""

from __future__ import annotations

import atexit
import subprocess
from typing import List

# Global state - tracked processes for cleanup
_tracked: List[subprocess.Popen] = []
_cleanup_registered = False


def tracked_processes() -> List[subprocess.Popen]:
    """Return list of currently tracked processes (read-only copy).

    Returns:
        List of subprocess.Popen objects being tracked
    """
    return list(_tracked)


def track_process(proc: subprocess.Popen) -> subprocess.Popen:
    """Track a cloudflared process for automatic cleanup on exit.

    Args:
        proc: Process returned by open_managed_tunnel() or similar

    Returns:
        The same process (for chaining)

    Example:
        from synth_ai.sdk.tunnels import open_managed_tunnel, track_process

        proc = track_process(open_managed_tunnel(token))
        # proc will be terminated automatically when Python exits
    """
    global _cleanup_registered
    _tracked.append(proc)

    if not _cleanup_registered:
        atexit.register(cleanup_all)
        _cleanup_registered = True

    return proc


def cleanup_all() -> None:
    """Stop all tracked cloudflared processes.

    This is called automatically on Python exit via atexit.
    You can also call it manually to clean up early.

    Processes are terminated gracefully (SIGTERM), with a fallback
    to SIGKILL if they don't exit within 5 seconds.
    """
    for proc in _tracked:
        try:
            if proc.poll() is None:  # Still running
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        except Exception:
            pass  # Best effort cleanup
    _tracked.clear()
