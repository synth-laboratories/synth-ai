"""Process tracking with automatic atexit cleanup.

This module tracks long-running tunnel-related subprocesses and ensures they
are terminated when Python exits (via atexit) or when cleanup_all() is called
explicitly.
"""

from __future__ import annotations

import atexit
from typing import List, Protocol


class _Terminable(Protocol):
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def poll(self) -> int | None: ...


# Global state - tracked processes for cleanup
_tracked: List[_Terminable] = []
_cleanup_registered = False


def tracked_processes() -> List[_Terminable]:
    """Return list of currently tracked processes (read-only copy).

    Returns:
        List of subprocess.Popen objects being tracked
    """
    return list(_tracked)


def track_process(proc: _Terminable) -> _Terminable:
    """Track a subprocess for automatic cleanup on exit.

    Args:
        proc: Managed process handle

    Returns:
        The same process (for chaining)
    """
    global _cleanup_registered
    _tracked.append(proc)

    if not _cleanup_registered:
        atexit.register(cleanup_all)
        _cleanup_registered = True

    return proc


def cleanup_all() -> None:
    """Stop all tracked processes.

    This is called automatically on Python exit via atexit.
    You can also call it manually to clean up early.

    Processes are terminated gracefully (SIGTERM), with a strict
    to SIGKILL if they don't exit within 5 seconds.
    """
    for proc in _tracked:
        try:
            if proc.poll() is None:  # Still running
                proc.terminate()
                try:
                    if hasattr(proc, "wait"):
                        try:
                            proc.wait(timeout=5)  # type: ignore[call-arg]
                        except TypeError:
                            proc.wait(5)  # type: ignore[call-arg]
                except Exception:
                    proc.kill()
        except Exception:
            pass  # Best effort cleanup
    _tracked.clear()
