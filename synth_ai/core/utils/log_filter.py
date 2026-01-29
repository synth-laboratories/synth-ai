"""Utilities for filtering noisy logs from stdout/stderr (Rust-backed)."""

from __future__ import annotations

import sys
from typing import TextIO

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for utils.log_filter.") from exc


def should_filter_log_line(line: str) -> bool:
    """Check if a log line should be filtered out."""
    fn = getattr(synth_ai_py, "should_filter_log_line", None)
    if fn is None:
        return False
    return fn(line)


class FilteredStream:
    """A file-like object that filters noisy log lines before writing."""

    def __init__(self, stream: TextIO):
        self._stream = stream
        self._buffer = ""
        # Store original write method to avoid recursion
        self._original_write = stream.write

    def write(self, text: str) -> int:
        """Write text, filtering out noisy log lines."""
        # Check if entire text matches filter (for unbuffered writes)
        if should_filter_log_line(text.strip()):
            return len(text)

        # Buffer incomplete lines
        self._buffer += text
        lines = self._buffer.split("\n")
        # Keep the last incomplete line in buffer
        self._buffer = lines[-1]
        # Process complete lines
        written = 0
        for line in lines[:-1]:
            if not should_filter_log_line(line):
                written += self._original_write(line + "\n")
        return len(text)

    def flush(self) -> None:
        """Flush any remaining buffered content."""
        if self._buffer and not should_filter_log_line(self._buffer):
            self._original_write(self._buffer)
            self._buffer = ""
        self._stream.flush()

    def __getattr__(self, name: str):
        """Delegate other attributes to the underlying stream."""
        return getattr(self._stream, name)


def install_log_filter() -> None:
    """Install stdout/stderr filters to suppress noisy logs.

    This wraps sys.stdout and sys.stderr to filter out verbose
    codex_otel logs before they're printed.
    """
    if not isinstance(sys.stdout, FilteredStream):
        sys.stdout = FilteredStream(sys.stdout)  # type: ignore[assignment]
    if not isinstance(sys.stderr, FilteredStream):
        sys.stderr = FilteredStream(sys.stderr)  # type: ignore[assignment]
