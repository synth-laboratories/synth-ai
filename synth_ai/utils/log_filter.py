"""Utilities for filtering noisy logs from stdout/stderr."""

from __future__ import annotations

import re
import sys
from typing import TextIO

# Patterns for noisy logs that should be filtered out
# Using simple substring matching for better performance and broader matching
_NOISY_LOG_SUBSTRINGS = [
    'codex_otel::otel_event_manager',
    'event.kind=response.reasoning_summary_text.delta',
    'event.name="codex.sse_event"',
    'codex_otel',
]

# Compiled regex patterns for more complex matching
_NOISY_LOG_PATTERNS = [
    # Filter all codex_otel verbose SSE event logs (most aggressive - catches everything from this module)
    re.compile(r'.*codex_otel::otel_event_manager.*', re.IGNORECASE),
    # Filter reasoning summary delta events
    re.compile(r'.*event\.kind=response\.reasoning_summary_text\.delta.*', re.IGNORECASE),
    # Filter any codex.sse_event logs
    re.compile(r'.*event\.name="codex\.sse_event".*', re.IGNORECASE),
    # Also catch logs that start with timestamp and have codex_otel
    re.compile(r'^\d{4}-\d{2}-\d{2}T.*codex_otel.*', re.IGNORECASE),
]


def should_filter_log_line(line: str) -> bool:
    """Check if a log line should be filtered out.
    
    Filters out noisy logs like verbose codex_otel SSE event logs.
    Uses both substring matching (faster) and regex (more precise).
    """
    if not line.strip():
        return False
    
    # Fast substring check first
    line_lower = line.lower()
    for substr in _NOISY_LOG_SUBSTRINGS:
        if substr.lower() in line_lower:
            return True
    
    # Fallback to regex for edge cases
    return any(pattern.search(line) for pattern in _NOISY_LOG_PATTERNS)


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
        lines = self._buffer.split('\n')
        # Keep the last incomplete line in buffer
        self._buffer = lines[-1]
        # Process complete lines
        written = 0
        for line in lines[:-1]:
            if not should_filter_log_line(line):
                written += self._original_write(line + '\n')
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

