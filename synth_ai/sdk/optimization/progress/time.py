"""Shared time formatting utilities for progress handlers."""


import time
from dataclasses import dataclass


@dataclass
class ProgressClock:
    """Track elapsed time and render a compact prefix."""

    start_time: float | None = None

    def now(self) -> float:
        current = time.time()
        if self.start_time is None:
            self.start_time = current
        return current

    def elapsed(self, now: float | None = None) -> float:
        if self.start_time is None:
            return 0.0
        if now is None:
            now = time.time()
        return max(0.0, now - self.start_time)

    def prefix(self, now: float | None = None, *, separator: str = "|") -> str:
        elapsed = int(self.elapsed(now))
        if elapsed >= 60:
            mins = elapsed // 60
            secs = elapsed % 60
            return f"{mins:2d}m {secs:02d}s {separator}"
        return f"    {elapsed:02d}s {separator}"

    def reset(self) -> None:
        self.start_time = None


class ProgressPrinter:
    """Centralized logging format for progress handlers."""

    def __init__(self, *, label: str | None = None, clock: ProgressClock | None = None) -> None:
        self._label = label
        self._clock = clock or ProgressClock()

    def now(self) -> float:
        return self._clock.now()

    def elapsed(self, now: float | None = None) -> float:
        return self._clock.elapsed(now)

    def log(self, message: str, *, now: float | None = None, separator: str = "|") -> None:
        timestamp = self._clock.now() if now is None else now
        prefix = self._clock.prefix(timestamp, separator=separator)
        if self._label:
            print(f"{prefix} [{self._label}] {message}")
        else:
            print(f"{prefix} {message}")
