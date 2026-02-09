"""Compatibility stop conditions for GEPA-style workflows."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

import contextlib
import os
import signal
import time
from typing import Any, Literal, Protocol, runtime_checkable

from gepa.core.state import GEPAState


@runtime_checkable
class StopperProtocol(Protocol):
    """Protocol for stop condition objects."""

    def __call__(self, gepa_state: GEPAState) -> bool:
        """Return True when optimization should stop."""
        ...


class TimeoutStopCondition(StopperProtocol):
    """Stop callback that stops after a specified timeout."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def __call__(self, gepa_state: GEPAState) -> bool:
        return time.time() - self.start_time > self.timeout_seconds


class FileStopper(StopperProtocol):
    """Stop callback that stops when a specific file exists."""

    def __init__(self, stop_file_path: str):
        self.stop_file_path = stop_file_path

    def __call__(self, gepa_state: GEPAState) -> bool:
        return os.path.exists(self.stop_file_path)

    def remove_stop_file(self) -> None:
        if os.path.exists(self.stop_file_path):
            os.remove(self.stop_file_path)


class ScoreThresholdStopper(StopperProtocol):
    """Stop callback that stops when a score threshold is reached."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, gepa_state: GEPAState) -> bool:
        current_best = max(gepa_state.program_full_scores_val_set, default=0.0)
        return current_best >= self.threshold


class NoImprovementStopper(StopperProtocol):
    """Stop callback that stops after max iterations without improvement."""

    def __init__(self, max_iterations_without_improvement: int):
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.best_score = float("-inf")
        self.iterations_without_improvement = 0

    def __call__(self, gepa_state: GEPAState) -> bool:
        current_score = max(gepa_state.program_full_scores_val_set, default=0.0)
        if current_score > self.best_score:
            self.best_score = current_score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        return self.iterations_without_improvement >= self.max_iterations_without_improvement

    def reset(self) -> None:
        self.iterations_without_improvement = 0


class SignalStopper(StopperProtocol):
    """Stop callback that stops when a signal is received."""

    def __init__(self, signals=None):
        self.signals = signals or [signal.SIGINT, signal.SIGTERM]
        self._stop_requested = False
        self._original_handlers: dict[int, Any] = {}
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            self._stop_requested = True

        for sig in self.signals:
            with contextlib.suppress(OSError, ValueError):
                self._original_handlers[sig] = signal.signal(sig, signal_handler)

    def __call__(self, gepa_state: GEPAState) -> bool:
        return self._stop_requested

    def cleanup(self) -> None:
        for sig, handler in self._original_handlers.items():
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, handler)


class MaxMetricCallsStopper(StopperProtocol):
    """Stop callback that stops after a maximum number of metric calls."""

    def __init__(self, max_metric_calls: int):
        self.max_metric_calls = max_metric_calls

    def __call__(self, gepa_state: GEPAState) -> bool:
        return gepa_state.total_num_evals >= self.max_metric_calls


class CompositeStopper(StopperProtocol):
    """Combine multiple stopping conditions."""

    def __init__(self, *stoppers: StopperProtocol, mode: Literal["any", "all"] = "any"):
        self.stoppers = stoppers
        self.mode = mode

    def __call__(self, gepa_state: GEPAState) -> bool:
        if self.mode == "any":
            return any(stopper(gepa_state) for stopper in self.stoppers)
        if self.mode == "all":
            return all(stopper(gepa_state) for stopper in self.stoppers)
        raise ValueError(f"Unknown mode: {self.mode}")
