"""Backward compatibility layer for synth_ai.experiment_queue.

This module has been moved to synth_ai.cli.local.experiment_queue/.
Imports from this location are deprecated and will be removed in v0.4.0.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "synth_ai.experiment_queue is deprecated. Use synth_ai.cli.local.experiment_queue instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from synth_ai.cli.local.experiment_queue import (
    Base,
    Experiment,
    ExperimentJob,
    ExperimentJobStatus,
    ExperimentStatus,
    Trial,
    TrialStatus,
    get_engine,
    get_session,
    init_db,
    session_scope,
)

__all__ = [
    "Base",
    "Experiment",
    "ExperimentJob",
    "ExperimentJobStatus",
    "ExperimentStatus",
    "Trial",
    "TrialStatus",
    "get_engine",
    "get_session",
    "init_db",
    "session_scope",
]

