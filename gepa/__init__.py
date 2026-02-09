"""GEPA compatibility layer backed by Synth AI."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

from .adapters import default_adapter
from .api import optimize
from .core.adapter import EvaluationBatch, GEPAAdapter
from .core.result import GEPAResult
from .examples import aime, banking77
from .utils.stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxMetricCallsStopper,
    NoImprovementStopper,
    ScoreThresholdStopper,
    SignalStopper,
    StopperProtocol,
    TimeoutStopCondition,
)

__all__ = [
    "CompositeStopper",
    "EvaluationBatch",
    "FileStopper",
    "GEPAAdapter",
    "GEPAResult",
    "MaxMetricCallsStopper",
    "NoImprovementStopper",
    "ScoreThresholdStopper",
    "SignalStopper",
    "StopperProtocol",
    "TimeoutStopCondition",
    "aime",
    "banking77",
    "default_adapter",
    "optimize",
]
