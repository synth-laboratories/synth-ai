"""Baseline file system for self-contained task evaluation.

This package provides abstractions for defining and executing baseline evaluations
without requiring deployed task apps. Supports both class-based and function-based
task runners with first-class train/val/test split support.
"""

from __future__ import annotations

from synth_ai.sdk.baseline.config import (
    BaselineConfig,
    BaselineResults,
    BaselineTaskRunner,
    DataSplit,
    TaskResult,
)

__all__ = [
    "BaselineConfig",
    "BaselineTaskRunner",
    "DataSplit",
    "TaskResult",
    "BaselineResults",
]

