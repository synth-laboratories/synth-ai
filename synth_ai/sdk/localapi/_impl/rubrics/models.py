"""Rubric and Criterion data models.

BACKWARD COMPATIBILITY SHIM: This module now re-exports from synth_ai.data.rubrics.
All classes have moved to synth_ai.data.rubrics as the canonical location.

For new code, import directly from synth_ai.data.rubrics:
    from synth_ai.data.rubrics import Criterion, Rubric

This module is preserved for backward compatibility with existing imports.
"""

from __future__ import annotations

# Re-export rubric classes from the canonical location
from synth_ai.data.rubrics import Criterion, Rubric

__all__ = [
    "Criterion",
    "Rubric",
]
