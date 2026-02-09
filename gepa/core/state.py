"""Minimal GEPA state containers for compatibility helpers."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ProgramIdx = int


@dataclass
class GEPAState:
    """Compatibility snapshot used by stop conditions."""

    program_full_scores_val_set: list[float] = field(default_factory=list)
    total_num_evals: int = 0
    program_candidates: list[dict[str, Any]] = field(default_factory=list)
