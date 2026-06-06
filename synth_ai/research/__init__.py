"""Public Research API for Synth (alpha).

Canonical entry: ``SynthClient().research``.

Alpha bootstrap delegates HTTP to ``managed-research`` until push 4 ports transport.
"""

from __future__ import annotations

from synth_ai.research.client import ResearchClient
from synth_ai.research.control import ResearchControlClient
from synth_ai.research.enums import ResearchHostKind, ResearchWorkMode
from synth_ai.research.errors import (
    ResearchApiError,
    ResearchConcurrentRunLimitExceededError,
    ResearchInsufficientCreditsError,
    ResearchLimitExceededError,
    ResearchProjectMonthlyBudgetExhaustedError,
    ResearchStructuredDenialError,
)
from synth_ai.research.models import (
    ResearchCreateProjectResult,
    ResearchProject,
    ResearchRun,
    ResearchRunbookPreset,
    ResearchWorkProduct,
)
from synth_ai.research.projects import ResearchProjectsAPI
from synth_ai.research.runs import ResearchRunHandle, ResearchRunsAPI

__all__ = [
    "ResearchApiError",
    "ResearchClient",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchControlClient",
    "ResearchCreateProjectResult",
    "ResearchHostKind",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchProject",
    "ResearchProjectMonthlyBudgetExhaustedError",
    "ResearchProjectsAPI",
    "ResearchRun",
    "ResearchRunHandle",
    "ResearchRunbookPreset",
    "ResearchRunsAPI",
    "ResearchStructuredDenialError",
    "ResearchWorkMode",
    "ResearchWorkProduct",
]
