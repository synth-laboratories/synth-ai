"""Public Research API for Synth.

Canonical entry: ``SynthClient().research``.
"""

from __future__ import annotations

from synth_ai.research.async_client import AsyncResearchClient
from synth_ai.research.client import ResearchClient
from synth_ai.research.enums import ResearchHostKind, ResearchWorkMode
from synth_ai.research.errors import (
    ResearchApiError,
    ResearchConcurrentRunLimitExceededError,
    ResearchInsufficientCreditsError,
    ResearchLimitExceededError,
    ResearchProjectMonthlyBudgetExhaustedError,
    ResearchStructuredDenialError,
)
from synth_ai.research.factories import ResearchFactoriesAPI
from synth_ai.research.limits import ResearchLimitsAPI
from synth_ai.research.models import (
    ResearchCreateProjectResult,
    ResearchProject,
    ResearchRun,
    ResearchRunbookPreset,
    ResearchWorkProduct,
)
from synth_ai.research.projects import ResearchProjectsAPI
from synth_ai.research.runs import ResearchRunHandle, ResearchRunsAPI, ResearchRunSession
from synth_ai.research.secrets import ResearchSecretsAPI

__all__ = [
    "AsyncResearchClient",
    "ResearchApiError",
    "ResearchClient",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchCreateProjectResult",
    "ResearchHostKind",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchProject",
    "ResearchProjectMonthlyBudgetExhaustedError",
    "ResearchFactoriesAPI",
    "ResearchLimitsAPI",
    "ResearchProjectsAPI",
    "ResearchRun",
    "ResearchRunHandle",
    "ResearchRunSession",
    "ResearchRunbookPreset",
    "ResearchRunsAPI",
    "ResearchSecretsAPI",
    "ResearchStructuredDenialError",
    "ResearchWorkMode",
    "ResearchWorkProduct",
]
