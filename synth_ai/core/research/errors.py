"""Research API errors (public names; ``Smr*`` compatibility aliases).

Catch these typed exceptions from ``SynthClient().research`` call sites.

| Exception | Typical cause |
| --- | --- |
| ``ResearchApiError`` | Base API error with structured ``message`` |
| ``ResearchStructuredDenialError`` | Policy or preflight denial |
| ``ResearchLimitExceededError`` | Org or project limit exceeded |
| ``ResearchConcurrentRunLimitExceededError`` | Too many concurrent runs |
| ``ResearchInsufficientCreditsError`` | Insufficient account credits |
| ``ResearchProjectMonthlyBudgetExhaustedError`` | Project monthly budget exhausted |
"""

from __future__ import annotations

from synth_ai.core.research._legacy.errors import (
    SmrApiError,
    SmrConcurrentRunLimitExceededError,
    SmrInsufficientCreditsError,
    SmrLimitExceededError,
    SmrProjectMonthlyBudgetExhaustedError,
    SmrStructuredDenialError,
)

ResearchApiError = SmrApiError
ResearchStructuredDenialError = SmrStructuredDenialError
ResearchLimitExceededError = SmrLimitExceededError
ResearchConcurrentRunLimitExceededError = SmrConcurrentRunLimitExceededError
ResearchInsufficientCreditsError = SmrInsufficientCreditsError
ResearchProjectMonthlyBudgetExhaustedError = SmrProjectMonthlyBudgetExhaustedError

# Compatibility aliases for migration from managed-research imports.
SmrApiError = ResearchApiError
SmrStructuredDenialError = ResearchStructuredDenialError
SmrLimitExceededError = ResearchLimitExceededError
SmrConcurrentRunLimitExceededError = ResearchConcurrentRunLimitExceededError
SmrInsufficientCreditsError = ResearchInsufficientCreditsError
SmrProjectMonthlyBudgetExhaustedError = ResearchProjectMonthlyBudgetExhaustedError

__all__ = [
    "ResearchApiError",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchProjectMonthlyBudgetExhaustedError",
    "ResearchStructuredDenialError",
    "SmrApiError",
    "SmrConcurrentRunLimitExceededError",
    "SmrInsufficientCreditsError",
    "SmrLimitExceededError",
    "SmrProjectMonthlyBudgetExhaustedError",
    "SmrStructuredDenialError",
]
