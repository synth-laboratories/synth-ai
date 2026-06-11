"""Research API errors (public names; Smr* compatibility aliases)."""

from __future__ import annotations

from synth_ai.managed_research.errors import (
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
