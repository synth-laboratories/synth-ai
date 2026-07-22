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

import importlib

from synth_ai.core.errors import (
    AuthorizationError,
    ConflictError,
    ContractMismatchError,
    RateLimitedError,
    ResearchOperationError,
    ResourceExhaustedError,
    RetryDirective,
    SynthError,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
    TransientServiceError,
)

ResearchApiError = SynthError
SmrApiError = ResearchApiError

_COMPATIBILITY_ERRORS = {
    "ResearchConcurrentRunLimitExceededError": "SmrConcurrentRunLimitExceededError",
    "ResearchInsufficientCreditsError": "SmrInsufficientCreditsError",
    "ResearchLimitExceededError": "SmrLimitExceededError",
    "ResearchProjectMonthlyBudgetExhaustedError": ("SmrProjectMonthlyBudgetExhaustedError"),
    "ResearchStructuredDenialError": "SmrStructuredDenialError",
    "SmrConcurrentRunLimitExceededError": "SmrConcurrentRunLimitExceededError",
    "SmrInsufficientCreditsError": "SmrInsufficientCreditsError",
    "SmrLimitExceededError": "SmrLimitExceededError",
    "SmrProjectMonthlyBudgetExhaustedError": ("SmrProjectMonthlyBudgetExhaustedError"),
    "SmrStructuredDenialError": "SmrStructuredDenialError",
}


def __getattr__(name: str) -> object:
    legacy_name = _COMPATIBILITY_ERRORS.get(name)
    if legacy_name is None:
        raise AttributeError(name)
    value = getattr(
        importlib.import_module("synth_ai.core.research._legacy.errors"),
        legacy_name,
    )
    globals()[name] = value
    return value


__all__ = [
    "ResearchApiError",
    "AuthorizationError",
    "ConflictError",
    "ContractMismatchError",
    "RateLimitedError",
    "ResearchOperationError",
    "ResourceExhaustedError",
    "RetryDirective",
    "SynthErrorCategory",
    "SynthErrorCode",
    "SynthFailure",
    "TransientServiceError",
    "SmrApiError",
]
