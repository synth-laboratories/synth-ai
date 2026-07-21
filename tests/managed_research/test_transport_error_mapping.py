"""Transport error mapping: backend 402 billing blocker codes -> typed SDK errors.

The backend never emits ``smr_insufficient_credits`` from the billing admission
gate; real credit denials carry the wallet/allowance ledger blocker codes built
by ``services/smr/billing/admission.py::billing_blocker_detail``. These tests
pin the SDK-side mapping of that family to ``SmrInsufficientCreditsError``
(aliased publicly as ``ResearchInsufficientCreditsError``) and prove that
factory budget codes and unknown codes remain ``SmrStructuredDenialError``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from synth_ai.managed_research.errors import (
    SmrConcurrentRunLimitExceededError,
    SmrInsufficientCreditsError,
    SmrStructuredDenialError,
)
from synth_ai.managed_research.transport.http import _raise_for_error_response
from synth_ai.research.errors import ResearchInsufficientCreditsError


def _billing_blocker_detail(error_code: str) -> dict[str, Any]:
    """Mirror of backend billing_blocker_detail (admission.py:184-205)."""
    return {
        "error_code": error_code,
        "message": (
            "SMR usage is exhausted for this organization. Wait for the next "
            "owner-approved allowance reset, top up flex credits, upgrade the "
            "plan, or ask an admin for a manual grant."
        ),
        "billing_preflight": {
            "org_id": "org_123",
            "surface": "run",
            "project_id": "proj_456",
            "run_id": None,
            "factory_effort_id": None,
            "dev_environment_id": None,
            "model_class": "premium",
            "estimated_customer_debit_microcents": 500_000_000,
            "allowed": False,
            "blocked_reason": error_code,
            "debit_pool_order": ["five_hour", "weekly", "wallet"],
            "selected_debit_pool": None,
            "available_microcents": 0,
            "wallet_balance_microcents": 0,
            "generated_at": "2026-07-20T12:00:00+00:00",
            "factory_run_admission": None,
        },
        "retryable": False,
    }


def _response(status_code: int, detail: dict[str, Any]) -> httpx.Response:
    request = httpx.Request("POST", "https://backend.example/api/smr/runs")
    return httpx.Response(
        status_code,
        request=request,
        content=json.dumps({"detail": detail}).encode(),
        headers={"content-type": "application/json"},
    )


INSUFFICIENT_CREDITS_CODES = [
    "smr_insufficient_credits",
    "smr_allowance_and_wallet_insufficient",
    "smr_allowance_and_wallet_exhausted",
    "smr_wallet_exhausted",
    "smr_allowance_manual_reset_required",
    "smr_billing_blocked",
    "smr_allowance_unprovisioned_premium",
    "smr_allowance_unprovisioned_value",
    "smr_window_exhausted_five_hour",
]


@pytest.mark.parametrize("code", INSUFFICIENT_CREDITS_CODES)
def test_wallet_allowance_402_codes_raise_insufficient_credits(code: str) -> None:
    detail = _billing_blocker_detail(code)
    response = _response(402, detail)
    with pytest.raises(SmrInsufficientCreditsError) as excinfo:
        _raise_for_error_response(response)
    exc = excinfo.value
    assert exc.status_code == 402
    assert exc.detail == detail
    assert exc.detail["billing_preflight"]["blocked_reason"] == code
    assert exc.detail["error_code"] == code
    assert "exhausted" in str(exc)


def test_public_alias_catches_mapped_402() -> None:
    response = _response(402, _billing_blocker_detail("smr_wallet_exhausted"))
    with pytest.raises(ResearchInsufficientCreditsError):
        _raise_for_error_response(response)


@pytest.mark.parametrize(
    "code",
    ["factory_budget_exhausted", "factory_run_budget_exhausted"],
)
def test_factory_budget_402_codes_stay_structured_denials(code: str) -> None:
    detail = _billing_blocker_detail(code)
    response = _response(402, detail)
    with pytest.raises(SmrStructuredDenialError) as excinfo:
        _raise_for_error_response(response)
    exc = excinfo.value
    assert not isinstance(exc, SmrInsufficientCreditsError)
    assert exc.status_code == 402
    assert exc.detail == detail


def test_unmapped_code_stays_structured_denial() -> None:
    detail = {"error_code": "smr_some_future_code", "message": "denied"}
    response = _response(403, detail)
    with pytest.raises(SmrStructuredDenialError) as excinfo:
        _raise_for_error_response(response)
    exc = excinfo.value
    assert not isinstance(exc, SmrInsufficientCreditsError)
    assert exc.status_code == 403
    assert exc.detail == detail


def test_concurrent_run_limit_429_mapping_unchanged() -> None:
    detail = {"error_code": "smr_concurrent_run_limit_exceeded", "message": "too many"}
    response = _response(429, detail)
    with pytest.raises(SmrConcurrentRunLimitExceededError) as excinfo:
        _raise_for_error_response(response)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == detail
