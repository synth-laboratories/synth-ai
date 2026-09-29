"""Opt-in wallet funding for keyed Index search (owner decision 2026-09-28).

A keyed caller of the MCP ``index_search`` tool gets the paid search only when its
organization has turned on wallet payments for the requested mode. Nobody is
charged without having opted in: without consent the tool refuses with
``index_wallet_consent_required`` before any paid request is sent.

The organization's consent and monthly cap are read from
``GET /api/v1/index/me/access-funding`` (``AccountAPI.access_funding``). The
backend stays authoritative: it re-checks consent and the per-call ceiling on
every paid search.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from synth_ai.core.errors import (
    RetryDirective,
    SynthError,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
)
from synth_ai.sdk.index.catalog import AccessFundingAccount, AccessFundingMode
from synth_ai.sdk.index.search import SearchMode

WALLET_CONSENT_REQUIRED_CODE = "index_wallet_consent_required"
#: Current Index wallet terms; the backend records the version an org accepted. It still
#: accepts the superseded ``synth-index-wallet-terms-2026-09-27`` and honours it for orgs
#: that consented to it; any other version is refused (422).
WALLET_TERMS_VERSION = "synth-index-wallet-terms-2026-09-28"
#: Published Fast price, used as the per-call ceiling for a wallet-funded Fast search.
FAST_WALLET_MAX_CHARGE_CENTS = 5
#: Published per-call ceiling for Deep (base price plus model cost), from the docs.
DEEP_WALLET_MAX_CHARGE_CEILING_CENTS = 25
WALLET_CONSENT_DOCS_URL = (
    "https://docs.usesynth.ai/synth-index/wallet-terms"
    "?utm_source=usesynth&utm_medium=mcp&utm_campaign=index-v02"
)
#: How long one access-funding read is reused before it is read again.
ACCESS_FUNDING_CACHE_SECONDS = 60.0


class WalletConsentReason(StrEnum):
    """Why a keyed search was not paid for."""

    MODE_UNAVAILABLE = "mode_unavailable"
    WALLET_OFF = "wallet_off"
    NO_CONSENT = "no_consent"
    CAP_TOO_LOW = "cap_too_low"


_REASON_TEXT: dict[WalletConsentReason, str] = {
    WalletConsentReason.MODE_UNAVAILABLE: "{mode} search is not available to your organization",
    WalletConsentReason.WALLET_OFF: "your organization has not turned on wallet payments for {mode} search",
    WalletConsentReason.NO_CONSENT: "your organization has not accepted the wallet terms for {mode} search",
    WalletConsentReason.CAP_TOO_LOW: "your organization's monthly limit for {mode} search is below the price of one search",
}


def wallet_consent_steps(mode: SearchMode) -> tuple[str, ...]:
    """Plain-language steps to turn on paid search for one mode."""
    return (
        f"Read the wallet terms: {WALLET_CONSENT_DOCS_URL}",
        f"An organization admin turns on wallet payments for {mode.value.title()} search, "
        "accepts the terms and sets a monthly limit (Python SDK: "
        "client.index.account.update_billing_policy(mode, BillingPolicyUpdate(...))).",
        "Make sure the wallet has funds, then run the search again.",
        "To search for free instead, use index_search without an API key.",
    )


class WalletConsentRequiredError(SynthError):
    """A keyed search was refused locally: the org has not opted in to paying for it.

    No paid request was sent and nothing was charged. ``detail`` carries the mode,
    the reason, the steps to enable it and the docs link.
    """

    def __init__(self, mode: SearchMode, reason: WalletConsentReason) -> None:
        why = _REASON_TEXT[reason].format(mode=mode.value.title())
        message = (
            f"Search with an API key is paid, and {why}, so nothing was run or charged. "
            f"To enable it, see {WALLET_CONSENT_DOCS_URL}"
        )
        super().__init__(
            message,
            failure=SynthFailure(
                code=SynthErrorCode(WALLET_CONSENT_REQUIRED_CODE),
                category=SynthErrorCategory.AUTHORIZATION,
                operation="index.search",
                request_id=None,
                correlation_id=None,
                retry=RetryDirective(retryable=False),
                status=None,
                detail=message,
                reason=reason.value,
            ),
        )
        self.mode = mode
        self.consent_reason = reason
        self.detail: dict[str, Any] = {
            "mode": mode.value,
            "reason": reason.value,
            "charged_cents": 0,
            "steps": list(wallet_consent_steps(mode)),
            "docs_url": WALLET_CONSENT_DOCS_URL,
        }


@dataclass(frozen=True, slots=True)
class WalletSearchGrant:
    """Consent is in place for ``mode``; charge at most ``max_charge_cents`` per call."""

    mode: SearchMode
    max_charge_cents: int
    consent_terms_version: str


def wallet_search_grant(account: AccessFundingAccount, mode: SearchMode) -> WalletSearchGrant:
    """Return the per-call grant for ``mode`` or raise ``WalletConsentRequiredError``."""
    funding: AccessFundingMode | None = next(
        (item for item in account.modes if item.mode == mode), None
    )
    if funding is None or not funding.access:
        raise WalletConsentRequiredError(mode, WalletConsentReason.MODE_UNAVAILABLE)
    if not funding.wallet_enabled:
        raise WalletConsentRequiredError(mode, WalletConsentReason.WALLET_OFF)
    if not funding.consent_terms_version:
        raise WalletConsentRequiredError(mode, WalletConsentReason.NO_CONSENT)
    if mode == SearchMode.FAST:
        ceiling = FAST_WALLET_MAX_CHARGE_CENTS
    else:
        ceiling = min(DEEP_WALLET_MAX_CHARGE_CEILING_CENTS, funding.monthly_cap_cents)
    if funding.monthly_cap_cents < FAST_WALLET_MAX_CHARGE_CENTS or ceiling < 1:
        raise WalletConsentRequiredError(mode, WalletConsentReason.CAP_TOO_LOW)
    return WalletSearchGrant(
        mode=mode,
        max_charge_cents=ceiling,
        consent_terms_version=funding.consent_terms_version,
    )


class AccessFundingCache:
    """Reuse one access-funding read for a short time; cheap per search call."""

    def __init__(
        self,
        ttl_seconds: float = ACCESS_FUNDING_CACHE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._value: AccessFundingAccount | None = None
        self._read_at = 0.0

    def get(self, read: Callable[[], AccessFundingAccount]) -> AccessFundingAccount:
        with self._lock:
            if self._value is not None and self._clock() - self._read_at < self._ttl:
                return self._value
        value = read()
        with self._lock:
            self._value = value
            self._read_at = self._clock()
        return value

    def invalidate(self) -> None:
        with self._lock:
            self._value = None
