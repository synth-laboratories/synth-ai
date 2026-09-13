"""Backend-authored Index allowance and spend; never contributor earnings.

See sibling docs/drafts/synth-index-pricing-economics-2026-09-12.md.
Mirror backend packages/contributions/usage.py without computing balances or
assuming promotional eligibility on the client.
"""

from typing import Annotated

from pydantic import AwareDatetime, Field, StrictInt

from .contracts import IndexContract

NonNegative = Annotated[StrictInt, Field(ge=0)]


class PublicUsage(IndexContract):
    searches_used: NonNegative
    searches_limit: NonNegative
    contents_used: NonNegative
    contents_limit: NonNegative
    requests_per_minute: NonNegative


class PrivateUsage(IndexContract):
    enabled: bool
    unit_price_cents: NonNegative
    requests_per_minute: NonNegative
    monthly_spend_cap_cents: NonNegative | None
    spent_cents: NonNegative
    successful_searches: NonNegative
    spendable_wallet_cents: NonNegative | None


class IndexUsageSummary(IndexContract):
    period_start: AwareDatetime
    resets_at: AwareDatetime
    public: PublicUsage
    private: PrivateUsage
