"""Capability discovery, tags, collections and account projections (wire mirrors).

``IndexUsageSummary`` mirrors backend ``packages/contributions/usage.py``: public
search stays free even for paid organizations; private spend is reported, not
inferred. Rewards are awarded cloud credits, never a cash balance or estimate.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, StrictInt, StringConstraints

from .contracts import ContributionAudience, Identifier, IndexContract

Count = Annotated[StrictInt, Field(ge=0)]
Label = Annotated[str, StringConstraints(min_length=1, max_length=200)]


class IndexLimits(IndexContract):
    max_results: Count
    max_excerpts_per_result: Count
    contents_max_bytes: Count
    query_max_bytes: Count


class IndexCapabilities(IndexContract):
    schema_versions: dict[str, str]
    search_modes: tuple[Literal["fast", "deep"], ...]
    visibilities: tuple[Literal["public", "private"], ...]
    limits: IndexLimits
    operations: tuple[str, ...] = Field(max_length=256)


class Tag(IndexContract):
    tag_id: Identifier
    label: Label


class TagList(IndexContract):
    items: tuple[Tag, ...] = Field(max_length=1000)


class Collection(IndexContract):
    collection_id: Identifier
    name: Label
    visibility: ContributionAudience


class CollectionList(IndexContract):
    items: tuple[Collection, ...] = Field(max_length=1000)


class PublicUsage(IndexContract):
    searches_used: Count
    searches_limit: Count
    contents_used: Count
    contents_limit: Count
    requests_per_minute: Count


class PrivateUsage(IndexContract):
    enabled: StrictBool
    unit_price_cents: Count
    requests_per_minute: Count
    monthly_spend_cap_cents: Count | None = None
    spent_cents: Count
    successful_searches: Count
    spendable_wallet_cents: Count | None = None


class IndexUsageSummary(IndexContract):
    period_start: AwareDatetime
    resets_at: AwareDatetime
    public: PublicUsage
    private: PrivateUsage


class RewardAward(IndexContract):
    award_id: Identifier
    program_id: Identifier
    amount_cents: Count
    status: Identifier
    awarded_at: AwareDatetime
    expires_at: AwareDatetime | None = None


class RewardsSummary(IndexContract):
    """Awarded cloud credits only; not cash earnings or attribution estimates."""

    balance_cents: Count
    items: tuple[RewardAward, ...] = Field(max_length=1000)
