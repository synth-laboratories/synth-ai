"""Capabilities, taxonomy, sharing, usage, profiles, rewards and contest wire mirrors.

Mirrors backend ``packages/contributions/{views,usage}.py``. Public search stays
free even for paid organizations. Rewards are awarded Synth cloud credits from
reviewed programs — never cash, earnings owed or attribution estimates.
"""

import datetime as dt
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, StrictInt, StringConstraints

from .contracts import (
    ContributionKind,
    ContributionReference,
    Identifier,
    IndexContract,
    ResearchArea,
    ShortText,
    WorkflowStage,
)
from .lifecycle import Capability, Digest, Title
from .search import SearchMode

Count = Annotated[StrictInt, Field(ge=0)]

# Capabilities ----------------------------------------------------------------


class PrivateSearchCapability(IndexContract):
    activated: bool
    price_cents_per_search: StrictInt | None = None


class FeatureCapability(IndexContract):
    enabled: bool


class SearchLimits(IndexContract):
    max_results: StrictInt
    max_excerpts_per_result: StrictInt
    query_max_bytes: StrictInt
    contents_max_bytes: StrictInt


class Capabilities(IndexContract):
    api_version: Literal["synth.index.api.v1"] = "synth.index.api.v1"
    contribution_schema_versions: tuple[Identifier, ...]
    taxonomy_version: Identifier
    modes: tuple[SearchMode, ...]
    search_modes: tuple[SearchMode, ...]
    visibilities: tuple[Literal["public", "private"], ...]
    deep_search: bool = False
    search_filters: bool
    private_search: PrivateSearchCapability
    upload: FeatureCapability
    review: FeatureCapability
    publication: FeatureCapability
    limits: SearchLimits
    contest_id: Identifier | None = None
    viewer_capabilities: tuple[Capability, ...] = ()


# Taxonomy and sharing ------------------------------------------------------------


class TagView(IndexContract):
    tag_id: Identifier
    slug: Identifier
    label: Title
    definition: ShortText
    public_count: Count | None = None


class TagRegistry(IndexContract):
    registry_version: Identifier
    tags: tuple[TagView, ...] = Field(max_length=256)
    kinds: tuple[ContributionKind, ...]
    research_areas: tuple[ResearchArea, ...]
    workflow_stages: tuple[WorkflowStage, ...]


class CollectionView(IndexContract):
    collection_id: Identifier
    contribution_id: Identifier
    name: Title
    audience: Literal["private", "org"]
    contribution_count: StrictInt = 1
    can_manage: bool = False


class Collections(IndexContract):
    items: tuple[CollectionView, ...] = Field(max_length=100)


class CollectionGrantSpec(IndexContract):
    """Only ``user`` subjects are supported; ``org`` is rejected by the backend."""

    subject_kind: Literal["user", "org"] = "user"
    subject_id: Identifier


class CollectionGrant(IndexContract):
    grant_id: Identifier
    collection_id: Identifier
    subject_kind: Literal["user"]
    subject_id: Identifier
    operations: tuple[str, ...]
    created_at: AwareDatetime


class CollectionGrants(IndexContract):
    items: tuple[CollectionGrant, ...] = Field(max_length=200)


class CollectionGrantRevoked(IndexContract):
    grant_id: Identifier
    revoked: Literal[True] = True


# Usage ---------------------------------------------------------------------------


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


PromoCreditStatus = Literal["active", "exhausted", "expired", "revoked", "campaign_ended"]


class PrivatePromoCredit(IndexContract):
    """Promotional private-search allowance: never cash, earnings or carryover.

    ``remaining_searches`` is what the balance can still fund at
    ``unit_price_cents``. When it reaches zero a private search is refused with
    ``IndexErrorCode.PRIVATE_CREDIT_EXHAUSTED`` unless the organization has paid
    authority, and ``resets_at`` says when the next allocation lands.
    """

    campaign_id: str
    status: PromoCreditStatus
    period_start: AwareDatetime
    resets_at: AwareDatetime | None
    expires_at: AwareDatetime
    allocated_cents: Count
    consumed_cents: Count
    remaining_cents: Count
    unit_price_cents: Count
    remaining_searches: Count
    paid_overflow_enabled: StrictBool
    carryover: Literal[False] = False
    withdrawable: Literal[False] = False


class PromoCreditSummary(IndexContract):
    """``credit`` is null when the organization is not enrolled in the promotion."""

    credit: PrivatePromoCredit | None


class AccessFundingMode(IndexContract):
    mode: SearchMode
    access: StrictBool
    wallet_enabled: StrictBool
    monthly_cap_cents: Count
    concurrency_limit: Annotated[StrictInt, Field(ge=1, le=32)]
    consent_terms_version: str | None = None
    policy_revision: Count | None = None


class DeepBetaFunding(IndexContract):
    grant_id: str
    cohort: str
    status: Literal["active", "revoked", "expired"]
    monthly_units: Count
    expires_at: AwareDatetime
    reserved_units: Count
    consumed_units: Count


class AccessFundingAccount(IndexContract):
    org_id: str
    can_manage_policy: StrictBool
    modes: tuple[AccessFundingMode, ...]
    deep_beta: DeepBetaFunding | None = None
    live_wallet_holds_microcents: Count
    wallet_available_microcents: Count
    generated_at: AwareDatetime


class BillingPolicyUpdate(IndexContract):
    wallet_enabled: StrictBool = False
    monthly_cap_cents: Count = 0
    concurrency_limit: Annotated[StrictInt, Field(ge=1, le=32)] = 1
    consent_terms_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]


# Profiles ------------------------------------------------------------------------


class ProfileSpec(IndexContract):
    display_name: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    github_login: (
        Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")] | None
    ) = None


class ProfilePinsSpec(IndexContract):
    contribution_ids: tuple[Identifier, ...] = Field(default=(), max_length=6)


class GithubIdentity(IndexContract):
    login: str
    verified: bool


class ProfileCard(IndexContract):
    reference: ContributionReference
    title: Title | None = None
    kind: ContributionKind | None = None
    published_at: AwareDatetime | None = None
    tag_ids: tuple[Identifier, ...] = Field(default=(), max_length=10)


class ActivityDay(IndexContract):
    date: dt.date
    count: Annotated[StrictInt, Field(ge=1)]


class ReuseSummary(IndexContract):
    """Counts below ``suppressed_below`` are returned as None for privacy."""

    window_days: StrictInt
    organic_search_appearances: Count | None = None
    organic_contents_reads: Count | None = None
    suppressed_below: StrictInt
    definition: ShortText


class ProfileView(IndexContract):
    principal_id: Identifier
    display_name: str | None = None
    github: GithubIdentity | None = None
    pinned: tuple[ProfileCard, ...] = Field(default=(), max_length=6)
    accepted: tuple[ProfileCard, ...] = Field(default=(), max_length=200)
    accepted_activity: tuple[ActivityDay, ...] = Field(default=(), max_length=366)
    reuse: ReuseSummary


# Rewards -------------------------------------------------------------------------

Cents = Annotated[StrictInt, Field(ge=1, le=100_000)]


class RewardAwardSpec(IndexContract):
    program_id: Identifier
    reference: ContributionReference
    amount_cents: Cents
    expires_at: AwareDatetime
    reason: ShortText


class RewardReverseSpec(IndexContract):
    reason: ShortText


class RewardAward(IndexContract):
    award_id: Identifier
    program_id: Identifier
    reference: ContributionReference
    recipient_user_id: Identifier
    amount_cents: Cents
    expires_at: AwareDatetime
    status: Literal["awarded", "reversed"]
    awarded_at: AwareDatetime
    reversed_at: AwareDatetime | None = None


class RewardEntry(IndexContract):
    award_id: Identifier
    program: Identifier
    credits_cents: Cents
    status: Literal["awarded", "expired", "reversed"]
    awarded_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    reference: ContributionReference | None = None
    receipt_id: Identifier


class MyRewards(IndexContract):
    """Awarded cloud credits; ``available_credits_cents`` is not reduced by later spend."""

    unit: Literal["synth_cloud_credit_cents"] = "synth_cloud_credit_cents"
    available_credits_cents: Count
    awarded_cents: Count
    reversed_cents: Count
    entries: tuple[RewardEntry, ...] = Field(max_length=500)
    note: str


# Contest -------------------------------------------------------------------------


class ContestSpec(IndexContract):
    contest_id: Identifier
    title: Title
    rules_digest: Digest
    closes_at: AwareDatetime


class ContestStatusSpec(IndexContract):
    status: Literal["open", "closed"]


class ContestView(IndexContract):
    contest_id: Identifier
    title: Title
    status: Literal["draft", "open", "closed"]
    rules_version: Digest
    rules_digest: Digest
    opens_at: AwareDatetime | None = None
    closes_at: AwareDatetime


class ContestEntrySpec(IndexContract):
    reference: ContributionReference


class ContestEntry(IndexContract):
    entry_id: Identifier
    contest_id: Identifier
    reference: ContributionReference
    manifest_digest: Digest
    submitted_by: Identifier
    status: Literal["submitted", "scored", "reviewed", "disqualified"]
    score: float | None = None
    evaluator_receipt_digest: Digest | None = None
    submitted_at: AwareDatetime


class ContestScoreSpec(IndexContract):
    score: Annotated[float, Field(ge=0, le=1e9, allow_inf_nan=False)]
    evaluator_receipt_digest: Digest


class ContestReviewSpec(IndexContract):
    decision: Literal["accept", "disqualify"]
    comments: ShortText


class LeaderboardEntry(IndexContract):
    rank: Annotated[StrictInt, Field(ge=1)]
    entry_id: Identifier
    reference: ContributionReference
    title: Title | None = None
    submitted_by: Identifier
    score: float
    score_label: str
    reviewed_at: AwareDatetime | None = None
    evaluation_receipt_id: Digest


class Leaderboard(IndexContract):
    contest_id: Identifier
    status: Literal["open", "closed"]
    entries: tuple[LeaderboardEntry, ...] = Field(max_length=500)
