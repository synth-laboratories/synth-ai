"""Contributor value wire mirrors; backend owns eligibility and scoring.

See backend notes/specifications/synth-index/contributor-value-visibility.md.
"""

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictBool, StrictInt, StringConstraints, model_validator

from .contracts import ContributionReference, Identifier, IndexContract

CLOUT_POLICY_VERSION = "synth.index.clout.v1"


def require_unique(values, name):
    """Reject duplicate identifiers in a bounded value field."""
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


class CloutEvidenceKind(StrEnum):
    """Canonical evidence kinds that can earn social recognition."""

    CONTRIBUTION = "qualified_contribution"
    REPRODUCTION = "verified_reproduction"
    REVIEW = "accepted_review"
    DELIVERED_USE = "organic_delivered_use"


class CloutEligibility(StrEnum):
    """Eligibility classifications supplied by the authoritative ledger."""

    QUALIFIED = "qualified"
    PENDING = "pending"
    SELF_USE = "self_use"
    INTERNAL = "internal_eval_demo"
    BOT = "bot"
    REVOKED = "revoked"


class CloutEvidence(IndexContract):
    """An exact-revision ledger event with identity and consent evidence."""

    policy_version: Literal["synth.index.clout.v1"] = CLOUT_POLICY_VERSION
    event_id: Identifier
    sequence: Annotated[StrictInt, Field(ge=1)]
    principal_id: Identifier
    reference: ContributionReference
    kind: CloutEvidenceKind
    receipt_id: Identifier
    eligibility: CloutEligibility
    identity_verified: StrictBool
    owner_consented: StrictBool
    occurred_at: AwareDatetime
    supersedes_event_id: Identifier | None = None

    @model_validator(mode="after")
    def complete_reference(self):
        """Require a revision and refuse an event that supersedes itself."""
        if self.reference.revision_id is None:
            raise ValueError("Clout evidence requires an exact revision")
        if self.supersedes_event_id == self.event_id:
            raise ValueError("Clout event cannot supersede itself")
        return self


class CloutDay(IndexContract):
    """Qualified social recognition earned on one UTC day."""

    day: date
    points: Annotated[StrictInt, Field(ge=1)]


class PublicClout(IndexContract):
    """Only the points and calendar approved for public disclosure."""

    policy_version: Literal["synth.index.clout.v1"] = CLOUT_POLICY_VERSION
    points: Annotated[StrictInt, Field(ge=0)]
    calendar: tuple[CloutDay, ...]


class OwnClout(IndexContract):
    """Separate authorized public and private points from pending evidence."""

    policy_version: Literal["synth.index.clout.v1"] = CLOUT_POLICY_VERSION
    public_points: Annotated[StrictInt, Field(ge=0)]
    private_points: Annotated[StrictInt, Field(ge=0)]
    pending_events: Annotated[StrictInt, Field(ge=0)]
    excluded_events: Annotated[StrictInt, Field(ge=0)]
    public_calendar: tuple[CloutDay, ...]
    private_calendar: tuple[CloutDay, ...] = ()


class ProfileAffiliation(IndexContract):
    """A self-declared affiliation with explicit disclosure consent."""

    label: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    public: StrictBool = False


class ProfileVisibility(IndexContract):
    """Private-by-default owner consent for individual profile surfaces."""

    public_identity: StrictBool = False
    public_display_name: StrictBool = False
    public_github: StrictBool = False
    public_calendar: StrictBool = False
    public_reuse: StrictBool = False
    public_clout: StrictBool = False
    public_card_ids: tuple[Identifier, ...] = Field(default=(), max_length=200)
    affiliations: tuple[ProfileAffiliation, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def unique_cards(self) -> Self:
        """Refuse repeated identifiers in the public card selection."""
        require_unique(self.public_card_ids, "public_card_ids")
        return self


class OwnCloutPage(IndexContract):
    """A bounded page of current-organization ledger evidence and totals."""

    summary: OwnClout
    items: tuple[CloutEvidence, ...] = Field(max_length=200)
    next_cursor: Identifier | None = None
    scope: str = "Currently authorized revisions; private reads follow the current organization"


class PublicProfileValue(IndexContract):
    """Consented public recognition and explicitly unverified affiliations."""

    clout: PublicClout | None = None
    affiliations: tuple[str, ...] = Field(default=(), max_length=16)
    affiliation_status: str = "Self-declared; not verified affiliations"


class StarterPreference(IndexContract):
    """Opt-in preference for manual research briefs."""

    enabled: StrictBool = False
    selected_brief_id: Identifier | None = None


class StarterBrief(IndexContract):
    """A bounded manual research brief with rights and QA requirements."""

    brief_id: Identifier
    title: str
    scope: str
    steps: tuple[str, ...]
    budget: str
    rights: str
    qa: str
    eligibility: str
    submission_path: Literal["/index/contribute"] = "/index/contribute"


class StarterState(IndexContract):
    """Current allowance and opt-in guidance without initiating work or charges."""

    preference: StarterPreference
    allowance_status: Literal["unknown", "available", "near_limit", "exhausted", "expired"]
    remaining_units: Annotated[StrictInt, Field(ge=0)] | None = None
    allowance_expires_at: AwareDatetime | None = None
    show_offer: bool
    briefs: tuple[StarterBrief, ...]
    note: str = "DEEP beta allowance includes live reservations. Other quotas are not inferred. No work or charge starts when selecting a brief."
