"""Contributor value wire mirrors; backend owns eligibility and scoring.

See backend notes/specifications/synth-index/contributor-value-visibility.md.
"""

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictBool, StrictInt, StringConstraints, model_validator

from .contracts import ContributionReference, Identifier, IndexContract

CLOUT_POLICY_VERSION = "synth.index.clout.v1"


def require_unique(values: tuple[str, ...], name: str) -> None:
    """Reject duplicate identifiers in a bounded value field.

    Args:
        values: Identifier strings in the bounded field; an empty tuple is valid.
        name: Field name included in the validation error.

    Raises:
        ValueError: The tuple contains a repeated identifier.
    """
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

    #: Backend policy version used to classify and score this evidence.
    policy_version: Literal["synth.index.clout.v1"] = CLOUT_POLICY_VERSION
    #: Stable identifier of the canonical ledger event.
    event_id: Identifier
    #: Positive ledger sequence used for ordered history and pagination.
    sequence: Annotated[StrictInt, Field(ge=1)]
    #: Contributor identity to which the evidence belongs.
    principal_id: Identifier
    #: Exact Contribution revision authorized for this event.
    reference: ContributionReference
    #: Kind of accepted work or organic delivered reuse.
    kind: CloutEvidenceKind
    #: Identifier of the authoritative receipt supporting this event.
    receipt_id: Identifier
    #: Backend classification; pending and excluded evidence earns no points.
    eligibility: CloutEligibility
    #: Whether the authoritative policy verified the contributor identity.
    identity_verified: StrictBool
    #: Whether the owner consented to the relevant recognition disclosure.
    owner_consented: StrictBool
    #: Timezone-aware timestamp of the ledger event.
    occurred_at: AwareDatetime
    #: Earlier event corrected by this event, or None for an original event.
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

    #: UTC calendar date on which qualified recognition was recorded.
    day: date
    #: Nonnegative social recognition total; calendar entries contain positive points.
    points: Annotated[StrictInt, Field(ge=1)]


class PublicClout(IndexContract):
    """Only the points and calendar approved for public disclosure."""

    #: Backend policy version used to classify and score this evidence.
    policy_version: Literal["synth.index.clout.v1"] = CLOUT_POLICY_VERSION
    #: Nonnegative social recognition total; calendar entries contain positive points.
    points: Annotated[StrictInt, Field(ge=0)]
    #: Publicly consented UTC-day recognition entries.
    calendar: tuple[CloutDay, ...]


class OwnClout(IndexContract):
    """Separate authorized public and private points from pending evidence."""

    #: Backend policy version used to classify and score this evidence.
    policy_version: Literal["synth.index.clout.v1"] = CLOUT_POLICY_VERSION
    #: Recognition total eligible for public disclosure.
    public_points: Annotated[StrictInt, Field(ge=0)]
    #: Recognition total visible only to the authorized owner.
    private_points: Annotated[StrictInt, Field(ge=0)]
    #: Count of evidence awaiting qualification; not awarded points.
    pending_events: Annotated[StrictInt, Field(ge=0)]
    #: Count of evidence excluded by the current policy.
    excluded_events: Annotated[StrictInt, Field(ge=0)]
    #: Consented UTC-day recognition entries; a visibility preference is not an award.
    public_calendar: tuple[CloutDay, ...]
    #: Owner-only UTC-day recognition entries; empty when none are available.
    private_calendar: tuple[CloutDay, ...] = ()


class ProfileAffiliation(IndexContract):
    """A self-declared affiliation with explicit disclosure consent."""

    #: Self-declared affiliation label, limited to 120 characters.
    label: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    #: Whether the owner explicitly permits this affiliation to be public.
    public: StrictBool = False


class ProfileVisibility(IndexContract):
    """Private-by-default owner consent for individual profile surfaces."""

    #: Whether the contributor identity may be disclosed publicly.
    public_identity: StrictBool = False
    #: Whether the display name may be disclosed publicly.
    public_display_name: StrictBool = False
    #: Whether the GitHub identity may be disclosed publicly.
    public_github: StrictBool = False
    #: Consented UTC-day recognition entries; a visibility preference is not an award.
    public_calendar: StrictBool = False
    #: Whether eligible reuse information may be disclosed publicly.
    public_reuse: StrictBool = False
    #: Whether qualified public clout may be disclosed publicly.
    public_clout: StrictBool = False
    #: Unique explicitly selected public card identifiers; at most 200.
    public_card_ids: tuple[Identifier, ...] = Field(default=(), max_length=200)
    #: Explicit affiliation selections, or consented labels in a public response; at most 16.
    affiliations: tuple[ProfileAffiliation, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def unique_cards(self) -> Self:
        """Refuse repeated identifiers in the public card selection."""
        require_unique(self.public_card_ids, "public_card_ids")
        return self


class OwnCloutPage(IndexContract):
    """A bounded page of current-organization ledger evidence and totals."""

    #: Separate public/private totals and pending/excluded counts.
    summary: OwnClout
    #: Authorized current-organization evidence entries; at most 200 per page.
    items: tuple[CloutEvidence, ...] = Field(max_length=200)
    #: Opaque continuation sequence, or None when no next page exists.
    next_cursor: Identifier | None = None
    #: Server explanation of the current organization or manual brief scope.
    scope: str = "Currently authorized revisions; private reads follow the current organization"


class PublicProfileValue(IndexContract):
    """Consented public recognition and explicitly unverified affiliations."""

    #: Consented public recognition, or None when not publicly visible.
    clout: PublicClout | None = None
    #: Explicit affiliation selections, or consented labels in a public response; at most 16.
    affiliations: tuple[str, ...] = Field(default=(), max_length=16)
    #: Explicit notice that affiliation labels are self-declared, not verified.
    affiliation_status: str = "Self-declared; not verified affiliations"


class StarterPreference(IndexContract):
    """Opt-in preference for manual research briefs."""

    #: Whether the owner opted in to manual starter suggestions; starts no work.
    enabled: StrictBool = False
    #: Chosen manual brief identifier, or None when no brief is selected.
    selected_brief_id: Identifier | None = None


class StarterBrief(IndexContract):
    """A bounded manual research brief with rights and QA requirements."""

    #: Stable identifier of this manual research brief.
    brief_id: Identifier
    #: Human-readable brief title.
    title: str
    #: Server explanation of the current organization or manual brief scope.
    scope: str
    #: Ordered suggested manual research steps; no automatic execution.
    steps: tuple[str, ...]
    #: Human-readable budget guidance, not a charge authorization or allowance grant.
    budget: str
    #: Rights and permission requirements for evidence used in the brief.
    rights: str
    #: Review and reproduction requirements before acceptance.
    qa: str
    #: Backend classification; pending and excluded evidence earns no points.
    eligibility: str
    #: Frontend intake path for submitting the resulting Contribution.
    submission_path: Literal["/index/contribute"] = "/index/contribute"


class StarterState(IndexContract):
    """Current allowance and opt-in guidance without initiating work or charges."""

    #: Current owner opt-in and selected manual brief.
    preference: StarterPreference
    #: Actual DEEP allowance state; unknown means no allowance is inferred.
    allowance_status: Literal["unknown", "available", "near_limit", "exhausted", "expired"]
    #: Available DEEP beta units including live reservations, or None when unknown.
    remaining_units: Annotated[StrictInt, Field(ge=0)] | None = None
    #: Timezone-aware allowance expiry, or None when unavailable.
    allowance_expires_at: AwareDatetime | None = None
    #: Whether the server permits displaying the manual starter offer.
    show_offer: bool
    #: Manual briefs offered by the server under current eligibility.
    briefs: tuple[StarterBrief, ...]
    #: Server guidance distinguishing allowance, quotas and manual selection.
    note: str = "DEEP beta allowance includes live reservations. Other quotas are not inferred. No work or charge starts when selecting a brief."
