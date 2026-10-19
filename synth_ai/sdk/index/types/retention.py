"""Customer-only retention observations. See backend SDK index_contract.md."""

from __future__ import annotations

from datetime import UTC, timedelta
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from synth_ai.types.base import ResponseBase, ResponseIdentifier

RetentionMode = Literal["zdr", "zdr_strict", "retained"]
# Content states this SDK knows. The backend may add more; unknown values are
# preserved on the model (never rejected) and treated as non-contradicting.
KNOWN_CONTENT_STATES = frozenset(
    {"processing", "available", "purged", "expired", "deleted", "unavailable"}
)
UNREADABLE_CONTENT_STATES = frozenset({"processing", "purged", "expired", "deleted", "unavailable"})


class SearchRetention(ResponseBase):
    """Tolerant retention observation: every field is optional.

    Older backends send ``policy_version``/``purpose``/``archived_at``/
    ``content_expires_at``/``retention_days``/``history_available``; newer ones add
    ``purges_at`` (zdr) and ``expires_at`` (retained). Absent fields are ``None``.
    """

    mode: RetentionMode | None = None
    policy_version: ResponseIdentifier | None = None
    purpose: Literal["customer_only"] | None = None
    content_state: str | None = Field(default=None, min_length=1, max_length=64)
    archived_at: AwareDatetime | None = None
    content_expires_at: AwareDatetime | None = None
    purges_at: AwareDatetime | None = None
    expires_at: AwareDatetime | None = None
    retention_days: Literal[30] | None = None
    history_available: bool = False

    @model_validator(mode="after")
    def validate_retention(self) -> Self:
        if self.mode in {"zdr", "zdr_strict"}:
            if (
                self.archived_at is not None
                or self.retention_days is not None
                or self.history_available
            ):
                raise ValueError("ZDR cannot advertise a retained archive")
            return self
        if self.mode != "retained" or self.retention_days is None:
            return self
        if self.history_available != (self.content_state == "available"):
            raise ValueError("history availability must match retained content state")
        if self.archived_at is None:
            if self.content_state == "available" or self.content_expires_at is not None:
                raise ValueError("available history requires its archive and expiry timestamps")
        elif self.content_expires_at != self.archived_at.astimezone(UTC) + timedelta(days=30):
            raise ValueError("retained content expiry must be exactly 30 days from archival")
        return self

    @property
    def content_unreadable(self) -> bool:
        """True only for a known state that forbids disclosing content."""
        return self.content_state in UNREADABLE_CONTENT_STATES


def validate_retention_observation(
    retention: SearchRetention | None, observed_at: AwareDatetime
) -> None:
    """Check archive lifetime against the server observation, never the client clock.

    See: the public contract snapshots in contracts/"""
    if retention is None:
        return
    if retention.archived_at is not None and retention.archived_at > observed_at:
        raise ValueError("retention cannot advertise future archival")
    content_expires_at = retention.content_expires_at or retention.expires_at
    if (
        retention.content_state == "available"
        and content_expires_at is not None
        and content_expires_at <= observed_at
    ):
        raise ValueError("retention cannot advertise expired content as available")
