"""Customer-only retention observations. See backend SDK index_contract.md."""

from __future__ import annotations

from datetime import UTC, timedelta
from typing import Literal, Self

from pydantic import AwareDatetime, model_validator

from synth_ai.types.base import Identifier, Model

RetentionMode = Literal["zdr", "retained"]


class SearchRetention(Model):
    mode: RetentionMode
    policy_version: Identifier
    purpose: Literal["customer_only"]
    content_state: Literal["processing", "available", "expired", "deleted", "unavailable"]
    archived_at: AwareDatetime | None
    content_expires_at: AwareDatetime | None
    retention_days: Literal[30] | None
    history_available: bool

    @model_validator(mode="after")
    def validate_retention(self) -> Self:
        if self.mode == "zdr":
            if (
                self.archived_at is not None
                or self.retention_days is not None
                or self.history_available
            ):
                raise ValueError("ZDR cannot advertise a retained archive")
            return self
        if self.retention_days != 30:
            raise ValueError("customer history requires a 30-day retention policy")
        if self.history_available != (self.content_state == "available"):
            raise ValueError("history availability must match retained content state")
        if self.archived_at is None:
            if self.content_state == "available" or self.content_expires_at is not None:
                raise ValueError("available history requires its archive and expiry timestamps")
        elif self.content_expires_at != self.archived_at.astimezone(UTC) + timedelta(days=30):
            raise ValueError("retained content expiry must be exactly 30 days from archival")
        return self


def validate_retention_observation(
    retention: SearchRetention | None, observed_at: AwareDatetime
) -> None:
    """Check archive lifetime against the server observation, never the client clock.

    See: backend/notes/specifications/sdk/index_contract.md."""
    if retention is None:
        return
    if retention.archived_at is not None and retention.archived_at > observed_at:
        raise ValueError("retention cannot advertise future archival")
    if (
        retention.content_state == "available"
        and retention.content_expires_at is not None
        and retention.content_expires_at <= observed_at
    ):
        raise ValueError("retention cannot advertise expired content as available")
