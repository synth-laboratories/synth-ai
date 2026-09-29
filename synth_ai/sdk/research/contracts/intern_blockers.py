"""Operator blocker handoffs; mirrors backend packages/intern/contracts.py."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .research_intern import (
    InternAsyncBlocker,
    InternAsyncCommandReceipt,
    InternProducedResourceReference,
    InternRuntimeBinding,
    InternSyncSession,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InternBlockerOpenSyncRequest(_Contract):
    """Idempotent request to open an operator handoff without resolving the blocker."""
    #: Caller retry key for opening this handoff; 1–512 characters.
    idempotency_key: str = Field(min_length=1, max_length=512)


class InternBlockerHandoffContext(_Contract):
    """Versioned snapshot of the blocked Async action for operator intervention."""
    #: Version discriminator for the async-blocker handoff context.
    schema_version: Literal["smr.intern-async-blocker-handoff-context.v1"] = (
        "smr.intern-async-blocker-handoff-context.v1"
    )
    #: Blocked action whose context is being handed off.
    blocker_id: str
    #: Async assignment that owns the blocked action.
    async_assignment_id: str
    #: Runtime binding carried into the operator handoff.
    binding: InternRuntimeBinding
    #: Schema version of the blocked action.
    action_schema_version: str
    #: Kind of action requiring operator intervention.
    action_kind: str
    #: 64-character lowercase SHA-256 hex digest of the action.
    action_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: Operator-facing description of the blocked action; at most 2,000 characters.
    summary: str = Field(min_length=1, max_length=2000)
    #: Reason for the handoff; at most 4,000 characters.
    rationale: str = Field(min_length=1, max_length=4000)
    #: Preauthorization rule associated with the blocked action.
    preauthorization_rule: str
    #: Produced-resource references supporting the handoff.
    evidence_resources: tuple[InternProducedResourceReference, ...] = ()
    #: Capability required of the operator handling this action.
    required_operator_capability: str


class InternBlockerHandoffReceipt(_Contract):
    """Durable evidence tying an Async blocker to its operator Sync session."""
    #: Version discriminator for the durable handoff receipt.
    schema_version: Literal["smr.intern-async-blocker-handoff-receipt.v1"] = (
        "smr.intern-async-blocker-handoff-receipt.v1"
    )
    #: Identity of the durable handoff receipt.
    handoff_receipt_id: str
    #: Blocker associated with this receipt.
    blocker_id: str
    #: Organization that owns the handoff.
    org_id: str
    #: Sync session opened for operator intervention.
    sync_session_id: str
    #: User recorded as opening the handoff.
    opened_by_user_id: str
    #: Retry key used to open this handoff.
    idempotency_key: str
    #: Snapshot of the blocked action and its runtime binding.
    context: InternBlockerHandoffContext
    #: 64-character lowercase SHA-256 hex digest of the context.
    context_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: Backend timestamp when the handoff receipt was created.
    created_at: datetime


class InternBlockerOpenSyncResponse(_Contract):
    """Opened operator handoff, its Sync session, and durable context receipt."""
    #: Version discriminator for the open-sync response.
    schema_version: Literal["smr.intern-async-blocker-open-sync.v1"] = (
        "smr.intern-async-blocker-open-sync.v1"
    )
    #: Blocker after the handoff was opened; opening alone does not resolve it.
    blocker: InternAsyncBlocker
    #: Sync session available for the operator handoff.
    sync_session: InternSyncSession
    #: Durable receipt tying the blocker, context, and Sync session together.
    handoff_receipt: InternBlockerHandoffReceipt


class InternBlockerResolveRequest(_Contract):
    """Explicit blocker disposition with optional supporting receipts."""
    #: Caller retry key for this resolution; 1–512 characters.
    idempotency_key: str = Field(min_length=1, max_length=512)
    #: Explicit completed, denied, or superseded disposition.
    outcome: Literal["completed", "denied", "superseded"]
    #: Optional operator explanation; at most 4,000 characters.
    comment: str | None = Field(default=None, max_length=4000)
    #: Up to 128 unique, non-blank supporting receipt IDs, each at most 512 characters.
    supporting_receipt_ids: tuple[str, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def validate_receipts(self):
        if any(not value.strip() or len(value) > 512 for value in self.supporting_receipt_ids):
            raise ValueError("supporting receipt IDs must be non-empty and at most 512 characters")
        if len(set(self.supporting_receipt_ids)) != len(self.supporting_receipt_ids):
            raise ValueError("supporting receipt IDs must be unique")
        return self


class InternBlockerResolveResponse(_Contract):
    """Resolved blocker and durable continuation-command receipt."""
    #: Version discriminator for the resolution response.
    schema_version: Literal["smr.intern-async-blocker-resolution.v1"] = (
        "smr.intern-async-blocker-resolution.v1"
    )
    #: Blocker carrying the durable resolution receipt.
    blocker: InternAsyncBlocker
    #: Durable command receipt for continuation of the owning Async assignment.
    continuation_command: InternAsyncCommandReceipt
