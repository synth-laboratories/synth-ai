"""Operator blocker handoffs; mirrors backend packages/intern/contracts.py."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .research_intern import (
    InternAsyncBlocker,
    InternAsyncCommandReceipt,
    InternRuntimeBinding,
    InternProducedResourceReference,
    InternSyncSession,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InternBlockerOpenSyncRequest(_Contract):
    idempotency_key: str = Field(min_length=1, max_length=512)


class InternBlockerHandoffContext(_Contract):
    schema_version: Literal["smr.intern-async-blocker-handoff-context.v1"] = (
        "smr.intern-async-blocker-handoff-context.v1"
    )
    blocker_id: str
    async_assignment_id: str
    binding: InternRuntimeBinding
    action_schema_version: str
    action_kind: str
    action_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: str = Field(min_length=1, max_length=2000)
    rationale: str = Field(min_length=1, max_length=4000)
    preauthorization_rule: str
    evidence_resources: tuple[InternProducedResourceReference, ...] = ()
    required_operator_capability: str


class InternBlockerHandoffReceipt(_Contract):
    schema_version: Literal["smr.intern-async-blocker-handoff-receipt.v1"] = (
        "smr.intern-async-blocker-handoff-receipt.v1"
    )
    handoff_receipt_id: str
    blocker_id: str
    org_id: str
    sync_session_id: str
    opened_by_user_id: str
    idempotency_key: str
    context: InternBlockerHandoffContext
    context_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


class InternBlockerOpenSyncResponse(_Contract):
    schema_version: Literal["smr.intern-async-blocker-open-sync.v1"] = (
        "smr.intern-async-blocker-open-sync.v1"
    )
    blocker: InternAsyncBlocker
    sync_session: InternSyncSession
    handoff_receipt: InternBlockerHandoffReceipt


class InternBlockerResolveRequest(_Contract):
    idempotency_key: str = Field(min_length=1, max_length=512)
    outcome: Literal["completed", "denied", "superseded"]
    comment: str | None = Field(default=None, max_length=4000)
    supporting_receipt_ids: tuple[str, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def validate_receipts(self):
        if any(not value.strip() or len(value) > 512 for value in self.supporting_receipt_ids):
            raise ValueError("supporting receipt IDs must be non-empty and at most 512 characters")
        if len(set(self.supporting_receipt_ids)) != len(self.supporting_receipt_ids):
            raise ValueError("supporting receipt IDs must be unique")
        return self


class InternBlockerResolveResponse(_Contract):
    schema_version: Literal["smr.intern-async-blocker-resolution.v1"] = (
        "smr.intern-async-blocker-resolution.v1"
    )
    blocker: InternAsyncBlocker
    continuation_command: InternAsyncCommandReceipt
