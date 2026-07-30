"""Typed Factory-B role-receipt contracts minted by the Research Intern.

Backend remains the contract authority. A role receipt binds one Factory Luna
role to one candidate through owner-authored runtime evidence and provenance,
so consumers verify receipts instead of trusting SDK-side claims.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_wire(cls, value: object) -> Self:
        return cls.model_validate(value)


FactoryLunaRole = Literal["researcher", "data_engineer", "reviewer"]


class FactoryRoleReceiptRuntimeEvidence(_StrictContract):
    run_id: str
    project_id: str
    actor_id: str
    actor_key: str
    actor_type: Literal["worker", "reviewer"]
    actor_subtype: str | None = None
    attempt_number: int = Field(ge=1)
    state: Literal["completed"]
    completion_status: Literal["succeeded"]
    started_at: datetime | None = None
    finished_at: datetime


class FactoryRoleReceiptProvenance(_StrictContract):
    schema_version: Literal["smr.factory-role-receipt.provenance.v1"]
    service_origin: Literal["urn:synth:research-intern"]
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    receipt_url: str = Field(min_length=1, max_length=2000)
    captured_at: datetime


class FactoryRoleReceiptMintRequest(_StrictContract):
    idempotency_key: str = Field(min_length=1, max_length=512)
    candidate_id: str = Field(min_length=1, max_length=255)
    role: FactoryLunaRole
    run_id: str = Field(min_length=1, max_length=255)


class FactoryRoleReceiptResponse(_StrictContract):
    schema_version: Literal["smr.factory-role-receipt.v1"]
    owner_service: Literal["research_intern"]
    authority: Literal["research_intern"]
    receipt_id: str
    receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    receipt_url: str = Field(min_length=1, max_length=2000)
    role: FactoryLunaRole
    candidate_id: str
    org_id: str
    research_intern_id: str
    factory_id: str
    idempotency_key: str
    runtime_evidence: FactoryRoleReceiptRuntimeEvidence
    provenance: FactoryRoleReceiptProvenance
    created_at: datetime


__all__ = [
    "FactoryLunaRole",
    "FactoryRoleReceiptMintRequest",
    "FactoryRoleReceiptProvenance",
    "FactoryRoleReceiptResponse",
    "FactoryRoleReceiptRuntimeEvidence",
]
