"""Mirrors backend packages/intern/resource_inventory.py.

See backend notes/specifications/tanha/current/systems/platform/intern_resource_inventory.md.
Incomplete coverage and unknown disposition are never inferred to mean settled.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class InternResourceDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    resource_kind: str
    resource_id: str
    cleanup_owner_run_id: str | None = None
    relation: Literal["self", "owned", "borrowed", "shared", "retained"]
    disposition: Literal["settled", "pending", "unknown", "excluded"]
    reason: str


class InternResourceInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    runtime_kind: Literal["sync", "async"]
    runtime_id: str
    observed_at: datetime
    coverage: Literal["registered-runtime-resources-v1"]
    coverage_complete: StrictBool
    incomplete_reasons: list[str] = Field(default_factory=list)
    resources: list[InternResourceDisposition]

    @classmethod
    def from_wire(cls, value):
        return cls.model_validate(value)
