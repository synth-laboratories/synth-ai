"""Mirrors backend packages/intern/resource_inventory.py.

See backend notes/specifications/tanha/current/systems/platform/intern_resource_inventory.md.
Incomplete coverage and unknown disposition are never inferred to mean settled.
``coverage_complete`` is creator coverage; ``disposition_complete`` is cleanup.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

InternResourceDispositionValue = Literal[
    "settled", "pending", "unknown", "excluded", "retained"
]
InternRuntimeClosureKind = Literal[
    "conversation_closed",
    "work_cancelled",
    "work_completed",
    "work_failed",
    "archived",
    "superseded_by_reopen",
]


class InternResourceDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    resource_kind: str
    resource_id: str
    cleanup_owner_run_id: str | None = None
    relation: Literal["self", "owned", "borrowed", "shared", "retained"]
    disposition: InternResourceDispositionValue
    reason: str
    cleanup_owner: str | None = None
    epoch: int | None = Field(default=None, ge=0)


class InternResourceStopEffect(BaseModel):
    """An effect actually performed when a runtime epoch closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    resource_kind: str
    resource_id: str
    # Close-time effects, then the real outcome of each internal run stop.
    effect: Literal[
        "stop_requested",
        "already_settled",
        "retained",
        "admission_fenced",
        "stopped",
        "already_terminal",
        "stop_accepted",
        "stop_failed",
    ]
    owner: str | None = None
    reason: str


class InternRuntimeResourceEpoch(BaseModel):
    """One admission epoch; closed epochs remain enumerated for recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    epoch: int = Field(ge=0)
    admission: Literal["open", "closed"]
    closure_kind: InternRuntimeClosureKind | None = None
    opened_at: datetime
    closed_at: datetime | None = None
    effects: list[InternResourceStopEffect] = Field(default_factory=list)


class InternResourceInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    runtime_kind: Literal["sync", "async"]
    runtime_id: str
    observed_at: datetime
    coverage: Literal["registered-runtime-resources-v1"]
    coverage_complete: StrictBool
    incomplete_reasons: list[str] = Field(default_factory=list)
    resources: list[InternResourceDisposition]
    creator_set: str | None = None
    profile: str | None = None
    disposition_complete: StrictBool = False
    epochs: list[InternRuntimeResourceEpoch] = Field(default_factory=list)

    @classmethod
    def from_wire(cls, value):
        inventory = cls.model_validate(value)
        # A complete cleanup can never rest on incomplete creator coverage.
        if inventory.disposition_complete and not inventory.coverage_complete:
            raise ValueError("Intern inventory reported cleanup without coverage")
        return inventory
