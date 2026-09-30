"""Mirrors backend packages/intern/resource_inventory.py.

See backend notes/specifications/tanha/current/systems/platform/intern_resource_inventory.md.
Incomplete coverage and unknown disposition are never inferred to mean settled.
``coverage_complete`` is creator coverage; ``disposition_complete`` is cleanup.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

InternResourceDispositionValue = Literal["settled", "pending", "unknown", "excluded", "retained"]
InternRuntimeClosureKind = Literal[
    "conversation_closed",
    "work_cancelled",
    "work_completed",
    "work_failed",
    "archived",
    "superseded_by_reopen",
]


class InternResourceDisposition(BaseModel):
    """Ownership relation and recorded cleanup disposition for one registered resource."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    #: Registered resource type.
    resource_kind: str
    #: Identity of the registered resource.
    resource_id: str
    #: Run responsible for cleanup, when known.
    cleanup_owner_run_id: str | None = None
    #: Ownership relation to this runtime: self, owned, borrowed, shared, or retained.
    relation: Literal["self", "owned", "borrowed", "shared", "retained"]
    #: Recorded cleanup disposition; unknown and pending do not mean settled.
    disposition: InternResourceDispositionValue
    #: Backend explanation for the recorded disposition.
    reason: str
    #: Cleanup owner label, when recorded.
    cleanup_owner: str | None = None
    #: Non-negative admission epoch associated with the resource, when known.
    epoch: int | None = Field(default=None, ge=0)


class InternResourceStopEffect(BaseModel):
    """An effect actually performed when a runtime epoch closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    #: Resource type affected by runtime closure.
    resource_kind: str
    #: Identity of the resource affected by closure.
    resource_id: str
    # Close-time effects, then the real outcome of each internal run stop.
    #: Recorded closure effect or actual internal-run stop outcome.
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
    #: Owner recorded for this effect, when available.
    owner: str | None = None
    #: Backend explanation of the closure effect.
    reason: str


class InternRuntimeResourceEpoch(BaseModel):
    """One admission epoch; closed epochs remain enumerated for recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    #: Non-negative admission epoch number.
    epoch: int = Field(ge=0)
    #: Whether the epoch accepts new resources.
    admission: Literal["open", "closed"]
    #: Reason admission closed; absent while no closure is recorded.
    closure_kind: InternRuntimeClosureKind | None = None
    #: Timestamp when the admission epoch opened.
    opened_at: datetime
    #: Timestamp when admission closed, if recorded.
    closed_at: datetime | None = None
    #: Effects recorded when this epoch closed, including internal-run stop outcomes.
    effects: list[InternResourceStopEffect] = Field(default_factory=list)
    # Null while the backend re-drive pass is still converging this epoch.
    #: Cleanup settlement timestamp; null while backend recovery is still converging.
    cleanup_settled_at: datetime | None = None


class InternResourceInventory(BaseModel):
    """Runtime resource observation; creator coverage and cleanup completeness are separate.

    ```python
    from synth_ai.sdk.research.contracts.intern_resources import InternResourceInventory

    inventory = InternResourceInventory.from_wire({
        "runtime_kind": "sync", "runtime_id": "session-1",
        "observed_at": "2026-09-29T12:00:00+00:00",
        "coverage": "registered-runtime-resources-v1",
        "coverage_complete": False, "resources": [],
    })
    assert not inventory.disposition_complete
    ```
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    #: Sync session or Async assignment inventory kind.
    runtime_kind: Literal["sync", "async"]
    #: Identity of the session or assignment whose resources were observed.
    runtime_id: str
    #: Backend timestamp of this inventory observation.
    observed_at: datetime
    #: Inventory contract discriminator: registered-runtime-resources-v1.
    coverage: Literal["registered-runtime-resources-v1"]
    #: Whether creator coverage is complete; separate from resource cleanup.
    coverage_complete: StrictBool
    #: Backend reasons creator coverage is incomplete.
    incomplete_reasons: list[str] = Field(default_factory=list)
    #: Registered resources and their individual dispositions.
    resources: list[InternResourceDisposition]
    #: Recorded creator-set identifier, when available.
    creator_set: str | None = None
    #: Inventory profile recorded by the backend, when available.
    profile: str | None = None
    #: Whether cleanup is complete; cannot be true with incomplete creator coverage.
    disposition_complete: StrictBool = False
    #: Admission epochs retained for recovery, including closed epochs.
    epochs: list[InternRuntimeResourceEpoch] = Field(default_factory=list)

    @classmethod
    def from_wire(cls, value: object) -> "InternResourceInventory":
        """Parse the backend wire representation of InternResourceInventory.

        Args:
            value: Backend response object to validate and convert.

        Returns:
            The parsed InternResourceInventory with backend values preserved.

        Raises:
            ValueError: Cleanup is reported complete despite incomplete creator coverage.
        """
        inventory = cls.model_validate(value)
        # A complete cleanup can never rest on incomplete creator coverage.
        if inventory.disposition_complete and not inventory.coverage_complete:
            raise ValueError("Intern inventory reported cleanup without coverage")
        return inventory
