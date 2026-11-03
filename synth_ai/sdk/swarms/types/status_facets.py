"""Independent run status facets (SYN-4849). See docs/swarms.md "Status facets".

The top-level ``RunSnapshot.state`` is unchanged. ``status_facets`` splits the
run into four independent facts, each read from its own authority, so a
terminal ``state`` never implies that outputs were published, money settled or
resources cleaned up:

* ``execution``: did the run start, and is it open, closing or closed?
* ``publication``: were the run's declared publications made, waived or refused?
* ``settlement``: is the customer charge and supplier settlement final?
* ``cleanup``: were sandboxes, traces and capacity released?

Each facet carries a typed ``cause`` (a stable code plus the authority it was
derived from) when the state needs one, and timestamps only when the server
recorded them. A fact the server did not observe is ``unknown``, never a zero or
a success. Servers that predate this field omit ``status_facets``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    Field,
    SerializerFunctionWrapHandler,
    StrictInt,
    model_serializer,
)

from synth_ai.types.base import ResponseBase, ResponseIdentifier
from synth_ai.types.unknown import UnknownValue

KnownExecutionFacetState = Literal[
    "queued", "running", "closing", "closed", "never_started", "unknown"
]
ExecutionFacetState = KnownExecutionFacetState | UnknownValue
KnownExecutionOutcome = Literal["completed", "partial", "failed", "cancelled"]
ExecutionOutcome = KnownExecutionOutcome | UnknownValue
KnownPublicationFacetState = Literal[
    "not_required", "pending", "published", "waived", "refused", "unknown"
]
PublicationFacetState = KnownPublicationFacetState | UnknownValue
KnownSettlementFacetState = Literal["pending", "in_doubt", "refused", "final", "unknown"]
SettlementFacetState = KnownSettlementFacetState | UnknownValue
KnownCleanupFacetState = Literal["pending", "complete", "failed", "unknown"]
CleanupFacetState = KnownCleanupFacetState | UnknownValue


class _OmitAbsent(ResponseBase):
    """Optional facet fields are omitted when absent, never sent as null."""

    @model_serializer(mode="wrap")
    def omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        return {key: item for key, item in value.items() if item is not None}


class FacetCause(ResponseBase):
    """Why a facet is in its state: a stable code and the authority it came from."""

    code: ResponseIdentifier
    authority: ResponseIdentifier


class ExecutionFacet(_OmitAbsent):
    state: ExecutionFacetState
    # Set once execution closed: how the run ended.
    outcome: ExecutionOutcome | None = None
    cause: FacetCause | None = None
    started_at: AwareDatetime | None = None
    closed_at: AwareDatetime | None = None


class PublicationCounts(ResponseBase):
    open: StrictInt = Field(ge=0)
    closed: StrictInt = Field(ge=0)
    waived: StrictInt = Field(ge=0)


class PublicationFacet(_OmitAbsent):
    state: PublicationFacetState
    cause: FacetCause | None = None
    # The owner's complete publication obligation counts; absent when unobserved.
    counts: PublicationCounts | None = None


class SettlementFacet(_OmitAbsent):
    state: SettlementFacetState
    cause: FacetCause | None = None
    customer_charge_final: bool | None = None
    supplier_settlement_final: bool | None = None
    # The owner's settlement operation, when a settled receipt names one.
    operation_id: ResponseIdentifier | None = None


class CleanupFacet(_OmitAbsent):
    state: CleanupFacetState
    cause: FacetCause | None = None
    completed_at: AwareDatetime | None = None


class RunStatusFacets(ResponseBase):
    execution: ExecutionFacet
    publication: PublicationFacet
    settlement: SettlementFacet
    cleanup: CleanupFacet
