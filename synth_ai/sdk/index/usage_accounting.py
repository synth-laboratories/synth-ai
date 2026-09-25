"""Typed Index operation usage receipts and aggregate export contracts.

The backend remains the wire and settlement authority. Customer receipts expose
physical consumption and retail accounting, never internal infrastructure costs.
See backend notes/specifications/synth-index/search-usage-accounting.md and
notes/specifications/synth-index/search-access-funding.md.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, model_validator

from .contracts import Identifier, IndexContract
from .search import SearchMode, SearchSettlementOutcome

NonNegative = Annotated[StrictInt, Field(ge=0)]


class UsageOperationKind(StrEnum):
    BM25_QUERY = "bm25_query"
    DENSE_QUERY = "dense_query"
    COLBERT_RERANK = "colbert_rerank"
    CONTENT_READ = "content_read"
    MODEL_CALL = "model_call"
    DEEP_STEP = "deep_step"
    TOOL_CALL = "tool_call"
    HOST_ALLOCATION = "host_allocation"


class UsageMetric(StrEnum):
    PHYSICAL_CALLS = "physical_calls"
    PHYSICAL_BATCHES = "physical_batches"
    CANDIDATES_RETURNED = "candidates_returned"
    CANDIDATES_SCORED = "candidates_scored"
    RESULTS_RETURNED = "results_returned"
    QUERY_ENCODINGS = "query_encodings"
    DOCUMENT_ENCODINGS = "document_encodings"
    QUERY_TOKENS = "query_tokens"
    DOCUMENT_TOKENS = "document_tokens"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    CACHED_INPUT_TOKENS = "cached_input_tokens"
    REASONING_TOKENS = "reasoning_tokens"
    VECTOR_COUNT = "vector_count"
    CACHE_HITS = "cache_hits"
    CACHE_MISSES = "cache_misses"
    READ_CALLS = "read_calls"
    BYTES_READ = "bytes_read"
    STEPS = "steps"
    TOOL_CALLS = "tool_calls"
    RETAINED_EVIDENCE = "retained_evidence"
    DISCARDED_EVIDENCE = "discarded_evidence"
    ELAPSED_MICROSECONDS = "elapsed_microseconds"
    ACCELERATOR_MICROSECONDS = "accelerator_microseconds"
    CPU_MICROSECONDS = "cpu_microseconds"
    STORAGE_BYTE_SECONDS = "storage_byte_seconds"
    EGRESS_BYTES = "egress_bytes"
    INFRASTRUCTURE_COST_USD_MICROS = "infrastructure_cost_usd_micros"
    UNALLOCATED_COST_USD_MICROS = "unallocated_cost_usd_micros"


class UsageEventStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNRECONCILED = "unreconciled"


class MeasurementState(StrEnum):
    COMPLETE = "complete"
    ESTIMATED = "estimated"
    PENDING = "pending"


class SettlementState(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    RESERVED = "reserved"
    SETTLED = "settled"
    RELEASED = "released"
    REFUNDED = "refunded"
    PENDING = "pending"


class UsageTotal(IndexContract):
    metric: UsageMetric
    quantity: NonNegative
    measured_quantity: NonNegative
    estimated_quantity: NonNegative
    pending_events: NonNegative


class OperationUsage(IndexContract):
    operation_id: Identifier
    parent_operation_id: Identifier | None
    operation_kind: UsageOperationKind
    physical_attempts: NonNegative
    statuses: tuple[UsageEventStatus, ...] = Field(default=(), max_length=5)
    totals: tuple[UsageTotal, ...] = Field(default=(), max_length=40)


class CustomerCharge(IndexContract):
    """Net customer settlement with corrections already applied by the server."""

    currency: str = "USD"
    price_version: Identifier | None = None
    reserved_microcents: NonNegative = 0
    settled_microcents: NonNegative = 0
    released_microcents: NonNegative = 0
    refunded_microcents: NonNegative = 0
    adjustment_microcents: StrictInt = 0
    settlement_state: SettlementState = SettlementState.NOT_APPLICABLE
    reservation_id: Identifier | None = None
    ledger_reference: str | None = None
    funding_source: Literal["none", "promo_credit", "deep_beta", "wallet"] | None = None
    terminal_outcome: SearchSettlementOutcome | None = None

    @model_validator(mode="after")
    def check_charge(self) -> Self:
        if (
            self.released_microcents + self.settled_microcents + self.refunded_microcents
            > self.reserved_microcents
        ):
            raise ValueError("net settlement, refunds, and releases exceed reservation")
        if self.adjustment_microcents > 0 or -self.adjustment_microcents > self.refunded_microcents:
            raise ValueError("signed corrections must reconcile outstanding refunds")
        if self.price_version is None and any((self.reserved_microcents, self.settled_microcents)):
            raise ValueError("unpriced usage cannot be charged")
        return self


class SearchUsageReceipt(IndexContract):
    schema_version: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    search_id: Identifier
    org_id: str
    requested_mode: SearchMode
    effective_mode: SearchMode
    event_count: NonNegative
    operation_count: NonNegative
    physical_attempt_count: NonNegative
    measurement_state: MeasurementState
    observed_consumption: tuple[UsageTotal, ...] = Field(default=(), max_length=40)
    operations: tuple[OperationUsage, ...] = Field(default=(), max_length=1024)
    charge: CustomerCharge
    generated_at: AwareDatetime


class SearchUsageSummaryRow(IndexContract):
    mode: SearchMode
    model_identity: str | None = None
    price_version: Identifier | None = None
    funding_source: Literal["none", "promo_credit", "deep_beta", "wallet"] | None = None
    terminal_outcome: SearchSettlementOutcome | None = None
    settlement_state: SettlementState = SettlementState.NOT_APPLICABLE
    search_count: NonNegative
    physical_attempt_count: NonNegative
    reserved_microcents: NonNegative = 0
    customer_charge_microcents: NonNegative
    released_microcents: NonNegative = 0
    refunded_microcents: NonNegative = 0
    adjustment_microcents: StrictInt = 0
    pending_event_count: NonNegative
    measurement_state: MeasurementState = MeasurementState.PENDING
    unmeasured_search_count: NonNegative = 0
    input_tokens: NonNegative | None = None
    output_tokens: NonNegative | None = None
    cached_input_tokens: NonNegative | None = None
    colbert_batches: NonNegative | None = None
    content_read_calls: NonNegative | None = None


class SearchUsageSummary(IndexContract):
    period_start: AwareDatetime
    period_end: AwareDatetime
    rows: tuple[SearchUsageSummaryRow, ...] = Field(default=(), max_length=500)
    total_rows: NonNegative = 0
    next_offset: NonNegative | None = None
