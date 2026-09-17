"""Typed Index operation usage receipts and aggregate export contracts.

The backend remains the wire and settlement authority. These models preserve
the separation between physical consumption, infrastructure cost, and the
customer charge visible on a search receipt.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictInt

from .contracts import Identifier, IndexContract
from .search import SearchMode

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
    currency: str = "USD"
    price_version: Identifier | None = None
    reserved_microcents: NonNegative = 0
    settled_microcents: NonNegative = 0
    released_microcents: NonNegative = 0
    refunded_microcents: NonNegative = 0
    settlement_state: SettlementState = SettlementState.NOT_APPLICABLE
    reservation_id: Identifier | None = None
    ledger_reference: str | None = None
    funding_source: Literal["index_deep_beta", "wallet"] | None = None
    intent_hash: str | None = None
    allowance_period_key: str | None = None
    terminal_outcome: Identifier | None = None


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
    operations: tuple[OperationUsage, ...] = Field(default=(), max_length=100)
    infrastructure_cost_usd_micros: NonNegative = 0
    unallocated_cost_usd_micros: NonNegative = 0
    charge: CustomerCharge
    generated_at: AwareDatetime


class SearchUsageSummaryRow(IndexContract):
    mode: SearchMode
    model_identity: str | None = None
    search_count: NonNegative
    physical_attempt_count: NonNegative
    infrastructure_cost_usd_micros: NonNegative
    customer_charge_microcents: NonNegative
    pending_event_count: NonNegative


class SearchUsageSummary(IndexContract):
    period_start: AwareDatetime
    period_end: AwareDatetime
    rows: tuple[SearchUsageSummaryRow, ...] = Field(default=(), max_length=500)
    total_rows: NonNegative = 0
    next_offset: NonNegative | None = None
