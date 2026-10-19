"""account.v1 observations; see backend account-billing-v023.md.

Exact monetary coefficients and event sequence strings survive JSON. Pending,
unknown and incomplete observations are data, never coerced into zero.
"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from .account import AccountPolicyLimit, AccountRequestLimits
from .base import Model
from .funding import FundingSource
from .money import Money

Product = Literal["index", "swarms", "monitor"]
BalanceClass = Literal["included", "sponsored", "purchased", "unattributed"]
EVENT_FACT_KEYS = frozenset(
    {
        "event_id",
        "search_id",
        "parent_search_id",
        "operation_id",
        "attempt_id",
        "metric",
        "quantity",
        "quality",
        "observed_at",
        "correction_of_event_id",
        "state",
        "created_at",
        "updated_at",
        "price_version",
        "currency",
        "settled_microcents",
        "refunded_microcents",
        "released_microcents",
        "settlement_state",
        "settlement_outcome",
        "funding_source",
        "ledger_reference",
        "adjustment_id",
        "amount_microcents",
        "usage_fact_id",
        "run_id",
        "meter_kind",
        "pricing_policy",
        "quantity_unit",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "billed_amount_pico_usd",
        "nominal_amount_pico_usd",
        "ledger_id",
        "occurred_at",
        "status",
        "started_at",
        "finished_at",
        "wallet_ledger_id",
        "customer_debit_microcents",
        "debit_pool",
        "fact_id",
        "outcome",
        "reason_code",
        "mode",
        "surface",
        "operation",
        "cumulative_limits",
        "decided_at",
        "receipt_id",
        "resource_kind",
        "owner",
        "price_revision",
        "cost_state",
        "cost_micros",
        "supersedes",
        "source_digest",
        "reservation_ref",
        "envelope_id",
        "receipt_digest",
    }
)


class AccountLimits(Model):
    observed_at: AwareDatetime
    limits: tuple[AccountPolicyLimit, ...]
    request_limits: AccountRequestLimits


class AccountContext(Model):
    org_id: str
    user_id: str
    api_origin: str | None
    credential_kind: Literal["session", "api_key", "oauth"]
    credential_id: str | None
    oauth_client_id: str | None = None
    scopes: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,127}$")], ...]
    observed_at: AwareDatetime


class CurrentPlan(Model):
    provider_observed_at: AwareDatetime | None = None
    projected_at: AwareDatetime | None = None
    wallet_observed_at: AwareDatetime | None = None
    observation_revision: str | None = None
    renews_at: AwareDatetime | None = None
    state: Literal["ready", "migration_required", "disabled", "unavailable"]
    observed_at: AwareDatetime
    code: str | None = None
    policy_version: str | None = None
    plan_id: str | None = None
    name: str | None = None
    monthly_price: Money | None = None
    weekly_included: Money | None = None
    issued_this_week: Money | None = None
    used_this_week: Money | None = None
    remaining_this_week: Money | None = None
    period_start: AwareDatetime | None = None
    resets_at: AwareDatetime | None = None
    cancels_at: AwareDatetime | None = None


class FundingClassBalance(Model):
    funding_class: BalanceClass
    available: Money
    reserved: Money | None = None
    expires_at: AwareDatetime | None
    non_expiring: Money


class BalanceTotal(Model):
    balance: Money
    reserved: Money
    available: Money


class AccountBalance(Model):
    observed_at: AwareDatetime
    ledger_updated_at: AwareDatetime | None
    total: BalanceTotal
    classes: tuple[FundingClassBalance, ...] = Field(min_length=4, max_length=4)
    ledger_net: Money
    reserved_attribution: Literal["not_attributed"] = "not_attributed"
    consumption_order: tuple[BalanceClass, ...]


class UsageSourceState(Model):
    product: Product
    state: Literal["available", "partial", "unavailable"]
    code: str | None = None
    coverage_start: AwareDatetime | None = None
    watermark: str | None = None
    projected_at: AwareDatetime | None = None


class FundingTotal(Model):
    source: FundingSource
    amount: Money
    item_count: int = Field(ge=0)
    unknown_amount_item_count: int = Field(ge=0)


class UsageMetricTotals(Model):
    name: str
    unit: str
    state: Literal["measured", "estimated", "pending", "unknown"]
    quantity: Decimal | None
    item_count: int = Field(ge=0)
    unknown_quantity_item_count: int = Field(ge=0)


class UsageTotals(Model):
    metrics: tuple[UsageMetricTotals, ...] = ()
    item_count: int = Field(ge=0)
    measured_charge: Money
    measured_charge_item_count: int = Field(ge=0)
    estimated_charge: Money
    estimated_charge_item_count: int = Field(ge=0)
    pending_charge_item_count: int = Field(ge=0)
    unknown_charge_item_count: int = Field(ge=0)
    final_item_count: int = Field(ge=0)
    pending_item_count: int = Field(ge=0)
    unknown_item_count: int = Field(ge=0)
    included_item_count: int = Field(default=0, ge=0)
    list_price: Money
    unknown_list_price_item_count: int = Field(ge=0)
    funding: tuple[FundingTotal, ...]


class ProductUsageTotals(Model):
    product: Product
    totals: UsageTotals


class UsageAction(Model):
    id: str
    label: str
    product: Product


class ActionUsageTotals(Model):
    product: Product
    component: str
    action: UsageAction
    totals: UsageTotals


class DailyUsageTotals(Model):
    day: date
    product: Product
    totals: UsageTotals


class AccountUsageSummary(Model):
    period_start: AwareDatetime
    period_end: AwareDatetime
    observed_at: AwareDatetime
    sources: tuple[UsageSourceState, ...]
    products: tuple[ProductUsageTotals, ...]
    actions: tuple[ActionUsageTotals, ...] = ()
    daily: tuple[DailyUsageTotals, ...] = ()
    total: UsageTotals


class AccountUsageEvent(Model):
    sequence: str = Field(pattern=r"^[0-9]{1,19}$")
    product: Product
    source: Literal[
        "index.usage",
        "index.execution",
        "index.settlement",
        "index.adjustment",
        "swarms.usage",
        "swarms.execution",
        "swarms.debit",
        "swarms.native_usage",
        "swarms.native_settlement",
        "monitor.decision",
    ]
    source_id: str = Field(min_length=1, max_length=512)
    source_revision: str = Field(pattern=r"^[a-f0-9]{32}$")
    operation_id: str = Field(min_length=1, max_length=256)
    parent_id: str | None
    actor_id: str | None
    occurred_at: AwareDatetime
    recorded_at: AwareDatetime
    facts: dict[str, str | bool | None]

    @model_validator(mode="after")
    def content_free_source(self):
        if not self.source.startswith(self.product + ".") or not set(self.facts) <= EVENT_FACT_KEYS:
            raise ValueError("Usage event has unexpected metadata or product identity")
        return self


class EventCoverage(Model):
    product: Product
    state: Literal["complete", "partial", "unavailable"]
    code: str | None = None
    captured_from: AwareDatetime | None = None
    backfilled: bool = False
    observed_at: AwareDatetime | None = None
    delivery_lag_seconds: float | None = Field(default=None, ge=0)


class AccountUsageEventPage(Model):
    period_start: AwareDatetime
    period_end: AwareDatetime
    observed_at: AwareDatetime
    watermark: str = Field(pattern=r"^[0-9]{1,19}$")
    sources: tuple[EventCoverage, ...]
    items: tuple[AccountUsageEvent, ...] = Field(max_length=200)
    next_cursor: str | None
    complete: bool


class CreditAlertState(Model):
    enabled: bool
    threshold_dollars: Annotated[int, Field(ge=1, le=10000)]
    recipient: str | None
    state: Literal["disabled", "armed", "low", "pending", "sent", "unknown", "unavailable"]
    checked_at: AwareDatetime | None
    last_sent_at: AwareDatetime | None
    observed_at: AwareDatetime
