"""Typed models for ``synth-backend/contracts/public/swarms-v1.openapi.yaml``.

Hand-written to match the contract. Response models allow extra fields so that
additive server changes do not break callers; request models forbid them.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Response(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Errors


class ErrorDetail(_Response):
    code: str
    service: str
    stage: str
    message: str | None = None
    upstream_status: int | None = None
    upstream_code: str | None = None
    upstream_kind: str | None = None
    cause: str | None = None
    outcome_unknown: bool | None = None
    detail: dict[str, str] | None = None


class ErrorBody(BaseModel):
    # Swarms routes send SwarmsErrorBody (error + outcome); other routes send error only.
    model_config = ConfigDict(extra="allow", frozen=True)
    error: ErrorDetail
    outcome: str | None = None  # Swarms routes: not_accepted | unknown


# Account


class Limit(_Response):
    resource: str
    metric: str
    window: str
    cap: float


class Org(_Response):
    id: str
    name: str
    personal: bool
    plan: str
    access_tier: str
    created_at: datetime
    clerk_org_id: str | None = None


class Principal(_Response):
    org_id: str
    credential_kind: Literal["api_key", "session"]
    credential_purpose: Literal["personal", "automation", "legacy"]
    principal_class: str
    scopes: list[str]
    permissions: list[str]
    authorization_version: int = Field(ge=1)
    access_tier: str
    user_id: str | None = None
    credential_id: str | None = None
    org_role: Literal["owner", "admin", "member"] | None = None


class Account(_Response):
    org: Org
    principal: Principal
    limits: list[Limit]


class ApiKey(_Response):
    key_id: str
    prefix: Literal["sk_synth_user_"]
    last4: str
    key_type: str
    is_active: bool
    purpose: Literal["personal", "automation"]
    name: str | None
    scopes: list[str]
    authorization_version: int = Field(ge=1)
    rotation_family_id: str | None
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None


class ApiKeyList(_Response):
    keys: list[ApiKey]


# Runs


class Money(_Request):
    currency: Literal["USD"] = "USD"
    coefficient: int = Field(ge=0)
    exponent: int

    @classmethod
    def usd(cls, amount: Decimal | int | str) -> Money:
        """Exact decimal USD, e.g. ``Money.usd("2.50")`` -> coefficient 250, exponent -2."""
        value = Decimal(str(amount))
        if not value.is_finite() or value < 0:
            raise ValueError("budget must be a finite, non-negative USD amount")
        sign, digits, exponent = value.normalize().as_tuple()
        assert isinstance(exponent, int)
        coefficient = int("".join(map(str, digits))) if digits else 0
        return cls(coefficient=coefficient, exponent=exponent if coefficient else 0)


class Limits(_Request):
    maximum_charge: Money | None
    wall_time_seconds: int = Field(ge=1, le=86400)
    maximum_tasks: int = Field(ge=1, le=1000)
    maximum_parallel_tasks: int = Field(ge=1, le=1000)


class InputFile(_Request):
    file_id: str
    revision_id: str
    project_id: str
    digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Inputs(_Request):
    files: list[InputFile] = Field(default_factory=list, max_length=32)


class LaunchRequest(_Request):
    project_id: str
    limits: Limits
    message: str | None = Field(default=None, min_length=1, max_length=8192)
    inputs: Inputs | None = None
    providers: dict[str, Any] | None = None
    models: dict[str, Any] | None = None
    method: dict[str, Any] | None = None
    program: dict[str, Any] | None = None
    profile_overrides: dict[str, Any] | None = None


class ControlRef(_Response):
    operation_id: str
    command: str  # cancel | pause | resume; unknown values forwarded
    state: str  # accepted | applied | quiesced | refused; unknown values forwarded


class ControlReceipt(_Response):
    operation_id: str
    resource_id: str
    state: str
    reason: str
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool
    command: str | None = None


TERMINAL_STATES = frozenset({"completed", "partial", "failed", "cancelled"})


class RunSnapshot(_Response):
    metadata: dict[str, Any]
    run_id: str
    project_id: str
    labels: dict[str, str] | None
    labels_revision: int
    objectives: list[Any] | None
    guidance_coverage: list[Any] | None
    state: str
    model: str
    provider: str
    result_available: bool
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool
    problems: list[Any] | None
    problems_observed: bool
    control: ControlRef | None = None
    method: Any = None
    failure: Any = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def failure_code(self) -> str | None:
        if isinstance(self.failure, dict) and isinstance(self.failure.get("code"), str):
            return self.failure["code"]
        return None


class _Page(_Response):
    metadata: dict[str, Any]
    next_cursor: str | None
    complete: bool


class RunPage(_Page):
    items: list[RunSnapshot]


class RunEvent(_Response):
    event_id: str
    run_id: str
    sequence: int = Field(ge=0)
    kind: str
    original_kind: str
    usage: Any
    observed_at: datetime
    approval_id: str | None


class EventPage(_Page):
    items: list[RunEvent]
    resume_cursor: str | None


class RunOutput(_Response):
    run_id: str
    output_id: str
    state: str
    reference: dict[str, str] | None
    filename: str


class OutputPage(_Page):
    items: list[RunOutput]


class RunResult(_Response):
    """What ``swarms.run`` returns: the closed run and its outputs."""

    run: RunSnapshot
    outputs: list[RunOutput]


# Billing (core/internal/billing and core/internal/stripe on synth-backend main)


class Balance(_Response):
    org_id: str
    ledger_microcents: int
    held_microcents: int
    headroom_microcents: int
    available_microcents: int
    available_usd_cents: int


class Plan(_Response):
    plan_id: str
    name: str
    kind: str  # free | recurring | credit_pack | internal; unknown values forwarded
    price_cents: int
    weekly_included_microcents: int
    interval: str | None = None


class PlanCatalog(_Response):
    policy_version: str
    plans: list[Plan]


class BillingHistoryEntry(_Response):
    entry_id: str
    cause: str  # grant | hold | settle | release | refund | clawback | adjustment | expire | ...
    delta_microcents: int
    held_delta_microcents: int
    funding_class: str | None
    hold_kind: str | None
    expires_at: datetime | None
    created_at: datetime


class BillingHistoryPage(_Response):
    items: list[BillingHistoryEntry]
    next_cursor: str | None


# The core decodes checkout bodies strictly: exactly one of plan_id or credit_cents.
MIN_CREDIT_CENTS = 500
MAX_CREDIT_CENTS = 1_000_000


class CheckoutRequest(_Request):
    plan_id: str | None = None
    credit_cents: int | None = Field(default=None, ge=MIN_CREDIT_CENTS, le=MAX_CREDIT_CENTS)

    @model_validator(mode="after")
    def _exactly_one(self) -> CheckoutRequest:
        if (self.plan_id is None) == (self.credit_cents is None):
            raise ValueError("checkout takes exactly one of plan_id or credit_cents")
        return self


class CheckoutSession(_Response):
    session_id: str
    url: str
