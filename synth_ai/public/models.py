"""Typed models for ``synth-backend/contracts/public/swarms-v1.openapi.yaml``.

Hand-written to match the contract. Response models allow extra fields so that
additive server changes do not break callers; request models forbid them.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(extra="forbid", frozen=True)
    error: ErrorDetail


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


class RunControlSummary(_Response):
    operation_id: str
    command: Literal["cancel", "pause", "resume"]
    state: Literal["accepted", "applied", "quiesced", "refused"]


class RunFailure(_Response):
    code: str
    retryable: bool
    problem_id: str | None = None


TERMINAL_STATES = frozenset({"completed", "partial", "failed", "cancelled"})


class RunSnapshot(_Response):
    run_id: str
    project_id: str
    state: str
    model: str
    provider: str
    result_available: bool
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool
    labels: dict[str, str] | None = None
    labels_revision: int | None = None
    control: RunControlSummary | None = None
    paused_at: datetime | None = None
    resume_by: datetime | None = None
    deadline_at: datetime | None = None
    pause_count: int | None = None
    failure: RunFailure | None = None
    objectives: list[dict[str, Any]] | None = None
    problems: list[dict[str, Any]] | None = None
    status_facets: dict[str, Any] | None = None
    method: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class RunPage(_Response):
    runs: list[RunSnapshot]
    next_cursor: str | None = None


class RunEvent(_Response):
    sequence: int = Field(ge=0)
    kind: str
    at: datetime
    data: dict[str, Any] | None = None


class RunEventPage(_Response):
    events: list[RunEvent]
    next_after: int | None = None


class RunOutput(_Response):
    output_id: str
    kind: str
    digest_sha256: str | None = None
    size_bytes: int | None = None


class RunOutputList(_Response):
    outputs: list[RunOutput]


class RunResult(_Response):
    """What ``swarms.run`` returns: the closed run and its outputs."""

    run: RunSnapshot
    outputs: list[RunOutput]


# Billing


class Balance(_Response):
    available_microcents: int
    included_microcents: int
    purchased_microcents: int
    period_ends_at: datetime | None = None


class Plan(_Response):
    plan_id: str
    monthly_price_microcents: int = Field(ge=0)
    weekly_included_microcents: int = Field(ge=0)


class PlanCatalog(_Response):
    plans: list[Plan]


class CheckoutRequest(_Request):
    kind: Literal["plan", "credits"]
    plan_id: str | None = None
    amount_microcents: int | None = Field(default=None, ge=1)


class CheckoutSession(_Response):
    checkout_url: str
    expires_at: datetime
