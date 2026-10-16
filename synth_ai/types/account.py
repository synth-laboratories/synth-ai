"""Per-organization discovery; see deployment/release/ACCOUNT_POLICY_ENGINE.md."""

from typing import Annotated, Literal, Self, get_args

from pydantic import AwareDatetime, Field, model_validator

from .base import Digest, Identifier, Model, ResponseModel

Subproduct = Literal[
    "index.fast",
    "index.deep",
    "index.private",
    "index.contributions",
    "swarms.docker",
    "swarms.daytona",
    "swarms.openrouter",
    "swarms.openai",
    "swarms.tinker",
    "swarms.synth",
    "monitor",
]
Condition = Literal[
    "activation_required",
    "own_credential_required",
    "byok_forbidden",
    "plan_required",
    "unavailable",
]


class AccountPolicyLimit(Model):
    limit_id: Identifier
    product: Literal["index", "swarms"]
    value: Annotated[int, Field(ge=0)] | None
    default_value: Annotated[int, Field(ge=0)] | None
    source: Identifier
    revision: Identifier | None
    expires_at: AwareDatetime | None
    dimension: Literal["requests", "concurrent", "spend_cents"]
    window: Literal["token_bucket", "concurrent", "day", "month"]
    observed_at: AwareDatetime
    state: Literal["ready", "unavailable"]
    code: Identifier | None
    used: Annotated[int, Field(ge=0)] | None
    remaining: Annotated[int, Field(ge=0)] | None
    resets_at: AwareDatetime | None
    burst_capacity: Annotated[int, Field(ge=0)] | None
    available_requests: Annotated[int, Field(ge=0)] | None
    next_request_at: AwareDatetime | None


class AccountEntitlement(Model):
    subproduct: Subproduct
    product: Literal["index", "swarms", "monitor"]
    stage: Literal["available", "planned", "unavailable"]
    kind: Literal["search", "content", "compute", "provider", "monitor"]
    supply: Literal["platform", "customer", "mixed"]
    access: Literal["allowed", "denied", "conditional", "unavailable"]
    code: Identifier
    conditions: tuple[Condition, ...] = Field(max_length=5)
    source: Literal["baseline", "override", "unavailable"]
    grant_revision: Annotated[int, Field(ge=1)] | None
    expires_at: AwareDatetime | None
    limits: tuple[AccountPolicyLimit, ...] = Field(max_length=8)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.product != self.subproduct.split(".")[0]:
            raise ValueError("subproduct group mismatch")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("duplicate policy condition")
        if len({item.limit_id for item in self.limits}) != len(self.limits):
            raise ValueError("duplicate policy limit")
        if self.access == "allowed" and (self.code != "allowed" or self.stage != "available"):
            raise ValueError("unavailable subproduct cannot grant access")
        return self


class AccountRequestLimits(Model):
    max_results: Annotated[int, Field(ge=1)]
    max_excerpts_per_result: Annotated[int, Field(ge=1)]
    query_max_bytes: Annotated[int, Field(ge=1)]
    contents_max_bytes: Annotated[int, Field(ge=1)]


class AccountEntitlements(ResponseModel):
    tenant_id: Identifier
    policy_revision: Annotated[int, Field(ge=0)]
    definition_version: Literal["account-policy.v1"] = "account-policy.v1"
    capability_sha256: Digest | None
    limits_observed_at: AwareDatetime
    request_limits: AccountRequestLimits
    subproducts: tuple[AccountEntitlement, ...] = Field(min_length=10, max_length=11)

    @model_validator(mode="after")
    def complete(self) -> Self:
        actual = {item.subproduct for item in self.subproducts}
        current = set(get_args(Subproduct))
        # During rollout, absence of the new provider row grants nothing.
        # Preserve complete legacy discovery without inventing an OpenAI row.
        if len(actual) != len(self.subproducts) or actual not in (
            current,
            current - {"swarms.openai"},
        ):
            raise ValueError("account policy must cover every subproduct exactly once")
        return self
