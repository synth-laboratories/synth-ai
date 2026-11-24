"""Declarative E2E specifications shared across Sync SDK, Async SDK, CLI, and MCP."""

from __future__ import annotations

from dataclasses import dataclass

from synth_ai.errors import (
    InvalidRequestError,
    LaunchBudgetBelowActorFloorError,
    NotFoundError,
    ProviderPresetNotAllowedError,
    RateLimitedError,
    SpendEntitlementExceededError,
    SwarmsAPIError,
    UnauthenticatedError,
    UpstreamUnavailableError,
)


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """Declarative launch contract exercised on every surface."""

    name: str
    topology: str
    version: int
    goal: str
    budget: str
    max_tasks: int | None = None
    max_repairs_per_task: int | None = None


@dataclass(frozen=True, slots=True)
class RefusalSpec:
    """Declarative server refusal contract (401/402/403/404/422/429/503)."""

    name: str
    status: int
    code: str
    error_class: type[SwarmsAPIError]
    retry_after: int | None = None
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class CliExitSpec:
    """Declarative CLI exit-code and stderr/stdout JSON contract."""

    name: str
    expected_exit: int
    expected_code: str | None


LAUNCH_SPECS: tuple[LaunchSpec, ...] = (
    LaunchSpec(
        name="swarms_v2",
        topology="swarms_v2",
        version=1,
        goal="Synthesize benchmark findings for swarms_v2.",
        budget="2.50",
        max_tasks=4,
        max_repairs_per_task=2,
    ),
    LaunchSpec(
        name="fork_and_flush",
        topology="fork_flush",
        version=1,
        goal="Explore parallel branches with fork_and_flush.",
        budget="3.00",
    ),
)

REFUSAL_SPECS: tuple[RefusalSpec, ...] = (
    RefusalSpec(
        name="unauthenticated_401",
        status=401,
        code="api_key_invalid",
        error_class=UnauthenticatedError,
    ),
    RefusalSpec(
        name="payment_required_402",
        status=402,
        code="spend_entitlement_exceeded",
        error_class=SpendEntitlementExceededError,
    ),
    RefusalSpec(
        name="forbidden_403",
        status=403,
        code="provider_preset_not_allowed",
        error_class=ProviderPresetNotAllowedError,
    ),
    RefusalSpec(
        name="not_found_404",
        status=404,
        code="run_not_found",
        error_class=NotFoundError,
    ),
    RefusalSpec(
        name="unprocessable_422",
        status=422,
        code="launch_budget_below_actor_floor",
        error_class=LaunchBudgetBelowActorFloorError,
    ),
    RefusalSpec(
        name="rate_limited_429",
        status=429,
        code="rate_limit_exceeded",
        error_class=RateLimitedError,
        retry_after=2,
        retryable=False,
    ),
    RefusalSpec(
        name="unavailable_503",
        status=503,
        code="overloaded",
        error_class=UpstreamUnavailableError,
        retryable=False,
    ),
)

CLI_EXIT_SPECS: tuple[CliExitSpec, ...] = (
    CliExitSpec(name="success_0", expected_exit=0, expected_code=None),
    CliExitSpec(name="usage_error_1", expected_exit=1, expected_code="usage_error"),
    CliExitSpec(name="wait_timeout_2", expected_exit=2, expected_code="wait_timeout"),
    CliExitSpec(name="run_failed_3", expected_exit=3, expected_code=None),
    CliExitSpec(
        name="server_refusal_4",
        expected_exit=4,
        expected_code="launch_budget_below_actor_floor",
    ),
    CliExitSpec(name="run_paused_5", expected_exit=5, expected_code="run_paused"),
)

# Keep unused-import-free alias check for base 422 compatibility.
_BASE_INVALID_REQUEST: type[SwarmsAPIError] = InvalidRequestError
