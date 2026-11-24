"""Typed convenience request construction; wire models remain generated from Core."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import ValidationError

from ._generated.models import (
    LaunchBudget,
    LaunchRequest,
    LaunchRequestDomicileRequirement,
    Limits,
    OptimizerRequest,
    OwnerResourceRequest,
    RoleAssignment,
    RoutingConstraints,
    RunGuidance,
    RunInputs,
)
from .errors import LaunchValidationError, local_problem

if TYPE_CHECKING:
    import builtins
    from collections.abc import Mapping

DomicileRequirement: TypeAlias = Literal["us_only", "us_or_europe", "unrestricted"]


def _launch_invalid(detail: str) -> LaunchValidationError:
    return LaunchValidationError(
        local_problem("invalid_launch_request", "launch_validation", status=422, detail=detail)
    )


def launch_request(
    message: str,
    *,
    budget: str | Decimal,
    domicile_requirement: DomicileRequirement | None = None,
    wall_time_seconds: int | None,
    maximum_tokens: int | None = None,
    pause_retention_seconds: int | None = None,
    parallel_workers: int | None,
    inputs: RunInputs | None,
    labels: dict[str, str] | None,
    resources: builtins.list[OwnerResourceRequest] | None,
    profile: str | None,
    research_task: str | None,
    routing: RoutingConstraints | None,
    guidance: builtins.list[RunGuidance] | None,
    profile_requirements: Mapping[str, bool],
    default_profile: str,
    role_assignments: Mapping[str, RoleAssignment] | None = None,
    launch_menu_revision: str | None = None,
    optimizer: OptimizerRequest | None = None,
    continue_from: str | None = None,
) -> LaunchRequest:
    # Core contract 1.3.0 requires a message; refuse locally before any request.
    if not isinstance(message, str):
        raise _launch_invalid("launch requires a message (str); Core refuses a launch without one")
    requires_task = profile_requirements.get(profile if profile is not None else default_profile)
    if requires_task is True and research_task is None:
        raise _launch_invalid("research profile requires research_task")
    if requires_task is False and research_task is not None:
        raise _launch_invalid("research_task requires a research profile")
    # Core contract 1.26.0: choices are made against one read menu revision.
    if role_assignments is not None and launch_menu_revision is None:
        raise _launch_invalid("role_assignments require launch_menu_revision from launch_choices()")
    if not isinstance(budget, (str, Decimal)):
        raise _launch_invalid("budget must be a decimal string or Decimal")
    try:
        amount = Decimal(budget)
    except InvalidOperation:
        raise _launch_invalid("budget must be a decimal string") from None
    if not amount.is_finite() or amount <= 0:
        raise _launch_invalid("budget must be positive")
    try:
        domicile = (
            LaunchRequestDomicileRequirement(domicile_requirement)
            if domicile_requirement is not None
            else None
        )
    except ValueError:
        raise _launch_invalid(f"invalid domicile_requirement: {domicile_requirement!r}") from None
    try:
        request = LaunchRequest(
            domicile_requirement=domicile,
            limits=Limits(
                maximum_charge=LaunchBudget(amount=format(amount, "f"), currency="USD"),
                parallel_workers=parallel_workers,
                wall_time_seconds=wall_time_seconds,
                maximum_tokens=maximum_tokens,
                pause_retention_seconds=pause_retention_seconds,
            ),
            message=message,
            inputs=inputs,
            labels=labels,
            resources=resources,
            profile=profile,
            research_task=research_task,
            routing=routing,
            guidance=guidance,
            role_assignments=dict(role_assignments) if role_assignments is not None else None,
            launch_menu_revision=launch_menu_revision,
            optimizer=optimizer,
            continue_from=continue_from,
        )
        # Omitted convenience options stay absent on the wire, not explicit nulls.
        return LaunchRequest.model_validate(request.model_dump(exclude_none=True))
    except ValidationError as error:
        fields = sorted(
            {str(item["loc"][0]) for item in error.errors(include_input=False) if item["loc"]}
        )
        raise _launch_invalid(
            "Invalid launch fields: "
            + ", ".join(fields)
            + (
                "; message must be nonblank, contain no NUL and fit the UTF-8 limit"
                if "message" in fields
                else ""
            )
        ) from None
