"""Typed convenience request construction; wire models remain generated from Core."""

import builtins
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Literal, TypeAlias

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

DomicileRequirement: TypeAlias = Literal["us_only", "us_or_europe", "unrestricted"]


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
        raise ValueError("launch requires a message (str); Core refuses a launch without one")
    requires_task = profile_requirements.get(profile if profile is not None else default_profile)
    if requires_task is True and research_task is None:
        raise ValueError("research profile requires research_task")
    if requires_task is False and research_task is not None:
        raise ValueError("research_task requires a research profile")
    # Core contract 1.26.0: choices are made against one read menu revision.
    if role_assignments is not None and launch_menu_revision is None:
        raise ValueError("role_assignments require launch_menu_revision from launch_choices()")
    if not isinstance(budget, (str, Decimal)):
        raise ValueError("budget must be a decimal string or Decimal")
    try:
        amount = Decimal(budget)
        if not amount.is_finite() or amount <= 0:
            raise ValueError("budget must be positive")
    except InvalidOperation as error:
        raise ValueError("budget must be a decimal string") from error
    try:
        request = LaunchRequest(
            domicile_requirement=LaunchRequestDomicileRequirement(domicile_requirement)
            if domicile_requirement is not None
            else None,
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
        request = LaunchRequest.model_validate(request.model_dump(exclude_none=True))
        return request
    except ValidationError as error:
        fields = sorted(
            {str(item["loc"][0]) for item in error.errors(include_input=False) if item["loc"]}
        )
        raise LaunchValidationError(
            local_problem(
                "invalid_launch_request",
                "launch_validation",
                status=422,
                detail="Invalid launch fields: "
                + ", ".join(fields)
                + (
                    "; message must be nonblank, contain no NUL and fit the UTF-8 limit"
                    if "message" in fields
                    else ""
                ),
            )
        ) from None
