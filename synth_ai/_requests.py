"""Typed convenience request construction; wire models remain generated from Core."""

import builtins
from decimal import Decimal, InvalidOperation

from ._generated.models import (
    LaunchBudget,
    LaunchRequest,
    Limits,
    OwnerResourceRequest,
    RoutingConstraints,
    RunGuidance,
    RunInputs,
)


def launch_request(
    message: str | None,
    *,
    budget: str | Decimal,
    wall_time_seconds: int | None,
    inputs: RunInputs | None,
    labels: dict[str, str] | None,
    resources: builtins.list[OwnerResourceRequest] | None,
    profile: str | None,
    research_task: str | None,
    routing: RoutingConstraints | None,
    guidance: builtins.list[RunGuidance] | None,
) -> LaunchRequest:
    if profile in {"fork-flush", "forgetting"} and research_task is None:
        raise ValueError("research profile requires research_task")
    if (profile is None or profile == "swarms-v2") and research_task is not None:
        raise ValueError("research_task requires a research profile")
    if not isinstance(budget, (str, Decimal)):
        raise ValueError("budget must be a decimal string or Decimal")
    try:
        amount = Decimal(budget)
        if not amount.is_finite() or amount <= 0:
            raise ValueError("budget must be positive")
    except InvalidOperation as error:
        raise ValueError("budget must be a decimal string") from error
    request = LaunchRequest(
        limits=Limits(
            maximum_charge=LaunchBudget(amount=format(amount, "f"), currency="USD"),
            wall_time_seconds=wall_time_seconds,
        ),
        message=message,
        inputs=inputs,
        labels=labels,
        resources=resources,
        profile=profile,
        research_task=research_task,
        routing=routing,
        guidance=guidance,
    )
    # Omitted convenience options stay absent on the wire, not explicit nulls.
    request = LaunchRequest.model_validate(request.model_dump(exclude_none=True))
    return request
