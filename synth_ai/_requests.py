"""Typed convenience request construction; wire models remain generated from Core."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import ValidationError

from ._generated.models import (
    LaunchBudget,
    LaunchChoices,
    LaunchRequest,
    LaunchRequestDomicileRequirement,
    Limits,
    Money,
    OptimizerRequest,
    OwnerResourceRequest,
    RoleAssignment,
    RoutingConstraints,
    RunGuidance,
    RunInputs,
    RunLimitChanges,
    RunLimitUpdate,
)
from ._validation import safe_validation_detail
from .errors import LaunchValidationError, check_upload_format, local_problem

if TYPE_CHECKING:
    import builtins
    from collections.abc import Mapping, Sequence

    from pydantic import JsonValue

DomicileRequirement: TypeAlias = Literal["us_only", "us_or_europe", "unrestricted"]


def _launch_invalid(detail: str) -> LaunchValidationError:
    return LaunchValidationError(
        local_problem("invalid_launch_request", "launch_validation", status=422, detail=detail)
    )


def _parse_budget(budget: str | Decimal) -> Decimal:
    if not isinstance(budget, (str, Decimal)):
        raise _launch_invalid("budget must be a decimal string or Decimal")
    try:
        amount = Decimal(budget)
    except InvalidOperation:
        raise _launch_invalid("budget must be a decimal string") from None
    if not amount.is_finite() or amount <= 0:
        raise _launch_invalid("budget must be positive")
    return amount


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
    amount = _parse_budget(budget)
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
        detail = safe_validation_detail(error)
        raise _launch_invalid(
            "Invalid launch fields: "
            + ", ".join(fields)
            + (
                "; message must be nonblank, contain no NUL and fit the UTF-8 limit"
                if "message" in fields
                else ""
            )
            + (f" ({detail})" if detail else "")
        ) from None


def limit_update_request(
    *,
    expected_revision: int,
    budget: str | Decimal | None = None,
    wall_time_seconds: int | None = None,
    maximum_tokens: int | None = None,
    pause_retention_seconds: int | None = None,
    resume: bool = False,
    reason: str | None = None,
) -> RunLimitUpdate:
    """Validate and construct a typed RunLimitUpdate payload."""
    if budget is not None:
        if not isinstance(budget, (str, Decimal)):
            raise ValueError("budget must be a positive decimal string or Decimal")
        try:
            amount = Decimal(budget)
        except InvalidOperation:
            raise ValueError("budget must be a positive decimal string or Decimal") from None
        if not amount.is_finite() or amount <= 0:
            raise ValueError("budget must be a positive decimal string or Decimal")
        maximum_charge = Money(amount=format(amount, "f"), currency="USD")
    else:
        maximum_charge = None
    changes = RunLimitChanges(
        maximum_charge=maximum_charge,
        maximum_tokens=maximum_tokens,
        wall_time_seconds=wall_time_seconds,
        pause_retention_seconds=pause_retention_seconds,
    )
    if not changes.model_dump(exclude_none=True):
        raise ValueError("at least one limit is required")
    return RunLimitUpdate(
        expected_revision=expected_revision, limits=changes, resume=resume, reason=reason
    )


def upload_text_body(*, name: str, media_type: str, content: str) -> dict[str, JsonValue]:
    """Validate and build the inline uploadFile request body."""
    return {"name": name, "media_type": check_upload_format(media_type), "content": content}


def prepare_upload_body(
    *, name: str, media_type: str, size_bytes: int, digest_sha256: str
) -> dict[str, JsonValue]:
    """Validate and build the createUpload request body."""
    return {
        "name": name,
        "media_type": check_upload_format(media_type),
        "size_bytes": size_bytes,
        "digest_sha256": digest_sha256,
    }


def launch_choices_query(
    *,
    profile: str | None = None,
    domicile_requirement: DomicileRequirement | None = None,
    parallel_workers: int | None = None,
    wall_time_seconds: int | None = None,
    budget: str | Decimal | None = None,
) -> dict[str, str | int | None]:
    """Build query parameters for getLaunchChoices."""
    maximum_charge = format(_parse_budget(budget), "f") if budget is not None else None
    return {
        "profile": profile,
        "domicile_requirement": domicile_requirement,
        "parallel_workers": parallel_workers,
        "wall_time_seconds": wall_time_seconds,
        "maximum_charge": maximum_charge,
    }


def role_assignments_from_choices(
    choices: LaunchChoices,
    *,
    sandbox_profile: str | None = None,
    inference_profile: str | None = None,
    sandbox_option: str | None = None,
    roles: Sequence[str] | None = None,
) -> dict[str, RoleAssignment]:
    """Build validated `role_assignments` from a `LaunchChoices` menu."""
    if sandbox_profile is None and inference_profile is None and sandbox_option is None:
        raise ValueError(
            "at least one of sandbox_profile, inference_profile, or sandbox_option is required"
        )
    available_actors = [role.actor_id for role in choices.roles]
    if roles is None:
        target_actors = available_actors
    else:
        target_actors = list(roles)
        if not target_actors:
            raise ValueError("roles must not be empty")
        known_actors = set(available_actors)
        unknown_actors = sorted(set(target_actors) - known_actors)
        if unknown_actors:
            raise ValueError(
                "unknown role(s): "
                + ", ".join(unknown_actors)
                + ("; available: " + ", ".join(available_actors) if available_actors else "")
            )
    if not target_actors:
        raise ValueError("choices declares no roles to assign")

    if inference_profile is not None:
        inference_ids = [profile.profile_id for profile in choices.inference_profiles]
        if inference_profile not in inference_ids:
            raise ValueError(
                f"unknown inference_profile {inference_profile!r}"
                + ("; available: " + ", ".join(inference_ids) if inference_ids else "")
            )

    resolved_sandbox_profile = sandbox_profile
    resolved_sandbox_option = sandbox_option
    if sandbox_profile is not None:
        sandbox_by_id = {profile.profile_id: profile for profile in choices.sandbox_profiles}
        chosen_sandbox = sandbox_by_id.get(sandbox_profile)
        if chosen_sandbox is None:
            available_sandboxes = list(sandbox_by_id)
            raise ValueError(
                f"unknown sandbox_profile {sandbox_profile!r}"
                + ("; available: " + ", ".join(available_sandboxes) if available_sandboxes else "")
            )
        option_ids = [option.option_id for option in chosen_sandbox.options]
        if sandbox_option is not None:
            if sandbox_option not in option_ids:
                raise ValueError(
                    f"unknown sandbox_option {sandbox_option!r} for sandbox_profile {sandbox_profile!r}"
                    + ("; available: " + ", ".join(option_ids) if option_ids else "")
                )
        elif chosen_sandbox.option_required:
            if len(option_ids) == 1:
                resolved_sandbox_option = option_ids[0]
            elif len(option_ids) > 1:
                raise ValueError(
                    f"sandbox_profile {sandbox_profile!r} requires sandbox_option; "
                    + "choose one of: "
                    + ", ".join(option_ids)
                )
            else:
                raise ValueError(
                    f"sandbox_profile {sandbox_profile!r} requires sandbox_option "
                    + "but no options are available"
                )
    elif sandbox_option is not None:
        matches = [
            profile.profile_id
            for profile in choices.sandbox_profiles
            if any(option.option_id == sandbox_option for option in profile.options)
        ]
        if not matches:
            raise ValueError(f"unknown sandbox_option {sandbox_option!r}")
        if len(matches) == 1:
            resolved_sandbox_profile = matches[0]

    assignment = RoleAssignment(
        inference_profile=inference_profile,
        sandbox_profile=resolved_sandbox_profile,
        sandbox_option=resolved_sandbox_option,
    )
    return dict.fromkeys(target_actors, assignment)
