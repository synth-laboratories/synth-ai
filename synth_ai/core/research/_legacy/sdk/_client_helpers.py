"""Shared internal coercion helpers for the Managed Research SDK client."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, cast

from synth_ai.core.research._legacy.errors import SmrApiError, SmrHostedModelOverridesError
from synth_ai.core.research._legacy.models.run_timeline import (
    SmrBranchMode,
    SmrRunBranchRequest,
)


class SmrLaunchMode(StrEnum):
    """Legacy wire launch mode retained for compatibility."""

    HOSTED = "hosted"
    LOCAL = "local"


def derive_launch_mode(*, local_execution: Mapping[str, Any] | None) -> SmrLaunchMode:
    return SmrLaunchMode.LOCAL if local_execution is not None else SmrLaunchMode.HOSTED


def _hosted_launch_surface_enforced() -> bool:
    value = os.getenv("SYNTH_SMR_HOSTED_OVERRIDE_ENFORCEMENT")
    return str(value or "").strip().lower() == "on"


def assert_hosted_launch_surface(
    *,
    local_execution: Mapping[str, Any] | None,
    agent_model: Any | None = None,
    agent_profile: str | None = None,
    agent_harness: Any | None = None,
    agent_kind: Any | None = None,
    agent_model_params: Mapping[str, Any] | None = None,
    actor_model_overrides: Any | None = None,
    roles: Any | None = None,
    execution_profile: Any | None = None,
    host_kind: Any | None = None,
) -> None:
    """Reject local-only actor overrides on hosted launches when opted in."""
    if derive_launch_mode(local_execution=local_execution) is SmrLaunchMode.LOCAL:
        return
    if not _hosted_launch_surface_enforced():
        return
    rejected = [
        name
        for name, present in (
            ("agent_model", agent_model is not None),
            ("agent_profile", bool(str(agent_profile or "").strip())),
            ("agent_harness", agent_harness is not None),
            ("agent_kind", agent_kind is not None),
            ("agent_model_params", bool(agent_model_params)),
            ("actor_model_overrides", bool(actor_model_overrides)),
            ("roles", bool(roles)),
            ("execution_profile", bool(execution_profile)),
            (
                "host_kind",
                str(getattr(host_kind, "value", host_kind) or "").strip().lower()
                in {"docker", "local"},
            ),
        )
        if present
    ]
    if rejected:
        raise SmrHostedModelOverridesError(
            "actor execution overrides require local_execution; rejected: " + ", ".join(rejected),
            rejected_fields=rejected,
            detail={"rejected_fields": rejected},
        )


def _coerce_dict(payload: Any, *, label: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    raise SmrApiError(f"Expected object response for {label}, received {type(payload).__name__}")


def _coerce_dict_list(payload: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise SmrApiError(f"Expected {label} entries to be objects")
        return cast(list[dict[str, Any]], payload)
    raise SmrApiError(f"Expected list response for {label}, received {type(payload).__name__}")


def _require_non_empty_string(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_mapping(
    payload: Mapping[str, Any] | dict[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    return dict(payload)


def _coerce_branch_request(
    *,
    checkpoint_id: str | None = None,
    checkpoint_record_id: str | None = None,
    checkpoint_uri: str | None = None,
    mode: SmrBranchMode | str = SmrBranchMode.EXACT,
    message: str | None = None,
    reason: str | None = None,
    title: str | None = None,
    source_node_id: str | None = None,
) -> SmrRunBranchRequest:
    normalized_mode = mode if isinstance(mode, SmrBranchMode) else SmrBranchMode(str(mode).strip())
    return SmrRunBranchRequest(
        checkpoint_id=checkpoint_id,
        checkpoint_record_id=checkpoint_record_id,
        checkpoint_uri=checkpoint_uri,
        mode=normalized_mode,
        message=message,
        reason=reason,
        title=title,
        source_node_id=source_node_id,
    )
