"""Shared internal coercion helpers for the Managed Research SDK client."""

from __future__ import annotations

import mimetypes
import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, cast

from synth_ai.sdk.research.contracts.run_timeline import (
    SmrBranchMode,
    SmrRunBranchRequest,
)
from synth_ai.sdk.research.contracts.smr_providers import (
    ProviderBinding,
    ResourceProvider,
)
from synth_ai.sdk.research.contracts.swarms import normalize_provider_selection
from synth_ai.sdk.research.errors import ResearchApiError, ResearchHostedModelOverridesError


def _guess_content_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _is_source_bundle_entry(path: str, entry: Mapping[str, Any]) -> bool:
    kind = str(entry.get("kind") or "").strip().lower()
    content_type = str(entry.get("content_type") or _guess_content_type(path)).strip().lower()
    return (
        kind == "source_bundle"
        or path.lower().endswith(".zip")
        or content_type
        in {
            "application/zip",
            "application/x-zip",
            "application/x-zip-compressed",
            "multipart/x-zip",
        }
    )


def _positive_int_env(name: str, default_value: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default_value
    try:
        value = int(raw)
    except ValueError:
        return default_value
    return value if value > 0 else default_value


def _fencing_headers(fencing_token: int | None) -> dict[str, str] | None:
    """``X-Fencing-Token`` header for mutating CloudDeployment ops, or None."""
    if fencing_token is None:
        return None
    if isinstance(fencing_token, bool):
        raise ValueError("fencing_token must be an integer when provided")
    return {"X-Fencing-Token": str(int(fencing_token))}


def _require_fencing_headers(fencing_token: int) -> dict[str, str]:
    if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
        raise ValueError("fencing_token must be a positive integer")
    if fencing_token < 1:
        raise ValueError("fencing_token must be a positive integer")
    return {"X-Fencing-Token": str(fencing_token)}


def _optional_non_empty_string(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


class SmrLaunchMode(StrEnum):
    """Legacy wire launch mode retained for compatibility."""

    HOSTED = "hosted"
    LOCAL = "local"


def provider_selection_payload(
    provider: str | tuple[str, ...] | list[str] | None,
) -> str | list[str] | None:
    normalized = normalize_provider_selection(provider)
    return list(normalized) if isinstance(normalized, tuple) else normalized


def reject_deprecated_provider_bindings(
    bindings: Sequence[ProviderBinding],
) -> None:
    if any(binding.provider is ResourceProvider.TINKER for binding in bindings):
        raise ValueError("tinker is deprecated and cannot be selected for new runs")


def _payload_selects_provider(payload: Any, *, provider: str) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == "provider" and isinstance(value, Mapping):
                selected = (
                    str(
                        value.get("provider_id") or value.get("provider") or value.get("kind") or ""
                    )
                    .strip()
                    .lower()
                )
                if selected == provider.lower():
                    return True
            if _payload_selects_provider(value, provider=provider):
                return True
    elif isinstance(payload, Sequence) and not isinstance(
        payload,
        (str, bytes, bytearray),
    ):
        return any(_payload_selects_provider(value, provider=provider) for value in payload)
    return False


def reject_deprecated_provider_payload(payload: Any) -> None:
    if _payload_selects_provider(payload, provider=ResourceProvider.TINKER.value):
        raise ValueError("tinker is deprecated and cannot be selected for new runs")


def reject_deprecated_run_policy_payload(payload: Mapping[str, Any]) -> None:
    access = payload.get("access")
    if isinstance(access, Mapping) and any(
        ResourceProvider.TINKER.value in (access.get(field_name) or ())
        for field_name in (
            "credential_providers",
            "inference_providers",
            "tool_providers",
        )
    ):
        raise ValueError("tinker is deprecated and cannot be selected for new runs")


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
        raise ResearchHostedModelOverridesError(
            "actor execution overrides require local_execution; rejected: " + ", ".join(rejected),
            rejected_fields=rejected,
            detail={"rejected_fields": rejected},
        )


def _coerce_dict(payload: Any, *, label: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    raise ResearchApiError(
        f"Expected object response for {label}, received {type(payload).__name__}"
    )


def _coerce_dict_list(payload: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ResearchApiError(f"Expected {label} entries to be objects")
        return cast(list[dict[str, Any]], payload)
    raise ResearchApiError(f"Expected list response for {label}, received {type(payload).__name__}")


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
