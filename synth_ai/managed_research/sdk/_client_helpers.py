"""Shared internal coercion helpers for the Managed Research SDK client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.models.run_timeline import (
    SmrBranchMode,
    SmrRunBranchRequest,
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
