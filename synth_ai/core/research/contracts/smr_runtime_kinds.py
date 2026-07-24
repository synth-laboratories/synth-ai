"""Public Managed Research runtime-kind enum."""

from __future__ import annotations

from enum import StrEnum


class SmrRuntimeKind(StrEnum):
    SANDBOX_AGENT = "sandbox_agent"
    HORIZONS = "horizons"
    CODEX = "codex"
    REACT = "react"
    CONTAINER_HTTP = "container_http"
    IMAGE_REF = "image_ref"
    SOURCE_BUILD = "source_build"


SMR_RUNTIME_KIND_VALUES: tuple[str, ...] = tuple(kind.value for kind in SmrRuntimeKind)


def coerce_smr_runtime_kind(
    value: SmrRuntimeKind | str | None,
    *,
    field_name: str = "runtime_kind",
) -> SmrRuntimeKind | None:
    if value is None:
        return None
    if isinstance(value, SmrRuntimeKind):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    try:
        return SmrRuntimeKind(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_RUNTIME_KIND_VALUES)}"
        ) from exc


__all__ = [
    "SMR_RUNTIME_KIND_VALUES",
    "SmrRuntimeKind",
    "coerce_smr_runtime_kind",
]
