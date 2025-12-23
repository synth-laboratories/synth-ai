"""Validation helpers for eval options."""

from __future__ import annotations


def _parse_seeds(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return [int(part) for part in parts]


def _parse_metadata(values: list[str]) -> dict[str, str]:
    if not values:
        return {}
    parsed: dict[str, str] = {}
    for entry in values:
        if "=" not in entry:
            raise ValueError("Metadata filter must be key=value")
        key, value = entry.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_ops(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def validate_eval_options(options: dict[str, object]) -> dict[str, object]:
    normalized = dict(options)
    seeds = normalized.get("seeds") or ""
    normalized["seeds"] = _parse_seeds(seeds)  # type: ignore[arg-type]

    metadata = normalized.get("metadata") or []
    normalized["metadata"] = _parse_metadata(metadata)  # type: ignore[arg-type]

    for key in ("max_turns", "max_llm_calls", "concurrency"):
        if key in normalized and normalized[key] not in (None, ""):
            normalized[key] = int(normalized[key])  # type: ignore[arg-type]

    if "ops" in normalized:
        normalized["ops"] = _parse_ops(normalized.get("ops") or "")  # type: ignore[arg-type]

    if "poll" in normalized and normalized["poll"] not in (None, ""):
        normalized["poll"] = float(normalized["poll"])  # type: ignore[arg-type]

    if "return_trace" in normalized:
        value = str(normalized["return_trace"]).lower()
        normalized["return_trace"] = value in ("true", "1", "yes")

    return normalized


__all__ = ["validate_eval_options"]
