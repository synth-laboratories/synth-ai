from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any

__all__ = ["validate_eval_options"]

_SEED_RANGE = re.compile(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_seeds(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    elif isinstance(value, list | tuple | set):
        chunks = list(value)
    else:
        chunks = [value]
    seeds: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, int):
            seeds.append(chunk)
        else:
            text = str(chunk).strip()
            if not text:
                continue
            match = _SEED_RANGE.match(text)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                if start > end:
                    raise ValueError(f"Invalid seed range '{text}': start must be <= end")
                seeds.extend(range(start, end + 1))
            else:
                seeds.append(int(text))
    return seeds


def _normalize_metadata(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, MutableMapping):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        result: dict[str, str] = {}
        for item in value:
            if isinstance(item, str) and "=" in item:
                key, val = item.split("=", 1)
                result[key.strip()] = val.strip()
        return result
    if isinstance(value, str) and "=" in value:
        key, val = value.split("=", 1)
        return {key.strip(): val.strip()}
    return {}


def _ensure_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, MutableMapping):
        return dict(value)
    return {}


def validate_eval_options(options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Validate and normalise eval configuration options."""

    result: dict[str, Any] = dict(options)

    if "seeds" in result:
        result["seeds"] = _parse_seeds(result.get("seeds"))

    for field in ("max_turns", "max_llm_calls", "concurrency"):
        try:
            result[field] = _coerce_int(result.get(field))
        except Exception as exc:
            raise ValueError(f"Invalid value for {field}: {result.get(field)}") from exc

    if result.get("max_llm_calls") is None:
        result["max_llm_calls"] = 10
    if result.get("concurrency") is None:
        result["concurrency"] = 1

    if "return_trace" in result:
        result["return_trace"] = _coerce_bool(result.get("return_trace"))

    metadata_value = result.get("metadata")
    result["metadata"] = _normalize_metadata(metadata_value)

    if "ops" in result:
        ops_list = _ensure_list(result.get("ops"))
        result["ops"] = ops_list

    result["env_config"] = _ensure_dict(result.get("env_config"))
    result["policy_config"] = _ensure_dict(result.get("policy_config"))

    trace_format = result.get("trace_format")
    if trace_format is not None:
        result["trace_format"] = str(trace_format)

    metadata_sql = result.get("metadata_sql")
    if metadata_sql is not None and not isinstance(metadata_sql, str):
        result["metadata_sql"] = str(metadata_sql)

    model = result.get("model")
    if model is not None:
        result["model"] = str(model)

    app_id = result.get("app_id")
    if app_id is not None:
        result["app_id"] = str(app_id)

    return result
