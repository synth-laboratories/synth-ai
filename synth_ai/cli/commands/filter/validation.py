from __future__ import annotations

import math
from collections.abc import MutableMapping
from typing import Any

__all__ = ["validate_filter_options"]


def validate_filter_options(options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Validate parameters passed to the filter CLI command."""
    # Coerce optional collections to the expected container types and strip blanks
    result: dict[str, Any] = dict(options)

    def _coerce_list(key: str) -> None:
        value = result.get(key)
        if value is None:
            result[key] = []
        elif isinstance(value, list | tuple | set):
            result[key] = [str(item).strip() for item in value if str(item).strip()]
        else:
            result[key] = [str(value).strip()] if str(value).strip() else []

    def _coerce_dict(key: str) -> None:
        value = result.get(key)
        if value is None:
            result[key] = {}
        elif isinstance(value, MutableMapping):
            normalized: dict[str, float] = {}
            for k, v in value.items():
                if k is None:
                    continue
                try:
                    number = float(v)
                    if math.isnan(number) or math.isinf(number):
                        continue
                    normalized[str(k).strip()] = number
                except Exception:
                    continue
            result[key] = normalized
        else:
            result[key] = {}

    _coerce_list("splits")
    _coerce_list("task_ids")
    _coerce_list("models")
    _coerce_dict("min_judge_scores")
    _coerce_dict("max_judge_scores")

    for duration_key in ("min_official_score", "max_official_score"):
        value = result.get(duration_key)
        if value is None or value == "":
            result[duration_key] = None
        else:
            try:
                result[duration_key] = float(value)
            except Exception:
                result[duration_key] = None

    for int_key in ("limit", "offset", "shuffle_seed"):
        value = result.get(int_key)
        if value is None or value == "":
            result[int_key] = None
        else:
            try:
                result[int_key] = int(value)
            except Exception:
                result[int_key] = None

    shuffle_value = result.get("shuffle")
    if isinstance(shuffle_value, str):
        result["shuffle"] = shuffle_value.strip().lower() in {"1", "true", "yes"}
    else:
        result["shuffle"] = bool(shuffle_value)

    # Preserve extra keys (e.g., min_created_at) as-is for downstream handling
    return result
