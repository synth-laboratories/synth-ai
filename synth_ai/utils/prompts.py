from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

__all__ = ["ensure_required_args"]


def ensure_required_args(
    args: Namespace,
    prompts: Mapping[str, str],
    *,
    coerce: Mapping[str, Callable[[Any], Any]] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> Namespace:
    """Ensure required CLI arguments are populated.

    Legacy helper that historically prompted users. Our tests rely on it to
    populate defaults and perform simple coercions, so we implement a minimal,
    non-interactive version that mirrors that behaviour.
    """

    coerce_map = dict(coerce or {})
    default_map: MutableMapping[str, Any] = dict(defaults or {})

    for key, label in prompts.items():
        value = getattr(args, key, None)
        if value in (None, "") and key in default_map:
            value = default_map[key]
        if value in (None, ""):
            raise ValueError(f"{label} is required")
        if key in coerce_map:
            try:
                value = coerce_map[key](value)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Failed to normalize {label}: {exc}") from exc
        setattr(args, key, value)
    return args
