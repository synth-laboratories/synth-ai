"""Generic dictionary utilities."""

from typing import Any, Mapping, MutableMapping


def deep_update(
    base: MutableMapping[str, Any], overrides: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Deep update with support for dot-notation keys (e.g., 'a.b.c')."""
    for key, value in overrides.items():
        parts = str(key).split(".")
        current: MutableMapping[str, Any] = base
        for part in parts[:-1]:
            nested = current.get(part)
            if not isinstance(nested, MutableMapping):
                nested = {}
                current[part] = nested
            current = nested
        leaf = parts[-1]
        existing = current.get(leaf)
        if isinstance(existing, MutableMapping) and isinstance(value, Mapping):
            deep_update(existing, value)
        else:
            current[leaf] = value
    return base
