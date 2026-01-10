"""Generic dictionary utilities."""

import copy
from typing import Any, Mapping, MutableMapping


def deep_update(
    base: MutableMapping[str, Any], overrides: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Deep update with support for dot-notation keys (e.g., 'a.b.c').

    Dot-notation keys are split and create nested dictionaries.
    Regular keys are updated normally.
    """
    for key, value in overrides.items():
        if "." in key:
            keys = key.split(".")
            current = base
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], MutableMapping):
                    current[k] = {}
                current = current[k]
            final_key = keys[-1]
            if (
                isinstance(value, Mapping)
                and isinstance(current.get(final_key), MutableMapping)
                and not isinstance(value, str | bytes)
            ):
                nested = copy.deepcopy(dict(current[final_key]))
                current[final_key] = deep_update(nested, value)
            else:
                current[final_key] = copy.deepcopy(value)
        else:
            if (
                isinstance(value, Mapping)
                and isinstance(base.get(key), MutableMapping)
                and not isinstance(value, str | bytes)
            ):
                nested = copy.deepcopy(dict(base[key]))
                base[key] = deep_update(nested, value)
            else:
                base[key] = copy.deepcopy(value)
    return base
