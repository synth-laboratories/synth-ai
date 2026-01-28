"""Generic dictionary utilities backed by Rust core."""

from typing import Any, Mapping, MutableMapping

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for utils.dict.") from exc


def deep_update(
    base: MutableMapping[str, Any], overrides: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Deep update with support for dot-notation keys (e.g., 'a.b.c')."""
    updated = synth_ai_py.deep_update(base, overrides)
    if isinstance(base, dict) and isinstance(updated, dict):
        base.clear()
        base.update(updated)
        return base
    return updated
