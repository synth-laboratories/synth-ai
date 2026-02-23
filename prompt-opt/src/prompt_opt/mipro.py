"""Python API wrapper around the shared Rust prompt-opt core."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


try:
    import prompt_opt_rust as _rust
except Exception:  # pragma: no cover
    _rust = None


def proposer_backends() -> list[str]:
    """Return supported proposer backends from the Rust core."""
    if _rust is None:
        return ["single_prompt", "rlm"]
    return list(_rust.proposer_backends())


def run_mipro(
    *,
    config: dict[str, Any],
    initial_policy: dict[str, Any],
    dataset: dict[str, Any],
    task_llm: Callable[[str], str],
) -> dict[str, Any]:
    """Run MIPRO through the Rust core and return JSON-decoded result."""
    if _rust is None:
        raise RuntimeError(
            "Rust bindings are unavailable. Build/install prompt_opt_rust "
            "from prompt-opt/rust_py to use the shared Rust core from Python."
        )
    result_json = _rust.run_mipro_json(
        json.dumps(config),
        json.dumps(initial_policy),
        json.dumps(dataset),
        task_llm,
    )
    return json.loads(result_json)
