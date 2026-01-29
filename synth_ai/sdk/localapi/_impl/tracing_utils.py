"""Utilities for wiring tracing_v3 into task apps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import synth_ai_py


def tracing_env_enabled(default: bool = False) -> bool:
    """Return True when tracing is enabled for task apps via environment variable."""

    return synth_ai_py.localapi_tracing_env_enabled(default)


def resolve_tracing_db_url() -> str | None:
    """Resolve tracing database URL using centralized logic."""

    return synth_ai_py.localapi_resolve_tracing_db_url()


def build_tracer_factory(
    make_tracer: Callable[..., Any], *, enabled: bool, db_url: str | None
) -> Callable[[], Any] | None:
    """Return a factory that instantiates a tracer when enabled, else None."""

    if not enabled:
        return None

    def _factory() -> Any:
        return make_tracer(db_url=db_url) if db_url else make_tracer()

    return _factory


def resolve_sft_output_dir() -> str | None:
    """Resolve location for writing SFT records, creating directory if requested."""

    return synth_ai_py.localapi_resolve_sft_output_dir()


def unique_sft_path(base_dir: str, *, run_id: str):
    """Return a unique JSONL path for an SFT record batch."""

    return synth_ai_py.localapi_unique_sft_path(base_dir, run_id)
