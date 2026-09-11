"""Canonical async container-pool client, re-exported without a second transport.

See: evals/docs/handoffs/EVAL_EXECUTION_STREAMING_DELIVERY_PLAN_2026-09-10.md §3.
Prefer ``AsyncSynthClient().pools`` to inherit the configured backend identity;
``PoolClient`` is available for callers managing its asynchronous lifetime.
"""

from __future__ import annotations

try:
    from synth_containers.pools import (
        PoolClient,
        PoolClientError,
        PoolRolloutTimeout,
        PoolStateError,
    )
except ModuleNotFoundError as error:
    if error.name not in {"synth_containers", "synth_containers.pools"}:
        raise
    raise ImportError(
        "Container pools require synth-ai[pools] and its matching synth-containers artifact"
    ) from error

__all__ = ["PoolClient", "PoolClientError", "PoolRolloutTimeout", "PoolStateError"]
