"""Streaming helpers for Graph Evolve jobs."""

from __future__ import annotations

from typing import Any, Sequence

from synth_ai.core.streaming import (
    GraphEvolveHandler,
    JobStreamer,
    StreamConfig,
    StreamEndpoints,
    StreamType,
)


def build_graph_evolve_streamer(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    handlers: Sequence[Any] | None,
    interval: float,
    timeout: float,
) -> JobStreamer:
    config = StreamConfig(
        enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
        max_events_per_poll=500,
        deduplicate=True,
    )

    if handlers is None:
        handlers = [GraphEvolveHandler()]

    return JobStreamer(
        base_url=backend_url,
        api_key=api_key,
        job_id=job_id,
        endpoints=StreamEndpoints.graph_evolve(job_id),
        config=config,
        handlers=list(handlers),
        interval_seconds=interval,
        timeout_seconds=timeout,
    )


__all__ = ["build_graph_evolve_streamer"]
