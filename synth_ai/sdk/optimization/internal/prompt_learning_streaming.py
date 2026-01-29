"""Streaming helpers for prompt learning jobs."""

from __future__ import annotations

from typing import Any, Sequence

from synth_ai.core.streaming import (
    JobStreamer,
    PromptLearningHandler,
    StreamConfig,
    StreamEndpoints,
    StreamType,
)


def build_prompt_learning_streamer(
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
        handlers = [PromptLearningHandler()]

    return JobStreamer(
        base_url=backend_url,
        api_key=api_key,
        job_id=job_id,
        endpoints=StreamEndpoints.prompt_learning(job_id),
        config=config,
        handlers=list(handlers),
        interval_seconds=interval,
        timeout_seconds=timeout,
    )


__all__ = ["build_prompt_learning_streamer"]
