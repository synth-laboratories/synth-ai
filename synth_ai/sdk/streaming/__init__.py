"""Streaming SDK - consume training events.

This module provides utilities for streaming training events
during job execution for real-time monitoring.

Example:
    from synth_ai.sdk.streaming import JobStreamer, StreamConfig
    
    streamer = JobStreamer(config)
    async for message in streamer.stream():
        print(f"Event: {message}")
"""

from __future__ import annotations

# Re-export from existing location
from synth_ai.streaming import (
    JobStreamer,
    StreamConfig,
    StreamEndpoints,
    StreamMessage,
    StreamType,
    StreamHandler,
    CallbackHandler,
)

__all__ = [
    "JobStreamer",
    "StreamConfig",
    "StreamEndpoints",
    "StreamMessage",
    "StreamType",
    "StreamHandler",
    "CallbackHandler",
]

