from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import StreamType


@dataclass(slots=True)
class StreamConfig:
    """Configuration describing which streams to consume and how to filter them."""

    enabled_streams: set[StreamType] = field(default_factory=lambda: set(StreamType))
    event_types: set[str] | None = None  # Whitelist: only include these event types
    event_types_exclude: set[str] | None = None  # Blacklist: exclude these event types
    event_levels: set[str] | None = None
    metric_names: set[str] | None = None
    metric_phases: set[str] | None = None
    timeline_phases: set[str] | None = None
    sample_rate: float = 1.0
    max_events_per_poll: int | None = None
    deduplicate: bool = True

    @classmethod
    def default(cls) -> StreamConfig:
        """Return a configuration representing the default (all streams) view."""
        return cls(
            event_types_exclude={
                # Filter out noisy events that just announce what metrics already show
                "sft.progress",  # Generic "Training progress" with no data
                "sft.loss",      # Generic "Loss update" with no data
                "sft.upstream.status",  # Very verbose status echo events
            }
        )

    @classmethod
    def minimal(cls) -> StreamConfig:
        """Return a configuration streaming status updates only."""
        return cls(enabled_streams={StreamType.STATUS})

    @classmethod
    def verbose(cls) -> StreamConfig:
        """Return a configuration with all streams and events (no filters)."""
        return cls()

    @classmethod
    def progress_only(cls) -> StreamConfig:
        """Return a configuration tailored to show training progress."""
        return cls(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={"sft.progress", "rl.train.step", "sft.validation.summary"},
            metric_names={"train.loss", "eval.reward_mean"},
        )

    @classmethod
    def errors_only(cls) -> StreamConfig:
        """Return a configuration that focuses on heightened severity signals."""
        return cls(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS},
            event_levels={"error", "warning"},
        )

    def should_include_event(self, event: dict[str, Any]) -> bool:
        """Determine whether an event message should be included."""
        event_type = event.get("type")
        
        # Apply blacklist first (takes precedence)
        if self.event_types_exclude and event_type in self.event_types_exclude:
            return False
        
        # Then apply whitelist
        if self.event_types and event_type not in self.event_types:
            return False
        
        if self.event_levels:
            return event.get("level") in self.event_levels
        return True

    def should_include_metric(self, metric: dict[str, Any]) -> bool:
        """Determine whether a metric point should be included."""
        if self.metric_names and metric.get("name") not in self.metric_names:
            return False
        if self.metric_phases:
            return metric.get("phase") in self.metric_phases
        return True

    def should_include_timeline(self, timeline_entry: dict[str, Any]) -> bool:
        """Determine whether a timeline entry should be included."""
        if self.timeline_phases:
            return timeline_entry.get("phase") in self.timeline_phases
        return True


__all__ = ["StreamConfig"]
