"""GEPA Progress Tracking Module.

Provides unified progress tracking for GEPA optimization jobs with:
- Centralized event parsing (handles [MASKED], variants, etc.)
- Typed event dataclasses for type-safe access
- Configurable display modes (silent, minimal, tqdm, rich)
- Modular result saving

Example usage:

    from synth_ai.sdk.api.train.progress import GEPAProgressTracker, DisplayMode

    # Terminal streaming with tqdm
    tracker = GEPAProgressTracker(
        display_mode=DisplayMode.TQDM,
        env_name="banking77",
        max_rollouts=500,
    )

    async for event in job.stream_events():
        tracker.update(event)

    # Access results
    print(f"Best: {tracker.best_score:.2%}")
    tracker.save_all("./results")

    # Headless tracking
    tracker = GEPAProgressTracker(display_mode=DisplayMode.SILENT)
    async for event in job.stream_events():
        tracker.update(event)
    results = tracker.to_analysis_dict()
"""

from .dataclasses import (
    BaselineInfo,
    CandidateInfo,
    FrontierUpdate,
    GenerationSummary,
    GEPAProgress,
    RolloutSample,
)
from .events import (
    BaselineEvent,
    CandidateEvent,
    CompleteEvent,
    EventCategory,
    EventParser,
    FrontierEvent,
    GenerationEvent,
    ParsedEvent,
    ProgressEvent,
    TerminationEvent,
    UsageEvent,
)
from .results import (
    save_candidates,
    save_pareto_history,
    save_raw_events,
    save_results,
    save_seed_analysis,
    save_summary_json,
    save_summary_txt,
)
from .tracker import DisplayMode, GEPAProgressTracker

__all__ = [
    # Tracker
    "GEPAProgressTracker",
    "DisplayMode",
    # Dataclasses
    "CandidateInfo",
    "FrontierUpdate",
    "BaselineInfo",
    "GEPAProgress",
    "GenerationSummary",
    "RolloutSample",
    # Events
    "EventParser",
    "EventCategory",
    "ParsedEvent",
    "BaselineEvent",
    "CandidateEvent",
    "FrontierEvent",
    "ProgressEvent",
    "GenerationEvent",
    "CompleteEvent",
    "TerminationEvent",
    "UsageEvent",
    # Results
    "save_results",
    "save_candidates",
    "save_pareto_history",
    "save_summary_json",
    "save_seed_analysis",
    "save_summary_txt",
    "save_raw_events",
]
