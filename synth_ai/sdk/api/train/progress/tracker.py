"""Unified GEPA Progress Tracker with configurable display modes.

This replaces the various ad-hoc trackers scattered across research/ and cookbooks/.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from .dataclasses import (
    BaselineInfo,
    CandidateInfo,
    FrontierUpdate,
    GenerationSummary,
    GEPAProgress,
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


class DisplayMode(Enum):
    """Display modes for terminal output."""

    SILENT = "silent"  # No output
    MINIMAL = "minimal"  # One-liners only
    TQDM = "tqdm"  # tqdm progress bars
    RICH = "rich"  # Rich console (future)


@dataclass
class GEPAProgressTracker:
    """Unified GEPA progress tracker with configurable display.

    Example usage:

        # Simple terminal streaming (local dev)
        tracker = GEPAProgressTracker(
            display_mode=DisplayMode.TQDM,
            env_name="banking77",
            max_rollouts=500,
        )

        async for event in job.stream_events():
            tracker.update(event)

        # Access final state
        print(f"Best: {tracker.best_score:.2%}")
        print(f"Candidates: {len(tracker.candidates)}")

        # Generate analysis
        tracker.save_all("./results")
        tracker.print_summary()

        # Headless (no terminal output, just tracking)
        tracker = GEPAProgressTracker(display_mode=DisplayMode.SILENT)
        async for event in job.stream_events():
            tracker.update(event)
        results = tracker.to_analysis_dict()
    """

    # Configuration
    display_mode: DisplayMode = DisplayMode.TQDM
    env_name: str = "gepa"
    max_rollouts: int = 500
    max_trials: int = 100
    job_id: str | None = None  # Optional job ID for log file naming
    log_dir: str | Path | None = None  # Directory for log file (defaults to cwd)
    enable_log_file: bool = False  # Whether to write events to a log file

    # Tracking state (initialized in __post_init__)
    candidates: list[CandidateInfo] = field(default_factory=list)
    pareto_history: list[FrontierUpdate] = field(default_factory=list)
    generation_history: list[GenerationSummary] = field(default_factory=list)
    baseline: BaselineInfo | None = None
    progress: GEPAProgress = field(default_factory=GEPAProgress)
    usage_data: dict[str, Any] | None = None
    raw_events: list[dict[str, Any]] = field(default_factory=list)

    # Internal state
    _start_time: float = field(default_factory=time.time)
    _rollout_pbar: Any = None
    _trial_pbar: Any = None
    _candidate_ids: set[str] = field(default_factory=set)
    _tqdm_available: bool = False
    _tqdm_module: Any = None
    _log_file: TextIO | None = field(default=None, repr=False)
    _log_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize progress bars if using tqdm mode."""
        self._start_time = time.time()
        self._candidate_ids = set()

        if self.display_mode == DisplayMode.TQDM:
            self._init_tqdm()

        # Initialize log file if enabled
        if self.enable_log_file:
            self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize event log file."""
        log_dir = Path(self.log_dir) if self.log_dir else Path.cwd()
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if self.job_id:
            filename = f"gepa_{self.job_id}_{timestamp}.log"
        else:
            filename = f"gepa_{self.env_name}_{timestamp}.log"

        self._log_path = log_dir / filename

        try:
            # Long-lived log file; closed explicitly in close()
            self._log_file = open(self._log_path, "w")  # noqa: SIM115
            # Write header
            header = {
                "type": "log_header",
                "timestamp": time.time(),
                "env_name": self.env_name,
                "job_id": self.job_id,
                "max_rollouts": self.max_rollouts,
                "max_trials": self.max_trials,
            }
            self._log_file.write(json.dumps(header) + "\n")
            self._log_file.flush()

            if self.display_mode != DisplayMode.SILENT:
                self._print(f"Logging events to: {self._log_path}")
        except Exception as e:
            if self.display_mode != DisplayMode.SILENT:
                self._print(f"Warning: Could not create log file: {e}")
            self._log_file = None
            self._log_path = None

    def _log_event(self, event: dict[str, Any]) -> None:
        """Write event to log file."""
        if self._log_file is not None:
            try:
                # Add timestamp if not present
                log_entry = {
                    "logged_at": time.time(),
                    "elapsed_seconds": time.time() - self._start_time,
                    **event,
                }
                self._log_file.write(json.dumps(log_entry) + "\n")
                self._log_file.flush()
            except Exception:
                pass  # Silently ignore log write errors

    def set_job_id(self, job_id: str) -> None:
        """Set job ID after tracker creation (useful when job ID isn't known until submission).

        If logging is enabled and no log file exists yet, this will create one.
        If a log file already exists, this updates the job_id in the header.
        """
        self.job_id = job_id

        # If logging is enabled but log file wasn't created (no job_id at init), create it now
        if self.enable_log_file and self._log_file is None:
            self._init_log_file()
        elif self._log_file is not None:
            # Write job_id update event
            self._log_event({"type": "job_id_set", "job_id": job_id})

    def _init_tqdm(self) -> None:
        """Initialize tqdm progress bars."""
        try:
            from tqdm import tqdm

            self._tqdm_available = True
            self._tqdm_module = tqdm

            self._print_banner()

            self._rollout_pbar = tqdm(
                total=self.max_rollouts,
                desc="Rollouts",
                unit="rollout",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
            self._trial_pbar = tqdm(
                total=self.max_trials,
                desc="Trials  ",
                unit="trial",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        except ImportError:
            self._tqdm_available = False
            if self.display_mode == DisplayMode.TQDM:
                print("Warning: tqdm not available, falling back to minimal output")
                self.display_mode = DisplayMode.MINIMAL

    def _print_banner(self) -> None:
        """Print startup banner."""
        if self.display_mode == DisplayMode.SILENT:
            return

        banner = f"""
{'=' * 60}
GEPA Optimization Progress - {self.env_name}
{'=' * 60}"""
        print(banner)

    def _print(self, msg: str) -> None:
        """Print message respecting display mode."""
        if self.display_mode == DisplayMode.SILENT:
            return

        if self._tqdm_available and self._tqdm_module:
            self._tqdm_module.write(msg)
        else:
            print(msg, flush=True)

    def _update_progress_bars(self) -> None:
        """Update tqdm progress bars."""
        if not self._tqdm_available:
            return

        if self._rollout_pbar:
            self._rollout_pbar.n = min(self.progress.rollouts_completed, self.max_rollouts)
            self._rollout_pbar.refresh()
        if self._trial_pbar:
            self._trial_pbar.n = min(self.progress.candidates_evaluated, self.max_trials)
            self._trial_pbar.refresh()

    def update(self, event: dict[str, Any]) -> None:
        """Update tracker from raw SSE event.

        Args:
            event: Raw SSE event dict with 'type' and 'data' keys
        """
        # Store raw event
        self.raw_events.append(event)

        # Log event to file if enabled
        self._log_event(event)

        # Parse event
        parsed = EventParser.parse(event)

        # Update elapsed time
        self.progress.elapsed_seconds = time.time() - self._start_time

        # Dispatch based on category
        if isinstance(parsed, BaselineEvent):
            self._handle_baseline(parsed)
        elif isinstance(parsed, CandidateEvent):
            self._handle_candidate(parsed)
        elif isinstance(parsed, FrontierEvent):
            self._handle_frontier(parsed)
        elif isinstance(parsed, ProgressEvent):
            self._handle_progress(parsed)
        elif isinstance(parsed, GenerationEvent):
            self._handle_generation(parsed)
        elif isinstance(parsed, CompleteEvent):
            self._handle_complete(parsed)
        elif isinstance(parsed, TerminationEvent):
            self._handle_termination(parsed)
        elif isinstance(parsed, UsageEvent):
            self._handle_usage(parsed)
        elif parsed.category == EventCategory.VALIDATION or parsed.category == EventCategory.UNKNOWN and (
            parsed.data.get("baseline_val_accuracy") is not None or
            parsed.data.get("is_baseline") is True or
            "validation" in parsed.event_type.lower()
        ):
            self._handle_validation(parsed)
        # ✅ ADD: Also handle job.event type events that contain validation data
        # (postgrest_emitter wraps events as job.event, so we need to check data for validation indicators)
        elif parsed.category == EventCategory.UNKNOWN and (
            parsed.data.get("baseline_val_accuracy") is not None or
            parsed.data.get("is_baseline") is True or
            "validation" in parsed.event_type.lower()
        ):
            self._handle_validation(parsed)

    def _handle_baseline(self, event: BaselineEvent) -> None:
        """Handle baseline evaluation event."""
        self.baseline = BaselineInfo(
            accuracy=event.accuracy,
            instance_scores=event.instance_scores or [],
            prompt=event.prompt,
        )
        self.progress.baseline_score = event.accuracy

        if self.display_mode != DisplayMode.SILENT:
            acc_str = f"{event.accuracy:.2%}" if event.accuracy else "N/A"
            self._print(f"Baseline: {acc_str}")

    def _handle_candidate(self, event: CandidateEvent) -> None:
        """Handle candidate evaluation event."""
        # Check if this is a baseline candidate
        is_baseline = event.data.get("is_baseline", False) or event.data.get("parent_id") is None
        
        # Extract baseline info from baseline candidate events
        if is_baseline and not self.baseline:
            instance_scores = event.data.get("instance_scores", [])
            seeds_evaluated = event.data.get("seeds_evaluated", [])
            prompt = event.data.get("prompt") or event.data.get("prompt_text") or event.data.get("transformation")
            
            self.baseline = BaselineInfo(
                accuracy=event.accuracy,
                instance_scores=instance_scores if isinstance(instance_scores, list) else [],
                seeds_evaluated=seeds_evaluated if isinstance(seeds_evaluated, list) else [],
                prompt=prompt if isinstance(prompt, dict) else None,
            )
            if event.accuracy is not None:
                self.progress.baseline_score = event.accuracy
            
            if self.display_mode != DisplayMode.SILENT:
                acc_str = f"{event.accuracy:.2%}" if event.accuracy else "N/A"
                self._print(f"Baseline: {acc_str}")
        
        # Avoid duplicates
        if event.candidate_id in self._candidate_ids:
            return
        self._candidate_ids.add(event.candidate_id)

        # Create candidate info
        candidate = CandidateInfo.from_event_data(event.data, event.candidate_id)
        self.candidates.append(candidate)

        # Update progress
        self.progress.candidates_evaluated = len(self.candidates)
        if event.accuracy is not None and event.accuracy > self.progress.best_score:
            self.progress.best_score = event.accuracy

        self._update_progress_bars()

    def _handle_frontier(self, event: FrontierEvent) -> None:
        """Handle pareto frontier update."""
        # Get timestamp_ms from event data if available
        timestamp_ms = event.data.get("timestamp_ms") if event.data else None

        update = FrontierUpdate(
            timestamp=time.time() - self._start_time,
            added=event.added or [],
            removed=event.removed or [],
            frontier=event.frontier or [],
            frontier_scores=event.frontier_scores or {},
            frontier_size=event.frontier_size,
            optimistic_score=event.best_score,
            generation=event.data.get("generation") if event.data else None,
            baseline_score=event.data.get("baseline_score") if event.data else None,
            timestamp_ms=timestamp_ms,
        )
        self.pareto_history.append(update)

        if event.best_score is not None and event.best_score > self.progress.best_score:
            self.progress.best_score = event.best_score

        if self.display_mode != DisplayMode.SILENT:
            lift = self.progress.lift
            lift_str = f" (lift: {lift:+.2%})" if lift is not None else ""
            self._print(
                f"Frontier [{event.frontier_size}]: best={event.best_score:.2%}{lift_str}"
            )

    def _handle_progress(self, event: ProgressEvent) -> None:
        """Handle progress update event."""
        if event.rollouts_completed > self.progress.rollouts_completed:
            self.progress.rollouts_completed = event.rollouts_completed

        if event.rollouts_total:
            self.progress.rollouts_total = event.rollouts_total

        if event.trials_completed > self.progress.candidates_evaluated:
            self.progress.candidates_evaluated = event.trials_completed

        if event.best_score is not None and event.best_score > self.progress.best_score:
            self.progress.best_score = event.best_score

        if event.baseline_score is not None and self.progress.baseline_score is None:
            self.progress.baseline_score = event.baseline_score

        self._update_progress_bars()

    def _handle_generation(self, event: GenerationEvent) -> None:
        """Handle generation complete event."""
        summary = GenerationSummary(
            generation=event.generation,
            candidates_proposed=event.candidates_proposed,
            candidates_accepted=event.candidates_accepted,
            best_accuracy=event.best_accuracy,
            timestamp=time.time() - self._start_time,
        )
        self.generation_history.append(summary)
        self.progress.generations_completed = event.generation

        if self.display_mode != DisplayMode.SILENT:
            self._print(f"Generation {event.generation} complete (best: {event.best_accuracy:.2%})")

    def _handle_complete(self, event: CompleteEvent) -> None:
        """Handle optimization complete event."""
        self.progress.phase = "complete"

        if event.best_score is not None:
            self.progress.best_score = event.best_score

        if event.baseline_score is not None:
            self.progress.baseline_score = event.baseline_score

        if event.finish_reason:
            self.progress.finish_reason = event.finish_reason

        if self.display_mode != DisplayMode.SILENT:
            self._print("Optimization complete!")

    def _handle_termination(self, event: TerminationEvent) -> None:
        """Handle termination event."""
        self.progress.finish_reason = event.reason

        if self.display_mode != DisplayMode.SILENT:
            self._print(f"Termination triggered: {event.reason}")

    def _handle_usage(self, event: UsageEvent) -> None:
        """Handle usage/cost event."""
        self.usage_data = event.data

        if self.display_mode != DisplayMode.SILENT:
            self._print(f"Cost: ${event.total_usd:.4f}")

    def _handle_validation(self, event: ParsedEvent) -> None:
        """Handle validation scored event."""
        data = event.data
        if data.get("is_baseline", False) or data.get("baseline_val_accuracy") is not None:
            # Baseline validation result
            # Try to get baseline_val_accuracy from top level first, then accuracy
            val_accuracy = data.get("baseline_val_accuracy") or data.get("accuracy")
            if val_accuracy is not None:
                # Store in baseline info if available
                if self.baseline:
                    self.baseline.val_accuracy = val_accuracy
                # Also store in progress for backwards compatibility
                if self.progress.baseline_score is None:
                    self.progress.baseline_score = val_accuracy

    # === Public API ===

    @property
    def best_score(self) -> float:
        """Get best score achieved."""
        return self.progress.best_score

    @property
    def baseline_score(self) -> float | None:
        """Get baseline score."""
        return self.progress.baseline_score

    @property
    def current_frontier(self) -> list[str]:
        """Get current pareto frontier candidate IDs."""
        if self.pareto_history:
            return self.pareto_history[-1].frontier
        return []

    def get_candidate(self, candidate_id: str) -> CandidateInfo | None:
        """Get candidate by ID."""
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        return None

    def get_pareto_candidates(self) -> list[CandidateInfo]:
        """Get candidates currently in pareto frontier."""
        frontier_ids = set(self.current_frontier)
        return [c for c in self.candidates if c.candidate_id in frontier_ids]

    def print_summary(self) -> None:
        """Print final summary to terminal."""
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)

        print(f"Status: {self.progress.phase}")
        print()

        print("Scoring Summary:")
        print(f"  Baseline:     {self.baseline_score:.2%}" if self.baseline_score else "  Baseline:     N/A")
        print(f"  Best:         {self.best_score:.2%}")
        if self.progress.lift is not None:
            print(f"  Lift:         {self.progress.lift:+.2%}")
        print()

        print("Optimization Stats:")
        print(f"  Rollouts:     {self.progress.rollouts_completed}")
        print(f"  Candidates:   {len(self.candidates)}")
        print(f"  Generations:  {self.progress.generations_completed}")
        print(f"  Frontier:     {len(self.current_frontier)}")
        print(f"  Finish:       {self.progress.finish_reason or 'N/A'}")
        print()

        print(f"Time: {self.progress.elapsed_seconds:.1f}s ({self.progress.elapsed_seconds/60:.1f} min)")

        if self.usage_data:
            total = self.usage_data.get("total_usd", 0)
            print(f"Cost: ${total:.4f}")

        print("=" * 60)

    def to_analysis_dict(self) -> dict[str, Any]:
        """Export all tracking data for analysis.

        Includes first-class program structure (stages), seed_info, token_usage, etc.
        """
        candidates_list = []
        for c in self.candidates:
            candidate_dict: dict[str, Any] = {
                "candidate_id": c.candidate_id,
                "accuracy": c.accuracy,
                "val_accuracy": c.val_accuracy,
                "train_accuracy": c.train_accuracy,
                "generation": c.generation,
                "parent_id": c.parent_id,
                "is_pareto": c.candidate_id in self.current_frontier,
                "accepted": c.accepted,
                "instance_scores": c.instance_scores,
                "mutation_type": c.mutation_type,
                "mutation_params": c.mutation_params,
                "prompt_summary": c.prompt_summary or c.get_prompt_summary(),
                "timestamp_ms": c.timestamp_ms,
            }

            # Add stages (first-class program structure)
            if c.stages:
                candidate_dict["stages"] = {
                    stage_id: stage.to_dict()
                    for stage_id, stage in c.stages.items()
                }

            # Add seed_scores
            if c.seed_scores:
                candidate_dict["seed_scores"] = c.seed_scores

            # Add token_usage
            if c.token_usage:
                candidate_dict["token_usage"] = c.token_usage.to_dict()

            # Add cost
            if c.cost_usd is not None:
                candidate_dict["cost_usd"] = c.cost_usd

            candidates_list.append(candidate_dict)

        return {
            "env_name": self.env_name,
            "progress": {
                "phase": self.progress.phase,
                "rollouts_completed": self.progress.rollouts_completed,
                "candidates_evaluated": self.progress.candidates_evaluated,
                "generations_completed": self.progress.generations_completed,
                "best_score": self.progress.best_score,
                "baseline_score": self.progress.baseline_score,
                "lift": self.progress.lift,
                "elapsed_seconds": self.progress.elapsed_seconds,
                "finish_reason": self.progress.finish_reason,
            },
            "baseline": {
                "accuracy": self.baseline.accuracy if self.baseline else None,
                "val_accuracy": self.baseline.val_accuracy if self.baseline else None,
                "instance_scores": self.baseline.instance_scores if self.baseline else [],
            },
            "candidates": candidates_list,
            "pareto_history": [
                {
                    "timestamp": u.timestamp,
                    "timestamp_ms": u.timestamp_ms,
                    "added": u.added,
                    "removed": u.removed,
                    "frontier": u.frontier,
                    "frontier_size": u.frontier_size,
                    "frontier_scores": u.frontier_scores,
                    "optimistic_score": u.optimistic_score,
                    "baseline_score": u.baseline_score,
                    "generation": u.generation,
                }
                for u in self.pareto_history
            ],
            "generation_history": [
                {
                    "generation": g.generation,
                    "candidates_proposed": g.candidates_proposed,
                    "candidates_accepted": g.candidates_accepted,
                    "best_accuracy": g.best_accuracy,
                    "timestamp": g.timestamp,
                }
                for g in self.generation_history
            ],
            "usage": self.usage_data,
        }

    @property
    def log_path(self) -> Path | None:
        """Get path to the log file if logging is enabled."""
        return self._log_path

    def close(self) -> None:
        """Close progress bars, log file, and clean up."""
        if self._rollout_pbar:
            self._rollout_pbar.close()
        if self._trial_pbar:
            self._trial_pbar.close()

        # Close log file
        if self._log_file is not None:
            try:
                # Write footer with final stats
                footer = {
                    "type": "log_footer",
                    "timestamp": time.time(),
                    "elapsed_seconds": time.time() - self._start_time,
                    "total_events": len(self.raw_events),
                    "best_score": self.best_score,
                    "baseline_score": self.baseline_score,
                    "candidates_count": len(self.candidates),
                    "phase": self.progress.phase,
                }
                self._log_file.write(json.dumps(footer) + "\n")
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
