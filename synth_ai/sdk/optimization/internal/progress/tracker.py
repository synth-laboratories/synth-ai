"""Unified GEPA Progress Tracker with configurable display modes.

This replaces the various ad-hoc trackers scattered across research/ and cookbooks/.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from .dataclasses import BaselineInfo, CandidateInfo, GEPAProgress
from .emitter import GEPAProgressEmitter
from .events import (
    BaselineEvent,
    CandidateEvent,
    CompleteEvent,
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
        print(f"Best: {tracker.best_reward:.2%}")
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

    emitter: GEPAProgressEmitter = field(init=False)

    # Internal state
    _rollout_pbar: Any = None
    _trial_pbar: Any = None
    _tqdm_available: bool = False
    _tqdm_module: Any = None
    _log_file: TextIO | None = field(default=None, repr=False)
    _log_path: Path | None = field(default=None, repr=False)
    _baseline_printed: bool = field(default=False, init=False)
    # NOTE: Rust ProgressTracker is owned by the emitter (SSOT).
    # No duplicate tracker here.

    def __post_init__(self) -> None:
        """Initialize progress bars if using tqdm mode."""
        self.emitter = GEPAProgressEmitter(
            env_name=self.env_name,
            max_rollouts=self.max_rollouts,
            max_trials=self.max_trials,
            job_id=self.job_id,
        )

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
                    "elapsed_seconds": self.emitter.progress.elapsed_seconds,
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
        self.emitter.job_id = job_id

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
                self._print("Warning: tqdm not available, falling back to minimal output")
                self.display_mode = DisplayMode.MINIMAL

    def _print_banner(self) -> None:
        """Print startup banner."""
        if self.display_mode == DisplayMode.SILENT:
            return

        banner = f"""
{"=" * 60}
GEPA Optimization Progress - {self.env_name}
{"=" * 60}"""
        self._print(banner)

    def _print(self, msg: str) -> None:
        """Print message respecting display mode."""
        if self.display_mode == DisplayMode.SILENT:
            return

        if self._tqdm_available and self._tqdm_module:
            self._tqdm_module.write(msg)
        else:
            logging.getLogger(__name__).info(msg)

    def _update_progress_bars(self) -> None:
        """Update tqdm progress bars."""
        if not self._tqdm_available:
            return

        if self._rollout_pbar:
            self._rollout_pbar.n = min(
                self.emitter.progress.rollouts_completed,
                self.max_rollouts,
            )
            self._rollout_pbar.refresh()
        if self._trial_pbar:
            self._trial_pbar.n = min(
                self.emitter.progress.candidates_evaluated,
                self.max_trials,
            )
            self._trial_pbar.refresh()

    def update(self, event: dict[str, Any]) -> None:
        """Update tracker from raw SSE event.

        Args:
            event: Raw SSE event dict with 'type' and 'data' keys
        """
        # Emitter feeds event to Rust core ProgressTracker (SSOT)
        # and syncs all state back. No duplicate tracking here.
        parsed = self.emitter.update(event)
        self._log_event(event)
        self._update_progress_bars()

        # Dispatch to UI handlers based on parsed event category
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

    def _handle_baseline(self, event: BaselineEvent) -> None:
        """Handle baseline evaluation event."""
        if self.display_mode == DisplayMode.SILENT or self._baseline_printed:
            return
        acc_str = f"{event.reward:.2%}" if event.reward else "N/A"
        self._print(f"Baseline: {acc_str}")
        self._baseline_printed = True

    def _handle_candidate(self, event: CandidateEvent) -> None:
        """Handle candidate evaluation event."""
        if self.display_mode == DisplayMode.SILENT or self._baseline_printed:
            return
        is_baseline = event.data.get("is_baseline", False) or event.data.get("parent_id") is None
        if is_baseline and self.emitter.baseline:
            acc_str = f"{event.reward:.2%}" if event.reward else "N/A"
            self._print(f"Baseline: {acc_str}")
            self._baseline_printed = True

    def _handle_frontier(self, event: FrontierEvent) -> None:
        """Handle pareto frontier update."""
        if self.display_mode != DisplayMode.SILENT:
            lift = self.emitter.progress.lift
            lift_str = f" (lift: {lift:+.2%})" if lift is not None else ""
            best_reward = event.best_reward
            if best_reward is None:
                best_reward = self.emitter.progress.best_reward
            self._print(f"Frontier [{event.frontier_size}]: best={best_reward:.2%}{lift_str}")

    def _handle_progress(self, event: ProgressEvent) -> None:
        """Handle progress update event."""
        return

    def _handle_generation(self, event: GenerationEvent) -> None:
        """Handle generation complete event."""
        if self.display_mode != DisplayMode.SILENT:
            self._print(f"Generation {event.generation} complete (best: {event.best_reward:.2%})")

    def _handle_complete(self, event: CompleteEvent) -> None:
        """Handle optimization complete event."""
        if self.display_mode != DisplayMode.SILENT:
            self._print("Optimization complete!")

    def _handle_termination(self, event: TerminationEvent) -> None:
        """Handle termination event."""
        if self.display_mode != DisplayMode.SILENT:
            self._print(f"Termination triggered: {event.reason}")

    def _handle_usage(self, event: UsageEvent) -> None:
        """Handle usage/cost event."""
        if self.display_mode != DisplayMode.SILENT:
            self._print(f"Cost: ${event.total_usd:.4f}")

    def _handle_validation(self, event: ParsedEvent) -> None:
        """Handle validation scored event."""
        return

    # === Public API ===

    @property
    def candidates(self) -> list[CandidateInfo]:
        """Tracked candidate summaries."""
        return self.emitter.candidates

    @property
    def pareto_history(self) -> list[Any]:
        """Tracked pareto frontier updates."""
        return self.emitter.pareto_history

    @property
    def generation_history(self) -> list[Any]:
        """Tracked generation summaries."""
        return self.emitter.generation_history

    @property
    def baseline(self) -> BaselineInfo | None:
        """Baseline summary (if available)."""
        return self.emitter.baseline

    @property
    def progress(self) -> GEPAProgress:
        """Progress snapshot."""
        return self.emitter.progress

    @property
    def usage_data(self) -> dict[str, Any] | None:
        """Usage/cost payload if emitted by the backend."""
        return self.emitter.usage_data

    @property
    def raw_events(self) -> list[dict[str, Any]]:
        """Raw events seen so far."""
        return self.emitter.raw_events

    @property
    def best_reward(self) -> float:
        """Get best reward achieved."""
        return self.emitter.progress.best_reward

    @property
    def best_score(self) -> float:
        """Deprecated: use best_reward instead."""
        return self.best_reward

    @property
    def baseline_reward(self) -> float | None:
        """Get baseline reward."""
        return self.emitter.progress.baseline_reward

    @property
    def baseline_score(self) -> float | None:
        """Deprecated: use baseline_reward instead."""
        return self.baseline_reward

    @property
    def current_frontier(self) -> list[str]:
        """Get current pareto frontier candidate IDs."""
        if self.emitter.pareto_history:
            return self.emitter.pareto_history[-1].frontier
        return []

    def get_candidate(self, candidate_id: str) -> CandidateInfo | None:
        """Get candidate by ID."""
        for c in self.emitter.candidates:
            if c.candidate_id == candidate_id:
                return c
        return None

    def get_pareto_candidates(self) -> list[CandidateInfo]:
        """Get candidates currently in pareto frontier."""
        frontier_ids = set(self.current_frontier)
        return [c for c in self.emitter.candidates if c.candidate_id in frontier_ids]

    def print_summary(self) -> None:
        """Print final summary to terminal."""
        self._print("")
        self._print("=" * 60)
        self._print("RESULTS")
        self._print("=" * 60)

        self._print(f"Status: {self.emitter.progress.phase}")
        self._print("")

        self._print("Scoring Summary:")
        self._print(
            f"  Baseline:     {self.baseline_reward:.2%}"
            if self.baseline_reward
            else "  Baseline:     N/A"
        )
        self._print(f"  Best:         {self.best_reward:.2%}")
        if self.emitter.progress.lift is not None:
            self._print(f"  Lift:         {self.emitter.progress.lift:+.2%}")
        self._print("")

        self._print("Optimization Stats:")
        self._print(f"  Rollouts:     {self.emitter.progress.rollouts_completed}")
        self._print(f"  Candidates:   {len(self.emitter.candidates)}")
        self._print(f"  Generations:  {self.emitter.progress.generations_completed}")
        self._print(f"  Frontier:     {len(self.current_frontier)}")
        self._print(f"  Finish:       {self.emitter.progress.finish_reason or 'N/A'}")
        self._print("")

        self._print(
            "Time: "
            f"{self.emitter.progress.elapsed_seconds:.1f}s "
            f"({self.emitter.progress.elapsed_seconds / 60:.1f} min)"
        )

        if self.emitter.usage_data:
            total = self.emitter.usage_data.get("total_usd", 0)
            self._print(f"Cost: ${total:.4f}")

        self._print("=" * 60)

    def to_analysis_dict(self) -> dict[str, Any]:
        """Export all tracking data for analysis.

        Includes first-class program structure (stages), seed_info, token_usage, etc.
        """
        candidates_list = []
        for c in self.emitter.candidates:
            candidate_dict: dict[str, Any] = {
                "candidate_id": c.candidate_id,
                "reward": c.reward,
                "objectives": c.objectives,
                "val_reward": c.val_reward,
                "train_reward": c.train_reward,
                "generation": c.generation,
                "parent_id": c.parent_id,
                "is_pareto": c.candidate_id in self.current_frontier,
                "accepted": c.accepted,
                "instance_rewards": c.instance_rewards,
                "instance_objectives": c.instance_objectives,
                "mutation_type": c.mutation_type,
                "mutation_params": c.mutation_params,
                "prompt_summary": c.prompt_summary or c.get_prompt_summary(),
                "timestamp_ms": c.timestamp_ms,
            }

            # Add stages (first-class program structure)
            if c.stages:
                candidate_dict["stages"] = {
                    stage_id: stage.to_dict() for stage_id, stage in c.stages.items()
                }

            # Add seed_rewards
            if c.seed_rewards:
                candidate_dict["seed_rewards"] = c.seed_rewards

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
                "phase": self.emitter.progress.phase,
                "rollouts_completed": self.emitter.progress.rollouts_completed,
                "candidates_evaluated": self.emitter.progress.candidates_evaluated,
                "generations_completed": self.emitter.progress.generations_completed,
                "best_reward": self.emitter.progress.best_reward,
                "baseline_reward": self.emitter.progress.baseline_reward,
                "lift": self.emitter.progress.lift,
                "elapsed_seconds": self.emitter.progress.elapsed_seconds,
                "finish_reason": self.emitter.progress.finish_reason,
            },
            "baseline": {
                "reward": self.emitter.baseline.reward if self.emitter.baseline else None,
                "objectives": self.emitter.baseline.objectives if self.emitter.baseline else None,
                "val_reward": self.emitter.baseline.val_reward if self.emitter.baseline else None,
                "instance_rewards": (
                    self.emitter.baseline.instance_rewards if self.emitter.baseline else []
                ),
                "instance_objectives": (
                    self.emitter.baseline.instance_objectives if self.emitter.baseline else None
                ),
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
                    "frontier_rewards": u.frontier_rewards,
                    "optimistic_reward": u.optimistic_reward,
                    "baseline_reward": u.baseline_reward,
                    "generation": u.generation,
                }
                for u in self.emitter.pareto_history
            ],
            "generation_history": [
                {
                    "generation": g.generation,
                    "candidates_proposed": g.candidates_proposed,
                    "candidates_accepted": g.candidates_accepted,
                    "best_reward": g.best_reward,
                    "timestamp": g.timestamp,
                }
                for g in self.emitter.generation_history
            ],
            "usage": self.emitter.usage_data,
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
                    "elapsed_seconds": self.emitter.progress.elapsed_seconds,
                    "total_events": len(self.emitter.raw_events),
                    "best_reward": self.best_reward,
                    "baseline_reward": self.baseline_reward,
                    "candidates_count": len(self.emitter.candidates),
                    "phase": self.emitter.progress.phase,
                }
                self._log_file.write(json.dumps(footer) + "\n")
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
