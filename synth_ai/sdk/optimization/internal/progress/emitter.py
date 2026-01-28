"""UI-agnostic progress tracking for GEPA optimization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .dataclasses import (
    BaselineInfo,
    CandidateInfo,
    FrontierUpdate,
    GenerationSummary,
    GEPAProgress,
    RolloutSample,
    SeedInfo,
    StageInfo,
    TokenUsage,
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

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.progress.emitter.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "ProgressTracker"):
        raise RuntimeError("Rust core progress tracker required; synth_ai_py is unavailable.")
    return synth_ai_py


def _looks_like_validation(parsed: ParsedEvent) -> bool:
    if parsed.category == EventCategory.VALIDATION:
        return True
    if parsed.category != EventCategory.UNKNOWN:
        return False
    data = parsed.data
    if data.get("baseline_val_accuracy") is not None:
        return True
    if data.get("is_baseline") is True:
        return True
    return "validation" in parsed.event_type.lower()


@dataclass
class GEPAProgressEmitter:
    """Track progress state without any rendering or side effects."""

    env_name: str = "gepa"
    max_rollouts: int = 500
    max_trials: int = 100
    job_id: str | None = None

    candidates: list[CandidateInfo] = field(default_factory=list)
    pareto_history: list[FrontierUpdate] = field(default_factory=list)
    generation_history: list[GenerationSummary] = field(default_factory=list)
    baseline: BaselineInfo | None = None
    progress: GEPAProgress = field(default_factory=GEPAProgress)
    usage_data: dict[str, Any] | None = None
    raw_events: list[dict[str, Any]] = field(default_factory=list)

    _start_time: float = field(default_factory=time.time)
    _candidate_ids: set[str] = field(default_factory=set)
    _rust_tracker: Any | None = field(default=None, init=False, repr=False)

    def update(self, event: dict[str, Any]) -> ParsedEvent:
        """Update state from a raw SSE event and return the parsed event."""
        self.raw_events.append(event)
        rust = _require_rust()
        parsed = EventParser.parse(event)

        if self._rust_tracker is None:
            self._rust_tracker = rust.ProgressTracker()

        state = self._rust_tracker.update(event)
        if isinstance(state, dict):
            self._apply_rust_state(state)

        if isinstance(parsed, UsageEvent):
            self._apply_usage(parsed)

        return parsed

    def _apply_rust_state(self, state: dict[str, Any]) -> None:
        progress = state.get("progress") if isinstance(state.get("progress"), dict) else {}
        self.progress = GEPAProgress(**progress)

        baseline = state.get("baseline") if isinstance(state.get("baseline"), dict) else None
        if baseline:
            rollout = baseline.get("rollout_sample")
            if isinstance(rollout, list):
                baseline["rollout_sample"] = [
                    RolloutSample.from_dict(item) for item in rollout if isinstance(item, dict)
                ]
            self.baseline = BaselineInfo(**baseline)
        else:
            self.baseline = None

        candidates = state.get("candidates") if isinstance(state.get("candidates"), list) else []
        parsed_candidates: list[CandidateInfo] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            stages = candidate.get("stages")
            if isinstance(stages, dict):
                stage_map: dict[str, StageInfo] = {}
                for stage_id, stage in stages.items():
                    if isinstance(stage, dict):
                        stage_map[str(stage_id)] = StageInfo.from_dict(stage)
                candidate["stages"] = stage_map
            seed_info = candidate.get("seed_info")
            if isinstance(seed_info, list):
                candidate["seed_info"] = [
                    SeedInfo.from_dict(item) for item in seed_info if isinstance(item, dict)
                ]
            rollout = candidate.get("rollout_sample")
            if isinstance(rollout, list):
                candidate["rollout_sample"] = [
                    RolloutSample.from_dict(item) for item in rollout if isinstance(item, dict)
                ]
            token_usage = candidate.get("token_usage")
            if isinstance(token_usage, dict):
                candidate["token_usage"] = TokenUsage.from_dict(token_usage)
            parsed = CandidateInfo(**candidate)
            if self._rust_tracker is not None:
                parsed.raw_data = {}
            parsed_candidates.append(parsed)
        self.candidates = parsed_candidates

        frontier_updates = (
            state.get("frontier_history") if isinstance(state.get("frontier_history"), list) else []
        )
        self.pareto_history = [
            FrontierUpdate(**update) for update in frontier_updates if isinstance(update, dict)
        ]

        generations = (
            state.get("generation_history")
            if isinstance(state.get("generation_history"), list)
            else []
        )
        self.generation_history = [
            GenerationSummary(**gen) for gen in generations if isinstance(gen, dict)
        ]

    def _apply_baseline(self, event: BaselineEvent) -> None:
        self.baseline = BaselineInfo(
            accuracy=event.accuracy,
            instance_scores=event.instance_scores or [],
            prompt=event.prompt,
        )
        self.progress.baseline_score = event.accuracy

    def _apply_candidate(self, event: CandidateEvent) -> None:
        is_baseline = event.data.get("is_baseline", False) or event.data.get("parent_id") is None

        if is_baseline and not self.baseline:
            instance_scores = event.data.get("instance_scores", [])
            seeds_evaluated = event.data.get("seeds_evaluated", [])
            prompt = (
                event.data.get("prompt")
                or event.data.get("prompt_text")
                or event.data.get("transformation")
            )

            self.baseline = BaselineInfo(
                accuracy=event.accuracy,
                instance_scores=instance_scores if isinstance(instance_scores, list) else [],
                seeds_evaluated=seeds_evaluated if isinstance(seeds_evaluated, list) else [],
                prompt=prompt if isinstance(prompt, dict) else None,
            )
            if event.accuracy is not None:
                self.progress.baseline_score = event.accuracy

        if event.candidate_id in self._candidate_ids:
            return
        self._candidate_ids.add(event.candidate_id)

        candidate = CandidateInfo.from_event_data(event.data, event.candidate_id)
        self.candidates.append(candidate)

        self.progress.candidates_evaluated = len(self.candidates)
        if event.accuracy is not None and event.accuracy > self.progress.best_score:
            self.progress.best_score = event.accuracy

    def _apply_frontier(self, event: FrontierEvent) -> None:
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

    def _apply_progress(self, event: ProgressEvent) -> None:
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

    def _apply_generation(self, event: GenerationEvent) -> None:
        summary = GenerationSummary(
            generation=event.generation,
            candidates_proposed=event.candidates_proposed,
            candidates_accepted=event.candidates_accepted,
            best_accuracy=event.best_accuracy,
            timestamp=time.time() - self._start_time,
        )
        self.generation_history.append(summary)
        self.progress.generations_completed = event.generation

    def _apply_complete(self, event: CompleteEvent) -> None:
        self.progress.phase = "complete"
        if event.best_score is not None:
            self.progress.best_score = event.best_score
        if event.baseline_score is not None:
            self.progress.baseline_score = event.baseline_score
        if event.finish_reason:
            self.progress.finish_reason = event.finish_reason

    def _apply_termination(self, event: TerminationEvent) -> None:
        self.progress.finish_reason = event.reason

    def _apply_usage(self, event: UsageEvent) -> None:
        self.usage_data = event.data

    def _apply_validation(self, event: ParsedEvent) -> None:
        data = event.data
        if data.get("is_baseline", False) or data.get("baseline_val_accuracy") is not None:
            val_accuracy = data.get("baseline_val_accuracy") or data.get("accuracy")
            if val_accuracy is not None:
                if self.baseline:
                    self.baseline.val_accuracy = val_accuracy
                if self.progress.baseline_score is None:
                    self.progress.baseline_score = val_accuracy


__all__ = ["GEPAProgressEmitter"]
