"""Stream handlers for GEPA optimization and eval progress."""


import json
import textwrap
import time
from typing import Any, Callable

from synth_ai.core.streaming.handlers import StreamHandler
from synth_ai.core.streaming.types import StreamType
from synth_ai.sdk.optimization.progress.time import ProgressClock, ProgressPrinter


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_reward(event_data: dict[str, Any]) -> float | None:
    if not isinstance(event_data, dict):
        return None
    reward = _coerce_float(event_data.get("reward"))
    score_obj = event_data.get("score")
    if reward is None and isinstance(score_obj, dict):
        reward = _coerce_float(score_obj.get("reward"))
    if reward is None:
        reward = _coerce_float(event_data.get("accuracy"))
    return reward


class IdleStatusTicker:
    """Log status when output has been idle for a configured interval."""

    def __init__(self) -> None:
        self._last_status_time: float | None = None

    def maybe_log(
        self,
        *,
        status: str,
        now: float,
        last_output_time: float | None,
        min_idle_seconds: float,
        log_fn: Callable[[str, float], None],
    ) -> None:
        if last_output_time is None:
            return
        idle_seconds = now - last_output_time
        if idle_seconds < min_idle_seconds:
            return
        if self._last_status_time is None or self._last_status_time < last_output_time:
            next_due = last_output_time + min_idle_seconds
        else:
            next_due = self._last_status_time + min_idle_seconds
        if now < next_due:
            return
        log_fn(f"Status: {status}", now)
        self._last_status_time = now


class GEPAStreamProgressHandler(StreamHandler):
    """GEPA progress handler with compact, human-friendly formatting."""

    SKIP_EVENT_SUBSTRINGS = ("rollout.concurrency", "billing")

    def __init__(
        self,
        total_generations: int | None,
        *,
        initial_population_size: int | None = None,
        children_per_generation: int | None = None,
        job_id: str | None = None,
        clock: ProgressClock | None = None,
        debug: bool = False,
    ) -> None:
        self.total_generations = total_generations
        self.initial_population_size = initial_population_size
        self.children_per_generation = children_per_generation
        self.job_id = job_id  # Filter events to this job only
        self.debug = debug
        self._printer = ProgressPrinter(clock=clock)
        self._phase: str | None = None
        self._phase_generation: int | None = None
        self._phase_start_time: float | None = None
        self._phase_candidates = 0
        self._phase_best_score = 0.0
        self._overall_best = 0.0
        self._last_best_prompt_sig: str | None = None
        self._last_best_candidate_sig: str | None = None
        self._seen_rollout_concurrency = False
        self._past_initial_phase = False
        self._generation_offset: int | None = None
        self._last_output_time: float | None = None
        self._last_status_time: float | None = None
        self._status_ticker = IdleStatusTicker()
        self._final_generation_complete_seen = False
        self._final_generation_complete_seen_at: float | None = None
        self._seen_candidates_by_gen: dict[int | None, set[str]] = {}
        self._seen_event_ids: set[str] = set()
        self._current_generation: int | None = None
        self._last_generation_completed: int | None = None
        self._last_run_id: str | None = None
        self.best_prompt: str | None = None
        self.best_score: float | None = None

    def _log(
        self,
        message: str,
        *,
        now: float | None = None,
        separator: str = "|",
        update_idle_timer: bool = True,
    ) -> None:
        timestamp = self._printer.now() if now is None else now
        self._printer.log(message, now=timestamp, separator=separator)
        if update_idle_timer:
            self._last_output_time = timestamp

    def _log_continuation(self, message: str, *, width: int = 100) -> None:
        """Print continuation lines aligned with text after '| ' in clock prefix."""
        # Align with text after "| ": "    49s | " or " 1m 05s | " = 10 chars
        margin = " " * 10
        wrapped = textwrap.fill(
            message,
            width=width,
            initial_indent=margin,
            subsequent_indent=margin,
            break_on_hyphens=False,
        )
        print(wrapped)

    def should_handle(self, message) -> bool:
        # Filter to only handle events from the target job
        if self.job_id is not None and hasattr(message, "job_id"):
            return message.job_id == self.job_id
        return True

    def _display_generation(self, backend_generation: Any) -> int | None:
        generation = _coerce_int(backend_generation)
        if generation is None:
            return None
        if generation == 0:
            self._generation_offset = 1
        elif self._generation_offset is None:
            if self.total_generations is not None and generation > self.total_generations:
                self._generation_offset = 1
            else:
                self._generation_offset = 0
        return generation + (self._generation_offset or 0)

    def _dedupe_event(self, event_type: str, event_data: dict[str, Any], run_id: str | None) -> bool:
        if not event_type.startswith("learning.policy.gepa."):
            return False

        # Run restarts: reset state when run_id changes
        if run_id and self._last_run_id and run_id != self._last_run_id:
            self._seen_candidates_by_gen.clear()
            self._seen_event_ids.clear()
            self._current_generation = None
            self._last_generation_completed = None
        if run_id:
            self._last_run_id = run_id

        generation = _coerce_int(event_data.get("generation"))

        if "candidate" in event_type:
            candidate_id = (
                event_data.get("candidate_id")
                or event_data.get("version_id")
                or event_data.get("id")
            )
            if not candidate_id and isinstance(event_data.get("program_candidate"), dict):
                candidate_id = event_data.get("program_candidate", {}).get("candidate_id")
            if candidate_id:
                seen = self._seen_candidates_by_gen.setdefault(generation, set())
                if str(candidate_id) in seen:
                    return True
                seen.add(str(candidate_id))

        # Drop out-of-order generation repeats (e.g., replays after completion)
        if event_type.endswith("generation.started") and generation is not None:
            if self._last_generation_completed is not None and generation <= self._last_generation_completed:
                return True
        if event_type.endswith("generation.completed") and generation is not None:
            if self._last_generation_completed is not None and generation <= self._last_generation_completed:
                return True

        return False

    def _phase_label(self) -> str:
        if self._phase == "initial":
            return "Initial population"
        if self._phase_generation is None:
            return "Generation"
        if self.total_generations:
            return f"Generation {self._phase_generation}/{self.total_generations}"
        return f"Generation {self._phase_generation}"

    def _print_phase_complete(self, now: float) -> None:
        if self._phase is None:
            return
        self._log(
            f"{self._phase_label()} complete. Best reward: {self._phase_best_score:.2f}",
            now=now,
            separator="└─",
        )

    def _start_phase(self, phase: str, generation: int | None, now: float) -> None:
        if self._phase == phase and self._phase_generation == generation:
            return
        if self._phase is not None:
            self._print_phase_complete(now)
        self._phase = phase
        self._phase_generation = generation
        self._phase_start_time = now
        self._phase_candidates = 0
        self._phase_best_score = 0.0
        self._current_generation = generation
        if phase == "initial":
            self._log("Initial population started", now=now, separator="┌─", update_idle_timer=False)
        else:
            self._log(f"{self._phase_label()} started", now=now, separator="┌─", update_idle_timer=False)

    def _format_best_prompt(self, prompt: Any) -> str:
        if prompt is None:
            return ""
        if isinstance(prompt, str):
            # Handle JSON strings that might be wrapped
            text = prompt.strip()
            # Strip "json" prefix if present (e.g., "json { ... }")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
            if text.startswith("{") and '"instruction"' in text:
                try:
                    parsed = json.loads(text)
                    return str(parsed.get("instruction", text)).strip()
                except Exception:
                    pass
            return text
        if isinstance(prompt, dict):
            # Direct instruction field (most common)
            instruction = prompt.get("instruction")
            if instruction:
                return str(instruction).strip()
            # Nested data.instruction
            data = prompt.get("data") if isinstance(prompt.get("data"), dict) else prompt
            if isinstance(data, dict) and data.get("instruction"):
                return str(data.get("instruction")).strip()
            # Messages format
            messages = data.get("messages") if isinstance(data, dict) else None
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "system":
                        text = msg.get("pattern") or msg.get("content")
                        if text:
                            return str(text).strip()
        try:
            return json.dumps(prompt, default=str)
        except Exception:
            return str(prompt)

    def _format_best_candidate(self, candidate: Any) -> str:
        if candidate is None:
            return ""
        if isinstance(candidate, dict):
            candidate_id = (
                candidate.get("candidate_id")
                or candidate.get("version_id")
                or candidate.get("key")
                or candidate.get("id")
            )
            reward = candidate.get("reward")
            score_obj = candidate.get("score") if isinstance(candidate.get("score"), dict) else {}
            if reward is None and score_obj:
                reward = score_obj.get("reward")
            prompt_text = candidate.get("prompt_text") or candidate.get("best_prompt_text")
            parts = []
            if candidate_id:
                parts.append(f"id={candidate_id}")
            if reward is not None:
                parts.append(
                    f"reward={reward:.3f}"
                    if isinstance(reward, (int, float))
                    else f"reward={reward}"
                )
            if prompt_text:
                parts.append(f"prompt={str(prompt_text).strip()}")
            return " | ".join(parts) if parts else json.dumps(candidate, default=str)
        return str(candidate)

    def _extract_instruction_text(self, event_data: dict[str, Any]) -> str | None:
        instruction = (
            event_data.get("instruction")
            or event_data.get("best_prompt")
            or event_data.get("prompt_text")
        )
        if instruction is None:
            return None
        text = self._format_best_prompt(instruction)
        if not text:
            return None
        return " ".join(str(text).split())

    def _maybe_print_best(self, now: float, event_data: dict[str, Any]) -> None:
        # Track best prompt internally without printing (shown via candidate lines)
        best_prompt = event_data.get("best_prompt")
        if best_prompt is not None:
            sig = json.dumps(best_prompt, default=str, sort_keys=True)
            if sig != self._last_best_prompt_sig:
                self._last_best_prompt_sig = sig
                text = self._format_best_prompt(best_prompt)
                if text:
                    self.best_prompt = text
        best_score = _coerce_float(event_data.get("best_score"))
        if best_score is not None:
            self.best_score = max(self.best_score or 0.0, best_score)

        # Track best candidate internally without printing (shown via candidate lines)
        best_candidate = event_data.get("best_candidate")
        if best_candidate is not None:
            sig = json.dumps(best_candidate, default=str, sort_keys=True)
            if sig != self._last_best_candidate_sig:
                self._last_best_candidate_sig = sig

    def handle(self, message) -> None:
        if not self.should_handle(message):
            return
        now = self._printer.now()

        data = message.data or {}
        if message.stream_type == StreamType.STATUS:
            status = str(data.get("status") or data.get("state") or "").lower()
            if status:
                self._status_ticker.maybe_log(
                    status=status,
                    now=now,
                    last_output_time=self._last_output_time,
                    min_idle_seconds=15.0,
                    log_fn=lambda msg, ts: self._log(msg, now=ts),
                )
            return
        event_type = str(data.get("type", ""))

        if "rollout.concurrency" in event_type and not self._seen_rollout_concurrency:
            self._seen_rollout_concurrency = True
            if self._phase is None:
                self._start_phase("initial", None, now)
            return

        if any(skip in event_type for skip in self.SKIP_EVENT_SUBSTRINGS):
            return

        event_data = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
        run_id = data.get("run_id")
        if self.debug:
            try:
                debug_payload = {
                    "stream_type": str(getattr(message, "stream_type", "")),
                    "job_id": getattr(message, "job_id", None),
                    "type": event_type,
                    "run_id": run_id,
                    "data": data,
                }
                self._log(
                    f"[DEBUG] GEPA message",
                    now=now,
                    update_idle_timer=False,
                )
                self._log_continuation(json.dumps(debug_payload, default=str, sort_keys=True))
            except Exception as exc:
                self._log(
                    f"[DEBUG] GEPA message (failed to serialize): {type(exc).__name__}: {exc}",
                    now=now,
                    update_idle_timer=False,
                )
        if self._dedupe_event(event_type, event_data, run_id):
            return
        self._maybe_print_best(now, event_data)

        if "generation.started" in event_type or "generation.start" in event_type:
            if run_id is None:
                return
            generation = self._display_generation(event_data.get("generation"))
            if generation is not None:
                self._past_initial_phase = True
                self._start_phase("gen", generation, now)
            return

        if "generation.completed" in event_type or "generation.complete" in event_type:
            if run_id is None:
                return
            generation = self._display_generation(event_data.get("generation"))
            if generation is not None:
                self._past_initial_phase = True
                self._start_phase("gen", generation, now)
                candidates_eval = _coerce_int(event_data.get("candidates_evaluated"))
                if candidates_eval is not None:
                    self._phase_candidates = candidates_eval
                self._print_phase_complete(now)
                self._phase = None
                self._phase_generation = None
                self._phase_start_time = None
                self._last_generation_completed = generation
                if self.total_generations and generation >= self.total_generations:
                    self._final_generation_complete_seen = True
                    self._final_generation_complete_seen_at = time.time()
            return

        if "candidate.evaluated" in event_type or "candidate_scored" in event_type or "proposal.scored" in event_type:
            if run_id is None:
                return
            generation = self._display_generation(event_data.get("generation"))
            mutation_type = event_data.get("mutation_type")
            if generation is None:
                if self._phase is None:
                    self._start_phase("initial", None, now)
                # If the backend doesn't attach generation info, infer when we've left initial population.
                if (
                    self._phase == "initial"
                    and self.initial_population_size
                    and self._phase_candidates >= self.initial_population_size
                    and self.children_per_generation
                ) or (self._phase == "initial" and mutation_type not in (None, "initial")):
                    self._past_initial_phase = True
                    next_gen = (self._last_generation_completed or 0) + 1
                    self._start_phase("gen", next_gen, now)
            else:
                self._past_initial_phase = True
                self._start_phase("gen", generation, now)

            if self._phase is None:
                self._start_phase("initial", None, now)

            self._phase_candidates += 1
            score = _extract_reward(event_data) or 0.0
            best_score = _coerce_float(event_data.get("best_score") or event_data.get("best_reward"))
            prev_best = self._overall_best
            if score > self._phase_best_score:
                self._phase_best_score = score
            is_new_best = False
            if best_score is not None:
                is_new_best = best_score > prev_best
                self._overall_best = max(self._overall_best, best_score)
            else:
                is_new_best = score > prev_best
                self._overall_best = max(self._overall_best, score)
            self.best_score = max(self.best_score or 0.0, self._overall_best)
            candidate_text = ""
            if is_new_best:
                candidate_text = self._extract_instruction_text(event_data) or ""
                if candidate_text:
                    self.best_prompt = candidate_text
                if best_score is not None:
                    self.best_score = max(self.best_score or 0.0, best_score)
                else:
                    self.best_score = max(self.best_score or 0.0, score)
            # Determine expected candidate count for this phase
            if self._phase == "initial":
                phase_total = self.initial_population_size
            else:
                phase_total = self.children_per_generation
            if phase_total:
                candidate_label = f"Candidate {self._phase_candidates}/{phase_total}"
            else:
                candidate_label = f"Candidate {self._phase_candidates}"
            self._log(f"{candidate_label}: mean_reward={score:.2f}", now=now)
            # Show instruction for new bests (candidate_text already extracted above)
            if is_new_best and candidate_text:
                self._log_continuation("New best prompt:")
                self._log_continuation(candidate_text)
            return

        if "job.completed" in event_type or event_type.endswith("job.completed"):
            if self._phase is not None:
                self._print_phase_complete(now)
            self._log(f"Job completed | overall best={self._overall_best:.2f}", now=now)
            return

        if "job.failed" in event_type or event_type.endswith("job.failed"):
            self._log(f"Job failed: {event_data}", now=now)
            return

    def flush(self) -> None:  # pragma: no cover - no buffered output
        return None

    def wants_event_backfill(self) -> bool:
        return self._final_generation_complete_seen

    def terminal_hint_ready(self, *, grace_seconds: float = 8.0) -> bool:
        if not self._final_generation_complete_seen_at:
            return False
        return (time.time() - self._final_generation_complete_seen_at) >= grace_seconds

    def status_tick(self, status: str) -> None:
        now = self._printer.now()
        self._log(f"Status: {status}", now=now)
        self._last_status_time = now

    def status_tick_if_idle(self, status: str, *, min_idle_seconds: float) -> None:
        now = self._printer.now()
        self._status_ticker.maybe_log(
            status=status,
            now=now,
            last_output_time=self._last_output_time,
            min_idle_seconds=min_idle_seconds,
            log_fn=lambda msg, ts: self._log(msg, now=ts),
        )


class EvalStreamProgressHandler(StreamHandler):
    """Eval progress handler with the same time formatting as GEPA."""

    def __init__(
        self,
        label: str,
        total_seeds: int,
        *,
        job_id: str | None = None,
        clock: ProgressClock | None = None,
        log_every: int = 10,
        debug: bool = False,
    ) -> None:
        self.label = label
        self.total_seeds = total_seeds
        self.job_id = job_id  # Filter events to this job only
        self.log_every = max(1, int(log_every))
        self.debug = debug
        self._printer = ProgressPrinter(label=label, clock=clock)
        self._completed_seeds = 0
        self._total_reward = 0.0
        self._rewards: list[float] = []
        self._started = False
        self._completed = False
        self._seen_seeds: set[int] = set()
        self._last_output_time: float | None = None
        self._status_ticker = IdleStatusTicker()

    def _log(self, message: str, *, now: float | None = None) -> None:
        timestamp = self._printer.now() if now is None else now
        self._printer.log(message, now=timestamp)
        self._last_output_time = timestamp

    def should_handle(self, message) -> bool:
        # Filter to only handle events from the target job
        if self.job_id is not None and hasattr(message, "job_id"):
            return message.job_id == self.job_id
        return True

    def _record_seed(self, reward: float | None, seed: int | None, now: float) -> None:
        if seed is not None:
            if seed in self._seen_seeds:
                return
            self._seen_seeds.add(seed)
        if reward is None:
            return
        self._completed_seeds += 1
        self._rewards.append(reward)
        self._total_reward += reward
        if (
            self._completed_seeds % self.log_every == 0
            or self._completed_seeds == self.total_seeds
        ):
            mean = self._total_reward / self._completed_seeds if self._completed_seeds else 0.0
            self._log(
                f"Progress: {self._completed_seeds}/{self.total_seeds} | mean_reward={mean:.3f}",
                now=now,
            )

    def record_rollout(self, seed: int, reward: float) -> None:
        now = self._printer.now()
        if not self._started:
            self._started = True
            self._log(f"Eval started: {self.total_seeds} seeds", now=now)
        self._record_seed(reward, seed, now)

    def finish(self) -> None:
        if self._completed:
            return
        self._completed = True
        now = self._printer.now()
        mean = self._total_reward / self._completed_seeds if self._completed_seeds else 0.0
        self._log(f"Eval completed: mean_reward={mean:.3f}", now=now)

    def handle(self, message) -> None:
        if not self.should_handle(message):
            return
        if self.debug:
            now = self._printer.now()
            try:
                debug_payload = {
                    "stream_type": str(getattr(message, "stream_type", "")),
                    "job_id": getattr(message, "job_id", None),
                    "type": str(getattr(message, "data", {}).get("type", "")),
                    "run_id": getattr(message, "data", {}).get("run_id"),
                    "data": getattr(message, "data", None),
                }
                self._log(
                    "[DEBUG] Eval message",
                    now=now,
                )
                self._log_continuation(json.dumps(debug_payload, default=str, sort_keys=True))
            except Exception as exc:
                self._log(
                    f"[DEBUG] Eval message (failed to serialize): {type(exc).__name__}: {exc}",
                    now=now,
                )
        if message.stream_type == StreamType.STATUS:
            status = str(message.data.get("status") or message.data.get("state") or "").lower()
            if status:
                now = self._printer.now()
                self._status_ticker.maybe_log(
                    status=status,
                    now=now,
                    last_output_time=self._last_output_time,
                    min_idle_seconds=15.0,
                    log_fn=lambda msg, ts: self._log(msg, now=ts),
                )
            return
        now = self._printer.now()
        data = message.data or {}
        event_type = str(data.get("type", ""))
        event_data = data.get("data", {}) if isinstance(data.get("data"), dict) else {}

        if event_type == "eval.policy.job.started":
            if not self._started:
                self._started = True
                seed_count = event_data.get("seed_count", self.total_seeds)
                self._log(f"Eval started: {seed_count} seeds", now=now)
            return

        if event_type == "eval.policy.seed.completed":
            seed = event_data.get("seed")
            reward = _coerce_float(event_data.get("reward"))
            self._record_seed(reward, seed, now)
            return

        if event_type == "eval.policy.job.completed":
            mean_reward = _coerce_float(event_data.get("mean_reward"))
            if mean_reward is not None:
                self._log(f"Eval completed: mean_reward={mean_reward:.3f}", now=now)
            self._completed = True
            return

        if event_type == "eval.policy.job.failed":
            error = event_data.get("error", "unknown error")
            self._log(f"Eval failed: {error}", now=now)
            self._completed = True
            return


class EvalStatusPrinter:
    """Compact, de-duplicated polling output for eval jobs."""

    def __init__(
        self,
        *,
        label: str,
        total_seeds: int | None = None,
        debug: bool = False,
    ) -> None:
        self._label = label
        self._total_seeds = total_seeds
        self._debug = debug
        self._printer = ProgressPrinter(label=label)
        self._started = False
        self._last_completed: int | None = None
        self._last_total: int | None = None
        self._last_output_time: float | None = None
        self._status_ticker = IdleStatusTicker()

    def _log(self, message: str) -> None:
        now = self._printer.now()
        self._printer.log(message, now=now)
        self._last_output_time = now

    def _extract_progress(self, status_data: dict[str, Any]) -> tuple[int, int, float | None]:
        results = status_data.get("results", {})
        completed = results.get("completed", 0) if isinstance(results, dict) else 0
        total = (
            results.get("total", self._total_seeds)
            if isinstance(results, dict)
            else self._total_seeds
        )
        if isinstance(results, dict):
            seed_results = results.get("seed_results")
            if isinstance(seed_results, list):
                completed = len(seed_results)
        total = total if total is not None else 0
        mean_reward = None
        if isinstance(results, dict):
            mean_reward = results.get("mean_reward")
            if mean_reward is None:
                summary = results.get("summary")
                if isinstance(summary, dict):
                    mean_reward = summary.get("mean_reward")
        return int(completed), int(total), mean_reward

    def log_start(self, *, total: int | None = None) -> None:
        if self._started:
            return
        total = total if total is not None else self._total_seeds
        total = total if total is not None else 0
        self._log(f"Eval started: {total} seeds")
        self._started = True

    def log_debug_config(self, message: str) -> None:
        if not self._debug:
            return
        self._log(f"[DEBUG] {message}")

    def handle_status(self, status_data: dict[str, Any]) -> None:
        if self._debug:
            try:
                self._log(f"[DEBUG] Eval status: {json.dumps(status_data, default=str, sort_keys=True)}")
            except Exception as exc:
                self._log(f"[DEBUG] Eval status (failed to serialize): {type(exc).__name__}: {exc}")
        status = str(status_data.get("status", "pending")).lower()
        completed, total, _ = self._extract_progress(status_data)
        if not self._started and status in {"pending", "running"}:
            self.log_start(total=total)
        self._last_completed = completed
        self._last_total = total
        if status in {"pending", "running"}:
            now = time.time()
            self._status_ticker.maybe_log(
                status=status,
                now=now,
                last_output_time=self._last_output_time,
                min_idle_seconds=15.0,
                log_fn=lambda msg, _ts: self._log(msg),
            )

    def log_terminal(
        self,
        *,
        status: str,
        mean_reward: float | None = None,
    ) -> None:
        if status in {"completed", "succeeded", "success"}:
            reward_str = f"{mean_reward:.3f}" if mean_reward is not None else "--"
            self._log(f"Eval completed: mean_reward={reward_str}")
        else:
            self._log(f"Eval {status}")

    def tick(self, *, min_idle_seconds: float = 15.0) -> None:
        """Print status if idle for min_idle_seconds."""
        now = time.time()
        self._status_ticker.maybe_log(
            status="running",
            now=now,
            last_output_time=self._last_output_time,
            min_idle_seconds=min_idle_seconds,
            log_fn=lambda msg, _ts: self._log(msg),
        )
