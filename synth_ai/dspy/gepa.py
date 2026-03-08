"""DSPy-compatible GEPA wrapper backed by Synth GEPA."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

import inspect
import math
import os
import warnings
from collections.abc import Callable
from typing import Any, Literal

from synth_ai import gepa as synth_gepa

from ._compat import (
    apply_system_prompt_to_student,
    attach_detailed_results,
    clone_student,
    coerce_model_identifier,
    extract_seed_candidate_from_student,
    infer_task_lm,
    materialize_dataset,
)

_AUTO_RUN_SETTINGS: dict[str, dict[str, int]] = {
    "light": {"n": 6},
    "medium": {"n": 12},
    "heavy": {"n": 18},
}

try:
    _SYNTH_GEPA_OPTIMIZE_PARAMS = frozenset(
        inspect.signature(synth_gepa.optimize).parameters.keys()
    )
except Exception:  # pragma: no cover - defensive for unusual runtime environments
    _SYNTH_GEPA_OPTIMIZE_PARAMS = None


def _count_predictors(student: Any) -> int:
    named_predictors = getattr(student, "named_predictors", None)
    if callable(named_predictors):
        try:
            return max(1, len(list(named_predictors())))
        except Exception:
            pass
    predictors = getattr(student, "predictors", None)
    if callable(predictors):
        try:
            return max(1, len(list(predictors())))
        except Exception:
            pass
    return 1


def _extract_best_prompt(result: Any) -> str | None:
    candidate = getattr(result, "best_candidate", None)
    if isinstance(candidate, dict):
        for key in ("system_prompt", "instruction", "prompt"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class GEPA:
    """Drop-in replacement for `dspy.GEPA` using Synth's GEPA compatibility API."""

    def __init__(
        self,
        metric: Callable[..., Any],
        *,
        auto: Literal["light", "medium", "heavy"] | None = None,
        max_full_evals: int | None = None,
        max_metric_calls: int | None = None,
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
        reflection_lm: Any | None = None,
        skip_perfect_score: bool = True,
        add_format_failure_as_feedback: bool = False,
        instruction_proposer: Any | None = None,
        component_selector: Any = "round_robin",
        use_merge: bool = True,
        max_merge_invocations: int | None = 5,
        num_threads: int | None = None,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        log_dir: str | None = None,
        track_stats: bool = False,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        track_best_outputs: bool = False,
        warn_on_score_mismatch: bool = True,
        use_mlflow: bool = False,
        seed: int | None = 0,
        gepa_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        budget_fields = (
            max_metric_calls is not None,
            max_full_evals is not None,
            auto is not None,
        )
        if sum(budget_fields) != 1:
            raise ValueError(
                "Exactly one of auto, max_full_evals, or max_metric_calls must be set."
            )
        if reflection_lm is None and instruction_proposer is None:
            raise ValueError("GEPA requires reflection_lm or instruction_proposer to be provided.")

        init_kwargs = dict(kwargs)
        if "module_selector" in init_kwargs and component_selector == "round_robin":
            component_selector = init_kwargs.pop("module_selector")
        if "run_dir" in init_kwargs and log_dir is None:
            log_dir = str(init_kwargs.pop("run_dir"))

        task_lm_override = init_kwargs.pop("task_lm", None)
        if task_lm_override is None:
            task_lm_override = init_kwargs.pop("task_model", None)

        passthrough_gepa_kwargs = dict(gepa_kwargs or {})
        if task_lm_override is not None:
            passthrough_gepa_kwargs["task_lm"] = task_lm_override

        for key in ("display_progress_bar", "logger", "callbacks", "raise_on_exception"):
            if key in init_kwargs:
                passthrough_gepa_kwargs[key] = init_kwargs.pop(key)

        if init_kwargs:
            warnings.warn(
                "Ignoring unsupported GEPA init argument(s): " + ", ".join(sorted(init_kwargs)),
                RuntimeWarning,
                stacklevel=2,
            )

        self.metric_fn = metric
        self.auto = auto
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        self.reflection_lm = reflection_lm
        self.skip_perfect_score = skip_perfect_score
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.instruction_proposer = instruction_proposer
        self.component_selector = component_selector
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.num_threads = num_threads
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.log_dir = log_dir
        self.track_stats = track_stats
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs
        self.track_best_outputs = track_best_outputs
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.use_mlflow = use_mlflow
        self.seed = int(seed or 0)
        self.gepa_kwargs = passthrough_gepa_kwargs
        if self.track_best_outputs and not self.track_stats:
            raise ValueError("track_best_outputs requires track_stats=True.")

    def auto_budget(
        self,
        num_preds: int,
        num_candidates: int,
        valset_size: int,
        minibatch_size: int = 35,
        full_eval_steps: int = 5,
    ) -> int:
        """Estimate GEPA metric-call budget using DSPy-style heuristics."""
        if num_preds < 1 or num_candidates < 1 or valset_size < 1:
            raise ValueError("num_preds, num_candidates, and valset_size must be >= 1.")
        if minibatch_size < 1 or full_eval_steps < 1:
            raise ValueError("minibatch_size and full_eval_steps must be >= 1.")

        num_trials = int(
            max(2 * (num_preds * 2) * math.log2(max(num_candidates, 2)), 1.5 * num_candidates)
        )
        total = valset_size + (num_candidates * 5) + (num_trials * minibatch_size)
        periodic_fulls = (num_trials + 1) // full_eval_steps + 1
        extra_final = 1 if num_trials < full_eval_steps else 0
        total += (periodic_fulls + extra_final) * valset_size
        return max(1, total)

    def _resolve_max_metric_calls(self, student: Any, train_size: int, val_size: int) -> int:
        if self.max_metric_calls is not None:
            return max(1, int(self.max_metric_calls))
        if self.max_full_evals is not None:
            return max(1, int(self.max_full_evals) * max(1, train_size + val_size))
        assert self.auto is not None
        auto_n = _AUTO_RUN_SETTINGS[self.auto]["n"]
        return self.auto_budget(
            num_preds=_count_predictors(student),
            num_candidates=auto_n,
            valset_size=max(1, val_size),
        )

    def compile(
        self,
        student: Any,
        *,
        trainset: list[Any],
        teacher: Any | None = None,
        valset: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run Synth GEPA through a DSPy-compatible `compile(...)` interface."""
        if teacher is not None:
            raise ValueError("Teacher is not supported in Synth DSPy GEPA compatibility mode.")
        if self.instruction_proposer is not None:
            warnings.warn(
                "instruction_proposer is not supported in Synth GEPA compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.add_format_failure_as_feedback:
            warnings.warn(
                "add_format_failure_as_feedback is not supported in Synth GEPA compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.num_threads is not None:
            warnings.warn(
                "num_threads is not supported in Synth GEPA compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.failure_score != 0.0:
            warnings.warn(
                "failure_score is not supported in Synth GEPA compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not self.warn_on_score_mismatch:
            warnings.warn(
                "warn_on_score_mismatch is not supported in Synth GEPA compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.metric_fn is not None:
            warnings.warn(
                "DSPy metric callbacks are not executed in Synth GEPA compatibility mode; "
                "dataset answer matching is used.",
                RuntimeWarning,
                stacklevel=2,
            )

        train_records = materialize_dataset(trainset)
        val_records = materialize_dataset(valset) if valset is not None else None
        seed_candidate, target_predictor_name = extract_seed_candidate_from_student(student)

        reflection_lm = coerce_model_identifier(self.reflection_lm)
        if reflection_lm is None:
            reflection_lm = os.getenv("SYNTH_DSPY_REFLECTION_LM", "").strip() or None

        extra_kwargs = dict(self.gepa_kwargs)
        extra_kwargs.update(kwargs)
        task_lm_override = extra_kwargs.pop("task_lm", None)
        if task_lm_override is None:
            task_lm_override = extra_kwargs.pop("task_model", None)
        task_lm = infer_task_lm(task_model=task_lm_override, student=student)
        max_calls = self._resolve_max_metric_calls(
            student,
            train_size=len(train_records),
            val_size=len(val_records) if val_records is not None else len(train_records),
        )

        component_selector: str = "round_robin"
        if isinstance(self.component_selector, str):
            component_selector = self.component_selector
        else:
            warnings.warn(
                "Custom component_selector objects are not supported; using 'round_robin'.",
                RuntimeWarning,
                stacklevel=2,
            )

        resolved_use_merge = False
        if self.use_merge:
            warnings.warn(
                "use_merge=True is not supported in Synth GEPA compatibility mode; "
                "running with use_merge=False.",
                RuntimeWarning,
                stacklevel=2,
            )

        optimize_kwargs: dict[str, Any] = {
            "seed_candidate": seed_candidate,
            "trainset": train_records,
            "valset": val_records,
            "task_lm": task_lm,
            "reflection_lm": reflection_lm,
            "candidate_selection_strategy": self.candidate_selection_strategy,
            "skip_perfect_score": self.skip_perfect_score,
            "reflection_minibatch_size": self.reflection_minibatch_size,
            "module_selector": component_selector,
            "use_merge": resolved_use_merge,
            "max_merge_invocations": self.max_merge_invocations or 5,
            "max_metric_calls": max_calls,
            "logger": None,
            "run_dir": self.log_dir,
            "callbacks": None,
            "use_wandb": self.use_wandb,
            "wandb_api_key": self.wandb_api_key,
            "wandb_init_kwargs": self.wandb_init_kwargs,
            "use_mlflow": self.use_mlflow,
            "track_best_outputs": self.track_best_outputs,
            "display_progress_bar": True,
            "seed": self.seed,
            "raise_on_exception": True,
        }
        if _SYNTH_GEPA_OPTIMIZE_PARAMS is not None:
            unsupported = sorted(k for k in extra_kwargs if k not in _SYNTH_GEPA_OPTIMIZE_PARAMS)
            for key in unsupported:
                extra_kwargs.pop(key, None)
            if unsupported:
                warnings.warn(
                    "Ignoring unsupported GEPA compile argument(s): " + ", ".join(unsupported),
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif extra_kwargs:
            warnings.warn(
                "GEPA passthrough argument filtering unavailable; ignoring extra GEPA compile args.",
                RuntimeWarning,
                stacklevel=2,
            )
            extra_kwargs = {}
        optimize_kwargs.update(extra_kwargs)

        result = synth_gepa.optimize(**optimize_kwargs)

        optimized_program = clone_student(student)
        best_prompt = _extract_best_prompt(result)
        if best_prompt:
            apply_system_prompt_to_student(
                optimized_program,
                best_prompt,
                target_predictor_name=target_predictor_name,
            )
        if self.track_stats:
            attach_detailed_results(optimized_program, result)
        return optimized_program
