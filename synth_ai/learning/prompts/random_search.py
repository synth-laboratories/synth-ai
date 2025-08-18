"""
Random-search prompt optimizer (BootstrapFewShotWithRandomSearch), DSPy-inspired.

Implements the high-level pseudocode of DSPy's Random Search optimizer in a
provider-agnostic, modular style. You can plug in your own student/program and
metric, and this module will explore baselines and bootstrapped few-shot variants.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# ---------------------------
# Protocol-like expectations (duck-typed)
# ---------------------------


class _ProgramLike:
    def reset_copy(self):  # zero-shot copy
        return self

    def deepcopy(self):  # deep copy
        return self

    def with_demos(self, demos: list[tuple[Any, Any]]):
        return self

    def run(self, x: Any) -> Any:
        raise NotImplementedError


# ---------------------------
# Helpers and lightweight components
# ---------------------------


@dataclass
class EvalResult:
    score: float
    subscores: list[float]


def evaluate(
    program: _ProgramLike, dataset: Sequence[tuple[Any, Any]], metric: Callable[[Any, Any], float]
) -> EvalResult:
    subs = []
    for x, y in dataset:
        subs.append(metric(program.run(x), y))
    return EvalResult(sum(subs) / max(1, len(subs)), subs)


class LabeledFewShot:
    def __init__(self, k: int):
        self.k = k

    def compile(
        self, student: _ProgramLike, trainset: Sequence[tuple[Any, Any]], sample: bool = True
    ) -> _ProgramLike:
        p = getattr(student, "deepcopy", student.reset_copy)()
        demos = list(trainset)
        if sample:
            random.shuffle(demos)
        p = p.with_demos(demos[: min(self.k, len(demos))])
        return p


class BootstrapFewShot:
    def __init__(
        self,
        *,
        metric: Callable[[Any, Any], float],
        metric_threshold: float | None = None,
        max_bootstrapped_demos: int = 8,
        max_labeled_demos: int = 0,
        teacher_settings: dict[str, Any] | None = None,
        max_rounds: int = 1,
    ):
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds

    def compile(
        self,
        student: _ProgramLike,
        teacher: _ProgramLike | None,
        trainset: Sequence[tuple[Any, Any]],
    ) -> _ProgramLike:
        p = getattr(student, "deepcopy", student.reset_copy)()
        rng = random.Random()
        # If bootstrapped demos disabled, return labeled-only few-shot quickly
        if self.max_bootstrapped_demos <= 0:
            demos: list[tuple[Any, Any]] = []
            if self.max_labeled_demos > 0:
                demos += rng.sample(list(trainset), k=min(self.max_labeled_demos, len(trainset)))
            return p.with_demos(demos)
        boot: list[tuple[Any, Any]] = []
        # Bootstrap demos by self consistency
        for _ in range(self.max_rounds):
            rng.shuffle(trainset := list(trainset))
            for x, y in trainset:
                yhat = p.run(x)
                ok = self.metric(yhat, y)
                if (self.metric_threshold is None and ok == 1) or (
                    self.metric_threshold is not None and ok >= self.metric_threshold
                ):
                    boot.append((x, y))
                if len(boot) >= self.max_bootstrapped_demos:
                    break
            if len(boot) >= self.max_bootstrapped_demos:
                break

        # Optionally add labeled demos
        demos = list(boot)
        if self.max_labeled_demos > 0:
            demos += rng.sample(list(trainset), k=min(self.max_labeled_demos, len(trainset)))

        return p.with_demos(demos)


# ---------------------------
# Random-search compile (BootstrapFewShotWithRandomSearch)
# ---------------------------


@dataclass
class Candidate:
    score: float
    subscores: list[float]
    seed: int
    program: _ProgramLike


def random_search_compile(
    student: _ProgramLike,
    trainset: Sequence[tuple[Any, Any]],
    valset: Sequence[tuple[Any, Any]],
    metric: Callable[[Any, Any], float],
    *,
    max_bootstrapped_demos: int = 8,
    max_labeled_demos: int = 4,
    max_rounds: int = 2,
    num_candidate_programs: int = 16,
    stop_at_score: float | None = None,
    evaluate_fn: Callable[[_ProgramLike, Sequence[tuple[Any, Any]], Callable[[Any, Any], float]], EvalResult] | None = None,
    on_candidate_evaluated: Callable[[int, float, EvalResult, dict[str, Any]], None] | None = None,
) -> tuple[_ProgramLike, list[dict[str, Any]]]:
    best_program: _ProgramLike | None = None
    best_score = float("-inf")
    candidates: list[Candidate] = []
    records: list[dict[str, Any]] = []

    seeds = list(range(num_candidate_programs))
    seeds = [-3, -2, -1] + seeds  # zero-shot, labeled few-shot, bootstrapped few-shot

    rng = random.Random(0)
    for idx, seed in enumerate(seeds):
        train_copy = list(trainset)

        if seed == -3:
            program = getattr(student, "reset_copy", student.deepcopy)()

        elif seed == -2:
            program = LabeledFewShot(k=max_labeled_demos).compile(student, train_copy, sample=True)

        else:
            if seed >= 0:
                rng.shuffle(train_copy)
            if max_bootstrapped_demos <= 0:
                size = 0
            else:
                size = (
                    max_bootstrapped_demos if seed == -1 else rng.randint(1, max_bootstrapped_demos)
                )
            program = BootstrapFewShot(
                metric=metric,
                metric_threshold=None,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                teacher_settings={},
                max_rounds=max_rounds,
            ).compile(student, teacher=None, trainset=train_copy)

        res = (
            evaluate_fn(program, valset, metric)
            if evaluate_fn
            else evaluate(program, valset, metric)
        )
        cand = Candidate(score=res.score, subscores=res.subscores, seed=seed, program=program)
        candidates.append(cand)
        # Record an intervention summary for reproducibility
        intervention: dict[str, Any] = {"seed": seed}
        if hasattr(program, "demos"):
            try:
                intervention["demos"] = program.demos  # type: ignore
            except Exception:
                intervention["demos"] = None
        # Type of candidate
        if seed == -3:
            intervention["kind"] = "zero_shot"
            intervention["label"] = "zero-shot"
        elif seed == -2:
            intervention["kind"] = "labeled_few_shot"
            intervention["label"] = f"labeled-{max_labeled_demos}"
        else:
            intervention["kind"] = "bootstrapped_few_shot"
            intervention["label"] = f"boot-b{max_bootstrapped_demos}-l{max_labeled_demos}"
        record_obj = {
            "score": cand.score,
            "subscores": cand.subscores,
            "intervention": intervention,
        }
        records.append(record_obj)

        if res.score > best_score:
            best_score, best_program = res.score, program

        if stop_at_score is not None and best_score >= stop_at_score:
            break

        if on_candidate_evaluated is not None:
            with contextlib.suppress(Exception):
                on_candidate_evaluated(idx + 1, res.score, res, intervention)

    # Attach candidates for inspection
    if hasattr(best_program, "candidate_programs"):
        # If user object supports attribute assignment
        with contextlib.suppress(Exception):
            best_program.candidate_programs = sorted(
                candidates, key=lambda c: c.score, reverse=True
            )  # type: ignore[attr-defined]

    return (best_program or getattr(student, "deepcopy", student)(), records)


__all__ = [
    "random_search_compile",
    "LabeledFewShot",
    "BootstrapFewShot",
]
