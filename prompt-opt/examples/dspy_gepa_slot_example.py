"""Example showing GEPA slot-in via prompt-opt."""

from __future__ import annotations

from prompt_opt.adapters.synth_offline import LocalEvaluator, SynthOfflineLearningAdapter
from prompt_opt.dspy import gepa


def _score_fn(example, candidate) -> float:
    expected = str(example.get("answer", "")).strip().lower()
    prompt = " ".join(candidate.values()).lower()
    return 1.0 if expected and expected in prompt else 0.0


def main() -> None:
    adapter = SynthOfflineLearningAdapter(LocalEvaluator(score_fn=_score_fn))
    result = gepa.optimize(
        seed_candidate={"system_prompt": "Answer briefly."},
        trainset=[{"input": "Capital of France?", "answer": "paris"}],
        adapter=adapter,
        max_metric_calls=6,
    )
    print(result.best_candidate)


if __name__ == "__main__":
    main()
