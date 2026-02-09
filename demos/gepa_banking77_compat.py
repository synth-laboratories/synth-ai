"""Run a GEPA compatibility workflow on Banking77."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

import os

from synth_ai import gepa


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required.")
    return value


def _build_system_prompt(labels: list[str]) -> str:
    label_block = "\n".join(f"- {label}" for label in labels)
    return (
        "You are a banking support classifier. "
        "Pick the single best intent label from the list below and return only that label.\n\n"
        f"Intents:\n{label_block}\n\n"
        "Return exactly one label."
    )


def main() -> None:
    _require_env("SYNTH_API_KEY")

    trainset, valset, _ = gepa.examples.banking77.init_dataset()
    labels = gepa.examples.banking77.get_labels()
    system_prompt = _build_system_prompt(labels)

    task_lm = os.environ.get("SYNTH_TASK_LM", "openai/gpt-4.1-mini")
    reflection_lm = os.environ.get("SYNTH_REFLECTION_LM", "openai/gpt-5")
    max_metric_calls = int(os.environ.get("SYNTH_MAX_METRIC_CALLS", "120"))

    result = gepa.optimize(
        seed_candidate={"system_prompt": system_prompt},
        trainset=trainset,
        valset=valset,
        task_lm=task_lm,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
    )

    best_prompt = result.best_candidate["system_prompt"]
    best_score = result.val_aggregate_scores[result.best_idx]
    print("Best validation score:", best_score)
    print("Best prompt:\n", best_prompt)


if __name__ == "__main__":
    main()
