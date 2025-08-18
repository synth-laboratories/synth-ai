"""
Example: MIPROv2-style optimizer on Banking77 using Groq gpt-oss-20b.

Requires:
- .env with GROQ_API_KEY
- datasets

Run:
- uv run -q python -m synth_ai.learning.prompts.run_mipro_banking77
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from synth_ai.learning.prompts.mipro import ProgramAdapter, evaluate_program, mipro_v2_compile
from synth_ai.lm.core.main_v3 import LM, build_messages


def choose_label(pred: str, label_names: list[str]) -> str:
    norm = (pred or "").strip().lower()
    d = {ln.lower(): ln for ln in label_names}
    if norm in d:
        return d[norm]

    def score(cand: str) -> int:
        c = cand.lower()
        return sum(1 for w in c.split() if w in norm)

    return max(label_names, key=score)


def accuracy(pred: str, gold: str, labels: list[str]) -> float:
    return 1.0 if choose_label(pred, labels) == gold else 0.0


class NaivePromptModel:
    """Toy prompt model that returns simple instruction variants."""

    def generate_instructions(self, ctx: dict[str, Any], k: int = 8) -> list[str]:
        base = "Classify the Banking77 intent and return exactly one label."
        variants = [
            base,
            base + " Be concise.",
            base + " Use examples to guide your reasoning.",
            base + " Return only the label text.",
            base + " Follow the label names strictly.",
            base + " Do not include explanations.",
            base + " Think about similar intents before answering.",
            base + " Carefully consider the user's message.",
        ]
        random.shuffle(variants)
        return variants[:k]


def build_run_fn(lm: LM, label_names: list[str]):
    def run_fn(x: str, _model: Any | None = None) -> str:
        # Use instructions and demos from adapter state (set by set_instructions/set_demos)
        # The adapter passes state via closure; we rebuild messages here
        instructions = state_ref.get("instructions", {}).get(
            "main", "You are an intent classifier for Banking77."
        )
        examples = "\n".join(f"Input: {a}\nLabel: {b}" for a, b in state_ref.get("demos", []))
        sys = instructions
        user = (f"Examples:\n{examples}\n\n" if examples else "") + f"Message: {x}\nLabel:"
        messages = build_messages(sys, user, images_bytes=None, model_name=lm.model)

        async def _call():
            resp = await lm.respond_async(messages=messages)
            return (resp.raw_response or "").strip()

        return asyncio.run(_call())

    return run_fn


def set_instructions(new_instr: dict[str, str], state: dict[str, Any]) -> dict[str, Any]:
    state["instructions"] = {**state.get("instructions", {}), **new_instr}
    return state


def set_demos(demos: list[tuple[str, str]], state: dict[str, Any]) -> dict[str, Any]:
    state["demos"] = list(demos)
    return state


def main():
    load_dotenv()
    random.seed(0)

    model = os.getenv("MODEL", "openai/gpt-oss-20b")
    vendor = os.getenv("VENDOR", "groq")
    lm = LM(model=model, vendor=vendor, temperature=0.0)

    print("Loading Banking77 dataset (train/dev split of test for demo)...")
    ds = load_dataset("banking77")
    label_names: list[str] = ds["test"].features["label"].names  # type: ignore

    all_items = [(r["text"], label_names[int(r["label"])]) for r in ds["test"]]
    random.shuffle(all_items)
    trainset: Sequence[tuple[str, str]] = all_items[:80]
    valset: Sequence[tuple[str, str]] = all_items[80:160]

    global state_ref
    state_ref = {
        "instructions": {"main": "You are an intent classifier for Banking77."},
        "demos": [],
    }
    adapter = ProgramAdapter(
        run_fn=build_run_fn(lm, label_names),
        state=state_ref,
        _predictors=["main"],
        set_instructions=set_instructions,
        set_demos=set_demos,
    )

    def metric(yhat: str, y: str) -> float:
        return accuracy(yhat, y, label_names)

    prompt_model = NaivePromptModel()
    task_model = None  # not used in this minimal example

    print("Running MIPROv2-style optimizer...")
    best, records = mipro_v2_compile(
        student=adapter,
        trainset=trainset,
        valset=valset,
        metric=metric,
        prompt_model=prompt_model,
        task_model=task_model,
        max_bootstrapped_demos=6,
        max_labeled_demos=4,
        num_candidates=6,
        num_trials=12,
        minibatch=True,
        minibatch_size=16,
        minibatch_full_eval_steps=3,
        seed=0,
    )

    res = evaluate_program(best, valset, metric)
    print(
        f"Best program accuracy on val: {res.score:.2%} ({sum(res.subscores)}/{len(res.subscores)})"
    )

    out = {
        "context": {
            "model": model,
            "vendor": vendor,
            "train_size": len(trainset),
            "val_size": len(valset),
        },
        "trials": records,
    }
    out_dir = Path(__file__).parent
    fname = str(out_dir / f"mipro_banking77_{int(time.time())}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved trial records to {fname}")


if __name__ == "__main__":
    main()
