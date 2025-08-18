"""
Banking77 in-context injection evals (async, not tests)

Samples a handful of Banking77 prompts and evaluates multiple override
contexts in parallel, printing simple accuracy for each.

Usage
- Keys in .env (GROQ_API_KEY, etc.)
- Run: uv run -q python -m synth_ai.learning.prompts.banking77_injection_eval
  Optional env:
    - N_SAMPLES=20 (default)
    - MODEL=openai/gpt-oss-20b (default)
    - VENDOR=groq (default)
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from synth_ai.lm.core.main_v3 import LM, build_messages
from synth_ai.lm.overrides import LMOverridesContext


async def classify_one(lm: LM, text: str, label_names: list[str]) -> str:
    labels_joined = ", ".join(label_names)
    system_message = (
        "You are an intent classifier for the Banking77 dataset. "
        "Given a customer message, respond with exactly one label from the list. "
        "Return only the label text with no extra words.\n\n"
        f"Valid labels: {labels_joined}"
    )
    user_message = f"Message: {text}\nLabel:"
    messages = build_messages(system_message, user_message, images_bytes=None, model_name=lm.model)
    resp = await lm.respond_async(messages=messages)
    return (resp.raw_response or "").strip()


def choose_label(pred: str, label_names: list[str]) -> str:
    norm_pred = pred.strip().lower()
    label_lookup = {ln.lower(): ln for ln in label_names}
    mapped = label_lookup.get(norm_pred)
    if mapped is not None:
        return mapped

    # Fallback: choose the label with the highest naive token overlap
    def score(cand: str) -> int:
        c = cand.lower()
        return sum(1 for w in c.split() if w in norm_pred)

    return max(label_names, key=score)


async def eval_context(
    lm: LM,
    items: list[tuple[str, str]],
    label_names: list[str],
    ctx_name: str,
    specs: list[dict[str, Any]],
) -> tuple[str, int, int]:
    correct = 0
    with LMOverridesContext(specs):
        tasks = [classify_one(lm, text, label_names) for text, _ in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for (text, gold), pred in zip(items, results, strict=False):
        if isinstance(pred, Exception):
            # Treat exceptions as incorrect
            continue
        mapped = choose_label(pred, label_names)
        correct += int(mapped == gold)
    return (ctx_name, correct, len(items))


async def main() -> None:
    load_dotenv()

    n = int(os.getenv("N_SAMPLES", "20"))
    model = os.getenv("MODEL", "openai/gpt-oss-20b")
    vendor = os.getenv("VENDOR", "groq")

    lm = LM(model=model, vendor=vendor, temperature=0.0)

    print("Loading Banking77 dataset (split='test')...")
    ds = load_dataset("banking77", split="test")
    label_names: list[str] = ds.features["label"].names  # type: ignore

    idxs = random.sample(range(len(ds)), k=min(n, len(ds)))
    items = [
        (ds[i]["text"], label_names[int(ds[i]["label"])])  # (text, gold_label)
        for i in idxs
    ]

    # Define a few override contexts to compare
    contexts: list[dict[str, Any]] = [
        {
            "name": "baseline (no overrides)",
            "overrides": [],
        },
        {
            "name": "nonsense prompt injection (expected worse)",
            "overrides": [
                {
                    "match": {"contains": "", "role": "user"},
                    "injection_rules": [
                        # Heavily corrupt user text by replacing vowels
                        {"find": "a", "replace": "x"},
                        {"find": "e", "replace": "x"},
                        {"find": "i", "replace": "x"},
                        {"find": "o", "replace": "x"},
                        {"find": "u", "replace": "x"},
                        {"find": "A", "replace": "X"},
                        {"find": "E", "replace": "X"},
                        {"find": "I", "replace": "X"},
                        {"find": "O", "replace": "X"},
                        {"find": "U", "replace": "X"},
                    ],
                }
            ],
        },
        {
            "name": "injection: atm->ATM, txn->transaction",
            "overrides": [
                {
                    "match": {"contains": "atm", "role": "user"},
                    "injection_rules": [
                        {"find": "atm", "replace": "ATM"},
                        {"find": "txn", "replace": "transaction"},
                    ],
                }
            ],
        },
        {
            "name": "params: temperature=0.0",
            "overrides": [
                {"match": {"contains": ""}, "params": {"temperature": 0.0}},
            ],
        },
        {
            "name": "model override: 20b->120b",
            "overrides": [
                {"match": {"contains": ""}, "params": {"model": "openai/gpt-oss-120b"}},
            ],
        },
    ]

    print(f"\nEvaluating {len(contexts)} contexts on {len(items)} Banking77 samples (async)...")

    # Evaluate each context sequentially but batched (each context classifies in parallel)
    results: list[tuple[str, int, int]] = []
    for ctx in contexts:
        name = ctx["name"]
        specs = ctx["overrides"]
        print(f"Evaluating: {name} ...")
        res = await eval_context(lm, items, label_names, name, specs)
        results.append(res)

    print("\nResults:")
    for name, correct, total in results:
        acc = correct / total if total else 0.0
        print(f"- {name}: {correct}/{total} correct ({acc:.2%})")


if __name__ == "__main__":
    asyncio.run(main())
