#!/usr/bin/env python3
"""Evaluate baseline prompts on HoVer seeds with parallel execution."""

import argparse
import asyncio
import json

from datasets import load_dataset
from openai import AsyncOpenAI

# Default seed splits
TRAINING_SEEDS = list(range(150))
VALIDATION_SEEDS = list(range(150, 650))

LABEL_MAP = {0: "SUPPORTED", 1: "REFUTED"}

BASELINE_SYSTEM = """You are a fact verification assistant. Your task is to determine whether a claim is SUPPORTED or REFUTED by the given evidence.

Analyze the evidence carefully and determine if it supports or refutes the claim.
- SUPPORTED: The evidence confirms the claim is true
- REFUTED: The evidence shows the claim is false

Give only the verdict, no explanation."""

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "verify_claim",
        "description": "Verify whether the claim is supported or refuted by the evidence",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["SUPPORTED", "REFUTED"],
                    "description": "The verdict",
                }
            },
            "required": ["verdict"],
        },
    },
}


def normalize_verdict(verdict: str) -> str:
    """Normalize verdict for comparison."""
    verdict = verdict.strip().upper()
    if "SUPPORT" in verdict:
        return "SUPPORTED"
    if "REFUT" in verdict:
        return "REFUTED"
    return verdict


def verdicts_match(predicted: str, expected: str) -> bool:
    """Check if predicted verdict matches expected."""
    return normalize_verdict(predicted) == normalize_verdict(expected)


async def evaluate_single(
    client: AsyncOpenAI,
    model: str,
    seed: int,
    example: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool, str, str]:
    """Evaluate a single example. Returns (seed, is_correct, predicted, expected)."""
    async with semaphore:
        claim = example["claim"]
        evidence = example["evidence"].strip()
        label_idx = int(example.get("label", 0))
        expected = LABEL_MAP.get(label_idx, "SUPPORTED")

        user_content = f"""Evidence:
{evidence}

Claim: {claim}

Based on the evidence above, is this claim SUPPORTED or REFUTED?"""

        kwargs = {}
        if "gpt-5" in model:
            kwargs["temperature"] = 1.0
            kwargs["max_completion_tokens"] = 8192
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = 256

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                tools=[TOOL_SCHEMA],
                tool_choice={"type": "function", "function": {"name": "verify_claim"}},
                **kwargs,
            )

            predicted = None

            # Try tool call first
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                args = json.loads(tool_calls[0].function.arguments)
                predicted = args.get("verdict", "")
            # Fallback: use text content (for reasoning models)
            else:
                predicted = response.choices[0].message.content or ""

            if predicted:
                is_correct = verdicts_match(predicted, expected)
                return (seed, is_correct, predicted, expected)
            else:
                return (seed, False, None, expected)

        except Exception as e:
            return (seed, False, f"Error: {e}", expected)


async def evaluate_baseline(model: str, seeds: list, max_concurrency: int = 20) -> dict:
    """Evaluate baseline prompt on specified seeds in parallel."""
    client = AsyncOpenAI()

    print("Loading HoVer dataset...")
    dataset = load_dataset("Dzeniks/hover", split="test")

    # Filter valid seeds
    valid_seeds = [s for s in seeds if s < len(dataset)]
    if len(valid_seeds) < len(seeds):
        print(f"Warning: {len(seeds) - len(valid_seeds)} seeds out of range")

    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    # Create all tasks
    tasks = [evaluate_single(client, model, seed, dataset[seed], semaphore) for seed in valid_seeds]

    print(f"Running {len(tasks)} evaluations with concurrency={max_concurrency}...")

    # Run all in parallel
    results = await asyncio.gather(*tasks)

    # Sort by seed and count results
    results = sorted(results, key=lambda x: x[0])
    correct = 0
    total = len(results)

    # Print sample of results (not all 500)
    print_limit = 20
    for i, (seed, is_correct, predicted, expected) in enumerate(results):
        if predicted is None:
            if i < print_limit:
                print(f"[{seed:4d}] X No valid response (exp: {expected})")
        elif isinstance(predicted, str) and predicted.startswith("Error:"):
            if i < print_limit:
                print(f"[{seed:4d}] X {predicted}")
        else:
            if is_correct:
                correct += 1
            if i < print_limit:
                pred_norm = normalize_verdict(predicted)
                print(
                    f"[{seed:4d}] {'V' if is_correct else 'X'} pred={pred_norm:12s} exp={expected}"
                )

    if len(results) > print_limit:
        print(f"... ({len(results) - print_limit} more results)")

    accuracy = correct / total if total > 0 else 0
    return {"model": model, "correct": correct, "total": total, "accuracy": accuracy}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--seeds", default=None, help="Seed range (e.g., 0-149 or 0,10,20)")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Use predefined split: train (0-149) or val (150-649)",
    )
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")
    args = parser.parse_args()

    # Parse seeds
    if args.split == "train":
        seeds = TRAINING_SEEDS
    elif args.split == "val":
        seeds = VALIDATION_SEEDS
    elif args.seeds:
        seeds = []
        for part in args.seeds.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                seeds.extend(range(int(start), int(end) + 1))
            else:
                seeds.append(int(part))
    else:
        seeds = VALIDATION_SEEDS  # Default to validation

    print(f"\nEvaluating baseline for {args.model} on {len(seeds)} seeds...")
    result = await evaluate_baseline(args.model, seeds, args.concurrency)

    print(f"\n{'=' * 50}")
    print(f"BASELINE RESULT: {result['model']}")
    print(f"Accuracy: {result['accuracy'] * 100:.1f}% ({result['correct']}/{result['total']})")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
