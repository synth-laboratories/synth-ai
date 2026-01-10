#!/usr/bin/env python3
"""Evaluate baseline prompts on HotpotQA seeds with parallel execution."""

import argparse
import asyncio
import json
import re

from datasets import load_dataset
from openai import AsyncOpenAI

# Default seed splits
TRAINING_SEEDS = list(range(150))
VALIDATION_SEEDS = list(range(150, 650))

BASELINE_SYSTEM = """You are a question-answering assistant. Answer the question based on the provided context.
Give a short, direct answer - typically a few words or a short phrase. Do not explain your reasoning."""

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": "Provide the answer to the question",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer to the question"}
            },
            "required": ["answer"],
        },
    },
}


def format_context(context: dict) -> str:
    """Format context paragraphs for the prompt."""
    paragraphs = []
    for title, sentences in zip(context["title"], context["sentences"], strict=False):
        text = " ".join(sentences)
        paragraphs.append(f"**{title}**\n{text}")
    return "\n\n".join(paragraphs)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    answer = answer.lower().strip()
    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)
    # Normalize whitespace
    answer = " ".join(answer.split())
    return answer


def answers_match(predicted: str, expected: str) -> bool:
    """Check if predicted answer matches expected (fuzzy matching)."""
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    # Exact match after normalization
    if pred_norm == exp_norm:
        return True

    # Check if one contains the other
    return pred_norm in exp_norm or exp_norm in pred_norm


async def evaluate_single(
    client: AsyncOpenAI,
    model: str,
    seed: int,
    example: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool, str, str]:
    """Evaluate a single example. Returns (seed, is_correct, predicted, expected)."""
    async with semaphore:
        context = format_context(example["context"])
        expected = example["answer"]
        question = example["question"]

        user_content = f"""Context:
{context}

Question: {question}

Answer the question based on the context above. Give a short, direct answer."""

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
                tool_choice={"type": "function", "function": {"name": "answer_question"}},
                **kwargs,
            )

            predicted = None

            # Try tool call first
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                args = json.loads(tool_calls[0].function.arguments)
                predicted = args.get("answer", "")
            # Fallback: use text content (for reasoning models)
            else:
                predicted = response.choices[0].message.content or ""

            if predicted:
                is_correct = answers_match(predicted, expected)
                return (seed, is_correct, predicted, expected)
            else:
                return (seed, False, None, expected)

        except Exception as e:
            return (seed, False, f"Error: {e}", expected)


async def evaluate_baseline(model: str, seeds: list, max_concurrency: int = 20) -> dict:
    """Evaluate baseline prompt on specified seeds in parallel."""
    client = AsyncOpenAI()

    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")

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
                print(f"[{seed:4d}] ✗ No valid response (exp: {expected})")
        elif isinstance(predicted, str) and predicted.startswith("Error:"):
            if i < print_limit:
                print(f"[{seed:4d}] ✗ {predicted}")
        else:
            if is_correct:
                correct += 1
            if i < print_limit:
                pred_short = predicted[:40] + "..." if len(predicted) > 40 else predicted
                print(
                    f"[{seed:4d}] {'✓' if is_correct else '✗'} pred={pred_short:45s} exp={expected}"
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
