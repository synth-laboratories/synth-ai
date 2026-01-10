#!/usr/bin/env python3
"""Evaluate baseline prompts on Iris validation seeds."""

import asyncio
import json
import argparse
from openai import AsyncOpenAI
from datasets import load_dataset

# Iris has 150 samples: 0-49 (setosa), 50-99 (versicolor), 100-149 (virginica)
# Stratified validation: 20 from each class = 60 total
VALIDATION_SEEDS = list(range(30, 50)) + list(range(80, 100)) + list(range(130, 150))

BASELINE_SYSTEM = """You are a botany classification assistant. Based on the flower's measurements, classify the iris species. Respond with one of: setosa, versicolor, or virginica."""

IRIS_LABELS = ["setosa", "versicolor", "virginica"]

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "iris_classify",
        "description": "Classify the iris species",
        "parameters": {
            "type": "object",
            "properties": {"species": {"type": "string", "description": "The classified species"}},
            "required": ["species"],
        },
    },
}


def format_features(row: dict) -> str:
    """Format iris features for the prompt."""
    return (
        f"Sepal Length: {row['SepalLengthCm']} cm\n"
        f"Sepal Width: {row['SepalWidthCm']} cm\n"
        f"Petal Length: {row['PetalLengthCm']} cm\n"
        f"Petal Width: {row['PetalWidthCm']} cm"
    )


def normalize_species(species: str) -> str:
    """Normalize species name for comparison."""
    species = species.lower().strip()
    if species.startswith("iris-"):
        species = species[5:]
    return species


async def evaluate_single(
    client: AsyncOpenAI,
    model: str,
    seed: int,
    example: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool, str, str]:
    """Evaluate a single example. Returns (seed, is_correct, predicted, expected)."""
    async with semaphore:
        features = format_features(example)
        expected_species = normalize_species(example["Species"])

        user_content = f"""Flower Measurements:
{features}

Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."""

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
                tool_choice={"type": "function", "function": {"name": "iris_classify"}},
                **kwargs,
            )

            predicted = None

            # Try tool call first
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                args = json.loads(tool_calls[0].function.arguments)
                predicted = normalize_species(args.get("species", ""))
            # Fallback: extract from text content (for reasoning models)
            else:
                content = response.choices[0].message.content or ""
                content_lower = content.lower()
                for species in IRIS_LABELS:
                    if species in content_lower:
                        predicted = species
                        break

            if predicted:
                is_correct = predicted == expected_species
                return (seed, is_correct, predicted, expected_species)
            else:
                return (seed, False, None, expected_species)

        except Exception as e:
            return (seed, False, f"Error: {e}", expected_species)


async def evaluate_baseline(model: str, seeds: list, max_concurrency: int = 20) -> dict:
    """Evaluate baseline prompt on specified seeds in parallel."""
    client = AsyncOpenAI()

    print("Loading Iris dataset...")
    dataset = load_dataset("scikit-learn/iris", split="train")

    # Filter valid seeds and prepare tasks
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

    # Sort by seed and print results
    results = sorted(results, key=lambda x: x[0])
    correct = 0
    total = len(results)

    for seed, is_correct, predicted, expected in results:
        if predicted is None:
            print(f"[{seed:3d}] ✗ No valid response")
        elif predicted.startswith("Error:"):
            print(f"[{seed:3d}] ✗ {predicted}")
        else:
            if is_correct:
                correct += 1
            print(f"[{seed:3d}] {'✓' if is_correct else '✗'} pred={predicted:15s} exp={expected}")

    accuracy = correct / total if total > 0 else 0
    return {"model": model, "correct": correct, "total": total, "accuracy": accuracy}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--seeds", default=None, help="Seed range (e.g., 30-49 or 30,40,50)")
    args = parser.parse_args()

    # Parse seeds or use default stratified validation seeds
    if args.seeds:
        seeds = []
        for part in args.seeds.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                seeds.extend(range(int(start), int(end) + 1))
            else:
                seeds.append(int(part))
    else:
        seeds = VALIDATION_SEEDS

    print(f"\nEvaluating baseline for {args.model} on {len(seeds)} seeds...")
    result = await evaluate_baseline(args.model, seeds)

    print(f"\n{'=' * 50}")
    print(f"BASELINE RESULT: {result['model']}")
    print(f"Accuracy: {result['accuracy'] * 100:.1f}% ({result['correct']}/{result['total']})")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
