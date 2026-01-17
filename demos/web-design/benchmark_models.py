#!/usr/bin/env python3
"""Benchmark image generation speed for different image generation models.

NOTE: This tests IMAGE GENERATION models (models that create images from text),
not vision models (models that understand images). Only gemini-2.5-flash-image
and similar models can generate images.
"""

import asyncio
import os
import time

from datasets import load_dataset

try:
    import google.generativeai as genai

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except ImportError:
    genai = None
    GOOGLE_API_KEY = None

try:
    from openai import AsyncOpenAI

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
    AsyncOpenAI = None
    OPENAI_API_KEY = None
    openai_client = None

# Test models - only image GENERATION models (not vision models)
MODELS = [
    # Gemini Image Generation
    "gemini-2.5-flash-image",  # Current model used in demo
    "gemini-3-pro-image-preview",  # Gemini 3 pro image generation
    # GPT Image Generation
    "gpt-image-1.5",  # GPT Image 1.5
    "gpt-image-1",  # GPT Image 1
    "gpt-image-1-mini",  # GPT Image 1 Mini (faster/cheaper?)
    "chatgpt-image-latest",  # Latest ChatGPT image model
]

# Number of times to test each model (to measure variance)
NUM_RUNS = 2  # 2 runs per model (6 models = 12 total generations)

# Load one example from the dataset
print("Loading dataset...")
dataset = load_dataset("JoshPurtell/web-design-screenshots", split="train")
dataset = dataset.filter(lambda ex: ex["site_name"] == "astral", load_from_cache_file=False)
example = dataset[0]

functional_description = example["functional_description"]
print(f"Example: {example['page_name']}")
print(f"Description length: {len(functional_description)} chars\n")

# Test prompt
test_prompt = f"""You are generating a professional startup website screenshot.

VISUAL STYLE GUIDELINES:
- Use a clean, modern, minimalist design aesthetic
- Color Scheme: Light backgrounds with high contrast dark text
- Typography: Large, bold headings with clear hierarchy
- Layout: Spacious with generous padding and margins
- Branding: Professional, tech-forward visual identity

Create a webpage that feels polished, modern, and trustworthy.

Generate a webpage screenshot based on this functional description:

{functional_description}

Apply the visual style guidelines to match the original design."""


async def benchmark_model(model: str) -> dict:
    """Benchmark a single model by calling it directly."""
    start = time.time()

    try:
        # Gemini models
        if model.startswith("gemini"):
            if not genai or not GOOGLE_API_KEY:
                return {
                    "model": model,
                    "duration": None,
                    "error": "google-generativeai not installed or GOOGLE_API_KEY not set",
                }

            gen_model = genai.GenerativeModel(model)
            response = await asyncio.to_thread(
                gen_model.generate_content,
                test_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )

            duration = time.time() - start

            # Check if response has image
            has_image = False
            if hasattr(response, "parts"):
                for part in response.parts:
                    if hasattr(part, "inline_data"):
                        has_image = True
                        break

            return {
                "model": model,
                "duration": duration,
                "has_image": has_image,
                "error": None,
            }

        # GPT/OpenAI models
        elif model.startswith(("gpt", "chatgpt")):
            if not openai_client or not OPENAI_API_KEY:
                return {
                    "model": model,
                    "duration": None,
                    "error": "openai not installed or OPENAI_API_KEY not set",
                }

            # OpenAI image generation models use /images/generations
            response = await openai_client.images.generate(
                model=model,
                prompt=test_prompt,
                n=1,
                size="1024x1024",  # Standard size
            )

            duration = time.time() - start

            has_image = len(response.data) > 0 if response.data else False

            return {
                "model": model,
                "duration": duration,
                "has_image": has_image,
                "error": None,
            }

        else:
            return {
                "model": model,
                "duration": None,
                "error": f"Unknown model family: {model}",
            }

    except Exception as e:
        duration = time.time() - start
        return {
            "model": model,
            "duration": duration,
            "error": str(e),
        }


async def main():
    print("=" * 80)
    print("IMAGE GENERATION BENCHMARK")
    print("=" * 80)
    print(f"Google API Key: {'✓' if GOOGLE_API_KEY else '✗'}")
    print(f"OpenAI API Key: {'✓' if OPENAI_API_KEY else '✗'}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Runs per model: {NUM_RUNS}")
    print("=" * 80)
    print()

    results_by_model = {}

    for model in MODELS:
        print(f"Testing {model} ({NUM_RUNS} runs)...")
        results_by_model[model] = []

        for run in range(NUM_RUNS):
            print(f"  Run {run + 1}/{NUM_RUNS}...", end=" ", flush=True)
            result = await benchmark_model(model)
            results_by_model[model].append(result)

            if result["error"]:
                print(f"❌ {result['error']}")
                break  # Stop if we get an error
            else:
                print(f"✅ {result['duration']:.1f}s")

        print()

    # Summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for model, results in results_by_model.items():
        successful = [r for r in results if r["error"] is None and r["duration"] is not None]

        if not successful:
            print(f"\n{model}: ALL FAILED")
            for r in results:
                print(f"  ❌ {r['error']}")
            continue

        durations = [r["duration"] for r in successful]
        avg = sum(durations) / len(durations)
        min_time = min(durations)
        max_time = max(durations)

        print(f"\n{model}:")
        print(f"  Average: {avg:.1f}s")
        print(f"  Min:     {min_time:.1f}s")
        print(f"  Max:     {max_time:.1f}s")
        print(f"  Runs:    {len(successful)}/{NUM_RUNS} successful")

        if len(durations) > 1:
            variance = sum((d - avg) ** 2 for d in durations) / len(durations)
            stddev = variance**0.5
            print(f"  StdDev:  {stddev:.1f}s")

    # Overall comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    model_averages = []
    for model, results in results_by_model.items():
        successful = [r for r in results if r["error"] is None and r["duration"] is not None]
        if successful:
            avg = sum(r["duration"] for r in successful) / len(successful)
            model_averages.append((model, avg))

    if model_averages:
        model_averages.sort(key=lambda x: x[1])

        print("\nRanked by speed (fastest first):")
        for i, (model, avg) in enumerate(model_averages, 1):
            print(f"  {i}. {model:35s} {avg:6.1f}s")

        if len(model_averages) > 1:
            fastest = model_averages[0]
            slowest = model_averages[-1]
            speedup = slowest[1] / fastest[1]
            print(f"\nFastest: {fastest[0]} ({fastest[1]:.1f}s)")
            print(f"Slowest: {slowest[0]} ({slowest[1]:.1f}s)")
            print(f"Speed difference: {speedup:.1f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
