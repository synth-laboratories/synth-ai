#!/usr/bin/env python3
"""Score generated images using Gemini VLM verifier."""

import base64
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types

api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# Get an original image from the cached dataset
CACHE_DIR = Path.home() / ".cache" / "synth_ai" / "web_design" / "task_images" / "resized_384"


def get_original_image() -> Path:
    """Get the Astral homepage (about page slice 1) as reference."""
    # Use the about page slice 1 which shows the full homepage design
    target = CACHE_DIR / "astral_about_20260107_025234__slice_01_of_02_000.png"
    if target.exists():
        return target
    # Fallback to first available
    images = list(CACHE_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No original images found in {CACHE_DIR}")
    return images[0]


def encode_image(path: Path) -> str:
    """Encode image to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def score_image(generated_path: Path, original_path: Path) -> dict:
    """Score a generated image against original using VLM."""

    prompt = """You are evaluating how well a generated webpage screenshot matches a reference design.

Compare the GENERATED image (first) against the ORIGINAL image (second).

Score on a scale of 0.0 to 1.0 based on:
1. Layout fidelity - Does the generated image have similar section organization?
2. Visual style - Does it match the clean, modern, professional aesthetic?
3. Content accuracy - Are the key elements (logo, headlines, sections) present?
4. Overall quality - Is it a polished, production-ready design?

Respond with ONLY a JSON object in this exact format:
{"score": 0.X, "reasoning": "brief explanation"}
"""

    # Read both images
    with open(generated_path, "rb") as f:
        generated_bytes = f.read()
    with open(original_path, "rb") as f:
        original_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=generated_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=original_bytes, mime_type="image/png"),
        ],
    )

    return response.text


def main():
    comparison_dir = Path(__file__).parent / "comparison_images"
    baseline_path = comparison_dir / "baseline_generated.png"
    mutated_path = comparison_dir / "mutated_generated.png"

    original_path = get_original_image()
    print(f"Original image: {original_path.name}")
    print("=" * 60)

    print("\n[1/2] Scoring BASELINE generated image...")
    baseline_result = score_image(baseline_path, original_path)
    print(f"Baseline: {baseline_result}")

    print("\n[2/2] Scoring MUTATED generated image...")
    mutated_result = score_image(mutated_path, original_path)
    print(f"Mutated: {mutated_result}")

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
