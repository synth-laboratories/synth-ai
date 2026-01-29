#!/usr/bin/env python3
"""Run verifier evaluation on baseline images using graph completions.

No optimization - just scores the generated images against reference images.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from google import genai

DEMO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DEMO_DIR / "baseline_images"

RUBRIC = """
You are evaluating whether a generated MTG card illustration matches the style of a reference image.

## Scoring Criteria (0-1 scale)

### 1. Color Palette Match (0.25 weight)
- Does the generated image use similar color tones, saturation levels, and overall palette?
- Consider: warm vs cool tones, muted vs vibrant, contrast levels

### 2. Brushwork & Texture Style (0.25 weight)
- Does the brushwork feel similar? (painterly, smooth, textured, etc.)
- Consider: visible brush strokes, blending style, edge handling

### 3. Mood & Atmosphere (0.25 weight)
- Does it capture the same emotional tone? (moody, ethereal, dramatic, etc.)
- Consider: lighting direction, shadow depth, atmospheric effects

### 4. Compositional Style (0.25 weight)
- Does the composition follow similar principles?
- Consider: focal point placement, negative space usage, framing

## Output Format
Return a JSON object with:
{
  "color_palette": {"score": 0.0-1.0, "reasoning": "..."},
  "brushwork_texture": {"score": 0.0-1.0, "reasoning": "..."},
  "mood_atmosphere": {"score": 0.0-1.0, "reasoning": "..."},
  "compositional_style": {"score": 0.0-1.0, "reasoning": "..."},
  "overall_score": 0.0-1.0,
  "overall_reasoning": "..."
}
"""


def load_image_as_data_url(path: Path) -> str:
    """Load image as base64 data URL."""
    data = path.read_bytes()
    ext = path.suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext.lstrip("."), "image/jpeg"
    )
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def run_verifier(
    client: genai.Client,
    generated_path: Path,
    reference_path: Path,
    card_name: str,
    style_description: str,
) -> dict:
    """Run VLM verifier comparing generated to reference image."""

    gen_b64 = base64.b64encode(generated_path.read_bytes()).decode()
    ref_b64 = base64.b64encode(reference_path.read_bytes()).decode()

    prompt = f"""
{RUBRIC}

## Task
Compare the GENERATED image to the REFERENCE image and score how well the generated image captures the artist's style.

**Artist Style:** {style_description}
**Card:** {card_name}

The FIRST image is the GENERATED image (to be scored).
The SECOND image is the REFERENCE image (the gold standard to match).

Analyze carefully and return your scoring JSON.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": gen_b64}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}},
                ],
            }
        ],
    )

    # Parse JSON from response
    text = response.text
    # Find JSON in response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {"error": "Failed to parse JSON", "raw_response": text}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run verifier eval on baseline images")
    parser.add_argument("--artist", type=str, default="seb_mckinnon", help="Artist key")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print("GEMINI_API_KEY required")
        return

    # Load artist metadata
    data = json.loads((DEMO_DIR / "card_descriptions.json").read_text())
    artist_meta = data["artists"].get(args.artist, {})
    if not artist_meta:
        print(f"Unknown artist: {args.artist}")
        return

    style_description = artist_meta["style_description"]
    artist_dir = OUTPUT_DIR / args.artist

    # Find all generated images (not _reference, not _input)
    generated = [
        f for f in artist_dir.glob("*.png")
        if "_reference" not in f.name
    ]

    if not generated:
        print(f"No generated images found in {artist_dir}")
        return

    print(f"\nEvaluating {len(generated)} images for {artist_meta['name']}")
    print(f"Style: {style_description}\n")

    client = genai.Client(api_key=gemini_key)
    results = {}

    for gen_path in generated:
        card_name = gen_path.stem
        ref_path = artist_dir / f"{card_name}_reference.jpg"

        if not ref_path.exists():
            print(f"[{card_name}] No reference image, skipping")
            continue

        print(f"[{card_name}] Evaluating...")

        judgement = run_verifier(
            client, gen_path, ref_path, card_name, style_description
        )

        results[card_name] = {
            "generated_image": str(gen_path.name),
            "reference_image": str(ref_path.name),
            "judgement": judgement,
        }

        score = judgement.get("overall_score", "N/A")
        print(f"  Score: {score}")
        if "overall_reasoning" in judgement:
            print(f"  Reasoning: {judgement['overall_reasoning'][:100]}...")

        # Save individual judgement
        judgement_path = artist_dir / f"{card_name}_judgement.json"
        judgement_path.write_text(json.dumps(results[card_name], indent=2))
        print(f"  Saved: {judgement_path.name}")

    # Save combined results
    combined_path = artist_dir / "all_judgements.json"
    combined_path.write_text(json.dumps(results, indent=2))
    print(f"\nAll judgements saved to: {combined_path}")

    # Summary
    scores = [
        r["judgement"].get("overall_score", 0)
        for r in results.values()
        if isinstance(r["judgement"].get("overall_score"), (int, float))
    ]
    if scores:
        print(f"\nAverage score: {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()
