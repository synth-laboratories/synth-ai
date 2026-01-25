#!/usr/bin/env python3
"""Run baseline eval to generate images and see what the model produces."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DATA_PATH = DEMO_DIR / "card_descriptions.json"
OUTPUT_DIR = DEMO_DIR / "baseline_images"

DEFAULT_PROMPT = """Generate a Magic: The Gathering card illustration that matches the target artist style.

Card name: {card_name}
Card description: {card_description}
Target artist style: {target_artist_style}

Do NOT mention the artist name in any text. Return only the image."""


def generate_image_sync(
    gemini_client,
    card: dict,
    artist_meta: dict,
) -> tuple[str, bytes | None]:
    """Generate an image for a card and return (card_name, image_bytes)."""
    from google import genai

    prompt = DEFAULT_PROMPT.format(
        card_name=card["card_name"],
        card_description=card.get("art_description", ""),
        target_artist_style=artist_meta.get("style_description", ""),
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return card["card_name"], part.inline_data.data

        print(f"  No image in response for {card['card_name']}")
        return card["card_name"], None

    except Exception as e:
        print(f"  Error generating {card['card_name']}: {e}")
        return card["card_name"], None


def main():
    import argparse
    import shutil
    from google import genai

    parser = argparse.ArgumentParser(description="Run baseline image generation eval")
    parser.add_argument("--artist", type=str, default="seb_mckinnon", help="Artist key")
    parser.add_argument("--num-images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        print("GEMINI_API_KEY required. Set it in your environment.")
        return

    # Load data
    payload = json.loads(DATA_PATH.read_text())
    artists = payload.get("artists", {})
    cards = payload.get("cards", [])

    artist_meta = artists.get(args.artist, {})
    if not artist_meta:
        print(f"Unknown artist: {args.artist}")
        print(f"Available: {list(artists.keys())}")
        return

    artist_cards = [c for c in cards if c.get("artist_key") == args.artist]
    artist_cards = [c for c in artist_cards if c.get("art_description")][:args.num_images]

    print(f"\nGenerating {len(artist_cards)} baseline images for {artist_meta['name']}")
    print(f"Style: {artist_meta['style_description']}\n")

    out_base = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_base.mkdir(exist_ok=True)
    artist_dir = out_base / args.artist
    artist_dir.mkdir(exist_ok=True)

    client = genai.Client(api_key=gemini_api_key)
    gold_dir = DEMO_DIR / "gold_images" / args.artist

    results = []

    for i, card in enumerate(artist_cards):
        card_name = card["card_name"]
        safe_name = card_name.replace(" ", "_").replace("/", "_")[:50]

        print(f"[{i+1}/{len(artist_cards)}] Generating: {card_name}...")

        # Generate image
        _, img_bytes = generate_image_sync(client, card, artist_meta)
        if not img_bytes:
            print(f"  FAILED - skipping")
            continue

        # Save generated image
        gen_path = artist_dir / f"{safe_name}.png"
        gen_path.write_bytes(img_bytes)
        print(f"  Saved: {gen_path.name}")

        # Copy reference image
        ref_src = None
        if gold_dir.exists():
            # Find matching reference (snake_case with number suffix)
            for ref_file in gold_dir.iterdir():
                if safe_name.lower().replace("_", "") in ref_file.stem.lower().replace("_", ""):
                    ref_src = ref_file
                    break

        ref_path = None
        if ref_src:
            ref_path = artist_dir / f"{safe_name}_reference{ref_src.suffix}"
            shutil.copy(ref_src, ref_path)
            print(f"  Reference: {ref_path.name}")

        # Save input/prompt
        prompt = DEFAULT_PROMPT.format(
            card_name=card_name,
            card_description=card.get("art_description", ""),
            target_artist_style=artist_meta.get("style_description", ""),
        )
        input_path = artist_dir / f"{safe_name}_input.txt"
        input_path.write_text(prompt)

        # Build trace record
        trace = {
            "id": f"{args.artist}_{i:03d}",
            "card_name": card_name,
            "artist_key": args.artist,
            "artist_name": artist_meta["name"],
            "style_description": artist_meta["style_description"],
            "card_description": card.get("art_description", ""),
            "prompt": prompt,
            "generated_image": gen_path.name,
            "reference_image": ref_path.name if ref_path else None,
            "human_score": None,  # To be filled manually
            "human_notes": "",    # To be filled manually
        }
        results.append(trace)

    # Save manifest for grading
    manifest_path = artist_dir / "grading_manifest.json"
    manifest = {
        "artist_key": args.artist,
        "artist_name": artist_meta["name"],
        "style_description": artist_meta["style_description"],
        "num_samples": len(results),
        "grading_instructions": (
            "For each sample, set human_score (0.0-1.0) based on how well the "
            "generated image matches the reference artist's style. Add notes explaining your score."
        ),
        "samples": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*60}")
    print(f"Generated {len(results)} samples")
    print(f"Output directory: {artist_dir}")
    print(f"Grading manifest: {manifest_path}")
    print(f"\nNext steps:")
    print(f"1. Open {artist_dir} and review each generated image vs reference")
    print(f"2. Edit grading_manifest.json to add human_score (0.0-1.0) and human_notes")
    print(f"3. Run verifier optimization on the graded dataset")


if __name__ == "__main__":
    main()
