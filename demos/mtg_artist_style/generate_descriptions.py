#!/usr/bin/env python3
"""Generate style-neutral art descriptions for MTG card images.

Uses a VLM to describe the visual content of each card's art without
mentioning artistic style or the artist. These descriptions serve as
inputs for the style-matching optimization task.

Usage:
    uv run python demos/mtg_artist_style/generate_descriptions.py
    uv run python demos/mtg_artist_style/generate_descriptions.py --model gpt-4.1-mini
    uv run python demos/mtg_artist_style/generate_descriptions.py --artist seb_mckinnon
"""

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate style-neutral art descriptions")
parser.add_argument(
    "--artist",
    type=str,
    default=None,
    help="Only process a specific artist (default: all artists)",
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4.1-mini",
    help="VLM model for descriptions (default: gpt-4.1-mini)",
)
parser.add_argument(
    "--max-concurrent",
    type=int,
    default=5,
    help="Maximum concurrent API calls (default: 5)",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Regenerate descriptions even if they exist",
)
args = parser.parse_args()

demo_dir = Path(__file__).resolve().parent
synth_root = demo_dir.parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("' ")
        if key:
            os.environ[key] = value


_load_env_file(synth_root / ".env")

# Get API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable required for VLM descriptions")

DESCRIPTION_PROMPT = """Describe this Magic: The Gathering card art in a style-neutral way.

Focus on:
1. SUBJECT: What is depicted (creatures, characters, objects, landscapes)
2. COMPOSITION: How elements are arranged (foreground, background, focal point)
3. COLORS: Dominant color palette and lighting
4. MOOD: The emotional tone or atmosphere
5. DETAILS: Notable visual elements, textures, environmental features

DO NOT mention:
- The artist's name or any artist
- Artistic style, technique, or medium (no "painterly", "watercolor", etc.)
- Art historical references or comparisons
- The card's game mechanics or rules text

Write a single paragraph of 2-4 sentences that could serve as an image generation prompt.
Be specific and descriptive about the visual content."""


async def describe_image(
    image_path: Path,
    card_info: dict[str, Any],
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Generate a style-neutral description of a card's art."""
    async with semaphore:
        # Load and encode image
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("ascii")
        
        ext = image_path.suffix.lower()
        mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
            ext, "image/jpeg"
        )
        data_url = f"data:{mime_type};base64,{img_data}"

        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DESCRIPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }
        ]

        # Call OpenAI API
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.3,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()

        description = result["choices"][0]["message"]["content"].strip()

        return {
            "card_name": card_info["card_name"],
            "artist": card_info["artist"],
            "artist_key": card_info["artist_key"],
            "set_name": card_info.get("set_name", ""),
            "type_line": card_info.get("type_line", ""),
            "oracle_text": card_info.get("oracle_text", ""),
            "image_path": card_info["image_path"],
            "art_description": description,
            "description_model": model,
        }


async def process_artist(
    artist_key: str,
    cards: list[dict[str, Any]],
    model: str,
    semaphore: asyncio.Semaphore,
    existing_descriptions: dict[str, str],
    force: bool,
) -> list[dict[str, Any]]:
    """Process all cards for a single artist."""
    results = []
    tasks = []

    for card in cards:
        image_path = demo_dir / card["image_path"]
        if not image_path.exists():
            print(f"  [skip] Image not found: {image_path}")
            continue

        # Check if we already have a description
        cache_key = f"{card['artist_key']}_{card['card_name']}_{card['image_path']}"
        if not force and cache_key in existing_descriptions:
            results.append({
                **card,
                "art_description": existing_descriptions[cache_key],
                "description_model": model,
            })
            continue

        tasks.append((card, image_path))

    if not tasks:
        return results

    # Process in batches
    async_tasks = [
        describe_image(image_path, card, model, semaphore)
        for card, image_path in tasks
    ]

    for coro in tqdm(
        asyncio.as_completed(async_tasks),
        total=len(async_tasks),
        desc=f"  {artist_key}",
        leave=False,
    ):
        try:
            result = await coro
            results.append(result)
        except Exception as e:
            print(f"  [error] {e}")

    return results


async def main() -> None:
    # Load metadata
    metadata_path = demo_dir / "artist_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Artist metadata not found. Run fetch_artist_cards.py first."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load existing descriptions if any
    descriptions_path = demo_dir / "card_descriptions.json"
    existing_descriptions: dict[str, str] = {}
    if descriptions_path.exists() and not args.force:
        with open(descriptions_path) as f:
            existing_data = json.load(f)
        for card in existing_data.get("cards", []):
            cache_key = f"{card['artist_key']}_{card['card_name']}_{card['image_path']}"
            existing_descriptions[cache_key] = card.get("art_description", "")
        print(f"Loaded {len(existing_descriptions)} existing descriptions")

    # Filter artists if specified
    if args.artist:
        if args.artist not in metadata["artists"]:
            available = list(metadata["artists"].keys())
            raise ValueError(f"Unknown artist '{args.artist}'. Available: {available}")
        artists_to_process = [args.artist]
    else:
        artists_to_process = list(metadata["artists"].keys())

    print("=" * 60)
    print("MTG Card Art Description Generator")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Artists: {len(artists_to_process)}")
    print(f"Max concurrent: {args.max_concurrent}")
    print()

    semaphore = asyncio.Semaphore(args.max_concurrent)
    all_results: list[dict[str, Any]] = []

    for artist_key in artists_to_process:
        artist_info = metadata["artists"][artist_key]
        artist_cards = [c for c in metadata["cards"] if c["artist_key"] == artist_key]
        
        print(f"\nProcessing {artist_info['name']} ({len(artist_cards)} cards)...")
        
        results = await process_artist(
            artist_key,
            artist_cards,
            args.model,
            semaphore,
            existing_descriptions,
            args.force,
        )
        all_results.extend(results)
        
        # Small delay between artists to avoid rate limits
        await asyncio.sleep(0.5)

    # Merge with any existing descriptions for other artists
    if descriptions_path.exists() and args.artist:
        with open(descriptions_path) as f:
            existing_data = json.load(f)
        
        # Keep cards from other artists
        processed_artist_keys = set(artists_to_process)
        for card in existing_data.get("cards", []):
            if card["artist_key"] not in processed_artist_keys:
                all_results.append(card)

    # Save results
    output_data = {
        "version": "1.0",
        "description_model": args.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cards": len(all_results),
        "artists": {
            artist_key: {
                "name": info["name"],
                "style_description": info["style_description"],
                "num_cards": len([c for c in all_results if c["artist_key"] == artist_key]),
            }
            for artist_key, info in metadata["artists"].items()
        },
        "cards": all_results,
    }

    with open(descriptions_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for artist_key in artists_to_process:
        count = len([c for c in all_results if c["artist_key"] == artist_key])
        print(f"  {metadata['artists'][artist_key]['name']}: {count} descriptions")
    print(f"\nTotal: {len(all_results)} card descriptions")
    print(f"Saved to: {descriptions_path}")


if __name__ == "__main__":
    asyncio.run(main())
