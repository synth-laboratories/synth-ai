#!/usr/bin/env python3
"""Fetch MTG card art from Scryfall API for specific artists.

Usage:
    uv run python demos/mtg_artist_style/fetch_artist_cards.py

This script downloads card images from recognizable MTG artists for use in
style-matching optimization. It uses the Scryfall API to fetch card data
and downloads high-resolution art crops.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

# Scryfall API rate limit: max 10 requests/second
# We'll be conservative with 100ms between requests
SCRYFALL_DELAY = 0.15

# Artists with highly recognizable styles (18 total)
ARTISTS = {
    # Gothic / Surreal
    "seb_mckinnon": {
        "name": "Seb McKinnon",
        "query": 'a:"Seb McKinnon"',
        "style_description": "Moody, gothic, painterly surrealism with ethereal atmosphere and dark fantasy themes",
        "cards_to_fetch": 20,
    },
    # Watercolor / Storybook
    "rebecca_guay": {
        "name": "Rebecca Guay",
        "query": 'a:"Rebecca Guay"',
        "style_description": "Delicate watercolor and ink work with storybook and pre-Raphaelite aesthetics",
        "cards_to_fetch": 20,
    },
    # Iconic / Saturated
    "terese_nielsen": {
        "name": "Terese Nielsen",
        "query": 'a:"Terese Nielsen"',
        "style_description": "Saturated colors, iconic compositions with high-contrast fantasy lighting",
        "cards_to_fetch": 20,
    },
    # Luminous Landscapes
    "john_avon": {
        "name": "John Avon",
        "query": 'a:"John Avon"',
        "style_description": "Luminous landscapes with atmospheric depth - you can smell the air",
        "cards_to_fetch": 20,
    },
    # Eerie Realism
    "nils_hamm": {
        "name": "Nils Hamm",
        "query": 'a:"Nils Hamm"',
        "style_description": "Eerie, atmospheric realism with strong mood design and unsettling undertones",
        "cards_to_fetch": 20,
    },
    # Dark Fantasy / Metal
    "rk_post": {
        "name": "rk post",
        "query": 'a:"rk post"',
        "style_description": "Dark fantasy with sharp silhouettes and metal-album energy",
        "cards_to_fetch": 20,
    },
    # Gritty / Dynamic
    "kev_walker": {
        "name": "Kev Walker",
        "query": 'a:"Kev Walker"',
        "style_description": "Gritty, dynamic figures with bold brushwork - classic modern Magic look",
        "cards_to_fetch": 20,
    },
    # Horror / Grotesque
    "ron_spencer": {
        "name": "Ron Spencer",
        "query": 'a:"Ron Spencer"',
        "style_description": "Grotesque horror creature work with extremely identifiable organic textures",
        "cards_to_fetch": 20,
    },
    # Old-School Surreal
    "mark_tedin": {
        "name": "Mark Tedin",
        "query": 'a:"Mark Tedin"',
        "style_description": "Old-school surreal alien fantasy with early-Magic signature feel and cosmic weirdness",
        "cards_to_fetch": 20,
    },
    # Dreamy / Expressive
    "quinton_hoover": {
        "name": "Quinton Hoover",
        "query": 'a:"Quinton Hoover"',
        "style_description": "Loose, expressive brushwork with dreamy character pieces and soft edges",
        "cards_to_fetch": 20,
    },
    # Cinematic Realism
    "zoltan_boros": {
        "name": "Zoltan Boros",
        "query": 'a:"Zoltan Boros"',
        "style_description": "Dramatic realism with big cinematic scenes and dynamic lighting",
        "cards_to_fetch": 20,
    },
    # Polished Portraits
    "steve_argyle": {
        "name": "Steve Argyle",
        "query": 'a:"Steve Argyle"',
        "style_description": "Polished character portraits with crisp fantasy illustration and vibrant colors",
        "cards_to_fetch": 20,
    },
    # Dense / Kinetic
    "wayne_reynolds": {
        "name": "Wayne Reynolds",
        "query": 'a:"Wayne Reynolds"',
        "style_description": "Dense linework with kinetic busy compositions and explosive action",
        "cards_to_fetch": 20,
    },
    # Sleek Concept Art
    "aleksi_briclot": {
        "name": "Aleksi Briclot",
        "query": 'a:"Aleksi Briclot"',
        "style_description": "Sleek, cinematic concept-art style with dramatic poses and rich atmosphere",
        "cards_to_fetch": 20,
    },
    # Ultra-Detailed Realism
    "magali_villeneuve": {
        "name": "Magali Villeneuve",
        "query": 'a:"Magali Villeneuve"',
        "style_description": "Ultra-detailed elegant high-fantasy realism with exquisite character rendering",
        "cards_to_fetch": 20,
    },
    # Abstract / Textural
    "drew_tucker": {
        "name": "Drew Tucker",
        "query": 'a:"Drew Tucker"',
        "style_description": "Abstract experimental texture-heavy fantasy with unconventional compositions",
        "cards_to_fetch": 20,
    },
    # Clean / Iconic Artifacts
    "dan_frazier": {
        "name": "Dan Frazier",
        "query": 'a:"Dan Frazier"',
        "style_description": "Clean iconic artifact and iconography look with graphic design sensibility",
        "cards_to_fetch": 20,
    },
    # Stylized / Bold
    "chippy": {
        "name": "Chippy",
        "query": 'a:"Chippy"',
        "style_description": "Stylized bold shapes with slightly cartoon satirical edge and strong silhouettes",
        "cards_to_fetch": 20,
    },
}


async def search_scryfall(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search Scryfall for cards matching a query."""
    url = "https://api.scryfall.com/cards/search"
    params = {
        "q": f"{query} has:image unique:art",
        "order": "released",
        "dir": "desc",
    }

    cards = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        if resp.status_code == 404:
            print(f"  No cards found for query: {query}")
            return []
        resp.raise_for_status()
        data = resp.json()

        for card in data.get("data", []):
            # Only include cards with art_crop available
            image_uris = card.get("image_uris", {})
            if not image_uris.get("art_crop"):
                continue

            cards.append(card)
            if len(cards) >= limit:
                break

        # Handle pagination if needed
        while len(cards) < limit and data.get("has_more"):
            await asyncio.sleep(SCRYFALL_DELAY)
            next_page = data.get("next_page")
            if not next_page:
                break
            resp = await client.get(next_page)
            resp.raise_for_status()
            data = resp.json()

            for card in data.get("data", []):
                image_uris = card.get("image_uris", {})
                if not image_uris.get("art_crop"):
                    continue
                cards.append(card)
                if len(cards) >= limit:
                    break

    return cards[:limit]


async def download_image(url: str, output_path: Path) -> bool:
    """Download an image from URL to local path."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


async def fetch_artist_cards(
    artist_key: str, artist_config: dict[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    """Fetch and download card art for a specific artist."""
    artist_name = artist_config["name"]
    query = artist_config["query"]
    num_cards = artist_config["cards_to_fetch"]

    print(f"\nFetching cards for {artist_name}...")

    # Create artist directory
    artist_dir = output_dir / artist_key
    artist_dir.mkdir(parents=True, exist_ok=True)

    # Search for cards
    cards = await search_scryfall(query, limit=num_cards)
    print(f"  Found {len(cards)} cards with art")

    # Download art for each card
    card_metadata = []
    for i, card in enumerate(cards):
        card_name = card.get("name", f"card_{i}")
        safe_name = "".join(c if c.isalnum() or c in "_ -" else "_" for c in card_name)
        safe_name = safe_name.replace(" ", "_").lower()

        image_uris = card.get("image_uris", {})
        art_crop_url = image_uris.get("art_crop")

        if not art_crop_url:
            continue

        # Download image
        img_path = artist_dir / f"{safe_name}_{i:02d}.jpg"
        if img_path.exists():
            print(f"  [cached] {card_name}")
        else:
            await asyncio.sleep(SCRYFALL_DELAY)
            success = await download_image(art_crop_url, img_path)
            if success:
                print(f"  [downloaded] {card_name}")
            else:
                continue

        # Store metadata
        card_metadata.append(
            {
                "card_name": card_name,
                "artist": artist_name,
                "artist_key": artist_key,
                "set_name": card.get("set_name", ""),
                "image_path": str(img_path.relative_to(output_dir.parent)),
                "scryfall_uri": card.get("scryfall_uri", ""),
                "oracle_text": card.get("oracle_text", ""),
                "type_line": card.get("type_line", ""),
            }
        )

    return card_metadata


async def main() -> None:
    """Main entry point."""
    demo_dir = Path(__file__).parent
    output_dir = demo_dir / "gold_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MTG Artist Card Art Fetcher")
    print("=" * 60)

    all_metadata: dict[str, Any] = {"artists": {}, "cards": []}

    for artist_key, artist_config in ARTISTS.items():
        cards = await fetch_artist_cards(artist_key, artist_config, output_dir)
        all_metadata["artists"][artist_key] = {
            "name": artist_config["name"],
            "style_description": artist_config["style_description"],
            "num_cards": len(cards),
        }
        all_metadata["cards"].extend(cards)

    # Save metadata
    metadata_path = demo_dir / "artist_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for artist_key, info in all_metadata["artists"].items():
        print(f"  {info['name']}: {info['num_cards']} cards")
    print(f"\nTotal cards: {len(all_metadata['cards'])}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Images saved to: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
