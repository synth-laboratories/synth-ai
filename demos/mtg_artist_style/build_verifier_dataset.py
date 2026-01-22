#!/usr/bin/env python3
"""Build verifier calibration dataset for MTG artist style matching.

Creates a dataset where:
- Input: image + artist style info + distinguishing giveaways
- Ground truth: IS_ACTUALLY_ARTIST (1.0 if from target artist, 0.0 if not)
- Loss: |IS_ACTUALLY_ARTIST - VERIFIER_SCORE|

Usage:
    uv run python demos/mtg_artist_style/build_verifier_dataset.py
    uv run python demos/mtg_artist_style/build_verifier_dataset.py --artist seb_mckinnon
"""

import argparse
import base64
import json
import random
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(description="Build verifier calibration dataset")
parser.add_argument(
    "--artist",
    type=str,
    default=None,
    help="Build dataset for specific artist (default: all)",
)
parser.add_argument(
    "--positives-per-artist",
    type=int,
    default=10,
    help="Number of positive examples per artist (default: 10)",
)
parser.add_argument(
    "--negatives-per-artist",
    type=int,
    default=10,
    help="Number of negative examples per artist (default: 10)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for sampling (default: 42)",
)
args = parser.parse_args()

demo_dir = Path(__file__).resolve().parent
random.seed(args.seed)

# Artist-specific distinguishing characteristics / giveaways
ARTIST_GIVEAWAYS = {
    "seb_mckinnon": {
        "style": "Moody, gothic, painterly surrealism with ethereal atmosphere",
        "giveaways": [
            "Dreamlike, almost hallucinatory quality with soft edges bleeding into each other",
            "Recurring motifs: birds/wings, flowing fabric, figures seen from behind",
            "Muted earth tones punctuated by luminous highlights",
            "Subjects often appear to dissolve into or emerge from their surroundings",
            "Strong sense of melancholy and quiet contemplation",
        ],
    },
    "rebecca_guay": {
        "style": "Delicate watercolor and ink with storybook pre-Raphaelite aesthetics",
        "giveaways": [
            "Flowing, organic linework reminiscent of Art Nouveau",
            "Soft, muted color palette with gentle gradients",
            "Ethereal female figures with elongated proportions",
            "Nature motifs: flowers, vines, leaves integrated into compositions",
            "Dreamlike, fairy-tale quality with romantic undertones",
        ],
    },
    "terese_nielsen": {
        "style": "Saturated colors, iconic compositions with high-contrast fantasy lighting",
        "giveaways": [
            "Bold, saturated color choices with strong value contrasts",
            "Dynamic poses with theatrical lighting",
            "Intricate background patterns and symbolic elements",
            "Faces with distinctive, idealized features",
            "Rich textural details especially in clothing and magical effects",
        ],
    },
    "john_avon": {
        "style": "Luminous landscapes with atmospheric depth",
        "giveaways": [
            "Vast, sweeping vistas with incredible sense of scale",
            "Masterful use of atmospheric perspective and light",
            "Rich, saturated colors especially in skies and water",
            "Landscapes that feel lived-in yet fantastical",
            "Strong horizon lines with dramatic cloud formations",
        ],
    },
    "nils_hamm": {
        "style": "Eerie, atmospheric realism with unsettling undertones",
        "giveaways": [
            "Photorealistic rendering with subtle wrongness",
            "Creatures with disturbingly organic, fleshy textures",
            "Muted, desaturated palette with sickly undertones",
            "Subjects often partially obscured or emerging from darkness",
            "Uncanny valley effect - familiar forms made alien",
        ],
    },
    "rk_post": {
        "style": "Dark fantasy with sharp silhouettes and metal-album energy",
        "giveaways": [
            "High contrast black and white with dramatic shadows",
            "Sharp, angular forms and aggressive poses",
            "Horror and death imagery rendered with visceral impact",
            "Figures often skeletal, undead, or demonic",
            "Dynamic compositions with strong diagonal lines",
        ],
    },
    "kev_walker": {
        "style": "Gritty, dynamic figures with bold brushwork",
        "giveaways": [
            "Loose, energetic brushstrokes visible in the work",
            "Earthy, muted color palette with strong darks",
            "Muscular, solid figure rendering with weight and mass",
            "Action poses captured mid-movement",
            "Textured, almost tactile surface quality",
        ],
    },
    "ron_spencer": {
        "style": "Grotesque horror creatures with organic textures",
        "giveaways": [
            "Disturbing creature designs with biological horror elements",
            "Intricate, almost obsessive detail in organic forms",
            "Slimy, chitinous, or fleshy textures",
            "Creatures that feel genuinely alien and threatening",
            "Dark, claustrophobic compositions",
        ],
    },
    "mark_tedin": {
        "style": "Old-school surreal alien fantasy with cosmic weirdness",
        "giveaways": [
            "Bizarre, reality-bending architectural structures",
            "Cosmic and alien imagery with Lovecraftian undertones",
            "Intricate geometric patterns and impossible spaces",
            "Artifacts that feel ancient and powerful",
            "Color palette often emphasizing purples, blues, and metallics",
        ],
    },
    "quinton_hoover": {
        "style": "Loose expressive brushwork with dreamy character pieces",
        "giveaways": [
            "Soft, impressionistic rendering with visible brushwork",
            "Gentle, pastel color harmonies",
            "Characters with soft, approachable expressions",
            "Backgrounds that fade into abstract color fields",
            "Whimsical, storybook quality",
        ],
    },
    "zoltan_boros": {
        "style": "Dramatic realism with cinematic scenes",
        "giveaways": [
            "Movie-poster quality compositions with dramatic lighting",
            "Highly detailed, realistic rendering",
            "Epic scale with multiple figures in action",
            "Rich, saturated colors with strong rim lighting",
            "Dynamic perspectives and foreshortening",
        ],
    },
    "steve_argyle": {
        "style": "Polished character portraits with crisp fantasy illustration",
        "giveaways": [
            "Clean, polished digital rendering",
            "Strong focus on character faces and expressions",
            "Vibrant, saturated color palette",
            "Detailed costume and armor design",
            "Glamorous, attractive character designs",
        ],
    },
    "wayne_reynolds": {
        "style": "Dense linework with kinetic busy compositions",
        "giveaways": [
            "Incredibly detailed, busy compositions packed with elements",
            "Strong black linework defining forms",
            "Exaggerated, dynamic poses with explosive energy",
            "Characters loaded with gear, weapons, and accessories",
            "Warm color palette with lots of reds and oranges",
        ],
    },
    "aleksi_briclot": {
        "style": "Sleek cinematic concept-art style",
        "giveaways": [
            "Painterly digital style with concept art sensibility",
            "Dramatic, cinematic lighting and atmosphere",
            "Strong silhouettes and shape design",
            "Dark, moody color palettes with selective saturation",
            "Epic scale figures against vast backgrounds",
        ],
    },
    "magali_villeneuve": {
        "style": "Ultra-detailed elegant high-fantasy realism",
        "giveaways": [
            "Exquisite detail in faces, hair, and fabric",
            "Elegant, beautiful character designs",
            "Rich, jewel-tone color palette",
            "Soft, romantic lighting",
            "Highly realistic rendering with painterly touches",
        ],
    },
    "drew_tucker": {
        "style": "Abstract experimental texture-heavy fantasy",
        "giveaways": [
            "Heavy use of texture and mixed-media effects",
            "Abstract, non-representational elements",
            "Unusual color combinations and palettes",
            "Dreamlike, almost psychedelic quality",
            "Forms that dissolve into pure texture",
        ],
    },
    "dan_frazier": {
        "style": "Clean iconic artifact design with graphic sensibility",
        "giveaways": [
            "Simple, iconic object designs",
            "Clean linework with flat color areas",
            "Graphic design sensibility - logo-like quality",
            "Metallic objects rendered with simple highlights",
            "Straightforward, centered compositions",
        ],
    },
    "chippy": {
        "style": "Stylized bold shapes with cartoon satirical edge",
        "giveaways": [
            "Bold, simplified shapes and forms",
            "Strong graphic quality with clear silhouettes",
            "Slightly exaggerated, almost caricatured features",
            "Metallic and mechanical subjects rendered stylistically",
            "Clean color blocking with limited palette",
        ],
    },
}


def load_image_as_data_url(image_path: Path) -> str:
    """Load an image as a base64 data URL."""
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("ascii")
    ext = image_path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")
    return f"data:{mime};base64,{img_data}"


def build_verifier_example(
    task_id: str,
    image_path: Path,
    target_artist_key: str,
    actual_artist_key: str,
    card_name: str,
) -> dict[str, Any]:
    """Build a single verifier calibration example."""
    is_match = 1.0 if actual_artist_key == target_artist_key else 0.0
    
    target_info = ARTIST_GIVEAWAYS.get(target_artist_key, {})
    style_desc = target_info.get("style", "Unknown style")
    giveaways = target_info.get("giveaways", [])
    
    # Build the input for the verifier
    input_data = {
        "image_url": load_image_as_data_url(image_path),
        "target_artist_style": style_desc,
        "distinguishing_giveaways": giveaways,
        "card_name": card_name,
    }
    
    # Build rubric explaining the ground truth
    if is_match:
        rubric = (
            f"This image IS by the target artist. "
            f"It should exhibit: {style_desc}. "
            f"Key giveaways to look for: {'; '.join(giveaways[:3])}"
        )
    else:
        actual_info = ARTIST_GIVEAWAYS.get(actual_artist_key, {})
        actual_style = actual_info.get("style", "different style")
        rubric = (
            f"This image is NOT by the target artist. "
            f"It's actually in a {actual_style} style, "
            f"which differs from the target's {style_desc}."
        )
    
    return {
        "task_id": task_id,
        "input": input_data,
        "gold_score": is_match,
        "rubric": rubric,
        "metadata": {
            "target_artist": target_artist_key,
            "actual_artist": actual_artist_key,
            "is_match": bool(is_match),
            "card_name": card_name,
        },
    }


def main() -> None:
    # Load card metadata
    metadata_path = demo_dir / "artist_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Run fetch_artist_cards.py first")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Filter artists
    if args.artist:
        if args.artist not in metadata["artists"]:
            raise ValueError(f"Unknown artist: {args.artist}")
        artists_to_process = [args.artist]
    else:
        artists_to_process = list(metadata["artists"].keys())
    
    print("=" * 60)
    print("Building Verifier Calibration Dataset")
    print("=" * 60)
    print(f"Artists: {len(artists_to_process)}")
    print(f"Positives per artist: {args.positives_per_artist}")
    print(f"Negatives per artist: {args.negatives_per_artist}")
    print()
    
    all_examples = []
    
    for target_artist in artists_to_process:
        print(f"\nBuilding examples for {target_artist}...")
        
        # Get cards by artist
        target_cards = [c for c in metadata["cards"] if c["artist_key"] == target_artist]
        other_cards = [c for c in metadata["cards"] if c["artist_key"] != target_artist]
        
        if len(target_cards) < args.positives_per_artist:
            print(f"  Warning: only {len(target_cards)} cards available for positives")
        
        # Sample positive examples (same artist)
        positive_cards = random.sample(
            target_cards, 
            min(args.positives_per_artist, len(target_cards))
        )
        
        for i, card in enumerate(positive_cards):
            image_path = demo_dir / card["image_path"]
            if not image_path.exists():
                continue
            
            example = build_verifier_example(
                task_id=f"{target_artist}_pos_{i:03d}",
                image_path=image_path,
                target_artist_key=target_artist,
                actual_artist_key=target_artist,
                card_name=card["card_name"],
            )
            all_examples.append(example)
        
        # Sample negative examples (different artists)
        negative_cards = random.sample(
            other_cards,
            min(args.negatives_per_artist, len(other_cards))
        )
        
        for i, card in enumerate(negative_cards):
            image_path = demo_dir / card["image_path"]
            if not image_path.exists():
                continue
            
            example = build_verifier_example(
                task_id=f"{target_artist}_neg_{i:03d}",
                image_path=image_path,
                target_artist_key=target_artist,
                actual_artist_key=card["artist_key"],
                card_name=card["card_name"],
            )
            all_examples.append(example)
        
        pos_count = len([e for e in all_examples if e["metadata"]["target_artist"] == target_artist and e["metadata"]["is_match"]])
        neg_count = len([e for e in all_examples if e["metadata"]["target_artist"] == target_artist and not e["metadata"]["is_match"]])
        print(f"  Created {pos_count} positive, {neg_count} negative examples")
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Split into train/validation
    split_idx = int(len(all_examples) * 0.8)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Build dataset structure for Graph Evolve
    dataset = {
        "version": "1.0",
        "metadata": {
            "name": "mtg_artist_verifier_calibration",
            "description": (
                "Verifier calibration dataset for MTG artist style matching. "
                "Each example contains an image and target artist info. "
                "Gold score is 1.0 if image is by target artist, 0.0 otherwise."
            ),
            "task_description": (
                "Given an image and a target artist's style description with distinguishing giveaways, "
                "determine if the image was created by that artist. "
                "Output a score from 0.0 to 1.0 where 1.0 means definitely by this artist."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "Base64 image data URL"},
                    "target_artist_style": {"type": "string", "description": "Description of target artist's style"},
                    "distinguishing_giveaways": {"type": "array", "items": {"type": "string"}},
                    "card_name": {"type": "string"},
                },
                "required": ["image_url", "target_artist_style", "distinguishing_giveaways"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string"},
                    "giveaways_found": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["score"],
            },
        },
        "artists": {k: ARTIST_GIVEAWAYS.get(k, {}) for k in artists_to_process},
        "train_tasks": [
            {"id": e["task_id"], "input": e["input"]}
            for e in train_examples
        ],
        "train_gold_outputs": [
            {"task_id": e["task_id"], "output": {"score": e["gold_score"]}, "score": e["gold_score"], "rubric": e["rubric"]}
            for e in train_examples
        ],
        "val_tasks": [
            {"id": e["task_id"], "input": e["input"]}
            for e in val_examples
        ],
        "val_gold_outputs": [
            {"task_id": e["task_id"], "output": {"score": e["gold_score"]}, "score": e["gold_score"], "rubric": e["rubric"]}
            for e in val_examples
        ],
        "train_seeds": list(range(len(train_examples))),
        "val_seeds": list(range(len(train_examples), len(train_examples) + len(val_examples))),
        "total_examples": len(all_examples),
        "num_artists": len(artists_to_process),
    }
    
    # Save dataset
    output_path = demo_dir / "verifier_calibration_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total examples: {len(all_examples)}")
    print(f"  Train: {len(train_examples)}")
    print(f"  Validation: {len(val_examples)}")
    print(f"  Positives: {len([e for e in all_examples if e['metadata']['is_match']])}")
    print(f"  Negatives: {len([e for e in all_examples if not e['metadata']['is_match']])}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
