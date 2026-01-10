#!/usr/bin/env python3
"""Generate sample images to compare baseline quality."""

import importlib
import os
import sys
from pathlib import Path

from PIL import Image

# Add demo dir to path and import local module dynamically
demo_dir = Path(__file__).parent
sys.path.insert(0, str(demo_dir))

_verify_generation = importlib.import_module("verify_generation")
generate_image_from_description = _verify_generation.generate_image_from_description
load_astral_example = _verify_generation.load_astral_example

# Get API key
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
    sys.exit(1)

# Load a sample example
example = load_astral_example()
description = example["functional_description"]
original_image = example["image"]

print(f"Generating image for: {example.get('page_id', 'unknown')[:50]}...")
print(f"Description length: {len(description)} chars")

# Generate image
try:
    image_bytes = generate_image_from_description(description, api_key)

    # Save generated image
    output_dir = demo_dir / "sample_outputs"
    output_dir.mkdir(exist_ok=True)

    generated_path = output_dir / "generated_sample.png"
    with open(generated_path, "wb") as f:
        f.write(image_bytes)
    print(f"✓ Generated image saved to: {generated_path}")

    # Save original for comparison
    original_path = output_dir / "original_sample.png"
    if isinstance(original_image, Image.Image):
        original_image.save(original_path)
    else:
        # If it's bytes or other format
        original_image.save(original_path)
    print(f"✓ Original image saved to: {original_path}")

    print("\nCompare images:")
    print(f"  open {original_path}")
    print(f"  open {generated_path}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
