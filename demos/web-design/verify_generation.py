"""
Generate image from functional description and verify with contrastive scoring.
"""

import json
import os
from pathlib import Path

from datasets import load_from_disk
from google import genai
from PIL import Image


def load_astral_example():
    """Load an example from astral site."""
    dataset_path = Path(__file__).parent / "hf_dataset"
    dataset = load_from_disk(str(dataset_path))

    # Filter for astral examples
    astral_examples = [ex for ex in dataset if ex["site_name"] == "astral"]

    if not astral_examples:
        raise ValueError("No astral examples found in dataset")

    # Get homepage or pricing page if available
    example = None
    for ex in astral_examples:
        if "homepage" in ex["page_name"] or "pricing" in ex["page_name"]:
            example = ex
            break

    if not example:
        example = astral_examples[0]

    return example


def generate_image_from_description(description: str, api_key: str) -> bytes:
    """Generate webpage image directly from functional description using Gemini 2.5 Flash Image."""
    client = genai.Client(api_key=api_key)

    prompt = f"""Generate a modern, professional webpage screenshot based on this functional description:

{description}

Create a visually appealing, well-designed webpage that implements all the functionality described above.
The design should be modern, clean, and professional."""

    response = client.models.generate_content(model="gemini-2.5-flash-image", contents=prompt)

    # Extract image bytes from response
    if hasattr(response, "candidates") and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    # inline_data is a Blob object with data attribute
                    return part.inline_data.data

    # Try response.parts directly
    if hasattr(response, "parts"):
        for part in response.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                return part.inline_data.data

    raise ValueError("No image data in response")


def verify_with_vision_model(
    original_image: Image.Image,
    generated_image: Image.Image,
    functional_description: str,
    api_key: str,
) -> dict:
    """Use Gemini 2.5 Flash Image to score the generated webpage against original."""

    client = genai.Client(api_key=api_key)

    prompt = f"""You are evaluating how well a GENERATED webpage visually matches the ORIGINAL webpage.

You will be shown:
1. The ORIGINAL webpage screenshot (the ground truth)
2. The GENERATED webpage image (what the AI created)
3. The functional description that was used to generate it

FUNCTIONAL DESCRIPTION (for context):
{functional_description}

Your task is to COMPARE THE TWO IMAGES and evaluate how closely the generated page matches the original's VISUAL DESIGN.

Score the generated webpage on these criteria (0-10 each):

1. COLOR SCHEME & BRANDING:
   - Do the colors match (backgrounds, text, accents, gradients)?
   - Is the brand identity preserved (dark/light theme, color palette)?
   - Are backgrounds, borders, and overlays similar?

2. TYPOGRAPHY & TEXT STYLING:
   - Do font sizes, weights, and styles match?
   - Is text hierarchy (headings vs body) consistent?
   - Are text colors and contrast levels similar?

3. LAYOUT & SPACING:
   - Does the spatial arrangement match (padding, margins, gaps)?
   - Are sections positioned similarly?
   - Is the overall page flow and rhythm the same?

4. VISUAL ELEMENTS & GRAPHICS:
   - Do images, icons, and visual treatments match?
   - Are decorative elements (gradients, shadows, effects) similar?
   - Is the visual style consistent (modern/minimal/bold)?

5. OVERALL VISUAL FIDELITY:
   - How closely does the generated page resemble the original AT FIRST GLANCE?
   - Would someone mistake it for the real site?
   - Does it capture the brand's visual essence?

FOCUS ON VISUAL DESIGN, NOT FUNCTIONAL CONTENT. Compare colors, fonts, spacing, styling, and visual polish.

BE EXTREMELY SPECIFIC AND CONCRETE. Instead of saying "colors are different", say "Original uses #F5F5F5 background, generated uses #1A2B3C dark blue".

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "color_scheme_branding": <score 0-10>,
  "typography_styling": <score 0-10>,
  "layout_spacing": <score 0-10>,
  "visual_elements": <score 0-10>,
  "overall_visual_fidelity": <score 0-10>,
  "average_score": <average of all scores>,
  "visual_critiques": [
    "Specific issue 1 (e.g., 'Background: Original is off-white/cream, Generated is dark navy blue')",
    "Specific issue 2 (e.g., 'Hero heading: Original uses large purple text, Generated uses white text')",
    "Specific issue 3 (e.g., 'Logo missing in generated version')",
    "... continue with 10-15 specific, itemized visual differences ..."
  ]
}}

IMPORTANT: List 10-15 concrete, specific visual differences. Be precise about what's wrong."""

    # Create content with both PIL images
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            prompt,
            original_image,
            "ORIGINAL IMAGE ABOVE",
            generated_image,
            "GENERATED IMAGE ABOVE - Now provide your JSON evaluation:",
        ],
    )

    # Parse JSON from response

    text = response.text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    return json.loads(text)


def main():
    # Get API key
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not gemini_key:
        print("Error: GEMINI_API_KEY not set")
        exit(1)

    # Load astral example
    print("Loading astral example from dataset...")
    example = load_astral_example()

    print(f"\nExample: {example['site_name']} - {example['page_name']}")
    print(f"URL: {example['url']}")
    print(f"Description length: {len(example['functional_description'])} chars")

    # Generate image from description
    print("\n" + "=" * 80)
    print("Generating webpage image using Gemini 2.5 Flash Image (nano-banana)...")
    print("=" * 80)

    generated_image_bytes = generate_image_from_description(
        example["functional_description"], gemini_key
    )

    # Save generated image
    output_dir = Path(__file__).parent / "verification_results"
    output_dir.mkdir(exist_ok=True)

    image_file = output_dir / f"{example['site_name']}_{example['page_name']}_generated.png"
    with open(image_file, "wb") as f:
        f.write(generated_image_bytes)

    print(f"✓ Generated image saved to: {image_file}")

    # Get original image
    if not isinstance(example["image"], Image.Image):
        print("\nWarning: Unexpected image format in dataset")
        print("Skipping verification step")
        return

    original_image = example["image"]

    # Load generated image as PIL Image
    generated_image = Image.open(image_file)

    # Verify with vision model
    print("\n" + "=" * 80)
    print("Running contrastive verification using Gemini 2.5 Flash Image...")
    print("=" * 80)

    try:
        verification_result = verify_with_vision_model(
            original_image, generated_image, example["functional_description"], gemini_key
        )

        print("\n" + "=" * 80)
        print("VERIFICATION RESULTS")
        print("=" * 80)
        print(json.dumps(verification_result, indent=2))

        # Save results
        result_file = (
            output_dir / f"{example['site_name']}_{example['page_name']}_verification.json"
        )
        with open(result_file, "w") as f:
            json.dump(verification_result, f, indent=2)

        print(f"\n✓ Verification results saved to: {result_file}")

    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
