"""
Generate functional descriptions of web pages from screenshots using Gemini 2.5.

The goal is to create descriptions that capture WHAT is on the page (content, structure, purpose)
but NOT HOW it looks (colors, fonts, spacing, visual design).
"""

import base64
import json
import os
from pathlib import Path

import google.generativeai as genai

FUNCTIONAL_DESCRIPTION_PROMPT = """You are analyzing a screenshot of a web page. Your task is to create a detailed FUNCTIONAL description of the page.

CRITICAL RULES:
1. Describe WHAT is on the page, NOT how it looks
2. Focus on content, structure, and purpose
3. Do NOT mention: colors, fonts, spacing, layouts, visual design, animations, gradients, shadows, borders
4. Do NOT use visual adjectives like "bold", "large", "small", "centered", "aligned"
5. Do NOT use layout/positioning terms like: "cards", "grid", "row", "column", "modal", "drawer", "left", "right", "top", "bottom", "above", "below"
6. Think like you're describing the page to someone who will rebuild it from scratch with their own design

FORBIDDEN TERMS (use alternatives):
- "cards" → use "sections" or "areas"
- "modal" → use "dialog" or "overlay message"
- "grid" → use "collection" or "set"
- "row of logos" → use "list of logos" or "set of logos"
- "column" → use "section" or "list"
- Avoid any directional positioning: left/right/top/bottom/above/below/beside

WHAT TO INCLUDE:
- Page type and primary purpose
- Main sections and their content
- Navigation elements and their targets
- Text content (headings, body text, labels)
- Calls-to-action and their purpose
- Data/information being presented
- Interactive elements and their function
- Forms and their fields
- Lists and their items
- Links and where they lead

STRUCTURE YOUR RESPONSE:
1. Page Overview: Brief description of page type and main purpose
2. Sections: List each major section with its content and purpose
3. Interactive Elements: Describe forms, buttons, links
4. Content Details: Specific text, data points, feature lists

EXAMPLE OF GOOD FUNCTIONAL DESCRIPTION:
"Pricing page for a SaaS product. Header navigation includes links to Product, Docs, Blog, and Sign In. Main content has three pricing tiers: Free, Pro, and Enterprise. Free tier includes 100 API calls per month, basic support, and single user access. Pro tier includes unlimited API calls, priority support, team collaboration, and costs $29/month. Enterprise tier has custom pricing and includes dedicated support, SLA guarantees, and custom integrations. Each tier has a call-to-action button to start a trial or contact sales."

EXAMPLE OF BAD FUNCTIONAL DESCRIPTION (too visual/layout-focused):
"Pricing page with a dark background and white text. Three large cards are centered on the page with rounded corners and shadows. The middle card is highlighted with a blue gradient. Each card has a large heading in bold font, followed by bullet points in a lighter gray color. A row of company logos appears at the bottom in a grid layout."

Now, analyze this screenshot and provide the functional description following the rules above.
"""


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_functional_description(image_path: str, api_key: str) -> dict:
    """Generate functional description from screenshot using Gemini 2.5."""
    # Configure the API
    genai.configure(api_key=api_key)

    # Create model
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # Load image
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Create the prompt with image
    response = model.generate_content(
        [FUNCTIONAL_DESCRIPTION_PROMPT, {"mime_type": "image/png", "data": image_data}]
    )

    return {
        "image_path": str(image_path),
        "functional_description": response.text,
        "model": "gemini-2.0-flash-exp",
    }


def _to_demo_relative(path: Path, demo_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(demo_dir))
    except ValueError:
        return str(path)


def process_sample_images(image_paths: list, api_key: str, output_file: str = None):
    """Process a sample of images and generate descriptions."""
    results = []
    demo_dir = Path(__file__).parent.resolve()

    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {image_path}")

        try:
            result = generate_functional_description(image_path, api_key)
            result["image_path"] = _to_demo_relative(Path(image_path), demo_dir)
            results.append(result)

            print("\nFunctional Description:")
            print("=" * 80)
            print(result["functional_description"])
            print("=" * 80)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append(
                {"image_path": _to_demo_relative(Path(image_path), demo_dir), "error": str(e)}
            )

    # Save results if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {output_file}")

    return results


def select_diverse_samples(data_dir: Path, samples_per_site: int = 2) -> list:
    """Select diverse sample images from the dataset."""
    samples = []

    # Get all site directories
    site_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    for site_dir in sorted(site_dirs)[:5]:  # Test on first 5 sites
        images = sorted(site_dir.glob("*.png"))

        if not images:
            continue

        # Sample: homepage and one other page
        if len(images) >= samples_per_site:
            # Try to get homepage and pricing/docs page
            homepage = None
            other = None

            for img in images:
                if "homepage" in img.name or "home" in img.name:
                    homepage = img
                elif "pricing" in img.name or "docs" in img.name:
                    other = img

            if homepage:
                samples.append(homepage)
            if other:
                samples.append(other)
            elif not homepage:
                # If no homepage, just take first two
                samples.extend(images[:samples_per_site])
        else:
            samples.extend(images)

    return samples[:10]  # Limit to 10 samples for testing


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        exit(1)

    # Select sample images
    data_dir = Path(__file__).parent / "data"
    sample_images = select_diverse_samples(data_dir)

    print(f"Selected {len(sample_images)} sample images for testing:")
    for img in sample_images:
        print(f"  - {img.parent.name}/{img.name}")

    # Process samples
    output_file = Path(__file__).parent / "sample_functional_descriptions.json"
    results = process_sample_images(sample_images, api_key, str(output_file))

    print(f"\n{'=' * 80}")
    print("EVALUATION CHECKLIST:")
    print(f"{'=' * 80}")
    print("For each description above, verify:")
    print("  1. ✓ Describes WHAT is on the page (content, structure)")
    print("  2. ✓ Does NOT mention colors, fonts, spacing, visual design")
    print("  3. ✓ Sufficient detail to reconstruct page functionality")
    print("  4. ✓ Focuses on purpose and content, not aesthetics")
