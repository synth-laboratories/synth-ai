"""
Generate functional descriptions for ALL screenshots in parallel.
"""

import asyncio
import json
import os
from pathlib import Path

import google.generativeai as genai
from tqdm.asyncio import tqdm as async_tqdm

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


async def generate_functional_description(
    image_path: str, api_key: str, semaphore: asyncio.Semaphore
) -> dict:
    """Generate functional description from screenshot using Gemini."""
    async with semaphore:
        # Configure the API (thread-safe)
        genai.configure(api_key=api_key)

        # Create model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Load image
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Create the prompt with image
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                [FUNCTIONAL_DESCRIPTION_PROMPT, {"mime_type": "image/png", "data": image_data}]
            ),
        )

        return {
            "image_path": str(image_path),
            "functional_description": response.text,
            "model": "gemini-2.0-flash-exp",
        }


def find_all_screenshots(data_dir: Path) -> list:
    """Find all screenshots in the data directory."""
    screenshots = []
    for site_dir in sorted(data_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        for img in sorted(site_dir.glob("*.png")):
            screenshots.append(img)
    return screenshots


async def process_batch(
    screenshots: list,
    api_key: str,
    results: list,
    failed: list,
    output_file: Path,
    max_concurrent: int = 10,
):
    """Process a batch of screenshots in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(img_path):
        try:
            result = await generate_functional_description(str(img_path), api_key, semaphore)
            results.append(result)
            return result
        except Exception as e:
            error = {"image_path": str(img_path), "error": str(e)}
            failed.append(error)
            return None

    # Process all with progress bar
    tasks = [process_one(img) for img in screenshots]
    await async_tqdm.gather(*tasks, desc="Generating descriptions")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


async def main():
    # Get API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        exit(1)

    # Find all screenshots
    data_dir = Path(__file__).parent / "data"
    screenshots = find_all_screenshots(data_dir)

    print(f"Found {len(screenshots)} screenshots")

    # Check if we already have descriptions
    output_file = Path(__file__).parent / "all_functional_descriptions.json"
    if output_file.exists():
        with open(output_file) as f:
            existing_results = json.load(f)
        existing_paths = {r["image_path"] for r in existing_results}
        print(f"Found {len(existing_results)} existing descriptions")
    else:
        existing_results = []
        existing_paths = set()

    # Filter out already processed
    screenshots_to_process = [s for s in screenshots if str(s) not in existing_paths]
    print(f"Need to process {len(screenshots_to_process)} new screenshots")

    if not screenshots_to_process:
        print("All screenshots already have descriptions!")
        return

    # Process in parallel with concurrency limit
    results = existing_results.copy()
    failed = []

    # Process in batches to avoid overwhelming memory
    batch_size = 100
    max_concurrent = 20  # Adjust based on API rate limits

    for i in range(0, len(screenshots_to_process), batch_size):
        batch = screenshots_to_process[i : i + batch_size]
        print(
            f"\nProcessing batch {i // batch_size + 1}/{(len(screenshots_to_process) + batch_size - 1) // batch_size}"
        )
        print(f"Batch size: {len(batch)}, Concurrency: {max_concurrent}")

        await process_batch(batch, api_key, results, failed, output_file, max_concurrent)

        print(f"Progress: {len(results)}/{len(screenshots)} total")

    print(f"\n✓ Generated {len(results)} functional descriptions")
    print(f"✓ Saved to {output_file}")

    if failed:
        print(f"\n⚠ {len(failed)} images failed:")
        for f in failed[:5]:
            print(f"  - {f['image_path']}: {f['error']}")

        failed_file = Path(__file__).parent / "failed_descriptions.json"
        with open(failed_file, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"✓ Failed images saved to {failed_file}")


if __name__ == "__main__":
    asyncio.run(main())
