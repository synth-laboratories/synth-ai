"""
Generate functional descriptions with proper rate limiting.
"""

import asyncio
import json
import os
import time
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


class RateLimiter:
    """Rate limiter that ensures we don't exceed API quota."""

    def __init__(self, max_requests_per_minute: int = 9):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to stay within rate limit."""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [r for r in self.requests if now - r < 60]

            if len(self.requests) >= self.max_requests:
                # Wait until oldest request is more than 1 minute old
                sleep_time = 60 - (now - self.requests[0]) + 1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Clean up again after sleep
                    now = time.time()
                    self.requests = [r for r in self.requests if now - r < 60]

            self.requests.append(time.time())


async def generate_functional_description(
    image_path: str, api_key: str, rate_limiter: RateLimiter, model_name: str = "gemini-2.5-flash"
) -> dict:
    """Generate functional description from screenshot using Gemini."""
    await rate_limiter.acquire()

    # Configure the API
    genai.configure(api_key=api_key)

    # Create model - rotate between different models for better quota
    model = genai.GenerativeModel(model_name)

    # Load image
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Create the prompt with image
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
        "model": model_name,
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


def _resolve_image_path(path_str: str, demo_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (demo_dir / path).resolve()


def _to_demo_relative(path: Path, demo_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(demo_dir))
    except ValueError:
        return str(path)


async def main():
    # Get API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        exit(1)

    # Find all screenshots
    demo_dir = Path(__file__).parent.resolve()
    data_dir = demo_dir / "data"
    screenshots = find_all_screenshots(data_dir)

    print(f"Found {len(screenshots)} screenshots")

    # Check if we already have descriptions
    output_file = Path(__file__).parent / "all_functional_descriptions.json"
    if output_file.exists():
        with open(output_file) as f:
            existing_results = json.load(f)
        existing_paths = set()
        for row in existing_results:
            image_path = row.get("image_path")
            if not image_path:
                continue
            try:
                existing_paths.add(_resolve_image_path(image_path, demo_dir))
                row["image_path"] = _to_demo_relative(
                    _resolve_image_path(image_path, demo_dir), demo_dir
                )
            except Exception:
                continue
        print(f"Found {len(existing_results)} existing descriptions")
    else:
        existing_results = []
        existing_paths = set()

    # Filter out already processed
    screenshots_to_process = [s for s in screenshots if s.resolve() not in existing_paths]
    print(f"Need to process {len(screenshots_to_process)} new screenshots")

    if not screenshots_to_process:
        print("All screenshots already have descriptions!")
        return

    # Rotate between different models to spread quota
    models = ["gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp"]

    # Create rate limiter (50 requests per minute total across all models)
    rate_limiter = RateLimiter(max_requests_per_minute=50)

    # Process all with concurrency but rate limited
    results = existing_results.copy()
    failed = []

    async def process_one(img_path, idx):
        # Rotate through models
        model_name = models[idx % len(models)]
        try:
            result = await generate_functional_description(
                str(img_path), api_key, rate_limiter, model_name
            )
            result["image_path"] = _to_demo_relative(img_path, demo_dir)
            results.append(result)

            # Save incrementally every 10 images
            if len(results) % 10 == 0:
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)

            return result
        except Exception as e:
            error = {"image_path": _to_demo_relative(img_path, demo_dir), "error": str(e)}
            failed.append(error)
            return None

    # Process with limited concurrency (5 concurrent tasks max)
    semaphore = asyncio.Semaphore(5)

    async def process_with_semaphore(img, idx):
        async with semaphore:
            return await process_one(img, idx)

    tasks = [process_with_semaphore(img, i) for i, img in enumerate(screenshots_to_process)]
    await async_tqdm.gather(*tasks, desc="Generating descriptions")

    # Final save
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Generated {len(results)} functional descriptions")
    print(f"✓ Saved to {output_file}")

    if failed:
        print(f"\n⚠ {len(failed)} images failed:")
        for f in failed[:10]:
            print(f"  - {Path(f['image_path']).name}: {f['error']}")

        failed_file = Path(__file__).parent / "failed_descriptions.json"
        with open(failed_file, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"✓ Failed images saved to {failed_file}")


if __name__ == "__main__":
    asyncio.run(main())
