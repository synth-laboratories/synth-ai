"""
Screenshot capture tool for web design training data.
Captures full-page screenshots from a list of URLs.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright


async def capture_screenshot(page, url: str, output_dir: Path, name: str = None):
    """Capture a full-page screenshot of a URL."""
    try:
        print(f"Capturing: {url}")

        # Navigate to the URL
        await page.goto(url, wait_until="load", timeout=60000)

        # Additional wait for any animations or lazy-loaded content
        await page.wait_for_timeout(3000)

        # Generate filename
        if name is None:
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").replace("/", "_")
            name = path_parts if path_parts else "homepage"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = output_dir / filename

        # Take full-page screenshot
        await page.screenshot(path=str(filepath), full_page=True)

        print(f"  ✓ Saved: {filename}")
        demo_dir = Path(__file__).parent.resolve()
        try:
            return str(filepath.resolve().relative_to(demo_dir))
        except ValueError:
            return str(filepath)

    except Exception as e:
        print(f"  ✗ Error capturing {url}: {e}")
        return None


async def capture_all(urls: list, output_dir: Path, site_name: str = "site"):
    """Capture screenshots for all URLs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        )
        page = await context.new_page()

        results = []
        for item in urls:
            if isinstance(item, dict):
                url = item.get("url")
                name = item.get("name")
            else:
                url = item
                name = None

            result = await capture_screenshot(page, url, output_dir, name)
            if result:
                results.append({"url": url, "file": result})

        await browser.close()

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {"site": site_name, "captured_at": datetime.now().isoformat(), "screenshots": results},
            f,
            indent=2,
        )

    print(f"\n✓ Captured {len(results)} screenshots")
    print(f"✓ Manifest saved to: {manifest_path}")

    return results


if __name__ == "__main__":
    import sys

    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = Path(__file__).parent / "data" / "autumn"

    # Load URLs from discovered_urls.json if it exists, otherwise use defaults
    urls_file = Path(__file__).parent / "discovered_urls.json"
    if urls_file.exists():
        with open(urls_file) as f:
            urls_data = json.load(f)
        print(f"Loaded {len(urls_data)} URLs from {urls_file}")
    else:
        # Default fallback URLs
        urls_data = [
            {"url": "https://useautumn.com/", "name": "homepage"},
        ]
        print("Using default URLs")

    print("Screenshot capture starting...")
    print(f"Output directory: {output_path}")
    print(f"URLs to capture: {len(urls_data)}\n")

    asyncio.run(capture_all(urls_data, output_path, "autumn"))
