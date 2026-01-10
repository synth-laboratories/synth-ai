"""
Capture HTML source code from original webpages.
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def capture_html(url: str) -> str:
    """Capture the full HTML source of a webpage."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        try:
            # Load the page
            await page.goto(url, wait_until="load", timeout=60000)
            await page.wait_for_timeout(3000)

            # Get the HTML content
            html = await page.content()

            await browser.close()
            return html
        except Exception as e:
            await browser.close()
            raise e


async def process_screenshot_to_html_mapping():
    """Create mapping from screenshots to their HTML source."""
    data_dir = Path(__file__).parent / "data"

    # Load existing screenshots metadata if available
    results = []

    # We need to reconstruct URLs from screenshot filenames
    # This is a mapping of site names to base URLs
    site_urls = {
        "astral": "https://astral.sh",
        "autumn": "https://useautumn.com",
        "clerk": "https://clerk.com",
        "flyio": "https://fly.io",
        "kernel": "https://www.kernel.sh",
        "mintlify": "https://mintlify.com",
        "modal": "https://modal.com",
        "vercel": "https://vercel.com",
        "supabase": "https://supabase.com",
        "railway": "https://railway.app",
        "render": "https://render.com",
        "neon": "https://neon.tech",
        "planetscale": "https://planetscale.com",
        "turso": "https://turso.tech",
        "workos": "https://workos.com",
        "resend": "https://resend.com",
        "sentry": "https://sentry.io",
        "posthog": "https://posthog.com",
    }

    for site_name, base_url in site_urls.items():
        site_dir = data_dir / site_name
        if not site_dir.exists():
            continue

        # Look for homepage and pricing pages (most common)
        for screenshot in site_dir.glob("*.png"):
            filename = screenshot.stem

            # Try to determine the URL from the filename
            if "homepage" in filename or filename == f"{site_name}_":
                url = base_url
            elif "pricing" in filename:
                url = f"{base_url}/pricing"
            elif "docs" in filename:
                # Skip complex docs pages for now
                continue
            else:
                # Try to parse the URL from filename
                # This is imperfect, we'll handle common cases
                continue

            results.append(
                {"site": site_name, "screenshot": str(screenshot), "url": url, "filename": filename}
            )

    return results


async def capture_all_html():
    """Capture HTML for all mapped screenshots."""
    mappings = await process_screenshot_to_html_mapping()

    output_dir = Path(__file__).parent / "html_sources"
    output_dir.mkdir(exist_ok=True)

    results = []

    for i, mapping in enumerate(mappings, 1):
        print(f"\n[{i}/{len(mappings)}] Capturing: {mapping['site']} - {mapping['url']}")

        try:
            html = await capture_html(mapping["url"])

            # Save HTML
            output_file = output_dir / f"{mapping['site']}_{mapping['filename']}.html"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html)

            results.append(
                {
                    **mapping,
                    "html_file": str(output_file),
                    "html_length": len(html),
                    "success": True,
                }
            )

            print(f"  ✓ Captured {len(html)} characters")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({**mapping, "error": str(e), "success": False})

        # Small delay between requests
        await asyncio.sleep(2)

    # Save mapping
    mapping_file = Path(__file__).parent / "html_source_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved mapping to {mapping_file}")

    successful = sum(1 for r in results if r.get("success"))
    print(f"\nCaptured {successful}/{len(results)} HTML sources successfully")

    return results


if __name__ == "__main__":
    asyncio.run(capture_all_html())
