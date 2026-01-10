"""
URL discovery tool for capturing all pages from a website.
Crawls a site to find all internal URLs.
"""

import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright


async def discover_urls(base_url: str, max_pages: int = 50):
    """Discover all URLs on a website by crawling."""
    visited = set()
    to_visit = {base_url}
    discovered = []

    base_domain = urlparse(base_url).netloc

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop()

            if url in visited:
                continue

            try:
                print(f"Crawling: {url}")
                await page.goto(url, wait_until="load", timeout=60000)

                visited.add(url)

                # Get page title for naming
                title = await page.title()
                parsed = urlparse(url)
                path = parsed.path.strip("/").replace("/", "_") or "homepage"

                discovered.append({"url": url, "name": path, "title": title})

                # Find all links on the page
                links = await page.eval_on_selector_all(
                    "a[href]", "(elements) => elements.map(e => e.href)"
                )

                # Filter for internal links
                for link in links:
                    parsed_link = urlparse(link)

                    # Same domain, not already visited, not a file download
                    if (
                        parsed_link.netloc == base_domain
                        and link not in visited
                        and not any(link.endswith(ext) for ext in [".pdf", ".zip", ".jpg", ".png"])
                    ):
                        # Remove fragments and query params for deduplication
                        clean_url = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"
                        if clean_url not in visited:
                            to_visit.add(clean_url)

            except Exception as e:
                print(f"  Error crawling {url}: {e}")

        await browser.close()

    print(f"\n✓ Discovered {len(discovered)} URLs")
    return discovered


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.autumnai.com"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"Discovering URLs from: {base_url}")
    print(f"Max pages: {max_pages}\n")

    urls = asyncio.run(discover_urls(base_url, max_pages))

    # Save to JSON
    output_file = Path(__file__).parent / "discovered_urls.json"
    with open(output_file, "w") as f:
        json.dump(urls, f, indent=2)

    print(f"✓ Saved URLs to: {output_file}")
    print("\nDiscovered URLs:")
    for item in urls:
        print(f"  - {item['url']}")
