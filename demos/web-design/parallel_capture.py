"""
Parallel screenshot capture for multiple websites.
"""

import asyncio
from pathlib import Path

from capture_screenshots import capture_all
from discover_urls import discover_urls

SITES = [
    {"name": "supabase", "url": "https://supabase.com", "max_pages": 40},
    {"name": "railway", "url": "https://railway.app", "max_pages": 40},
    {"name": "render", "url": "https://render.com", "max_pages": 40},
    {"name": "flyio", "url": "https://fly.io", "max_pages": 40},
    {"name": "neon", "url": "https://neon.tech", "max_pages": 40},
    {"name": "planetscale", "url": "https://planetscale.com", "max_pages": 40},
    {"name": "turso", "url": "https://turso.tech", "max_pages": 40},
    {"name": "clerk", "url": "https://clerk.com", "max_pages": 40},
    {"name": "workos", "url": "https://workos.com", "max_pages": 40},
    {"name": "resend", "url": "https://resend.com", "max_pages": 40},
    {"name": "sentry", "url": "https://sentry.io", "max_pages": 40},
    {"name": "posthog", "url": "https://posthog.com", "max_pages": 40},
    {"name": "astral", "url": "https://astral.sh", "max_pages": 40},
]


async def process_site(site):
    """Discover and capture screenshots for a single site."""
    name = site["name"]
    url = site["url"]
    max_pages = site["max_pages"]

    print(f"\n[{name.upper()}] Starting discovery from {url}")

    try:
        # Discover URLs
        urls = await discover_urls(url, max_pages)
        print(f"[{name.upper()}] Discovered {len(urls)} URLs")

        if not urls:
            print(f"[{name.upper()}] No URLs discovered, skipping capture")
            return {"name": name, "count": 0, "success": False}

        # Capture screenshots
        output_dir = Path(__file__).parent / "data" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{name.upper()}] Starting screenshot capture")
        results = await capture_all(urls, output_dir, name)

        print(f"[{name.upper()}] ✓ Captured {len(results)} screenshots")
        return {"name": name, "count": len(results), "success": True}

    except Exception as e:
        print(f"[{name.upper()}] ✗ Error: {e}")
        return {"name": name, "count": 0, "success": False, "error": str(e)}


async def process_all_sites(sites):
    """Process all sites in parallel."""
    # Process sites in batches to avoid overwhelming the system
    batch_size = 4
    all_results = []

    for i in range(0, len(sites), batch_size):
        batch = sites[i : i + batch_size]
        print(f"\n{'=' * 80}")
        print(f"Processing batch {i // batch_size + 1}: {', '.join(s['name'] for s in batch)}")
        print(f"{'=' * 80}")

        tasks = [process_site(site) for site in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results.extend(results)

        # Small delay between batches
        if i + batch_size < len(sites):
            await asyncio.sleep(2)

    return all_results


if __name__ == "__main__":
    print("Starting parallel screenshot capture...")
    print(f"Total sites: {len(SITES)}")

    results = asyncio.run(process_all_sites(SITES))

    # Print summary
    print("\n" + "=" * 80)
    print("CAPTURE SUMMARY")
    print("=" * 80)

    successful = [r for r in results if not isinstance(r, Exception) and r.get("success")]
    failed = [r for r in results if isinstance(r, Exception) or not r.get("success")]

    for result in successful:
        print(f"✓ {result['name']:15s} - {result['count']:3d} screenshots")

    if failed:
        print("\nFailed:")
        for result in failed:
            if isinstance(result, Exception):
                print(f"✗ Error: {result}")
            else:
                print(f"✗ {result['name']:15s} - {result.get('error', 'Unknown error')}")

    total_screenshots = sum(r.get("count", 0) for r in results if not isinstance(r, Exception))
    print(f"\nTotal new screenshots: {total_screenshots}")
