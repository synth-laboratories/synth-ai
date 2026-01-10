"""
Capture screenshot of generated HTML and create a side-by-side comparison.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


async def capture_generated_page(html_file: Path, output_path: Path):
    """Capture screenshot of generated HTML page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        # Load the HTML file
        await page.goto(f"file://{html_file.absolute()}")

        # Wait for page to render
        await page.wait_for_timeout(2000)

        # Take full page screenshot
        await page.screenshot(path=str(output_path), full_page=True)

        await browser.close()
        print(f"✓ Captured screenshot: {output_path}")


async def main():
    # Find generated HTML files
    generated_dir = Path(__file__).parent / "generated_pages"
    html_files = list(generated_dir.glob("*.html"))

    if not html_files:
        print("No generated HTML files found!")
        return

    print("Generated HTML files:")
    for i, html_file in enumerate(html_files):
        print(f"  [{i}] {html_file.name}")

    choice = input("\nSelect index to capture (default 0): ").strip()
    index = int(choice) if choice else 0

    html_file = html_files[index]

    # Create output directory
    output_dir = Path(__file__).parent / "comparisons"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"generated_{html_file.stem}.png"

    print(f"\nCapturing screenshot of: {html_file.name}")
    await capture_generated_page(html_file, output_path)

    # Find original screenshot
    # Parse the filename to find original
    parts = html_file.stem.split("_", 1)
    if len(parts) == 2:
        site_name = parts[0]
        original_name = parts[1] + ".png"

        original_path = Path(__file__).parent / "data" / site_name / original_name

        if original_path.exists():
            print(f"\n{'=' * 80}")
            print("COMPARISON:")
            print(f"{'=' * 80}")
            print(f"Original:  {original_path}")
            print(f"Generated: {output_path}")
            print("\nOpen both images to compare:")
            print(f"  open {original_path}")
            print(f"  open {output_path}")
        else:
            print(f"\nCould not find original screenshot at: {original_path}")


if __name__ == "__main__":
    asyncio.run(main())
