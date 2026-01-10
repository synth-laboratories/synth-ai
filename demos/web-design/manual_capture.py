"""
Manual URL capture for sites that need specific URLs defined.
"""

import asyncio
from pathlib import Path

from capture_screenshots import capture_all

# Manually curated URLs for Kernel
KERNEL_URLS = [
    {"url": "https://www.kernel.sh", "name": "homepage"},
    {"url": "https://www.kernel.sh/docs", "name": "docs"},
    {"url": "https://www.kernel.sh/pricing", "name": "pricing"},
    {"url": "https://www.kernel.sh/about", "name": "about"},
    {"url": "https://www.kernel.sh/blog", "name": "blog"},
    {"url": "https://www.kernel.sh/features", "name": "features"},
    {"url": "https://www.kernel.sh/solutions", "name": "solutions"},
    {"url": "https://www.kernel.sh/customers", "name": "customers"},
    {"url": "https://www.kernel.sh/integrations", "name": "integrations"},
    {"url": "https://www.kernel.sh/security", "name": "security"},
    {"url": "https://www.kernel.sh/changelog", "name": "changelog"},
    {"url": "https://www.kernel.sh/contact", "name": "contact"},
    {"url": "https://www.kernel.sh/login", "name": "login"},
    {"url": "https://www.kernel.sh/signup", "name": "signup"},
    {"url": "https://www.kernel.sh/demo", "name": "demo"},
]


if __name__ == "__main__":
    urls = KERNEL_URLS
    output_path = Path(__file__).parent / "data" / "kernel"

    print("Manual screenshot capture starting...")
    print(f"Output directory: {output_path}")
    print(f"URLs to try: {len(urls)}\n")

    asyncio.run(capture_all(urls, output_path, "kernel"))
