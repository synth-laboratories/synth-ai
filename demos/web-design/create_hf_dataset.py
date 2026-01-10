"""
Create a Hugging Face dataset from functional descriptions + screenshots.

This is meant for *publishing* the demo dataset to the Hugging Face Hub, so that
the demo can download from a public dataset repo instead of using git-committed
screenshots.

Dataset schema:
- functional_description (string)
- image (Image)
- site_name (string)
- page_name (string)
- url (string)
- capture_date (string, ISO8601 or empty)
- model_used (string)
"""

import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image

# Mapping of site names to base URLs
SITE_URLS = {
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


def extract_url_from_filename(site_name: str, filename: str) -> str:
    """Extract probable URL from site name and filename."""
    base_url = SITE_URLS.get(site_name, f"https://{site_name}.com")

    # Remove timestamp from filename
    # Pattern: {page}_{YYYYMMDD}_{HHMMSS}.png
    match = re.match(r"(.+?)_\d{8}_\d{6}", filename)
    page_path = match.group(1) if match else filename.replace(".png", "")

    # Map common page types
    if page_path == "homepage" or page_path == site_name:
        return base_url
    elif "_" in page_path:
        # Convert underscores to slashes for paths like "docs_api_reference"
        page_path = page_path.replace("_", "/")

    return f"{base_url}/{page_path}"


def extract_date_from_filename(filename: str) -> str | None:
    """Extract date from filename in format YYYYMMDD_HHMMSS."""
    match = re.search(r"_(\d{8})_(\d{6})\.png$", filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        try:
            # Parse and format as ISO 8601
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return dt.isoformat()
        except ValueError:
            return None
    return None


def create_dataset(
    descriptions_file: Path,
    images_root: Optional[Path] = None,
    *,
    resize_size: int = 384,
    max_image_pixels: int = 12_000_000,
    slice_height: int | None = None,
    slice_overlap: int = 256,
) -> Dataset:
    """Create Hugging Face dataset from functional descriptions and screenshots."""

    # Load functional descriptions
    if not descriptions_file.exists():
        print(f"Error: {descriptions_file} not found!")
        print("Provide --descriptions-file or generate descriptions first.")
        exit(1)

    with open(descriptions_file) as f:
        descriptions_data = json.load(f)

    print(f"Loaded {len(descriptions_data)} functional descriptions")

    # Create dataset rows
    dataset_rows = []

    for item in descriptions_data:
        raw_image_path = Path(item["image_path"])
        if raw_image_path.is_absolute():
            image_path = raw_image_path
        else:
            base = images_root or descriptions_file.parent
            image_path = (base / raw_image_path).resolve()

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Extract metadata
        site_name = image_path.parent.name
        page_name = image_path.stem
        filename = image_path.name

        # Extract URL and date
        url = extract_url_from_filename(site_name, filename)
        capture_date = extract_date_from_filename(filename)

        # Load + (optionally) slice + downsample image so the published HF dataset is safe by default.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(image_path) as img:
                    width, height = img.size
                    pixels = int(width) * int(height)
                    if pixels > max_image_pixels:
                        print(
                            f"Skipping oversized image: {site_name}/{filename} "
                            f"({width}x{height}={pixels:,} px; limit={max_image_pixels:,})"
                        )
                        continue

                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Slice very tall pages into multiple vertical segments to preserve readability after downsampling.
                    if slice_height and height > slice_height:
                        overlap = max(0, int(slice_overlap))
                        step = max(1, int(slice_height) - overlap)
                        y0 = 0
                        slices: list[Image.Image] = []
                        while y0 < height:
                            y1 = min(height, y0 + int(slice_height))
                            crop = img.crop((0, y0, width, y1))
                            crop.thumbnail((resize_size, resize_size), Image.Resampling.LANCZOS)
                            slices.append(crop.copy())
                            if y1 >= height:
                                break
                            y0 += step

                        slice_count = len(slices)
                        for idx, crop_img in enumerate(slices):
                            dataset_rows.append(
                                {
                                    "functional_description": item["functional_description"],
                                    "image": crop_img,
                                    "site_name": site_name,
                                    "page_name": f"{page_name}__slice_{idx + 1:02d}_of_{slice_count:02d}",
                                    "url": url,
                                    "capture_date": capture_date,
                                    "model_used": item.get("model", "gemini-2.0-flash-exp"),
                                }
                            )
                        continue

                    # Non-sliced: just downsample
                    img.thumbnail((resize_size, resize_size), Image.Resampling.LANCZOS)
                    img = img.copy()
        except (Image.DecompressionBombWarning, OSError) as e:
            print(f"Skipping unreadable/unsafe image: {site_name}/{filename} ({e})")
            continue

        dataset_rows.append(
            {
                "functional_description": item["functional_description"],
                "image": img,
                "site_name": site_name,
                "page_name": page_name,
                "url": url,
                "capture_date": capture_date,
                "model_used": item.get("model", "gemini-2.0-flash-exp"),
            }
        )

    print(f"Created {len(dataset_rows)} dataset rows")

    # Define features schema
    features = Features(
        {
            "functional_description": Value("string"),
            "image": HFImage(),
            "site_name": Value("string"),
            "page_name": Value("string"),
            "url": Value("string"),
            "capture_date": Value("string"),
            "model_used": Value("string"),
        }
    )

    # Create dataset
    dataset = Dataset.from_dict(
        {
            "functional_description": [row["functional_description"] for row in dataset_rows],
            "image": [row["image"] for row in dataset_rows],
            "site_name": [row["site_name"] for row in dataset_rows],
            "page_name": [row["page_name"] for row in dataset_rows],
            "url": [row["url"] for row in dataset_rows],
            "capture_date": [row["capture_date"] for row in dataset_rows],
            "model_used": [row["model_used"] for row in dataset_rows],
        },
        features=features,
    )

    print(f"\nDataset created with {len(dataset)} examples")
    print(f"Features: {dataset.features}")

    # Save to disk
    output_dir = Path(__file__).parent / "hf_dataset"
    dataset.save_to_disk(str(output_dir))

    print(f"\n✓ Dataset saved to: {output_dir}")
    print("\nDataset info:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Sites represented: {len(set(dataset['site_name']))}")
    print("\nExample row:")
    print(f"  Site: {dataset[0]['site_name']}")
    print(f"  Page: {dataset[0]['page_name']}")
    print(f"  URL: {dataset[0]['url']}")
    print(f"  Capture date: {dataset[0]['capture_date']}")
    print(f"  Model used: {dataset[0]['model_used']}")
    print(f"  Description length: {len(dataset[0]['functional_description'])} chars")

    return dataset


def push_to_hub(dataset, repo_name: str, token: str = None):
    """Push dataset to Hugging Face Hub."""
    print(f"\nPushing dataset to Hugging Face Hub: {repo_name}")

    dataset.push_to_hub(
        repo_name,
        token=token,
        private=False,  # Set to True if you want a private dataset
    )

    print(f"✓ Dataset pushed to: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Hugging Face dataset")
    parser.add_argument(
        "--descriptions-file",
        type=str,
        default=str(Path(__file__).parent / "all_functional_descriptions.json"),
        help="JSON file containing functional descriptions (default: demos/web-design/all_functional_descriptions.json)",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Optional base directory for relative image_path values inside the descriptions JSON.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=384,
        help="Resize images so their longest side is at most this many pixels (default: 384).",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=12_000_000,
        help="Skip images larger than this many total pixels (default: 12,000,000).",
    )
    parser.add_argument(
        "--slice-height",
        type=int,
        default=None,
        help="If set, slice images taller than this many pixels into multiple vertical segments before resizing.",
    )
    parser.add_argument(
        "--slice-overlap",
        type=int,
        default=256,
        help="Vertical overlap (pixels) between slices when --slice-height is set (default: 256).",
    )
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--repo-name", type=str, help="HF repo name (e.g., username/dataset-name)")
    parser.add_argument("--token", type=str, help="HF token (or set HF_TOKEN env var)")

    args = parser.parse_args()

    # Create dataset
    descriptions_file = Path(args.descriptions_file).expanduser()
    images_root = Path(args.images_root).expanduser() if args.images_root else None
    dataset = create_dataset(
        descriptions_file=descriptions_file,
        images_root=images_root,
        resize_size=int(args.resize_size),
        max_image_pixels=int(args.max_image_pixels),
        slice_height=int(args.slice_height) if args.slice_height else None,
        slice_overlap=int(args.slice_overlap),
    )

    # Optionally push to hub
    if args.push:
        if not args.repo_name:
            print("Error: --repo-name required when using --push")
            exit(1)

        import os

        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("Error: HF token required. Set --token or HF_TOKEN env var")
            exit(1)

        push_to_hub(dataset, args.repo_name, token)
