"""
Create Hugging Face dataset from functional descriptions and screenshots.

Dataset structure:
- Input: functional_description (string)
- Output: image (PIL Image)
"""
import json
import re
from datetime import datetime
from pathlib import Path
from datasets import Dataset, Features, Value, Image as HFImage
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
    if match:
        page_path = match.group(1)
    else:
        page_path = filename.replace(".png", "")

    # Map common page types
    if page_path == "homepage" or page_path == site_name:
        return base_url
    elif "_" in page_path:
        # Convert underscores to slashes for paths like "docs_api_reference"
        page_path = page_path.replace("_", "/")

    return f"{base_url}/{page_path}"


def extract_date_from_filename(filename: str) -> str:
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


def create_dataset():
    """Create Hugging Face dataset from functional descriptions and screenshots."""

    # Load functional descriptions
    descriptions_file = Path(__file__).parent / "all_functional_descriptions.json"

    if not descriptions_file.exists():
        print(f"Error: {descriptions_file} not found!")
        print("Run generate_all_descriptions.py first")
        exit(1)

    with open(descriptions_file, 'r') as f:
        descriptions_data = json.load(f)

    print(f"Loaded {len(descriptions_data)} functional descriptions")

    # Create dataset rows
    dataset_rows = []

    for item in descriptions_data:
        image_path = Path(item["image_path"])

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

        dataset_rows.append({
            "functional_description": item["functional_description"],
            "image": str(image_path),
            "site_name": site_name,
            "page_name": page_name,
            "url": url,
            "capture_date": capture_date,
            "model_used": item.get("model", "gemini-2.0-flash-exp")
        })

    print(f"Created {len(dataset_rows)} dataset rows")

    # Define features schema
    features = Features({
        "functional_description": Value("string"),
        "image": HFImage(),
        "site_name": Value("string"),
        "page_name": Value("string"),
        "url": Value("string"),
        "capture_date": Value("string"),
        "model_used": Value("string")
    })

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
        features=features
    )

    print(f"\nDataset created with {len(dataset)} examples")
    print(f"Features: {dataset.features}")

    # Save to disk
    output_dir = Path(__file__).parent / "hf_dataset"
    dataset.save_to_disk(str(output_dir))

    print(f"\n✓ Dataset saved to: {output_dir}")
    print(f"\nDataset info:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Sites represented: {len(set(dataset['site_name']))}")
    print(f"\nExample row:")
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
        private=False  # Set to True if you want a private dataset
    )

    print(f"✓ Dataset pushed to: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Hugging Face dataset")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--repo-name", type=str, help="HF repo name (e.g., username/dataset-name)")
    parser.add_argument("--token", type=str, help="HF token (or set HF_TOKEN env var)")

    args = parser.parse_args()

    # Create dataset
    dataset = create_dataset()

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
