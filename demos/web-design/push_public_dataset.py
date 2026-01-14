#!/usr/bin/env python3
"""
Push the Web Design demo dataset to the Hugging Face Hub.

This is intentionally "local-only" automation:
- It reads the token from the HF_TOKEN environment variable
- It does NOT print the token
- It builds the dataset using `create_hf_dataset.py` logic and pushes it public
"""

import argparse
import os
from pathlib import Path
from typing import Final

from create_hf_dataset import create_dataset, push_to_hub
from huggingface_hub import HfApi

DEFAULT_REPO_ID: Final[str] = "synth-laboratories/web-design-screenshots"


def main() -> int:
    parser = argparse.ArgumentParser(description="Push web-design dataset to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id to push to (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--descriptions-file",
        type=str,
        default=str(Path(__file__).parent / "all_functional_descriptions.json"),
        help="JSON file containing functional descriptions",
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
    parser.add_argument(
        "--whoami",
        action="store_true",
        help="Print Hugging Face identity (username + orgs) for the loaded token, then exit.",
    )
    args = parser.parse_args()

    hf_token = (os.environ.get("HF_TOKEN") or "").strip()
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found.\n"
            "Set HF_TOKEN in your environment before running this script.\n"
        )

    if args.whoami:
        info = HfApi().whoami(token=hf_token)
        username = info.get("name") or info.get("user") or "unknown"
        orgs = info.get("orgs") or []
        print(f"HF username: {username}")
        print(f"HF orgs: {', '.join(orgs) if orgs else '(none)'}")
        return 0

    descriptions_file = Path(args.descriptions_file).expanduser()
    images_root = Path(args.images_root).expanduser() if args.images_root else None

    # Build dataset and push public. Do not print token.
    dataset = create_dataset(
        descriptions_file=descriptions_file,
        images_root=images_root,
        resize_size=int(args.resize_size),
        max_image_pixels=int(args.max_image_pixels),
        slice_height=int(args.slice_height) if args.slice_height else None,
        slice_overlap=int(args.slice_overlap),
    )
    push_to_hub(dataset, repo_name=args.repo_id, token=hf_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
