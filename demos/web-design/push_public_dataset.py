#!/usr/bin/env python3
"""
Push the Web Design demo dataset to the Hugging Face Hub using a token loaded from a dotenv file.

This is intentionally "local-only" automation:
- It reads your token from a dotenv file path you control (default: monorepo/.env.dev)
- It does NOT print the token
- It builds the dataset using `create_hf_dataset.py` logic and pushes it public
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Final

from create_hf_dataset import create_dataset, push_to_hub
from dotenv import dotenv_values
from huggingface_hub import HfApi

DEFAULT_ENV_FILE: Final[Path] = Path(
    "/Users/joshpurtell/Documents/GitHub/monorepo/backend/.env.dev"
)
DEFAULT_REPO_ID: Final[str] = "synth-laboratories/web-design-screenshots"

TOKEN_KEYS: Final[tuple[str, ...]] = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


def _load_hf_token(env_file: Path | None) -> str | None:
    """
    Load a token from a dotenv file.

    Note: we intentionally don't print values from the dotenv file.
    """
    if env_file is None:
        return None

    if not env_file.exists():
        return None

    values = dotenv_values(env_file)
    for key in TOKEN_KEYS:
        token = (values.get(key) or "").strip()
        if token:
            return token

    return None


def _candidate_env_files(primary: Path) -> list[Path]:
    """Try a few common monorepo locations to reduce footguns."""
    candidates = [primary]
    # Back-compat with older location
    candidates.append(Path("/Users/joshpurtell/Documents/GitHub/monorepo/.env.dev"))
    # Relative to synth-ai repo (../monorepo/backend/.env.dev)
    candidates.append(Path(__file__).resolve().parents[3] / "monorepo" / "backend" / ".env.dev")
    # Relative to synth-ai repo (../monorepo/.env.dev)
    candidates.append(Path(__file__).resolve().parents[3] / "monorepo" / ".env.dev")

    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            rp = p.expanduser()
        if rp not in seen:
            out.append(rp)
            seen.add(rp)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Push web-design dataset to Hugging Face Hub")
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(DEFAULT_ENV_FILE),
        help=f"dotenv file to load token from (default: {DEFAULT_ENV_FILE})",
    )
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

    env_file = Path(args.env_file).expanduser()
    hf_token = None
    for candidate in _candidate_env_files(env_file):
        hf_token = _load_hf_token(candidate)
        if hf_token:
            break

    hf_token = hf_token or os.environ.get("HF_TOKEN", "").strip()

    if not hf_token:
        keys = ", ".join(TOKEN_KEYS)
        raise RuntimeError(
            f"No Hugging Face token found.\n"
            f"- Looked in: {', '.join(str(p) for p in _candidate_env_files(env_file))}\n"
            f"- Tried keys: {keys}\n"
            f"- Also checked current environment for HF_TOKEN\n"
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
