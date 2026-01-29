#!/usr/bin/env python3
"""Run the full MTG artist style matching demo.

This runs:
1. Verifier optimization (Graph Evolve) - creates a verifier that penalizes artist name mentions
2. Prompt optimization (GEPA) - optimizes prompts to match style without naming the artist

Usage:
    uv run python demos/mtg_artist_style/run_demo.py --artist seb_mckinnon
    uv run python demos/mtg_artist_style/run_demo.py --artist seb_mckinnon --local
    uv run python demos/mtg_artist_style/run_demo.py --artist seb_mckinnon --skip-verifier-opt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

demo_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Run MTG artist style matching demo")
parser.add_argument(
    "--artist",
    type=str,
    default="seb_mckinnon",
    help="Artist key (see README for full list of 18 artists)",
)
parser.add_argument(
    "--local",
    action="store_true",
    help="Run in local mode: use localhost:8000 backend",
)
parser.add_argument(
    "--skip-verifier-opt",
    action="store_true",
    help="Skip verifier optimization (use existing artifact)",
)
parser.add_argument(
    "--skip-prompt-opt",
    action="store_true",
    help="Skip prompt optimization (only run verifier opt)",
)
parser.add_argument(
    "--fetch-images",
    action="store_true",
    help="Fetch artist card images before running (if not already fetched)",
)
args = parser.parse_args()


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and stream output."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()
    return process.returncode


def main() -> None:
    print("=" * 60)
    print("MTG Artist Style Matching Demo")
    print("=" * 60)

    # Check if metadata exists
    metadata_path = demo_dir / "artist_metadata.json"
    if not metadata_path.exists() or args.fetch_images:
        print("\nFetching artist card images...")
        ret = run_command(
            [sys.executable, str(demo_dir / "fetch_artist_cards.py")],
            "Fetch artist card images from Scryfall",
        )
        if ret != 0:
            print(f"ERROR: fetch_artist_cards.py failed with code {ret}")
            sys.exit(1)

    # Load metadata to verify artist exists
    with open(metadata_path) as f:
        metadata = json.load(f)

    if args.artist not in metadata["artists"]:
        available = list(metadata["artists"].keys())
        print(f"ERROR: Unknown artist '{args.artist}'")
        print(f"Available artists: {available}")
        sys.exit(1)

    artist_info = metadata["artists"][args.artist]
    print(f"\nArtist: {artist_info['name']}")
    print(f"Style: {artist_info['style_description']}")
    print(f"Cards: {artist_info['num_cards']}")

    # Build common args
    common_args = ["--artist", args.artist]
    if args.local:
        common_args.append("--local")

    # Step 1: Verifier optimization
    verifier_artifact = demo_dir / "artifacts" / args.artist / "verifier_opt.json"
    if args.skip_verifier_opt and verifier_artifact.exists():
        print(f"\nSkipping verifier optimization (using existing artifact)")
    else:
        ret = run_command(
            [sys.executable, str(demo_dir / "run_verifier_opt.py")] + common_args,
            f"Verifier Optimization for {artist_info['name']}",
        )
        if ret != 0:
            print(f"ERROR: Verifier optimization failed with code {ret}")
            sys.exit(1)

    # Step 2: Prompt optimization (GEPA)
    if args.skip_prompt_opt:
        print("\nSkipping prompt optimization")
    else:
        ret = run_command(
            [sys.executable, str(demo_dir / "run_prompt_opt.py")] + common_args,
            f"GEPA Prompt Optimization for {artist_info['name']}",
        )
        if ret != 0:
            print(f"ERROR: Prompt optimization failed with code {ret}")
            sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    artifacts_dir = demo_dir / "artifacts" / args.artist
    if verifier_artifact.exists():
        with open(verifier_artifact) as f:
            v_data = json.load(f)
        print(f"\nVerifier:")
        print(f"  Graph ID: {v_data.get('graph_id')}")
        print(f"  Best Score: {v_data.get('best_score')}")

    prompt_artifact = artifacts_dir / "prompt_opt.json"
    if prompt_artifact.exists():
        with open(prompt_artifact) as f:
            p_data = json.load(f)
        print(f"\nPrompt Optimization:")
        print(f"  Job ID: {p_data.get('gepa_job_id')}")
        print(f"  Best Score: {p_data.get('best_score')}")
        print(f"  Contains Artist Name: {p_data.get('contains_forbidden_pattern')}")

        if p_data.get("contains_forbidden_pattern"):
            print("\n  ⚠️  WARNING: Optimized prompt still mentions the artist name!")
        else:
            print("\n  ✅ SUCCESS: Prompt captures style without naming the artist!")

    print(f"\nArtifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
