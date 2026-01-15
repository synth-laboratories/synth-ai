#!/usr/bin/env python3
"""
Create a Daytona snapshot with EngineBench dependencies pre-installed.

This builds a Docker image via Daytona and creates a snapshot from it.
Subsequent runs will use this snapshot for fast (~10s) sandbox startup.

Usage:
    export DAYTONA_API_KEY=...
    uv run python demos/engine_bench/create_daytona_snapshot.py

    # Or specify a custom snapshot name:
    uv run python demos/engine_bench/create_daytona_snapshot.py --name synth-engine-bench-codex-v2
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def main():
    parser = argparse.ArgumentParser(description="Create Daytona snapshot for EngineBench")
    parser.add_argument(
        "--name",
        type=str,
        default="synth-engine-bench-codex-v1",
        help="Snapshot name (default: synth-engine-bench-codex-v1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CREATING DAYTONA SNAPSHOT FOR ENGINE_BENCH")
    print("=" * 60)
    print()

    if not os.environ.get("DAYTONA_API_KEY"):
        print("ERROR: DAYTONA_API_KEY not set")
        sys.exit(1)

    from daytona_sdk import CreateSnapshotParams, Daytona, DaytonaConfig, Image

    api_key = os.environ["DAYTONA_API_KEY"]
    client = Daytona(DaytonaConfig(api_key=api_key))

    # Build image from Dockerfile
    dockerfile_path = Path(__file__).parent / "Dockerfile.daytona"
    print(f"[1/2] Building image from: {dockerfile_path}")
    print("      This includes:")
    print("        - Python 3.11")
    print("        - Rust/cargo (pre-compiled deps)")
    print("        - Node.js + Codex CLI")
    print("        - synth-ai SDK")
    print("        - engine-bench repo")
    print("      (This may take several minutes...)")
    print()

    image = Image.from_dockerfile(dockerfile_path)

    # Create snapshot
    snapshot_name = args.name
    print(f"[2/2] Creating snapshot: {snapshot_name}")

    params = CreateSnapshotParams(name=snapshot_name, image=image)

    def on_logs(chunk):
        print(chunk, end="", flush=True)

    try:
        client.snapshot.create(params, on_logs=on_logs)
        print()
        print()
        print("=" * 60)
        print("SUCCESS!")
        print(f"Snapshot '{snapshot_name}' created.")
        print()
        print("To use per-rollout Daytona sandboxes, run:")
        print()
        print(f"  USE_DAYTONA_SANDBOXES=1 DAYTONA_SNAPSHOT_NAME={snapshot_name} \\")
        print(
            "    uv run python demos/engine_bench/run_eval.py --local --seeds 10 --model codex-5.1-mini --agent codex"
        )
        print()
        print("Or update daytona_helper.py SNAPSHOT_NAME constant.")
        print("=" * 60)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
