#!/usr/bin/env python3
"""
Browse existing Crafter evaluations and launch viewer for a selected run.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import asyncio
from tabulate import tabulate

from src.synth_env.examples.crafter_classic.agent_demos.full_enchilada import (
    set_current_eval_dir,
    app,
)
from fastapi.staticfiles import StaticFiles
import uvicorn


def list_evaluations(evals_dir: Path = Path("src/evals/crafter")):
    """List all available evaluations with summary info."""
    if not evals_dir.exists():
        print(f"No evaluations found at {evals_dir}")
        return []

    evaluations = []
    for run_dir in sorted(evals_dir.glob("run_*"), reverse=True):
        if run_dir.is_dir():
            summary_file = run_dir / "evaluation_summary.json"
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    summary = json.load(f)

                eval_info = {
                    "run_id": run_dir.name,
                    "timestamp": summary["evaluation_metadata"]["timestamp"],
                    "models": ", ".join(summary["models_evaluated"]),
                    "difficulties": ", ".join(summary["difficulties_evaluated"]),
                    "num_trajectories": summary["evaluation_metadata"]["num_trajectories"],
                    "path": run_dir,
                }
                evaluations.append(eval_info)

    return evaluations


async def view_evaluation(eval_dir: Path):
    """Launch viewer for a specific evaluation."""
    if not eval_dir.exists():
        print(f"Evaluation directory not found: {eval_dir}")
        return

    viewer_dir = eval_dir / "viewer"
    if not viewer_dir.exists():
        print(f"Viewer files not found in {eval_dir}")
        return

    print(f"\n📁 Viewing evaluation: {eval_dir}")
    print("🌐 Launching viewer at http://localhost:8000")
    print("   Press Ctrl+C to stop the viewer")

    # Set the current eval directory for the viewer
    set_current_eval_dir(eval_dir)

    # Mount static files from the viewer directory
    app.mount("/", StaticFiles(directory=str(viewer_dir), html=True), name="viewer")

    # Run viewer
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    parser = argparse.ArgumentParser(description="Browse Crafter evaluations")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="src/evals/crafter",
        help="Base directory for evaluations",
    )
    parser.add_argument(
        "--run-id", type=str, help="Specific run ID to view (e.g., run_20240115_143022)"
    )
    parser.add_argument("--latest", action="store_true", help="View the latest evaluation")

    args = parser.parse_args()
    evals_dir = Path(args.eval_dir)

    # List evaluations
    evaluations = list_evaluations(evals_dir)

    if not evaluations:
        return

    # Display table of evaluations
    if not args.run_id and not args.latest:
        print("\n📊 Available Crafter Evaluations:")
        table_data = []
        for i, eval_info in enumerate(evaluations):
            # Parse timestamp for cleaner display
            try:
                ts = datetime.fromisoformat(eval_info["timestamp"])
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            except:
                ts_str = eval_info["timestamp"]

            table_data.append(
                [
                    i + 1,
                    eval_info["run_id"],
                    ts_str,
                    eval_info["models"],
                    eval_info["difficulties"],
                    eval_info["num_trajectories"],
                ]
            )

        headers = ["#", "Run ID", "Timestamp", "Models", "Difficulties", "Trajectories"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Ask user to select
        print("\nEnter the number of the evaluation to view (or 'q' to quit): ", end="")
        choice = input().strip()

        if choice.lower() == "q":
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(evaluations):
                selected_eval = evaluations[idx]
                await view_evaluation(selected_eval["path"])
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")

    # View specific run
    elif args.run_id:
        eval_path = evals_dir / args.run_id
        await view_evaluation(eval_path)

    # View latest
    elif args.latest and evaluations:
        latest_eval = evaluations[0]
        await view_evaluation(latest_eval["path"])


if __name__ == "__main__":
    asyncio.run(main())
