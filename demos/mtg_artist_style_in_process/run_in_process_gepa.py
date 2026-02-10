#!/usr/bin/env python3
"""Run the MTG artist style in-process GEPA demo.

Usage:
    # Quick mode (~30-60s) for fast validation:
    uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py --quick

    # Full run (may take 10+ minutes):
    uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py

    # Custom timeout:
    uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py --timeout 1800
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from synth_ai.sdk import run_in_process_job_sync

from mtg_in_process_task_app import build_config

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MTG artist style in-process GEPA demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation run (~30-60s):
  %(prog)s --quick

  # Full optimization run:
  %(prog)s

  # Debug with verbose output:
  %(prog)s --quick --verbose
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "gepa_mtg_in_process.toml"),
        help="Path to GEPA config TOML",
    )
    parser.add_argument("--artist", type=str, default=None, help="Artist key override")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples override")
    parser.add_argument(
        "--customer-note",
        type=str,
        default=None,
        help="Customer style note override",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=None,
        help="Optional backend URL override (defaults to SYNTH_BACKEND_URL)",
    )
    parser.add_argument("--no-poll", action="store_true", help="Submit without polling")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: minimal settings for fast validation (~30-60s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Polling timeout in seconds (default: 120 for quick, infinite for full)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    # Quick mode: minimal viable settings for fast validation
    if args.quick:
        overrides.update(
            {
                # Minimal population
                "prompt_learning.gepa.population.initial_size": 1,
                "prompt_learning.gepa.population.num_generations": 1,
                "prompt_learning.gepa.population.children_per_generation": 1,
                # Minimal evaluation
                "prompt_learning.gepa.evaluation.seeds": [0, 1, 2],
                "prompt_learning.gepa.evaluation.validation_seeds": [3],
                # Minimal rollout budget
                "prompt_learning.gepa.rollout.budget": 2,
                # Use fewer examples
                "prompt_learning.env_config.max_examples": 3,
            }
        )

    # User-provided overrides
    if args.artist:
        overrides["prompt_learning.env_config.artist_key"] = args.artist
    if args.max_examples is not None:
        overrides["prompt_learning.env_config.max_examples"] = args.max_examples
    if args.customer_note:
        overrides["prompt_learning.env_config.customer_style_note"] = args.customer_note

    return overrides


def _on_status(status: Dict[str, Any]) -> None:
    """Callback for status updates during polling."""
    job_status = status.get("status", "unknown")
    progress = status.get("progress", {})
    current = progress.get("current", 0)
    total = progress.get("total", 0)
    best = status.get("best_score") or status.get("best_reward")

    parts = [f"[{job_status}]"]
    if total > 0:
        parts.append(f"progress={current}/{total}")
    if best is not None:
        parts.append(f"best_score={best:.4f}")

    print(f"  {' '.join(parts)}", file=sys.stderr)


def main() -> None:
    args = _parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config).expanduser().resolve()
    overrides = _build_overrides(args)

    # Determine timeout
    if args.timeout is not None:
        timeout = args.timeout
    elif args.quick:
        timeout = 120.0  # 2 minutes for quick mode
    else:
        timeout = float("inf")  # No limit for full run - wait as long as needed

    mode_str = "QUICK" if args.quick else "FULL"
    print(f"\n=== MTG In-Process GEPA Demo ({mode_str} mode) ===")
    print(f"Config: {config_path}")
    timeout_str = "infinite" if timeout == float("inf") else f"{timeout}s"
    print(f"Timeout: {timeout_str}")
    if overrides:
        print(f"Overrides: {json.dumps(overrides, indent=2)}")
    print()

    start_time = time.time()

    try:
        result = run_in_process_job_sync(
            job_type="prompt_learning",
            config_path=config_path,
            config_factory=build_config,
            backend_url=args.backend_url,
            overrides=overrides or None,
            poll=not args.no_poll,
            timeout=timeout,
            poll_interval=3.0 if args.quick else 5.0,
            on_status=_on_status if not args.no_poll else None,
            # cancel_on_timeout not yet supported by run_in_process_job
        )

        elapsed = time.time() - start_time
        status = result.status or {}

        print(f"\n=== Result (took {elapsed:.1f}s) ===")
        print(f"Job ID: {result.job_id}")
        print(f"Backend: {result.backend_url}")
        print(f"Task app URL: {result.task_app_url}")
        if result.task_app_worker_token:
            print("Worker token: [set]")

        job_status = status.get("status", str(status))
        print(f"Status: {job_status}")

        if isinstance(status, dict):
            best_score = status.get("best_score") or status.get("best_reward")
            if best_score is not None:
                print(f"Best score: {best_score:.4f}")

            # Show additional details in verbose mode
            if args.verbose:
                print(f"\nFull status: {json.dumps(status, indent=2, default=str)}")

        # Exit with appropriate code
        if job_status in ("completed", "success"):
            sys.exit(0)
        elif job_status in ("cancelled", "timeout"):
            print(f"\nJob did not complete: {job_status}", file=sys.stderr)
            sys.exit(1)
        elif job_status == "failed":
            error = status.get("error") or status.get("message", "Unknown error")
            print(f"\nJob failed: {error}", file=sys.stderr)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"\n=== Error (after {elapsed:.1f}s) ===", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
