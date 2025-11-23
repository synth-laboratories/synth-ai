#!/usr/bin/env python3
"""Integration test script for MIPRO with proxy models.

This script emulates the monorepo test_mipro_banking77.py but uses the synth-ai SDK
to build and submit jobs to the local backend.

Run with:
    REDIS_URL=redis://127.0.0.1:6379/0 \
    uv run python -m tests.integration.prompt_learning.test_mipro_proxy_local \
      --config path/to/banking77_mipro_lowrisk_proxy.toml

Or use the default config from monorepo:
    REDIS_URL=redis://127.0.0.1:6379/0 \
    uv run python -m tests.integration.prompt_learning.test_mipro_proxy_local
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Add monorepo backend to path for config loading
# Try multiple possible paths
MONOREPO_BACKEND = None
for parent_count in [5, 6, 7]:
    candidate = Path(__file__).resolve().parents[parent_count] / "monorepo" / "backend"
    if candidate.exists():
        MONOREPO_BACKEND = candidate
        sys.path.insert(0, str(candidate))
        break

# Also try relative to synth-ai root
if MONOREPO_BACKEND is None:
    synth_ai_root = Path(__file__).resolve().parents[3]
    candidate = synth_ai_root.parent / "monorepo" / "backend"
    if candidate.exists():
        MONOREPO_BACKEND = candidate
        sys.path.insert(0, str(candidate))

from synth_ai.api.train.prompt_learning import PromptLearningJob


def _fmt_float(value: float, precision: int = 4) -> str:
    """Format float with specified precision."""
    return f"{value:.{precision}f}"


def _fmt_int(value: int) -> str:
    """Format integer with thousands separator."""
    return f"{value:,}"


def _print_metrics_table(rows: Sequence[Tuple[str, str]]) -> None:
    """Pretty-print a simple metrics table."""
    if not rows:
        return
    col_width = max(len(label) for label, _ in rows)
    print(f"{'Metric'.ljust(col_width)}  Value")
    print("-" * (col_width + 7))
    for label, value in rows:
        print(f"{label.ljust(col_width)}  {value}")


def main():
    parser = argparse.ArgumentParser(description="Run MIPRO optimization with proxy models")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (defaults to monorepo banking77_mipro_lowrisk_proxy.toml)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll for job completion (default: False)",
    )
    args = parser.parse_args()

    # Find config file
    if args.config:
        config_path = Path(args.config)
    else:
        # Default to monorepo config
        if MONOREPO_BACKEND and MONOREPO_BACKEND.exists():
            config_path = (
                MONOREPO_BACKEND
                / "app"
                / "routes"
                / "prompt_learning"
                / "configs"
                / "banking77_mipro_lowrisk_proxy.toml"
            )
        else:
            # Try relative to synth-ai root
            synth_ai_root = Path(__file__).resolve().parents[3]
            config_path = (
                synth_ai_root.parent
                / "monorepo"
                / "backend"
                / "app"
                / "routes"
                / "prompt_learning"
                / "configs"
                / "banking77_mipro_lowrisk_proxy.toml"
            )

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Please specify --config or ensure monorepo config exists")
        sys.exit(1)

    print("=" * 80)
    print("MIPRO Test Run: Banking77 (Proxy Models)")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Backend: {args.backend}")
    print("=" * 80)
    print()

    # Create job using synth-ai SDK
    # For local backend, use dummy API keys if not set
    import os
    api_key = os.environ.get("SYNTH_API_KEY", "local-test-key")
    task_app_api_key = os.environ.get("ENVIRONMENT_API_KEY", "local-test-key")
    
    print("Creating prompt learning job...")
    try:
        job = PromptLearningJob.from_config(
            config_path=config_path,
            backend_url=args.backend,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
            overrides={"backend": args.backend},
        )
    except Exception as e:
        print(f"ERROR: Failed to create job: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print(f"✅ Job created successfully")
    
    # Build payload to get task URL and verify config
    build_result = job._build_payload()
    print(f"   Task URL: {build_result.task_url}")
    
    # Verify config includes proxy models
    config_body = build_result.payload["config_body"]
    pl_config = config_body.get("prompt_learning", {})
    mipro_config = pl_config.get("mipro", {})
    print(f"   Algorithm: {pl_config.get('algorithm', 'unknown')}")
    print()

    proxy_models = pl_config.get("proxy_models") or mipro_config.get("proxy_models")
    adaptive_pool = mipro_config.get("adaptive_pool")

    print("Configuration verification:")
    if proxy_models:
        print(f"   ✅ Proxy models: {proxy_models.get('lo_model')} (LO) / {proxy_models.get('hi_model')} (HI)")
    else:
        print("   ⚠️  Proxy models: Not configured")
    if adaptive_pool:
        level = adaptive_pool.get("level", "UNKNOWN")
        print(f"   ✅ Adaptive pool: {level}")
    else:
        print("   ⚠️  Adaptive pool: Not configured")

    # MIPRO-specific config
    num_iterations = mipro_config.get("num_iterations")
    num_evaluations = mipro_config.get("num_evaluations_per_iteration")
    batch_size = mipro_config.get("batch_size")
    if num_iterations:
        print(f"   ✅ Iterations: {num_iterations}")
    if num_evaluations:
        print(f"   ✅ Evaluations per iteration: {num_evaluations}")
    if batch_size:
        print(f"   ✅ Batch size: {batch_size}")
    print()

    # Submit job to backend
    print("Submitting job to backend...")
    start_time = time.time()

    try:
        job_id = job.submit()
        if not job_id:
            print(f"ERROR: No job_id returned from submit()")
            sys.exit(1)

        print(f"✅ Job submitted: {job_id}")
        print()

        if args.poll:
            print("Polling for job completion...")
            print("=" * 80)

            # Poll until complete
            try:
                results = job.poll_until_complete(timeout=3600.0, interval=5.0)
                final_status_value = "completed"

            except Exception as poll_error:
                print(f"⚠️  Polling failed or timed out: {poll_error}")
                final_status_value = "unknown"
                results = {}

            print()
            print("=" * 80)
            if final_status_value == "completed":
                print("✅ Job Completed Successfully!")
            elif final_status_value == "failed":
                print("❌ Job Failed")
            else:
                print(f"⚠️  Job Status: {final_status_value}")

            # Extract metrics from results
            if results:
                print()
                print("Job Results:")
                print("-" * 80)

                # Extract metrics if available
                metrics_rows: list[Tuple[str, str]] = []

                # Basic job info
                metrics_rows.append(("Job ID", job_id))
                metrics_rows.append(("Status", final_status_value))
                metrics_rows.append(("Total Time", f"{time.time() - start_time:.1f}s"))

                # Try to extract optimization metrics from results
                if isinstance(results, dict):
                    best_score = results.get("best_score") or results.get("final_score")
                    baseline_score = results.get("baseline_score")
                    if best_score is not None:
                        metrics_rows.append(("Best Score", _fmt_float(best_score, 4)))
                    if baseline_score is not None:
                        metrics_rows.append(("Baseline Score", _fmt_float(baseline_score, 4)))
                        if best_score is not None:
                            lift = best_score - baseline_score
                            metrics_rows.append(("Lift", f"{lift:+.4f}"))

                    # Proxy model stats
                    proxy_stats = results.get("proxy_stats") or results.get("proxy_models")
                    if proxy_stats:
                        hi_count = proxy_stats.get("hi_count", 0)
                        lo_count = proxy_stats.get("lo_count", 0)
                        both_count = proxy_stats.get("both_count", 0)
                        total_decisions = hi_count + lo_count + both_count
                        if total_decisions > 0:
                            metrics_rows.append(
                                (
                                    "Proxy Decisions",
                                    f"HI: {hi_count}, LO: {lo_count}, BOTH: {both_count}",
                                )
                            )
                        net_gain = proxy_stats.get("net_gain_usd")
                        if net_gain is not None:
                            metrics_rows.append(("Proxy Net Gain", f"${net_gain:.4f}"))

                        # Correlation stats
                        r2 = proxy_stats.get("r2")
                        pearson = proxy_stats.get("pearson")
                        if r2 is not None:
                            metrics_rows.append(("Proxy R²", _fmt_float(r2, 4)))
                        if pearson is not None:
                            metrics_rows.append(("Proxy Pearson", _fmt_float(pearson, 4)))

                    # Adaptive pool stats
                    adaptive_pool_stats = results.get("adaptive_pool_stats")
                    if adaptive_pool_stats:
                        net_gain = adaptive_pool_stats.get("net_gain_usd")
                        if net_gain is not None:
                            metrics_rows.append(("Adaptive Pool Net Gain", f"${net_gain:.4f}"))

                    # Cost stats
                    cost_stats = results.get("cost_stats") or results.get("costs")
                    if cost_stats:
                        total_cost = cost_stats.get("total_cost_usd")
                        rollout_cost = cost_stats.get("rollout_cost_usd")
                        proposal_cost = cost_stats.get("proposal_cost_usd")
                        if total_cost is not None:
                            metrics_rows.append(("Total Cost", f"${total_cost:.4f}"))
                        if rollout_cost is not None:
                            metrics_rows.append(("Rollout Cost", f"${rollout_cost:.4f}"))
                        if proposal_cost is not None:
                            metrics_rows.append(("Proposal Cost", f"${proposal_cost:.4f}"))

                if metrics_rows:
                    print()
                    _print_metrics_table(metrics_rows)
                    print()

                # Print full results if available
                print("Full Results:")
                print("-" * 80)
                import json

                print(json.dumps(results, indent=2, default=str))
            else:
                print("No results available")

        else:
            print(f"Job submitted. Use --poll to wait for completion.")
            print(f"Check status with: curl {args.backend}/api/prompt-learning/online/jobs/{job_id}")

    except Exception as e:
        print(f"ERROR: Failed to submit job: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

