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
import os
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

# Load environment variables from .env files if available
try:
    from dotenv import load_dotenv
    
    # Try synth-ai/.env
    synth_ai_root = Path(__file__).resolve().parents[3]
    synth_ai_env = synth_ai_root / ".env"
    if synth_ai_env.exists():
        load_dotenv(synth_ai_env, override=False)
    
    # Try synth-ai/examples/rl/.env
    rl_env = synth_ai_root / "examples" / "rl" / ".env"
    if rl_env.exists():
        load_dotenv(rl_env, override=False)
        
except ImportError:
    pass  # dotenv not available, skip
except Exception:
    pass  # Best effort

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
    # Load API keys from environment (loaded from .env files above)
    api_key = os.environ.get("SYNTH_API_KEY")
    task_app_api_key = os.environ.get("ENVIRONMENT_API_KEY")
    
    if not api_key:
        print("WARNING: SYNTH_API_KEY not found in environment. Using dummy key.")
        api_key = "local-test-key"
    if not task_app_api_key:
        print("WARNING: ENVIRONMENT_API_KEY not found in environment. Using dummy key.")
        task_app_api_key = "local-test-key"
    
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
                print("=" * 80)
                print("Metrics Report")
                print("-" * 80)

                # Extract metrics if available
                metrics_rows: list[Tuple[str, str]] = []

                # Basic job info
                elapsed_time = time.time() - start_time
                metrics_rows.append(("Total Time", f"{elapsed_time:.1f}s"))

                # Try to extract optimization metrics from results
                if isinstance(results, dict):
                    best_score = results.get("best_score") or results.get("final_score")
                    baseline_score = results.get("baseline_score")
                    
                    if baseline_score is not None:
                        metrics_rows.append(("Baseline Score", _fmt_float(baseline_score, 4)))
                    if best_score is not None:
                        metrics_rows.append(("Final Score", _fmt_float(best_score, 4)))
                        if baseline_score is not None:
                            lift = best_score - baseline_score
                            metrics_rows.append(("Lift", f"{lift:+.4f}"))

                    # Proxy model stats
                    proxy_stats = results.get("proxy_stats") or results.get("proxy_models") or results.get("proxy")
                    if proxy_stats:
                        hi_count = proxy_stats.get("hi_count", proxy_stats.get("total_hi", 0))
                        lo_count = proxy_stats.get("lo_count", proxy_stats.get("total_lo", 0))
                        both_count = proxy_stats.get("both_count", proxy_stats.get("total_both", 0))
                        total_decisions = hi_count + lo_count + both_count
                        if total_decisions > 0:
                            hi_pct = (hi_count / total_decisions) * 100
                            lo_pct = (lo_count / total_decisions) * 100
                            both_pct = (both_count / total_decisions) * 100
                            metrics_rows.append(
                                (
                                    "Proxy Seed Decisions",
                                    f"HI-only: {hi_count}/{total_decisions} ({hi_pct:.1f}%), "
                                    f"LO-only: {lo_count}/{total_decisions} ({lo_pct:.1f}%), "
                                    f"BOTH: {both_count}/{total_decisions} ({both_pct:.1f}%)",
                                )
                            )
                            metrics_rows.append(
                                (
                                    "Proxy Decisions",
                                    f"HI-only: {hi_count}, LO-only: {lo_count}, BOTH: {both_count} (total: {total_decisions})",
                                )
                            )
                        
                        # Proxy evaluations
                        hi_evals = proxy_stats.get("hi_evals", hi_count + both_count)
                        lo_evals = proxy_stats.get("lo_evals", lo_count + both_count)
                        metrics_rows.append(
                            (
                                "Proxy Evaluations",
                                f"HI evals: {hi_evals}, LO evals: {lo_evals} (BOTH counts as both)",
                            )
                        )
                        
                        net_gain = proxy_stats.get("net_gain_usd") or proxy_stats.get("total_net_gain_usd")
                        if net_gain is not None:
                            if net_gain >= 0:
                                metrics_rows.append(("Proxy Net Gain (USD)", f"${net_gain:.4f} (saved)"))
                            else:
                                metrics_rows.append(("Proxy Net Gain (USD)", f"-${abs(net_gain):.4f} (cost)"))

                        # Correlation stats
                        corr_stats = proxy_stats.get("correlation") or {}
                        r2 = corr_stats.get("r2") or proxy_stats.get("r2")
                        pearson = corr_stats.get("pearson") or corr_stats.get("correlation") or proxy_stats.get("pearson")
                        rmse = corr_stats.get("rmse") or proxy_stats.get("rmse")
                        n_pairs = corr_stats.get("n_pairs") or proxy_stats.get("n_pairs")
                        mean_lo = corr_stats.get("mean_lo") or proxy_stats.get("mean_lo")
                        mean_hi = corr_stats.get("mean_hi") or proxy_stats.get("mean_hi")
                        
                        if r2 is not None:
                            metrics_rows.append(("Proxy Correlation (R²)", _fmt_float(r2, 4)))
                        if pearson is not None:
                            metrics_rows.append(("Proxy Correlation (Pearson)", _fmt_float(pearson, 4)))
                        if rmse is not None:
                            metrics_rows.append(("Proxy Correlation (RMSE)", _fmt_float(rmse, 4)))
                        if n_pairs is not None:
                            metrics_rows.append(("Proxy Correlation Pairs", str(n_pairs)))
                        if mean_lo is not None and mean_hi is not None:
                            metrics_rows.append(("Proxy Score Means", f"LO: {mean_lo:.3f}, HI: {mean_hi:.3f}"))

                    # Adaptive pool stats
                    adaptive_pool_stats = results.get("adaptive_pool_stats") or results.get("adaptive_pool")
                    if adaptive_pool_stats:
                        net_gain = adaptive_pool_stats.get("net_gain_usd") or adaptive_pool_stats.get("total_net_gain_usd")
                        if net_gain is not None:
                            if net_gain >= 0:
                                metrics_rows.append(("Adaptive Pool Net Gain (USD)", f"${net_gain:.4f} (saved)"))
                            else:
                                metrics_rows.append(("Adaptive Pool Net Gain (USD)", f"-${abs(net_gain):.4f} (cost)"))
                        
                        effective_size = adaptive_pool_stats.get("effective_size")
                        if effective_size is not None:
                            metrics_rows.append(("Adaptive Pool Effective Size", f"{effective_size:.1f} seeds"))
                        pool_init_size = adaptive_pool_stats.get("pool_init_size")
                        pool_final_size = adaptive_pool_stats.get("pool_final_size")
                        if pool_init_size is not None:
                            metrics_rows.append(("Adaptive Pool Initial Size", f"{pool_init_size} seeds"))
                        if pool_final_size is not None:
                            metrics_rows.append(("Adaptive Pool Final Size", f"{pool_final_size} seeds"))

                    # Cost stats
                    cost_stats = results.get("cost_stats") or results.get("costs") or results.get("cost")
                    if cost_stats:
                        total_cost = cost_stats.get("total_cost_usd") or cost_stats.get("total")
                        rollout_cost = cost_stats.get("rollout_cost_usd") or cost_stats.get("rollout")
                        proposal_cost = cost_stats.get("proposal_cost_usd") or cost_stats.get("proposal")
                        if total_cost is not None:
                            metrics_rows.append(("Total Cost (USD)", f"${total_cost:.4f}"))
                        if rollout_cost is not None:
                            metrics_rows.append(("Rollout Cost (USD)", f"${rollout_cost:.4f}"))
                        if proposal_cost is not None:
                            metrics_rows.append(("Proposal Cost (USD)", f"${proposal_cost:.4f}"))
                    
                    # Token stats
                    token_stats = results.get("token_stats") or results.get("tokens")
                    if token_stats:
                        total_tokens = token_stats.get("total") or token_stats.get("total_tokens")
                        hi_tokens = token_stats.get("hi") or token_stats.get("hi_tokens")
                        lo_tokens = token_stats.get("lo") or token_stats.get("lo_tokens")
                        if total_tokens is not None:
                            metrics_rows.append(("Rollout Tokens (Total)", _fmt_int(total_tokens)))
                        if hi_tokens is not None:
                            metrics_rows.append(("Rollout Tokens (HI)", _fmt_int(hi_tokens)))
                        if lo_tokens is not None:
                            metrics_rows.append(("Rollout Tokens (LO/Proxy)", _fmt_int(lo_tokens)))
                    
                    # Total net gain
                    total_net_gain = results.get("total_net_gain_usd") or results.get("net_gain")
                    if total_net_gain is not None:
                        if total_net_gain >= 0:
                            metrics_rows.append(("Total Net Gain (USD)", f"${total_net_gain:.4f} (saved)"))
                        else:
                            metrics_rows.append(("Total Net Gain (USD)", f"-${abs(total_net_gain):.4f} (cost)"))

                if metrics_rows:
                    print()
                    _print_metrics_table(metrics_rows)
                    print()
                else:
                    print("No metrics available in results")
                    print()
                
                # Print summary
                print("=" * 80)
                print(f"Job ID: {job_id}")
                print(f"Status: {final_status_value}")
                print("=" * 80)

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

