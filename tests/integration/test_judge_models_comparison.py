"""
Integration test comparing judge performance across different models.

Tests multiple models (Groq, OpenAI variants) and records:
- Total evaluation time
- Pearson correlations (event and outcome)
- Semaphore wait statistics
- Average API call times

Results are saved to .txt files for easy review.
"""

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest


# Models to test
MODELS_TO_TEST = [
    {"provider": "groq", "model": "qwen/qwen3-32b", "name": "groq-qwen3-32b"},
    {"provider": "groq", "model": "openai/gpt-oss-120b", "name": "gpt-oss-120b"},
    {"provider": "groq", "model": "openai/gpt-oss-20b", "name": "gpt-oss-20b"},
    {"provider": "openai", "model": "gpt-5-nano", "name": "gpt-5-nano"},
]

LIMIT = 10
BACKEND_URL = "http://localhost:8000"
DB_PATH = "traces/v3/synth_ai.db"
RUBRIC_CONFIG = "examples/multi_step/configs/crafter_rl_stepwise_hosted_judge.toml"
OUTPUT_DIR = Path("tests/integration/judge_model_results")


def parse_output(output: str) -> dict[str, Any]:
    """Parse the evaluation output to extract metrics."""
    result = {
        "total_time_s": None,
        "traces_succeeded": None,
        "traces_total": None,
        "event_pearson": None,
        "outcome_pearson": None,
        "avg_api_time_ms": None,
        "avg_semaphore_wait_ms": None,
        "semaphore_wait_pct": None,
        "rate_limit_errors": 0,
        "p0_ms": None,
        "p50_ms": None,
        "p90_ms": None,
        "p99_ms": None,
        "p100_ms": None,
    }
    
    for line in output.split("\n"):
        # Parse total time
        # Format: "✅ Evaluation complete: 10/10 traces succeeded in 47.56s"
        if "Evaluation complete:" in line and "traces succeeded in" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                # Look for the X/Y pattern
                if "/" in part and i > 0:
                    try:
                        result["traces_succeeded"] = int(part.split("/")[0])
                        result["traces_total"] = int(part.split("/")[1])
                    except (ValueError, IndexError):
                        pass
                # Look for time after "in"
                if part == "in" and i + 1 < len(parts):
                    try:
                        time_str = parts[i + 1].rstrip("s")
                        result["total_time_s"] = float(time_str)
                    except (ValueError, IndexError):
                        pass
        
        # Parse pearson correlations
        if "[event_avg_vs_outcome] samples=" in line and "pearson=" in line:
            parts = line.split("pearson=")
            if len(parts) > 1:
                result["event_pearson"] = float(parts[1].strip())
        
        if "[outcome] samples=" in line and "pearson=" in line:
            parts = line.split("pearson=")
            if len(parts) > 1:
                result["outcome_pearson"] = float(parts[1].strip())
        
        # Parse averages
        if "Avg API call time per window:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                result["avg_api_time_ms"] = float(parts[1].strip().rstrip("ms"))
        
        if "Avg semaphore wait per window:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                wait_info = parts[1].strip()
                wait_ms = wait_info.split("ms")[0].strip()
                result["avg_semaphore_wait_ms"] = float(wait_ms)
                # Extract percentage
                if "(" in wait_info and "% of API time)" in wait_info:
                    pct_str = wait_info.split("(")[1].split("%")[0]
                    result["semaphore_wait_pct"] = float(pct_str)
        
        if "Rate limit errors (429):" in line:
            parts = line.split(":")
            if len(parts) > 1:
                result["rate_limit_errors"] = int(parts[1].strip())
        
        # Parse percentiles
        if "p0  (min):" in line:
            parts = line.split(":")
            if len(parts) > 1:
                result["p0_ms"] = float(parts[1].strip().rstrip("ms"))
        if "p50 (med):" in line:
            parts = line.split(":")
            if len(parts) > 1:
                result["p50_ms"] = float(parts[1].strip().rstrip("ms"))
        if "p90:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                result["p90_ms"] = float(parts[1].strip().rstrip("ms"))
        if "p99:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                result["p99_ms"] = float(parts[1].strip().rstrip("ms"))
        if "p100 (max):" in line:
            parts = line.split(":")
            if len(parts) > 1:
                result["p100_ms"] = float(parts[1].strip().rstrip("ms"))
    
    return result


async def run_evaluation_async(provider: str, model: str, name: str) -> tuple[dict[str, Any], str]:
    """Run evaluation for a specific model asynchronously and return parsed results + raw output."""
    print(f"🚀 Starting {name}...")
    
    cmd_str = (
        f"set -a; source .env; uv run python -m rubrics_dev.judge_eval "
        f"--db {DB_PATH} "
        f"--backend-url {BACKEND_URL} "
        f"--rubric-config {RUBRIC_CONFIG} "
        f"--provider {provider} "
        f"--model {model} "
        f'--api-key "$SYNTH_API_KEY" '
        f"--limit {LIMIT}"
    )
    
    start = time.perf_counter()
    
    process = await asyncio.create_subprocess_exec(
        "bash", "-c", cmd_str,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/Users/joshpurtell/Documents/GitHub/synth-ai"
    )
    
    stdout, stderr = await process.communicate()
    elapsed = time.perf_counter() - start
    
    stdout_str = stdout.decode('utf-8')
    stderr_str = stderr.decode('utf-8')
    
    if process.returncode != 0:
        print(f"❌ {name} failed with exit code {process.returncode}")
        print(f"STDERR: {stderr_str[:500]}")
        return {"error": stderr_str, "elapsed_s": elapsed, "model_name": name}, stdout_str + "\n\n" + stderr_str
    
    print(f"✅ {name} completed in {elapsed:.2f}s")
    
    # Parse the output
    metrics = parse_output(stdout_str)
    metrics["wall_clock_time_s"] = elapsed
    metrics["model_name"] = name
    
    return metrics, stdout_str


def save_results(name: str, metrics: dict[str, Any], raw_output: str):
    """Save results to .txt file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = OUTPUT_DIR / f"{name}_results.txt"
    
    with output_file.open("w") as f:
        f.write(f"Model: {name}\n")
        f.write("=" * 60 + "\n\n")
        
        # Check if this was a failed run
        if "error" in metrics:
            f.write("❌ EVALUATION FAILED\n")
            f.write("-" * 60 + "\n")
            f.write(f"Error: {metrics['error'][:500]}\n")
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("RAW OUTPUT:\n")
            f.write("=" * 60 + "\n")
            f.write(raw_output)
            return
        
        # Summary metrics
        f.write("SUMMARY METRICS:\n")
        f.write("-" * 60 + "\n")
        wall_clock = metrics.get('wall_clock_time_s')
        f.write(f"Total Wall Clock Time: {wall_clock:.2f}s\n" if isinstance(wall_clock, (int, float)) else f"Total Wall Clock Time: {wall_clock}\n")
        total_time = metrics.get('total_time_s')
        f.write(f"Evaluation Time: {total_time:.2f}s\n" if isinstance(total_time, (int, float)) else f"Evaluation Time: {total_time}\n")
        f.write(f"Traces Succeeded: {metrics.get('traces_succeeded', 'N/A')}/{metrics.get('traces_total', 'N/A')}\n")
        f.write("\n")
        
        # Pearson correlations
        f.write("PEARSON CORRELATIONS:\n")
        f.write("-" * 60 + "\n")
        event_pearson = metrics.get('event_pearson')
        outcome_pearson = metrics.get('outcome_pearson')
        f.write(f"Event Pearson: {event_pearson if event_pearson is not None else 'N/A'}\n")
        f.write(f"Outcome Pearson: {outcome_pearson if outcome_pearson is not None else 'N/A'}\n")
        f.write("\n")
        
        # Performance metrics
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Avg API Call Time: {metrics.get('avg_api_time_ms', 'N/A')}ms\n")
        f.write(f"Avg Semaphore Wait: {metrics.get('avg_semaphore_wait_ms', 'N/A')}ms "
                f"({metrics.get('semaphore_wait_pct', 'N/A')}% of API time)\n")
        f.write(f"Rate Limit Errors: {metrics.get('rate_limit_errors', 0)}\n")
        f.write("\n")
        
        # Percentiles
        f.write("COMPLETION TIME DISTRIBUTION:\n")
        f.write("-" * 60 + "\n")
        f.write(f"p0  (min): {metrics.get('p0_ms', 'N/A')}ms\n")
        f.write(f"p50 (med): {metrics.get('p50_ms', 'N/A')}ms\n")
        f.write(f"p90:       {metrics.get('p90_ms', 'N/A')}ms\n")
        f.write(f"p99:       {metrics.get('p99_ms', 'N/A')}ms\n")
        f.write(f"p100 (max): {metrics.get('p100_ms', 'N/A')}ms\n")
        f.write("\n")
        
        # Raw output
        f.write("=" * 60 + "\n")
        f.write("RAW OUTPUT:\n")
        f.write("=" * 60 + "\n")
        f.write(raw_output)
    
    print(f"📝 Results saved to {output_file}")


def save_comparison_summary(all_metrics: list[dict[str, Any]]):
    """Save a comparison summary across all models."""
    output_file = OUTPUT_DIR / "00_comparison_summary.txt"
    
    with output_file.open("w") as f:
        f.write("JUDGE MODEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Separate succeeded and failed
        succeeded = [m for m in all_metrics if "error" not in m]
        failed = [m for m in all_metrics if "error" in m]
        
        if failed:
            f.write(f"⚠️  {len(failed)} model(s) failed:\n")
            for m in failed:
                f.write(f"   - {m.get('model_name', 'unknown')}: {m.get('error', 'Unknown')[:80]}...\n")
            f.write("\n")
        
        if not succeeded:
            f.write("❌ All models failed!\n")
            return
        
        # Table header
        f.write(f"{'Model':<20} {'Time(s)':<10} {'EventAvg r':<12} {'Outcome r':<12} "
                f"{'API(ms)':<10} {'Wait%':<8} {'Med(ms)':<10} {'Max(ms)':<10} {'429s':<6}\n")
        f.write("-" * 120 + "\n")
        
        # Sort by total time (only succeeded models)
        sorted_metrics = sorted(succeeded, key=lambda x: x.get('total_time_s', float('inf')))
        
        for m in sorted_metrics:
            name = m.get('model_name', 'Unknown')[:19]
            time_s = m.get('total_time_s', 0.0)
            event_r = m.get('event_pearson')
            outcome_r = m.get('outcome_pearson')
            api_ms = m.get('avg_api_time_ms', 0.0)
            wait_pct = m.get('semaphore_wait_pct', 0.0)
            rate_429 = m.get('rate_limit_errors', 0)
            
            event_str = f"{event_r:.4f}" if event_r is not None else "N/A"
            outcome_str = f"{outcome_r:.4f}" if outcome_r is not None else "N/A"
            
            median_ms = m.get('p50_ms')
            max_ms = m.get('p100_ms')
            median_str = f"{median_ms:.0f}" if isinstance(median_ms, (int, float)) else "N/A"
            max_str = f"{max_ms:.0f}" if isinstance(max_ms, (int, float)) else "N/A"
            f.write(f"{name:<20} {time_s:<10.2f} {event_str:<12} {outcome_str:<12} "
                    f"{api_ms:<10.1f} {wait_pct:<8.1f} {median_str:<10} {max_str:<10} {rate_429:<6}\n")
        
        f.write("\n")
        
        # Winner analysis (only for succeeded models)
        if sorted_metrics:
            f.write("\nANALYSIS:\n")
            f.write("=" * 80 + "\n")
            
            fastest = min(sorted_metrics, key=lambda x: x.get('total_time_s', float('inf')))
            f.write(f"⚡ Fastest: {fastest.get('model_name')} ({fastest.get('total_time_s'):.2f}s)\n")
            
            # Best outcome correlation
            best_outcome = max(sorted_metrics, key=lambda x: x.get('outcome_pearson', -1.0) or -1.0)
            f.write(f"🎯 Best Outcome Correlation: {best_outcome.get('model_name')} "
                    f"(r={best_outcome.get('outcome_pearson', 0.0):.4f})\n")
            
            # Most efficient (lowest wait %)
            most_efficient = min(sorted_metrics, key=lambda x: x.get('semaphore_wait_pct', float('inf')))
            f.write(f"⚙️  Most Efficient: {most_efficient.get('model_name')} "
                    f"({most_efficient.get('semaphore_wait_pct', 0.0):.1f}% wait)\n")
            
            # Check for rate limits
            rate_limit_issues = [m for m in sorted_metrics if m.get('rate_limit_errors', 0) > 0]
            if rate_limit_issues:
                f.write(f"\n⚠️  Rate Limit Issues:\n")
                for m in rate_limit_issues:
                    f.write(f"   - {m.get('model_name')}: {m.get('rate_limit_errors')} errors\n")
            else:
                f.write(f"\n✅ No rate limit errors detected\n")
    
    print(f"\n📊 Comparison summary saved to {output_file}")


async def run_all_models_parallel():
    """Run all model evaluations in parallel."""
    print(f"\n🚀 Starting parallel evaluation of {len(MODELS_TO_TEST)} models...")
    print(f"{'='*60}\n")
    
    # Create tasks for all models
    tasks = [
        run_evaluation_async(
            provider=model_config["provider"],
            model=model_config["model"],
            name=model_config["name"]
        )
        for model_config in MODELS_TO_TEST
    ]
    
    # Run all in parallel
    results = await asyncio.gather(*tasks)
    
    return results


@pytest.mark.integration
@pytest.mark.asyncio
async def test_judge_model_comparison():
    """
    Integration test comparing judge performance across multiple models.
    
    Tests Groq and OpenAI models on the same set of traces and records:
    - Timing metrics
    - Pearson correlations
    - Semaphore efficiency
    
    All models are evaluated IN PARALLEL for maximum efficiency.
    
    Results are saved to .txt files in tests/integration/judge_model_results/
    """
    overall_start = time.perf_counter()
    
    # Run all models in parallel
    results = await run_all_models_parallel()
    
    overall_elapsed = time.perf_counter() - overall_start
    print(f"\n{'='*60}")
    print(f"⏱️  Total wall-clock time for all models: {overall_elapsed:.2f}s")
    print(f"{'='*60}\n")
    
    # Save individual results
    all_metrics = []
    for metrics, raw_output in results:
        all_metrics.append(metrics)
        model_name = metrics.get("model_name", "unknown")
        save_results(model_name, metrics, raw_output)

    # Print quick comparison to stdout
    succeeded_inline = [m for m in all_metrics if "error" not in m]
    if succeeded_inline:
        print("Model Summary (time & correlations):")
        print("-" * 90)
        print(f"{'Model':<20} {'Time(s)':<10} {'EventAvg r':<12} {'Outcome r':<10} {'Median(ms)':<12} {'Max(ms)':<10}")
        for m in sorted(succeeded_inline, key=lambda x: x.get("total_time_s", float("inf"))):
            name = m.get("model_name", "unknown")[:19]
            time_s = m.get("total_time_s", 0.0)
            event_r = m.get("event_pearson")
            outcome_r = m.get("outcome_pearson")
            event_str = f"{event_r:.4f}" if event_r is not None else "N/A"
            outcome_str = f"{outcome_r:.4f}" if outcome_r is not None else "N/A"
            median_ms = m.get("p50_ms")
            max_ms = m.get("p100_ms")
            median_str = f"{median_ms:.0f}" if isinstance(median_ms, (int, float)) else "N/A"
            max_str = f"{max_ms:.0f}" if isinstance(max_ms, (int, float)) else "N/A"
            print(f"{name:<20} {time_s:<10.2f} {event_str:<12} {outcome_str:<10} {median_str:<12} {max_str:<10}")
        print()
    
    # Save comparison summary
    save_comparison_summary(all_metrics)
    
    # Check for failures
    failed = [m for m in all_metrics if "error" in m]
    if failed:
        print(f"\n⚠️  Warning: {len(failed)} model(s) failed:")
        for m in failed:
            print(f"   - {m.get('model_name')}: {m.get('error', 'Unknown error')[:100]}")
        print("\n💡 Tip: If you see 'unsupported_model' errors, restart the backend to pick up new model configs.")
    
    # Only fail if ALL models failed
    succeeded = [m for m in all_metrics if "error" not in m]
    assert len(succeeded) > 0, "All models failed! Check backend logs and configuration."


if __name__ == "__main__":
    # Can run directly for manual testing
    asyncio.run(run_all_models_parallel())
