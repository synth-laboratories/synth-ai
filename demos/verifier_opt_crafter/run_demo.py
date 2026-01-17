#!/usr/bin/env python3
"""Run the Crafter Verifier Optimization demo end-to-end.

This demo optimizes a verifier graph that scores fixed Crafter traces using GraphGen.

Usage:
    uv run python demos/verifier_opt_crafter/run_demo.py
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import nest_asyncio
from synth_ai.core.paths import REPO_ROOT
from synth_ai.core.urls import BACKEND_URL_BASE, backend_health_url
from synth_ai.sdk.api.train.graphgen import GraphGenJob
from synth_ai.sdk.api.train.graphgen_models import (
    GraphGenGoldOutput,
    GraphGenTask,
    GraphGenTaskSet,
    GraphGenTaskSetMetadata,
    GraphGenVerifierConfig,
)
from synth_ai.sdk.auth import get_or_mint_synth_api_key

parser = argparse.ArgumentParser(description="Run Crafter Verifier Optimization demo")
parser.add_argument(
    "--dataset-path",
    type=str,
    default=None,
    help="Path to Crafter trace dataset (JSON or JSONL). If not provided, will look for default locations.",
)
parser.add_argument(
    "--max-traces",
    type=int,
    default=30,
    help="Maximum number of traces to use for optimization (default: 30)",
)
parser.add_argument(
    "--generations",
    type=int,
    default=3,
    help="Number of generations for optimization (default: 3)",
)
parser.add_argument(
    "--children",
    type=int,
    default=3,
    help="Children per generation (default: 3)",
)
parser.add_argument(
    "--rollout-budget",
    type=int,
    default=100,
    help="Rollout budget for optimization (default: 100)",
)
args = parser.parse_args()

# Repo root (demos/verifier_opt_crafter/run_demo.py -> demos -> repo)
repo_root = REPO_ROOT

nest_asyncio.apply()

# Backend configuration
SYNTH_API_BASE = BACKEND_URL_BASE

print(f"Backend: {SYNTH_API_BASE}")

# Check backend health
try:
    r = httpx.get(backend_health_url(SYNTH_API_BASE), timeout=30)
    if r.status_code == 200:
        print(f"Backend health: {r.json()}")
    else:
        print(f"WARNING: Backend returned status {r.status_code}")
except Exception as e:
    print(f"WARNING: Could not check backend health: {e}")

# Get API Key
API_KEY = get_or_mint_synth_api_key(backend_url=SYNTH_API_BASE)
print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")


def _find_default_dataset() -> Optional[Path]:
    """Find default Crafter trace dataset locations."""
    # Common locations relative to repo root
    candidates = [
        Path(repo_root) / "demos" / "verifier_opt_crafter" / "crafter_judge_adas_dataset.json",
        Path(repo_root)
        / "demos"
        / "verifier_opt_crafter"
        / "crafter_verifier_graph_opt_dataset.json",
        Path(repo_root)
        / "demos"
        / "verifier_opt_crafter"
        / "data"
        / "crafter_judge_adas_dataset.json",
        Path(repo_root)
        / "demos"
        / "verifier_opt_crafter"
        / "data"
        / "crafter_verifier_graph_opt_dataset.json",
    ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    return None


def _load_adas_dataset(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load an ADAS-style dataset with tasks + gold_outputs.

    Returns:
        Tuple of (traces, gold_outputs_by_id) where gold_outputs_by_id maps task_id -> full gold output
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks") or []
    gold_outputs = data.get("gold_outputs") or []
    gold_by_task = {g.get("task_id"): g.get("output") for g in gold_outputs if isinstance(g, dict)}

    traces: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        trace_id = t.get("id")
        inp = t.get("input") or {}
        trace = inp.get("trace")
        gold = gold_by_task.get(trace_id) or {}
        gold_score = gold.get("score")
        if trace_id and trace is not None and gold_score is not None:
            traces.append(
                {
                    "trace_id": trace_id,
                    "trace": trace,
                    "gold_score": float(gold_score),
                    "gold_event_rewards": gold.get("event_rewards", []),
                    "gold_outcome_feedback": gold.get("outcome_feedback", ""),
                    "gold_event_feedback": gold.get("event_feedback", ""),
                }
            )
    return traces, gold_by_task


def _load_jsonl_traces(path: Path) -> List[Dict[str, Any]]:
    """Load traces from JSONL file."""
    traces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(json.loads(line))
    return traces


def _truncate_trace_for_context(trace: Any, max_items: int = 10, max_chars: int = 2000) -> Any:
    """Recursively truncate heavy trace structures and long strings.

    Crafter traces contain huge arrays and very long string content that can crash LLM contexts.
    """
    if isinstance(trace, list):
        if len(trace) > max_items:
            return [
                _truncate_trace_for_context(item, max_items, max_chars)
                for item in trace[:max_items]
            ] + [{"__truncated__": f"... {len(trace) - max_items} more items ..."}]
        return [_truncate_trace_for_context(item, max_items, max_chars) for item in trace]
    elif isinstance(trace, dict):
        return {k: _truncate_trace_for_context(v, max_items, max_chars) for k, v in trace.items()}
    elif isinstance(trace, str):
        if len(trace) > max_chars:
            return trace[:max_chars] + f"... [truncated {len(trace) - max_chars} chars]"
        return trace
    return trace


def _split_train_val(
    traces: List[Dict[str, Any]], train_ratio: float = 0.8
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split traces into train/validation sets."""
    split_idx = int(len(traces) * train_ratio)
    return traces[:split_idx], traces[split_idx:]


def _build_graphgen_dataset(traces: List[Dict[str, Any]]) -> GraphGenTaskSet:
    """Build GraphGen dataset from traces."""
    tasks = []
    gold_outputs = []

    for trace in traces:
        trace_id = trace.get("trace_id", f"trace_{len(tasks)}")
        # Truncate trace for context efficiency
        truncated_trace = _truncate_trace_for_context(
            trace.get("trace", {}), max_items=10, max_chars=2000
        )

        tasks.append(
            GraphGenTask(
                id=trace_id,
                input={
                    "trace": truncated_trace,
                    "trace_id": trace_id,
                },
            )
        )

        gold_outputs.append(
            GraphGenGoldOutput(
                task_id=trace_id,
                output={
                    "score": trace.get("gold_score", 0.0),
                    "event_rewards": trace.get("gold_event_rewards", []),
                    "outcome_feedback": trace.get("gold_outcome_feedback", ""),
                    "event_feedback": trace.get("gold_event_feedback", ""),
                },
            )
        )

    # Define schemas for the verifier
    input_schema = {
        "type": "object",
        "properties": {
            "trace": {
                "type": "object",
                "description": "Crafter v3/v4 trace with event_history or session_time_steps, metadata, etc.",
            },
            "trace_id": {"type": "string"},
        },
        "required": ["trace"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number", "description": "Final score [0, 1]"},
            "outcome_feedback": {"type": "string", "description": "Summary of entire episode"},
            "event_rewards": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Score for each event",
            },
            "event_feedback": {"type": "string", "description": "Detailed event-level feedback"},
        },
        "required": ["score", "outcome_feedback", "event_rewards", "event_feedback"],
    }

    return GraphGenTaskSet(
        metadata=GraphGenTaskSetMetadata(
            name="crafter_verifier_optimization",
            description="""Evaluate Crafter agent execution traces.

You are building a VERIFIER GRAPH that scores how well an agent played Crafter.
The input is a v3/v4 trace capturing:
- Per-turn observations (inventory, position, achievements)
- Actions taken by the agent
- Event rewards from the environment
- Termination status

Your verifier should output:
{
    "score": float 0-1,          # Overall quality score
    "outcome_feedback": str,      # Summary of entire episode
    "event_rewards": list[float], # One per decision/event
    "event_feedback": str         # Detailed event-level feedback
}

Primary objective: predict a score that correlates with the gold_score label.
Secondary objective: produce actionable feedback aligned with the score.""",
            input_schema=input_schema,  # Put schemas in metadata (backend expects them here)
            output_schema=output_schema,
        ),
        tasks=tasks,
        gold_outputs=gold_outputs,
        verifier_config=GraphGenVerifierConfig(
            mode="gold_examples",  # Use gold examples mode since we have gold scores
            model="gpt-4o-mini",
            provider="openai",
        ),
        input_schema=input_schema,  # Also at top level for backward compatibility
        output_schema=output_schema,
    )


async def main():
    """Main demo function."""

    # Timing helper
    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"

    timings: dict[str, float] = {}
    total_start = time.time()

    # Load traces
    print("\n" + "=" * 60)
    print("LOADING CRAFTER TRACES")
    print("=" * 60)

    dataset_path = None
    if args.dataset_path:
        dataset_path = Path(args.dataset_path).expanduser()
        if not dataset_path.exists():
            print(f"ERROR: Dataset path not found: {dataset_path}")
            sys.exit(1)
    else:
        dataset_path = _find_default_dataset()
        if not dataset_path:
            print("ERROR: Could not find default dataset. Please provide --dataset-path")
            print("\nExpected locations:")
            print("  - demos/verifier_opt_crafter/crafter_judge_adas_dataset.json")
            print("  - demos/verifier_opt_crafter/crafter_verifier_graph_opt_dataset.json")
            print("  - demos/verifier_opt_crafter/data/crafter_verifier_graph_opt_dataset.json")
            sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")

    traces: List[Dict[str, Any]] = []
    gold_outputs: Dict[str, Dict[str, Any]] = {}

    if dataset_path.suffix == ".json":
        traces, gold_outputs = _load_adas_dataset(dataset_path)
        print(f"Loaded {len(traces)} traces from ADAS dataset")
    elif dataset_path.suffix == ".jsonl":
        traces = _load_jsonl_traces(dataset_path)
        # Build minimal gold_outputs from traces
        gold_outputs = {
            t.get("trace_id", f"trace_{i}"): {"score": t.get("gold_score", 0.0)}
            for i, t in enumerate(traces)
        }
        print(f"Loaded {len(traces)} traces from JSONL")
    else:
        print(f"ERROR: Unsupported file format: {dataset_path.suffix}")
        sys.exit(1)

    if not traces:
        print("ERROR: No traces loaded")
        sys.exit(1)

    # Limit traces if requested
    if args.max_traces and len(traces) > args.max_traces:
        traces = traces[: args.max_traces]
        trace_ids = {t.get("trace_id") for t in traces}
        gold_outputs = {k: v for k, v in gold_outputs.items() if k in trace_ids}
        print(f"Limited to {len(traces)} traces")

    # Split train/val
    train_traces, val_traces = _split_train_val(traces, train_ratio=0.8)
    print(f"Train: {len(train_traces)} traces, Val: {len(val_traces)} traces")

    # Build GraphGen dataset
    print("\n" + "=" * 60)
    print("BUILDING GRAPHGEN DATASET")
    print("=" * 60)
    dataset = _build_graphgen_dataset(train_traces)
    print(f"Dataset built: {len(dataset.tasks)} tasks, {len(dataset.gold_outputs)} gold outputs")

    # Run GraphGen optimization
    print("\n" + "=" * 60)
    print("RUNNING VERIFIER GRAPH OPTIMIZATION")
    print("=" * 60)

    optimization_start = time.time()

    # Create GraphGen job for verifier optimization
    job = GraphGenJob.from_dataset(
        dataset=dataset,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
        graph_type="verifier",
        policy_models=["gpt-4o-mini"],  # Policy models for graph execution
        judge_model="gpt-4o-mini",  # Judge/verifier model for scoring traces
        rollout_budget=args.rollout_budget,
        proposer_effort="medium",
        num_generations=args.generations,
        population_size=max(
            2, args.children
        ),  # GraphGen uses population_size instead of children_per_generation
    )

    try:
        job_id = job.submit()
        print(f"GraphGen Job ID: {job_id.graphgen_job_id}")
        print(f"Graph Evolve Job ID: {job_id.graph_evolve_job_id}")
    except Exception as e:
        if hasattr(e, "errors"):
            print("\n" + "=" * 60)
            print("VALIDATION ERRORS")
            print("=" * 60)
            for error in e.errors:
                print(f"  {error.get('field', 'unknown')}: {error.get('error', 'unknown error')}")
                if "suggestion" in error:
                    print(f"    Suggestion: {error['suggestion']}")
        else:
            print(f"\nERROR: {e}")
        raise

    # Stream events and wait for completion
    print("\nStreaming optimization progress...")
    print("-" * 60)

    # Track generation timings
    generation_timings: Dict[int, Dict[str, Any]] = {}
    current_generation: Optional[int] = None
    generation_start_time: Optional[float] = None
    eval_count = 0
    initial_graph_time: Optional[float] = None
    initial_validation_time: Optional[float] = None
    holdout_start_time: Optional[float] = None

    def on_event_callback(event: Dict[str, Any]) -> None:
        nonlocal current_generation, generation_start_time, eval_count
        nonlocal initial_graph_time, initial_validation_time, holdout_start_time

        event_type = event.get("type", "")

        if event_type == "graph_evolve.initial_graph_generated":
            initial_graph_time = time.time()
            print(
                f"  [TIMING] Initial graph generated: {format_duration(initial_graph_time - optimization_start)}"
            )

        elif event_type == "graph_evolve.initial_validation":
            initial_validation_time = time.time()
            if initial_graph_time:
                print(
                    f"  [TIMING] Initial validation: {format_duration(initial_validation_time - initial_graph_time)}"
                )

        elif event_type == "graph_evolve.generation_started":
            gen_num = event.get("generation", len(generation_timings) + 1)
            current_generation = gen_num
            generation_start_time = time.time()
            eval_count = 0
            generation_timings[gen_num] = {"start": generation_start_time, "evals": 0}
            print(f"  [TIMING] Generation {gen_num} started")

        elif event_type == "graph_evolve.candidate_evaluated":
            eval_count += 1
            if current_generation and current_generation in generation_timings:
                generation_timings[current_generation]["evals"] = eval_count

        elif event_type == "graph_evolve.generation_completed":
            gen_num = event.get("generation", current_generation)
            if gen_num and gen_num in generation_timings:
                end_time = time.time()
                start = generation_timings[gen_num]["start"]
                duration = end_time - start
                evals = generation_timings[gen_num]["evals"]
                generation_timings[gen_num]["duration"] = duration
                generation_timings[gen_num]["end"] = end_time
                print(
                    f"  [TIMING] Generation {gen_num} completed: {format_duration(duration)} ({evals} evals)"
                )

        elif event_type == "graph_evolve.holdout_evaluation":
            holdout_start_time = time.time()
            print("  [TIMING] Holdout evaluation started")

    # Import handlers
    from synth_ai.sdk.streaming import CallbackHandler, GraphGenHandler

    try:
        # Stream until completion with timing callback
        # Use both GraphGenHandler (for CLI output) and CallbackHandler (for timing)
        timing_handler = CallbackHandler(on_event=on_event_callback)
        job.stream_until_complete(
            timeout=3600.0,
            handlers=[GraphGenHandler(), timing_handler],
        )
        timings["optimization"] = time.time() - optimization_start

        # Print holdout timing if we have it
        if holdout_start_time:
            holdout_duration = time.time() - holdout_start_time
            print(f"  [TIMING] Holdout evaluation: {format_duration(holdout_duration)}")

        # Get final result with best_snapshot_id (stream_until_complete may not include it)
        result = job.get_status()

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Best Score: {result.get('best_score', 'N/A')}")
        print(f"Best Snapshot ID: {result.get('best_snapshot_id', 'N/A')}")
        print(f"Duration: {format_duration(timings['optimization'])}")

        # Print generation timing summary
        if generation_timings:
            print("\nGeneration Timing Summary:")
            for gen_num in sorted(generation_timings.keys()):
                gen_data = generation_timings[gen_num]
                duration = gen_data.get("duration", 0)
                evals = gen_data.get("evals", 0)
                print(
                    f"  Gen {gen_num}: {format_duration(duration)} ({evals} candidates evaluated)"
                )

        if result.get("status") not in ("succeeded", "completed"):
            print(f"ERROR: Optimization failed: {result.get('error', 'Unknown error')}")
            return

    except Exception as e:
        print(f"\nERROR during optimization: {e}")
        import traceback

        traceback.print_exc()
        return

    # Validation: Display holdout evaluation results from backend
    print("\n" + "=" * 60)
    print("HOLDOUT EVALUATION RESULTS")
    print("=" * 60)

    # The backend runs holdout evaluation automatically using val_seeds (20% holdout)
    # Results are included in the job status
    train_score = result.get("best_score")
    val_score = result.get("val_score")
    best_snapshot_id = result.get("best_snapshot_id")

    print(f"Train Score: {train_score if train_score is not None else 'N/A'}")
    print(
        f"Validation Score: {val_score if val_score is not None else 'N/A (no val_seeds configured)'}"
    )
    print(f"Best Snapshot ID: {best_snapshot_id or 'N/A'}")

    if train_score is not None and val_score is not None:
        if val_score < train_score * 0.8:
            drop_pct = 100 * (train_score - val_score) / train_score if train_score > 0 else 0
            print(f"⚠️  Potential overfitting: {drop_pct:.1f}% drop from train to val")
        else:
            print("✓ Validation score within acceptable range")
    elif val_score is None:
        print("Note: Backend did not return val_score - check if val_seeds were configured")

    # Run inference on validation traces and show output vs gold
    print("\n" + "=" * 60)
    print("VALIDATION TRACE COMPARISONS (Output vs Gold)")
    print("=" * 60)

    for i, trace in enumerate(val_traces[:3]):  # Show first 3 val traces
        trace_id = trace.get("trace_id", f"trace_{i}")
        print(f"\n{'=' * 60}")
        print(f"Trace: {trace_id}")
        print(f"{'=' * 60}")

        # Get gold outputs for this trace
        gold = gold_outputs.get(trace_id, {})
        gold_score = gold.get("score", trace.get("gold_score", "N/A"))
        gold_outcome = gold.get("outcome_feedback", trace.get("gold_outcome_feedback", "N/A"))
        gold_event_fb = gold.get("event_feedback", trace.get("gold_event_feedback", "N/A"))
        gold_event_rewards = gold.get("event_rewards", trace.get("gold_event_rewards", []))

        # Run inference with optimized graph
        try:
            truncated_trace = _truncate_trace_for_context(
                trace.get("trace", {}), max_items=10, max_chars=2000
            )
            verifier_result = job.run_inference(
                input_data={
                    "trace": truncated_trace,
                    "trace_id": trace_id,
                }
            )
            output = verifier_result.get("output", {})

            # Show full predicted output
            print("\n--- PREDICTED OUTPUT ---")
            print(json.dumps(output, indent=2, default=str))

            # Show full gold output
            print("\n--- GOLD OUTPUT ---")
            gold_full = {
                "score": gold_score,
                "outcome_feedback": gold_outcome,
                "event_feedback": gold_event_fb,
                "event_rewards": gold_event_rewards,
            }
            print(json.dumps(gold_full, indent=2, default=str))

            # Score comparison
            print("\n--- COMPARISON ---")
            if isinstance(output.get("score"), (int, float)) and isinstance(
                gold_score, (int, float)
            ):
                diff = abs(output["score"] - gold_score)
                print(f"Predicted score: {output['score']:.3f}")
                print(f"Gold score:      {gold_score:.3f}")
                print(f"Difference:      {diff:.3f} {'✓' if diff < 0.1 else '⚠️'}")
            else:
                print(f"Predicted score: {output.get('score', 'N/A')}")
                print(f"Gold score:      {gold_score}")

        except Exception as e:
            print(f"  ERROR running inference: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Optimization:")
    print(f"  Train Score: {result.get('best_score', 'N/A')}")
    print(f"  Val Score: {result.get('val_score', 'N/A')}")
    print(f"  Duration: {format_duration(timings['optimization'])}")
    print(f"  Generations: {args.generations}")
    print(f"  Children/gen: {args.children}")
    print(f"  Rollout budget: {args.rollout_budget}")

    timings["total"] = time.time() - total_start
    print(f"\nTotal time: {format_duration(timings['total'])}")

    # Print the optimized graph
    print("\n" + "=" * 60)
    print("OPTIMIZED GRAPH")
    print("=" * 60)
    try:
        graph_txt = job.download_graph_txt()
        print(graph_txt)
    except Exception as e:
        print(f"Failed to download graph: {e}")
        # Try to get the graph record as fallback
        try:
            graph_record = job.get_graph_record()
            print(json.dumps(graph_record, indent=2, default=str))
        except Exception as e2:
            print(f"Failed to get graph record: {e2}")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
