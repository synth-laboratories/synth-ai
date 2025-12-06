"""Modular result saving for GEPA progress tracking.

Provides functions to save various result files from a GEPAProgressTracker.

## Output Files

- `candidates.json` - All candidates with stages, seed_info, token_usage, etc.
- `pareto_history.json` - Frontier evolution with frontier_scores, timestamps
- `summary.json` - Full analysis dict with all tracking data
- `seeds.json` - Aggregated seed manifest with query text and expected output
- `seed_analysis.json` - Baseline vs best comparison per seed
- `summary.txt` - Human-readable summary (optional)
- `raw_events.json` - Raw SSE events (optional, can be large)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tracker import GEPAProgressTracker


def save_results(
    tracker: GEPAProgressTracker,
    output_dir: str | Path,
    *,
    include_raw_events: bool = False,
    include_summary_txt: bool = True,
) -> dict[str, Path]:
    """Save all result files from tracker.

    Args:
        tracker: GEPAProgressTracker instance with tracking data
        output_dir: Directory to save files to
        include_raw_events: Whether to save raw SSE events (can be large)
        include_summary_txt: Whether to save human-readable summary

    Returns:
        Dict mapping file type to path of saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    # 1. Candidates JSON (with stages, seed_info, token_usage, etc.)
    files["candidates"] = save_candidates(tracker, output_dir)

    # 2. Pareto history JSON
    files["pareto_history"] = save_pareto_history(tracker, output_dir)

    # 3. Summary JSON (analysis dict)
    files["summary"] = save_summary_json(tracker, output_dir)

    # 4. Seeds JSON (aggregated seed manifest)
    files["seeds"] = save_seeds(tracker, output_dir)

    # 5. Seed analysis (disagreement between baseline and best)
    files["seed_analysis"] = save_seed_analysis(tracker, output_dir)

    # 6. Human-readable summary
    if include_summary_txt:
        files["summary_txt"] = save_summary_txt(tracker, output_dir)

    # 7. Raw events (optional, can be large)
    if include_raw_events:
        files["raw_events"] = save_raw_events(tracker, output_dir)

    return files


def save_candidates(tracker: GEPAProgressTracker, output_dir: Path) -> Path:
    """Save all candidates with their details.

    Includes first-class program structure (stages), seed_info, token_usage, etc.
    """
    filepath = output_dir / "candidates.json"

    # Get frontier IDs for marking pareto status
    frontier_ids = set(tracker.current_frontier)

    candidates_data = []
    for c in tracker.candidates:
        candidate_dict: dict[str, Any] = {
            "candidate_id": c.candidate_id,
            "accuracy": c.accuracy,
            "val_accuracy": c.val_accuracy,
            "train_accuracy": c.train_accuracy,
            "generation": c.generation,
            "parent_id": c.parent_id,
            "is_pareto": c.candidate_id in frontier_ids,
            "accepted": c.accepted,
            "mutation_type": c.mutation_type,
            "mutation_params": c.mutation_params,
            "instance_scores": c.instance_scores,
            "seeds_evaluated": c.seeds_evaluated,
            "prompt_summary": c.prompt_summary or c.get_prompt_summary(),
            "transformation": c.transformation,
            "timestamp": c.timestamp,
            "timestamp_ms": c.timestamp_ms,
        }

        # Add stages (first-class program structure)
        if c.stages:
            candidate_dict["stages"] = {
                stage_id: stage.to_dict()
                for stage_id, stage in c.stages.items()
            }

        # Add seed_scores [{seed, score}, ...]
        if c.seed_scores:
            candidate_dict["seed_scores"] = c.seed_scores

        # Add seed_info [{seed, query, expected}, ...]
        if c.seed_info:
            candidate_dict["seed_info"] = [s.to_dict() for s in c.seed_info]

        # Add token_usage
        if c.token_usage:
            candidate_dict["token_usage"] = c.token_usage.to_dict()

        # Add cost
        if c.cost_usd is not None:
            candidate_dict["cost_usd"] = c.cost_usd

        # Add evaluation duration
        if c.evaluation_duration_ms is not None:
            candidate_dict["evaluation_duration_ms"] = c.evaluation_duration_ms

        # Add minibatch scores
        if c.minibatch_scores:
            candidate_dict["minibatch_scores"] = c.minibatch_scores

        # Add skip reason
        if c.skip_reason:
            candidate_dict["skip_reason"] = c.skip_reason

        # Add rollout samples
        if c.rollout_sample:
            candidate_dict["rollout_sample"] = [
                {
                    "seed": r.seed,
                    "query": r.query,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "correct": r.correct,
                }
                for r in c.rollout_sample
            ]

        candidates_data.append(candidate_dict)

    # Sort by accuracy descending
    candidates_data.sort(key=lambda x: x.get("accuracy") or 0, reverse=True)

    with open(filepath, "w") as f:
        json.dump(candidates_data, f, indent=2)

    return filepath


def save_pareto_history(tracker: GEPAProgressTracker, output_dir: Path) -> Path:
    """Save pareto frontier evolution over time."""
    filepath = output_dir / "pareto_history.json"

    history = [
        {
            "timestamp": u.timestamp,
            "timestamp_ms": u.timestamp_ms,
            "added": u.added,
            "removed": u.removed,
            "frontier": u.frontier,
            "frontier_size": u.frontier_size,
            "frontier_scores": u.frontier_scores,
            "optimistic_score": u.optimistic_score,
            "baseline_score": u.baseline_score,
            "generation": u.generation,
        }
        for u in tracker.pareto_history
    ]

    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)

    return filepath


def save_summary_json(tracker: GEPAProgressTracker, output_dir: Path) -> Path:
    """Save full analysis dict as JSON."""
    filepath = output_dir / "summary.json"

    analysis = tracker.to_analysis_dict()
    analysis["saved_at"] = datetime.now().isoformat()

    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2)

    return filepath


def save_seeds(tracker: GEPAProgressTracker, output_dir: Path) -> Path:
    """Save aggregated seed manifest with query text and expected output.

    Aggregates seed_info from all candidates to build a complete seed dataset.
    Each seed appears once with its query text, expected output, and which
    candidates were evaluated on it.
    """
    filepath = output_dir / "seeds.json"

    # Aggregate seed_info from all candidates
    seeds_map: dict[int, dict[str, Any]] = {}

    for c in tracker.candidates:
        # From seed_info (preferred - has query text)
        for si in c.seed_info:
            seed = si.seed
            if seed not in seeds_map:
                seeds_map[seed] = {
                    "seed": seed,
                    "query": si.query,
                    "expected": si.expected,
                    "evaluated_by": [],
                }
            # Add this candidate to the list of evaluators
            seeds_map[seed]["evaluated_by"].append({
                "candidate_id": c.candidate_id,
                "score": si.score,
                "correct": si.correct,
            })

        # From rollout_sample (has query text)
        for rs in c.rollout_sample:
            seed = rs.seed
            if seed not in seeds_map:
                seeds_map[seed] = {
                    "seed": seed,
                    "query": rs.query,
                    "expected": rs.expected,
                    "evaluated_by": [],
                }
            elif not seeds_map[seed].get("query") and rs.query:
                # Fill in query if we have it
                seeds_map[seed]["query"] = rs.query
                seeds_map[seed]["expected"] = rs.expected

        # From seed_scores (has score but not query)
        for ss in c.seed_scores:
            seed = ss.get("seed", -1)
            if seed < 0:
                continue
            if seed not in seeds_map:
                seeds_map[seed] = {
                    "seed": seed,
                    "query": "",
                    "expected": "",
                    "evaluated_by": [],
                }

    # Convert to sorted list
    seeds_list = sorted(seeds_map.values(), key=lambda x: x["seed"])

    # Add summary stats
    output = {
        "total_seeds": len(seeds_list),
        "seeds_with_query": sum(1 for s in seeds_list if s.get("query")),
        "seeds": seeds_list,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def save_seed_analysis(tracker: GEPAProgressTracker, output_dir: Path) -> Path:
    """Save seed-level analysis (disagreement between baseline and best).

    Compares baseline and best candidate on per-seed basis to identify
    where they disagree and which performs better.
    """
    filepath = output_dir / "seed_analysis.json"

    # Get baseline instance scores
    baseline_scores = tracker.baseline.instance_scores if tracker.baseline else []

    # Get best candidate instance scores
    best_candidate = None
    best_accuracy = 0.0
    for c in tracker.candidates:
        acc = c.accuracy or 0.0
        if acc > best_accuracy:
            best_accuracy = acc
            best_candidate = c

    best_scores = best_candidate.instance_scores if best_candidate else []

    # Analyze disagreements
    analysis: dict[str, Any] = {
        "baseline_accuracy": tracker.baseline_score,
        "best_accuracy": tracker.best_score,
        "best_candidate_id": best_candidate.candidate_id if best_candidate else None,
        "num_baseline_seeds": len(baseline_scores),
        "num_best_seeds": len(best_scores),
        "disagreement_seeds": [],
        "baseline_wins": 0,
        "best_wins": 0,
    }

    # Compare only where we have both
    min_len = min(len(baseline_scores), len(best_scores))
    for i in range(min_len):
        baseline_pass = baseline_scores[i] > 0.5
        best_pass = best_scores[i] > 0.5

        if baseline_pass != best_pass:
            seed_info = {
                "seed_index": i,
                "baseline_score": baseline_scores[i],
                "best_score": best_scores[i],
                "baseline_pass": baseline_pass,
                "best_pass": best_pass,
                "winner": "baseline" if baseline_pass else "best",
            }
            analysis["disagreement_seeds"].append(seed_info)

            if baseline_pass:
                analysis["baseline_wins"] += 1
            else:
                analysis["best_wins"] += 1

    analysis["total_disagreements"] = len(analysis["disagreement_seeds"])
    analysis["net_improvement"] = analysis["best_wins"] - analysis["baseline_wins"]

    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2)

    return filepath


def save_summary_txt(tracker: GEPAProgressTracker, output_dir: Path) -> Path:
    """Save human-readable summary text file."""
    filepath = output_dir / "summary.txt"

    lines = [
        "=" * 80,
        f"GEPA Optimization Summary - {tracker.env_name}",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 80,
        "",
        "RESULTS",
        "-" * 40,
        f"Status: {tracker.progress.phase}",
        f"Finish Reason: {tracker.progress.finish_reason or 'N/A'}",
        "",
        "Scoring:",
        f"  Baseline:  {tracker.baseline_score:.2%}" if tracker.baseline_score else "  Baseline:  N/A",
        f"  Best:      {tracker.best_score:.2%}",
    ]

    if tracker.progress.lift is not None:
        lines.append(f"  Lift:      {tracker.progress.lift:+.2%}")

    lines.extend([
        "",
        "Stats:",
        f"  Rollouts:     {tracker.progress.rollouts_completed}",
        f"  Candidates:   {len(tracker.candidates)}",
        f"  Generations:  {tracker.progress.generations_completed}",
        f"  Frontier:     {len(tracker.current_frontier)}",
        f"  Time:         {tracker.progress.elapsed_seconds:.1f}s",
        "",
    ])

    # Pareto frontier candidates
    lines.extend([
        "PARETO FRONTIER",
        "-" * 40,
    ])

    frontier_candidates = tracker.get_pareto_candidates()
    frontier_candidates.sort(key=lambda c: c.accuracy or 0, reverse=True)

    for i, c in enumerate(frontier_candidates, 1):
        acc_str = f"{c.accuracy:.2%}" if c.accuracy else "N/A"
        lines.append(f"{i}. {c.candidate_id}: {acc_str} (gen {c.generation})")
        if c.prompt_summary:
            # Truncate long prompts
            summary = c.prompt_summary[:200] + "..." if len(c.prompt_summary) > 200 else c.prompt_summary
            lines.append(f"   {summary}")

    lines.extend(["", ""])

    # Top candidates
    lines.extend([
        "ALL CANDIDATES (sorted by accuracy)",
        "-" * 40,
    ])

    sorted_candidates = sorted(tracker.candidates, key=lambda c: c.accuracy or 0, reverse=True)

    for i, c in enumerate(sorted_candidates[:20], 1):  # Top 20
        acc_str = f"{c.accuracy:.2%}" if c.accuracy else "N/A"
        pareto_mark = "*" if c.candidate_id in tracker.current_frontier else " "
        lines.append(f"{pareto_mark}{i:2d}. {c.candidate_id:<20} acc={acc_str:<8} gen={c.generation}")

    if len(sorted_candidates) > 20:
        lines.append(f"   ... and {len(sorted_candidates) - 20} more")

    lines.extend(["", "=" * 80])

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


def save_raw_events(tracker: GEPAProgressTracker, output_dir: Path) -> Path:
    """Save raw SSE events (can be large)."""
    filepath = output_dir / "raw_events.json"

    with open(filepath, "w") as f:
        json.dump(tracker.raw_events, f, indent=2)

    return filepath
