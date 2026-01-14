#!/usr/bin/env python3
"""Build a verifier dataset from captured PTCG rollouts.

This script expects JSONL records written by the PTCG task app when
PTCG_TRACE_DIR is set. It computes heuristic labels and produces a
Graph Opt dataset for verifier optimization.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _compute_outcome_reward(result: Dict[str, Any]) -> float:
    winner = result.get("winner")
    if winner == "P1":
        return 1.0
    if winner == "P2":
        return 0.0
    p1_prizes = result.get("p1_prizes", 0)
    p2_prizes = result.get("p2_prizes", 0)
    if p1_prizes < p2_prizes:
        return 0.6
    if p1_prizes > p2_prizes:
        return 0.4
    return 0.5


def _score_rollout(record: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    result = record.get("result", {})
    trace_steps = record.get("trace_steps", []) or []
    decision_steps = result.get("decision_steps") or len(trace_steps) or 1

    def _count(predicate) -> int:
        return sum(1 for step in trace_steps if predicate(step))

    invalid_steps = _count(lambda s: not s.get("action_valid", True))
    attack_available = _count(lambda s: "DeclareAttack" in (s.get("available_actions") or []))
    attack_taken = _count(lambda s: s.get("action_type") == "DeclareAttack")
    energy_available = _count(lambda s: "AttachEnergy" in (s.get("available_actions") or []))
    energy_taken = _count(lambda s: s.get("action_type") == "AttachEnergy")
    evolve_taken = _count(lambda s: s.get("action_type") in {"ChooseActive", "ChooseBench"})
    basics_taken = _count(lambda s: s.get("action_type") == "PlayBasic")
    endturn_when_attack = _count(
        lambda s: s.get("action_type") == "EndTurn"
        and "DeclareAttack" in (s.get("available_actions") or [])
    )

    outcome_reward = float(record.get("outcome_reward") or _compute_outcome_reward(result))
    attack_rate = attack_taken / max(1, attack_available)
    energy_rate = energy_taken / max(1, energy_available)
    progress_rate = (attack_taken + energy_taken + evolve_taken + basics_taken) / max(1, decision_steps)
    stall_rate = endturn_when_attack / max(1, decision_steps)
    invalid_rate = invalid_steps / max(1, decision_steps)

    score = (
        0.6 * outcome_reward
        + 0.2 * attack_rate
        + 0.1 * energy_rate
        + 0.1 * progress_rate
        - 0.4 * invalid_rate
        - 0.2 * stall_rate
    )

    metrics = {
        "outcome_reward": outcome_reward,
        "attack_rate": attack_rate,
        "energy_rate": energy_rate,
        "progress_rate": progress_rate,
        "stall_rate": stall_rate,
        "invalid_rate": invalid_rate,
    }
    return _clamp(score), metrics


def _load_rollouts(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Rollout file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def _summarize_scores(scores: Iterable[float]) -> Dict[str, float]:
    values = list(scores)
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def build_dataset(records: List[Dict[str, Any]], drop_game_state: bool) -> Dict[str, Any]:
    tasks = []
    gold_outputs = []
    scores = []

    rubric_context = (
        "Score Pokemon TCG gameplay quality. Reward strong play: legal actions, "
        "attacking when available, attaching energy with intent, and developing the board. "
        "Penalize illegal actions, stalling, and missing obvious attacks."
    )

    for idx, record in enumerate(records):
        trace_steps = record.get("trace_steps", []) or []
        if drop_game_state:
            for step in trace_steps:
                step.pop("game_state", None)

        score, metrics = _score_rollout(record)
        scores.append(score)
        task_id = record.get("trace_id") or f"ptcg_trace_{idx:04d}"
        tasks.append(
            {
                "task_id": task_id,
                "input": {
                    "trace_steps": trace_steps,
                    "final_result": record.get("result", {}),
                    "metrics": metrics,
                    "rubric_guidance": rubric_context,
                },
            }
        )
        gold_outputs.append({"task_id": task_id, "output": {"score": score}, "score": score})

    summary = _summarize_scores(scores)
    print("Dataset summary:", json.dumps(summary, indent=2))

    return {
        "tasks": tasks,
        "gold_outputs": gold_outputs,
        "metadata": {
            "name": "ptcg_gameplay_verifier",
            "task_description": (
                "Score Pokemon TCG gameplay traces on a 0-1 scale. Use the trace to judge "
                "action legality, tempo, board development, and prize-taking intent."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "trace_steps": {"type": "array"},
                    "final_result": {"type": "object"},
                    "metrics": {"type": "object"},
                    "rubric_guidance": {"type": "string"},
                },
                "required": ["trace_steps", "final_result", "rubric_guidance"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "event_reviews": {"type": "array"},
                    "outcome_review": {
                        "type": "object",
                        "properties": {
                            "criteria": {"type": "object"},
                            "total": {"type": "number"},
                            "summary": {"type": "string"},
                        },
                        "required": ["criteria", "total"],
                    },
                    "event_totals": {"type": "array"},
                    "score": {"type": "number"},
                },
                "required": ["event_reviews", "outcome_review", "event_totals", "score"],
            },
            "output_config": {
                "format": "json",
                "strict": True,
                "extract_from": [
                    "parse_output_output",
                    "judge_gameplay_output",
                    "final_output",
                    "(root)",
                ],
                "schema": {
                    "type": "object",
                    "properties": {
                        "event_reviews": {"type": "array"},
                        "outcome_review": {
                            "type": "object",
                            "properties": {
                                "criteria": {"type": "object"},
                                "total": {"type": "number"},
                                "summary": {"type": "string"},
                            },
                            "required": ["criteria", "total"],
                        },
                        "event_totals": {"type": "array"},
                        "score": {"type": "number"},
                    },
                    "required": ["event_reviews", "outcome_review", "event_totals", "score"],
                },
            },
            "domain": "games",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PTCG verifier dataset")
    parser.add_argument(
        "--rollouts",
        type=str,
        default="demos/gepa_ptcg/artifacts/ptcg_rollouts.jsonl",
        help="Path to rollout JSONL file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="demos/gepa_ptcg/artifacts/ptcg_verifier_dataset.json",
        help="Path to write dataset JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rollouts")
    parser.add_argument(
        "--drop-game-state",
        action="store_true",
        help="Remove raw game_state strings from trace_steps",
    )
    args = parser.parse_args()

    rollout_path = Path(args.rollouts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_rollouts(rollout_path, limit=args.limit)
    dataset = build_dataset(records, drop_game_state=args.drop_game_state)

    out_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote dataset to {out_path}")


if __name__ == "__main__":
    main()
