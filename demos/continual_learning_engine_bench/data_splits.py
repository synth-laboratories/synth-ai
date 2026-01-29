#!/usr/bin/env python3
"""
Difficulty-based splits for EngineBench (Pokemon TCG) continual learning.

We compute a heuristic difficulty score per card instance and create two splits:
- easy: lower half by difficulty score
- hard: upper half by difficulty score

This lets us simulate a distribution shift from easy to hard cards.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ENGINE_BENCH_DIR = Path(
    os.getenv("ENGINE_BENCH_DIR", str(Path.home() / ".cache" / "engine-bench"))
).expanduser()
INSTANCES_DIR = ENGINE_BENCH_DIR / "data" / "instances" / "single"


def ensure_engine_bench_repo() -> None:
    """Ensure engine-bench data is available locally."""
    if not INSTANCES_DIR.exists():
        raise FileNotFoundError(
            f"EngineBench instances not found at {INSTANCES_DIR}. "
            "Set ENGINE_BENCH_DIR or run the EngineBench task app once to populate the cache."
        )


def load_instance(instance_id: str) -> dict:
    """Load instance spec from local EngineBench repo."""
    instance_path = INSTANCES_DIR / f"{instance_id}.json"
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance not found: {instance_path}")
    return json.loads(instance_path.read_text())


def load_instance_ids() -> List[str]:
    """Return sorted instance IDs from disk."""
    ensure_engine_bench_repo()
    return sorted([p.stem for p in INSTANCES_DIR.glob("*.json")])


DIFFICULTY_SPLITS = ("easy", "hard")


def _count_text(items: List[dict], key: str = "text") -> int:
    return sum(1 for item in items if str(item.get(key, "")).strip())


def compute_difficulty_score(instance: dict) -> int:
    """Compute a simple heuristic difficulty score for a card instance."""
    tests = instance.get("tests", []) or []
    cards = instance.get("cards", []) or []

    abilities = sum(len(card.get("abilities", []) or []) for card in cards)
    attacks = sum(len(card.get("attacks", []) or []) for card in cards)
    ability_text = sum(_count_text(card.get("abilities", []) or []) for card in cards)
    attack_text = sum(_count_text(card.get("attacks", []) or []) for card in cards)

    stage_bonus = 0
    for card in cards:
        stage = str(card.get("stage", "")).lower()
        if stage in ("stage1", "stage2", "stage 1", "stage 2"):
            stage_bonus += 1

    # Weighted heuristic: tests and abilities matter more than raw counts.
    return int(len(tests) * 2 + abilities * 2 + attacks + ability_text + attack_text + stage_bonus)


@dataclass(frozen=True)
class DifficultySplit:
    name: str
    instance_ids: List[str]
    threshold: int


class EngineBenchDifficultyDataset:
    """EngineBench dataset with difficulty-based splits (easy/hard)."""

    def __init__(self) -> None:
        ensure_engine_bench_repo()
        self._instances: Dict[str, dict] = {}
        self._scores: Dict[str, int] = {}
        self._splits: Dict[str, DifficultySplit] = {}
        instance_ids = load_instance_ids()

        for instance_id in instance_ids:
            instance = load_instance(instance_id)
            self._instances[instance_id] = instance
            self._scores[instance_id] = compute_difficulty_score(instance)

        self._build_splits()

    def _build_splits(self) -> None:
        scored: List[Tuple[str, int]] = sorted(
            self._scores.items(), key=lambda item: (item[1], item[0])
        )
        if not scored:
            raise RuntimeError("No EngineBench instances found.")

        cutoff = len(scored) // 2
        if cutoff == 0:
            cutoff = 1

        easy_ids = [instance_id for instance_id, _ in scored[:cutoff]]
        hard_ids = [instance_id for instance_id, _ in scored[cutoff:]]

        easy_threshold = scored[cutoff - 1][1]
        hard_threshold = scored[cutoff][1] if cutoff < len(scored) else easy_threshold

        self._splits["easy"] = DifficultySplit(
            name="easy", instance_ids=easy_ids, threshold=easy_threshold
        )
        self._splits["hard"] = DifficultySplit(
            name="hard", instance_ids=hard_ids, threshold=hard_threshold
        )

    def split_ids(self, split_name: str) -> List[str]:
        if split_name not in self._splits:
            raise ValueError(f"Unknown split: {split_name}. Use {DIFFICULTY_SPLITS}.")
        return list(self._splits[split_name].instance_ids)

    def split_size(self, split_name: str) -> int:
        return len(self.split_ids(split_name))

    def score(self, instance_id: str) -> int:
        return self._scores[instance_id]

    def sample(self, *, split: str, index: int) -> dict:
        """Return a sample dict for a given split and index."""
        ids = self.split_ids(split)
        if not ids:
            raise ValueError(f"No instances available for split: {split}")
        idx = index % len(ids)
        instance_id = ids[idx]
        instance = self._instances[instance_id]
        return {
            "index": idx,
            "split": split,
            "instance_id": instance_id,
            "difficulty_score": self._scores[instance_id],
            "card_name": instance.get("name", instance_id),
        }

    def split_stats(self, split: str) -> dict:
        ids = self.split_ids(split)
        scores = [self._scores[i] for i in ids]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        threshold = self._splits[split].threshold
        return {
            "count": len(ids),
            "avg_score": avg_score,
            "threshold": threshold,
        }


def print_split_info() -> None:
    dataset = EngineBenchDifficultyDataset()
    print("EngineBench Difficulty Splits")
    print("=" * 60)
    for split in DIFFICULTY_SPLITS:
        stats = dataset.split_stats(split)
        print(
            f"{split.upper():>4}: {stats['count']} instances | "
            f"avg score: {stats['avg_score']:.2f} | threshold: {stats['threshold']}"
        )


if __name__ == "__main__":
    print_split_info()
