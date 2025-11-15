"""Track learning curves across all frameworks."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    """Single checkpoint in learning curve."""

    rollout_count: int
    performance: float
    checkpoint_pct: float


@dataclass
class LearningCurve:
    """Learning curve for a framework on a benchmark."""

    framework: str
    benchmark: str
    checkpoints: list[Checkpoint] = field(default_factory=list)
    total_rollouts: int = 0

    def record(
        self, rollout_count: int, performance: float, checkpoint_pct: float
    ) -> None:
        """Record a checkpoint.

        Args:
            rollout_count: Number of rollouts completed so far
            performance: Performance metric (e.g., accuracy)
            checkpoint_pct: Checkpoint percentage (0.0-1.0)
        """
        self.checkpoints.append(
            Checkpoint(
                rollout_count=rollout_count,
                performance=performance,
                checkpoint_pct=checkpoint_pct,
            )
        )
        self.total_rollouts = max(self.total_rollouts, rollout_count)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework,
            "benchmark": self.benchmark,
            "total_rollouts": self.total_rollouts,
            "checkpoints": [
                {
                    "rollout_count": cp.rollout_count,
                    "performance": cp.performance,
                    "checkpoint_pct": cp.checkpoint_pct,
                }
                for cp in self.checkpoints
            ],
        }

    def to_csv(self, output_path: Path) -> None:
        """Export to CSV format.

        Args:
            output_path: Path to CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "framework",
                    "benchmark",
                    "rollout_count",
                    "performance",
                    "checkpoint_pct",
                ]
            )
            for cp in self.checkpoints:
                writer.writerow(
                    [
                        self.framework,
                        self.benchmark,
                        cp.rollout_count,
                        cp.performance,
                        cp.checkpoint_pct,
                    ]
                )

    @classmethod
    def from_csv(cls, csv_path: Path) -> LearningCurve:
        """Load from CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            LearningCurve instance
        """
        checkpoints = []
        framework = ""
        benchmark = ""

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not framework:
                    framework = row["framework"]
                    benchmark = row["benchmark"]
                checkpoints.append(
                    Checkpoint(
                        rollout_count=int(row["rollout_count"]),
                        performance=float(row["performance"]),
                        checkpoint_pct=float(row["checkpoint_pct"]),
                    )
                )

        curve = cls(framework=framework, benchmark=benchmark, checkpoints=checkpoints)
        if checkpoints:
            curve.total_rollouts = max(cp.rollout_count for cp in checkpoints)
        return curve


class LearningCurveTracker:
    """Tracker for learning curves with checkpoint management."""

    def __init__(
        self,
        framework: str,
        benchmark: str,
        total_budget: int,
        checkpoints: list[float] | None = None,
    ):
        """Initialize learning curve tracker.

        Args:
            framework: Framework name (e.g., "synth_gepa")
            benchmark: Benchmark name (e.g., "iris")
            total_budget: Total rollout budget
            checkpoints: List of checkpoint percentages (default: [0.1, 0.25, 0.5, 0.75, 1.0])
        """
        self.framework = framework
        self.benchmark = benchmark
        self.total_budget = total_budget
        self.checkpoints = checkpoints or [0.1, 0.25, 0.5, 0.75, 1.0]
        self.curve = LearningCurve(framework=framework, benchmark=benchmark)
        self.rollout_count = 0
        self.performance_history: list[float] = []
        self._checkpoint_recorded: set[float] = set()

    def record_evaluation(self, performance: float) -> None:
        """Record a single evaluation.

        Args:
            performance: Performance metric (e.g., accuracy)
        """
        self.rollout_count += 1
        self.performance_history.append(performance)

        # Check if we've hit any checkpoints
        current_pct = self.rollout_count / self.total_budget if self.total_budget > 0 else 0.0

        for checkpoint_pct in self.checkpoints:
            if checkpoint_pct not in self._checkpoint_recorded:
                if current_pct >= checkpoint_pct:
                    self.curve.record(
                        rollout_count=self.rollout_count,
                        performance=performance,
                        checkpoint_pct=checkpoint_pct,
                    )
                    self._checkpoint_recorded.add(checkpoint_pct)

    def get_checkpoint_rollout(self, checkpoint_pct: float) -> int:
        """Get rollout count for a checkpoint percentage.

        Args:
            checkpoint_pct: Checkpoint percentage (0.0-1.0)

        Returns:
            Rollout count at checkpoint
        """
        return int(self.total_budget * checkpoint_pct)

    def save(self, output_dir: Path) -> None:
        """Save learning curve to files.

        Args:
            output_dir: Directory to save files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_dir / f"{self.framework}_{self.benchmark}_learning_curve.csv"
        self.curve.to_csv(csv_path)

        # Save JSON
        import json

        json_path = output_dir / f"{self.framework}_{self.benchmark}_learning_curve.json"
        with open(json_path, "w") as f:
            json.dump(self.curve.to_dict(), f, indent=2)

