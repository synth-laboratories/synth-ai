"""Helpers for persisting trials and trace metadata gleaned from job outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from .models import Experiment, ExperimentJob, Trial, TrialStatus
from .results import ResultSummary


def persist_trials_from_summary(
    session: Session,
    job: ExperimentJob,
    summary: ResultSummary,
) -> list[Trial]:
    """Create Trial rows from learning curve checkpoints."""
    trials: list[Trial] = []
    if not summary.learning_curve_points:
        return trials

    now = datetime.now(UTC)
    for idx, point in enumerate(summary.learning_curve_points, start=1):
        metadata = {
            "rollout_count": point.rollout_count,
            "checkpoint_pct": point.checkpoint_pct,
            **point.metadata,
        }
        rewards = {}
        if point.performance is not None:
            rewards["performance"] = point.performance

        trial = Trial(
            trial_id=str(uuid4()),
            experiment_id=job.experiment_id,
            job_id=job.job_id,
            trial_number=idx,
            system_name=metadata.get("system_name") or f"{job.job_type}_trial_{idx}",
            reward=point.performance,
            aggregate_score=point.performance,
            rewards=rewards or None,
            metadata=metadata,
            status=TrialStatus.COMPLETED,
            completed_at=now,
        )
        session.add(trial)
        trials.append(trial)

    return trials


def update_experiment_metadata(
    experiment: Experiment,
    summary: ResultSummary,
) -> None:
    """Attach aggregate metrics from results to the experiment metadata."""
    experiment.metadata_json.setdefault("aggregate", {})
    aggregate = experiment.metadata_json["aggregate"]
    aggregate["best_score"] = summary.best_score
    aggregate["baseline_score"] = summary.baseline_score
    aggregate["total_rollouts"] = summary.total_rollouts
    aggregate["total_time"] = summary.total_time
    aggregate["artifacts"] = summary.artifacts
